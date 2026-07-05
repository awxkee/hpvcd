/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#![deny(unreachable_pub)]
mod bitreader;
mod cabac;
mod color;
mod config;
mod deblock;
mod decode;
mod decoder;
mod error;
mod fmt;
mod heif;
mod info;
mod intra;
mod limits;
mod metadata;
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon;
mod plane;
mod reconstruct;
mod sao;
#[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
mod sse;
mod threadpool;
mod transform;
mod wpp;
mod yuv;

pub use color::{Cicp, ColorMetadata, MatrixCoefficients, Primaries, TransferFunction};
pub use decoder::Decoder;
pub use error::DecodeError;
pub use fmt::{BitDepth, ChromaFormat, ImageBuffer, SampleBuf};
pub use info::{ImageInfo, read_heic_info, read_heic_info_with_limits};
pub use limits::ParseLimits;
pub use metadata::{CleanAperture, ContentLightLevel, Metadata, Orientation, PixelAspectRatio};

/// Convert a decoded u16 YUV plane to the appropriate typed buffer.
/// 8-bit images produce `SampleBuf::U8` (direct cast, no precision loss).
fn plane_to_buf(plane: Vec<u16>, bd: BitDepth) -> SampleBuf {
    if bd == BitDepth::Eight {
        SampleBuf::U8(plane.into_iter().map(|v| v as u8).collect())
    } else {
        SampleBuf::U16(plane)
    }
}

#[derive(Clone, Copy, Default)]
struct HevcVisibleCrop {
    left: usize,
    top: usize,
}

#[inline]
fn visible_crop_from_hvcc(hvcc: &[u8]) -> HevcVisibleCrop {
    config::parse_hvcc_full(hvcc)
        .ok()
        .map(|(sps, _)| HevcVisibleCrop {
            left: sps.crop_left as usize,
            top: sps.crop_top as usize,
        })
        .unwrap_or_default()
}

fn copy_plane_window(
    src: &[u16],
    src_w: usize,
    src_h: usize,
    crop_left: usize,
    crop_top: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; dst_w * dst_h];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 || src.is_empty() {
        return out;
    }

    let rows_in_src = (src.len() / src_w).min(src_h);
    if rows_in_src == 0 {
        return out;
    }

    let src_y0 = crop_top.min(rows_in_src - 1);
    let src_x0 = crop_left.min(src_w - 1);

    for (r, dst_row) in out.chunks_exact_mut(dst_w).enumerate() {
        let src_y = (src_y0 + r).min(rows_in_src - 1);
        let src_base = src_y * src_w;
        let available = (src.len() - src_base).min(src_w);
        if available == 0 {
            continue;
        }

        let src_x = src_x0.min(available - 1);
        let copy_w = dst_w.min(available - src_x);
        let (dst_copy, dst_edge) = dst_row.split_at_mut(copy_w);
        dst_copy.copy_from_slice(&src[src_base + src_x..src_base + src_x + copy_w]);
        if !dst_edge.is_empty() {
            dst_edge.fill(src[src_base + src_x + copy_w - 1]);
        }
    }

    out
}

fn copy_visible_yuv_planes(
    planes: &yuv::YuvPlanes,
    dw: usize,
    dh: usize,
    crop: HevcVisibleCrop,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let y = copy_plane_window(
        &planes.y,
        planes.width,
        planes.height,
        crop.left,
        crop.top,
        dw,
        dh,
    );

    if planes.chroma.is_monochrome() {
        return (y, Vec::new(), Vec::new());
    }

    let sub_w = planes.chroma.sub_w();
    let sub_h = planes.chroma.sub_h();
    let cw = dw.div_ceil(sub_w);
    let ch = dh.div_ceil(sub_h);
    let coded_cw = planes.width.div_ceil(sub_w);
    let coded_ch = planes.height.div_ceil(sub_h);
    let crop_cx = crop.left / sub_w;
    let crop_cy = crop.top / sub_h;

    let cb = copy_plane_window(&planes.cb, coded_cw, coded_ch, crop_cx, crop_cy, cw, ch);
    let cr = copy_plane_window(&planes.cr, coded_cw, coded_ch, crop_cx, crop_cy, cw, ch);
    (y, cb, cr)
}

/// A fully decoded HEIF/HEIC image.
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB pixels, typed to the source bit depth.
    pub pixels: ImageBuffer,
    pub alpha: Option<SampleBuf>,
    pub bit_depth: BitDepth,
    pub color: ColorMetadata,
    pub orientation: Orientation,
    pub content_light_level: Option<ContentLightLevel>,
    /// Clean aperture (`clap` property) read from the item properties, if present.
    pub clean_aperture: Option<CleanAperture>,
    /// Pixel aspect ratio (`pasp` property). `None` means assume 1:1 (square pixels).
    pub pixel_aspect_ratio: Option<PixelAspectRatio>,
    pub exif: Option<Vec<u8>>,
}

/// Raw YCbCr planes from a HEIF/HEIC file; no color conversion applied.
pub struct DecodedYuv {
    pub y: SampleBuf,
    pub cb: SampleBuf,
    pub cr: SampleBuf,
    /// Optional alpha plane (luma-only), decoded from the `auxl` auxiliary item.
    /// Same dimensions and bit depth as the luma plane.
    pub alpha: Option<SampleBuf>,
    /// Luma plane width (display-cropped; equals coded width minus conformance window).
    pub width: u32,
    /// Luma plane height.
    pub height: u32,
    pub bit_depth: BitDepth,
    pub chroma: ChromaFormat,
    pub color: ColorMetadata,
    pub orientation: Orientation,
    /// Clean aperture (`clap` property) read from the item properties, if present.
    pub clean_aperture: Option<CleanAperture>,
    /// Pixel aspect ratio (`pasp` property). `None` means assume 1:1 (square pixels).
    pub pixel_aspect_ratio: Option<PixelAspectRatio>,
    pub exif: Option<Vec<u8>>,
}

/// Decode the optional alpha auxiliary item (`auxl`) into a luma-only plane,
/// applying the alpha HEVC conformance-window offset and returning exactly
/// `dw*dh` display samples. `clap` is intentionally not applied here.
fn decode_alpha_plane(
    file: &[u8],
    heif: &heif::HeifFile,
    dw: usize,
    dh: usize,
) -> Option<SampleBuf> {
    let a = heif.alpha.as_ref()?;
    if a.hvcc.is_empty() {
        return None;
    }
    let astart = a.data_offset as usize;
    let aend = astart.checked_add(a.data_length as usize)?;
    if aend > file.len() {
        return None;
    }
    decode_hevc_item(&file[astart..aend], &a.hvcc, None)
        .ok()
        .map(|(ap, _)| {
            let crop = visible_crop_from_hvcc(&a.hvcc);
            let plane = copy_plane_window(&ap.y, ap.width, ap.height, crop.left, crop.top, dw, dh);
            plane_to_buf(plane, ap.bit_depth)
        })
}

/// Decode a HEIF/HEIC file and return raw YCbCr planes (no color conversion),
/// using a default [`Decoder`]. For a display-ready 8-bit image use
/// [`decode_heic_rgb8`].
pub fn decode_heic_yuv(file: &[u8]) -> Result<DecodedYuv, DecodeError> {
    Decoder::default().decode_yuv(file)
}

pub(crate) fn decode_heic_yuv_with(
    decoder: &Decoder,
    file: &[u8],
) -> Result<DecodedYuv, DecodeError> {
    let heif = heif::parse(file, decoder.limits())?;

    if let Some(grid) = &heif.grid {
        return decode_grid_yuv(decoder, file, grid, &heif);
    }

    // Single-tile path
    let start = heif.primary.data_offset as usize;
    let Some(end) = start.checked_add(heif.primary.data_length as usize) else {
        return Err(DecodeError::Bitstream(
            "image data offset/length overflows usize".into(),
        ));
    };
    if end > file.len() {
        return Err(DecodeError::Bitstream(
            "image data extends past file end".into(),
        ));
    }
    let (planes, _) =
        decode_hevc_item(&file[start..end], &heif.primary.hvcc, Some(decoder.pool()))?;
    let dw = heif.primary.display_w as usize;
    let dh = heif.primary.display_h as usize;
    decoder.check_dims(dw, dh)?;
    // HEIF `ispe` gives the consumer-visible dimensions; HEVC conformance
    // window gives the origin inside the coded planes.  Do not apply `clap`
    // here; it is exposed as metadata for the caller.
    let crop = visible_crop_from_hvcc(&heif.primary.hvcc);
    let (y_out, cb_out, cr_out) = copy_visible_yuv_planes(&planes, dw, dh, crop);

    let bd = planes.bit_depth;
    Ok(DecodedYuv {
        y: plane_to_buf(y_out, bd),
        cb: plane_to_buf(cb_out, bd),
        cr: plane_to_buf(cr_out, bd),
        alpha: decode_alpha_plane(file, &heif, dw, dh),
        width: dw as u32,
        height: dh as u32,
        bit_depth: bd,
        chroma: planes.chroma,
        color: heif.primary.color,
        orientation: heif.primary.orientation,
        clean_aperture: heif.primary.clap,
        pixel_aspect_ratio: heif.primary.pasp,
        exif: heif.exif,
    })
}

/// Immutable per-grid context for the YUV band workers.
struct YuvGridCtx<'a> {
    file: &'a [u8],
    tiles: &'a [heif::HeifItem],
    fallback_hvcc: &'a [u8],
    cols: usize,
    out_w: usize,
    out_h: usize,
    cw: usize,
    ch: usize,
    tile_w: usize,
    tile_h: usize,
    tile_cw: usize,
    tile_ch: usize,
    tile_crop_left: usize,
    tile_crop_top: usize,
    sub_w: usize,
    sub_h: usize,
    has_chroma: bool,
}

/// Stitch all tiles of grid-row `band_row` into the three plane bands. Each band
/// is that grid-row's contiguous slice of its plane; local row 0 maps to the
/// band's first global row, so vertical offsets within a band are just `y`.
/// Edge tiles are right/bottom-padded with their last sample (matching the
/// original serial behavior).
fn stitch_yuv_band(
    ctx: &YuvGridCtx<'_>,
    band_row: usize,
    y_band: &mut [u16],
    cb_band: &mut [u16],
    cr_band: &mut [u16],
) {
    for col in 0..ctx.cols {
        let tile_idx = band_row * ctx.cols + col;
        let tile = match ctx.tiles.get(tile_idx) {
            Some(t) => t,
            None => break,
        };
        let start = tile.data_offset as usize;
        let Some(end) = start.checked_add(tile.data_length as usize) else {
            continue;
        };
        if end > ctx.file.len() {
            continue;
        }
        let hvcc = if !tile.hvcc.is_empty() {
            &tile.hvcc
        } else {
            ctx.fallback_hvcc
        };
        if hvcc.is_empty() {
            continue;
        }
        let (planes, _) = match decode_hevc_item(&ctx.file[start..end], hvcc, None) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let p_cw = planes.width.div_ceil(ctx.sub_w);
        let p_ch = planes.height.div_ceil(ctx.sub_h);

        let dst_x = col * ctx.tile_w;
        let dst_y_base = band_row * ctx.tile_h;
        let copy_w = ctx.tile_w.min(ctx.out_w.saturating_sub(dst_x));
        let copy_h = ctx.tile_h.min(ctx.out_h.saturating_sub(dst_y_base));

        let tile_crop_left = ctx.tile_crop_left.min(planes.width.saturating_sub(1));
        let tile_crop_top = ctx.tile_crop_top.min(planes.height.saturating_sub(1));

        for y in 0..copy_h {
            let dst_start = y * ctx.out_w + dst_x;
            let dst = &mut y_band[dst_start..dst_start + copy_w];
            if planes.width == 0 || planes.height == 0 || planes.y.is_empty() {
                continue;
            }
            let src_y = (tile_crop_top + y).min(planes.height - 1);
            let src_base = src_y * planes.width;
            if src_base >= planes.y.len() {
                continue;
            }
            let available = (planes.y.len() - src_base).min(planes.width);
            if available == 0 {
                continue;
            }
            let src_x = tile_crop_left.min(available - 1);
            let copy = copy_w.min(available - src_x);
            let (exact, pad) = dst.split_at_mut(copy);
            exact.copy_from_slice(&planes.y[src_base + src_x..src_base + src_x + copy]);
            if !pad.is_empty() {
                pad.fill(planes.y[src_base + src_x + copy - 1]);
            }
        }

        if !ctx.has_chroma || planes.cb.is_empty() {
            continue;
        }

        let c_dst_x = col * ctx.tile_cw;
        let c_dst_y_base = band_row * ctx.tile_ch;
        let c_copy_w = ctx.tile_cw.min(ctx.cw.saturating_sub(c_dst_x));
        let c_copy_h = ctx.tile_ch.min(ctx.ch.saturating_sub(c_dst_y_base));

        let crop_cx = ctx.tile_crop_left / ctx.sub_w;
        let crop_cy = ctx.tile_crop_top / ctx.sub_h;

        for y in 0..c_copy_h {
            let c_start = y * ctx.cw + c_dst_x;
            if p_cw == 0 || p_ch == 0 || planes.cb.is_empty() || planes.cr.is_empty() {
                continue;
            }
            let src_y = (crop_cy + y).min(p_ch - 1);
            let src_base = src_y * p_cw;
            if src_base >= planes.cb.len() || src_base >= planes.cr.len() {
                continue;
            }
            let cb_available = (planes.cb.len() - src_base).min(p_cw);
            let cr_available = (planes.cr.len() - src_base).min(p_cw);
            let available = cb_available.min(cr_available);
            if available == 0 {
                continue;
            }
            let src_x = crop_cx.min(available - 1);
            let copy = c_copy_w.min(available - src_x);

            let cb_row = &mut cb_band[c_start..c_start + c_copy_w];
            let (cb_exact, cb_pad) = cb_row.split_at_mut(copy);
            cb_exact.copy_from_slice(&planes.cb[src_base + src_x..src_base + src_x + copy]);
            if !cb_pad.is_empty() {
                cb_pad.fill(planes.cb[src_base + src_x + copy - 1]);
            }

            let cr_row = &mut cr_band[c_start..c_start + c_copy_w];
            let (cr_exact, cr_pad) = cr_row.split_at_mut(copy);
            cr_exact.copy_from_slice(&planes.cr[src_base + src_x..src_base + src_x + copy]);
            if !cr_pad.is_empty() {
                cr_pad.fill(planes.cr[src_base + src_x + copy - 1]);
            }
        }
    }
}

/// Allocate the three YUV planes and fill them band-by-band. Grid rows are
/// stitched concurrently via the decoder's work-stealing pool, each band getting
/// exclusive `&mut` regions from three [`threadpool::DisjointMut`] wrappers (one
/// per plane). Single-threaded pools and single-band images take a serial path
/// that skips pool dispatch; the output is byte-identical either way.
fn fill_grid_yuv_bands(
    decoder: &Decoder,
    y_total: usize,
    c_total: usize,
    rows: usize,
    ctx: &YuvGridCtx<'_>,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let y_stride = ctx.tile_h * ctx.out_w;
    let c_stride = ctx.tile_ch * ctx.cw;

    let pool = decoder.pool();
    if pool.threads() > 1 && rows > 1 {
        let y_dm = threadpool::DisjointMut::new(vec![0u16; y_total]);
        let cb_dm = threadpool::DisjointMut::new(vec![0u16; c_total]);
        let cr_dm = threadpool::DisjointMut::new(vec![0u16; c_total]);
        threadpool::parallel_for(pool, rows, |r| {
            let y_lo = r * y_stride;
            let y_hi = ((r + 1) * y_stride).min(y_total);
            let c_lo = r * c_stride;
            let c_hi = ((r + 1) * c_stride).min(c_total);
            if y_lo >= y_hi {
                return;
            }
            let mut y_band = y_dm.slice_mut(y_lo..y_hi);
            if c_total > 0 && c_lo < c_hi {
                let mut cb_band = cb_dm.slice_mut(c_lo..c_hi);
                let mut cr_band = cr_dm.slice_mut(c_lo..c_hi);
                stitch_yuv_band(ctx, r, &mut y_band, &mut cb_band, &mut cr_band);
            } else {
                stitch_yuv_band(ctx, r, &mut y_band, &mut [], &mut []);
            }
        });
        return (y_dm.into_inner(), cb_dm.into_inner(), cr_dm.into_inner());
    }

    // Serial path — used for single-threaded pools or single-band images, where
    // dispatching to the pool would only add overhead.
    let mut out_y = vec![0u16; y_total];
    let mut out_cb = vec![0u16; c_total];
    let mut out_cr = vec![0u16; c_total];
    for r in 0..rows {
        let y_lo = r * y_stride;
        let y_hi = ((r + 1) * y_stride).min(y_total);
        let c_lo = r * c_stride;
        let c_hi = ((r + 1) * c_stride).min(c_total);
        if y_lo >= y_hi {
            continue;
        }
        if c_total > 0 && c_lo < c_hi {
            stitch_yuv_band(
                ctx,
                r,
                &mut out_y[y_lo..y_hi],
                &mut out_cb[c_lo..c_hi],
                &mut out_cr[c_lo..c_hi],
            );
        } else {
            let empty: &mut [u16] = &mut [];
            let empty2: &mut [u16] = &mut [];
            stitch_yuv_band(ctx, r, &mut out_y[y_lo..y_hi], empty, empty2);
        }
    }
    (out_y, out_cb, out_cr)
}

/// Composite a tiled grid into a single YUV image (no RGB conversion).
fn decode_grid_yuv(
    decoder: &Decoder,
    file: &[u8],
    grid: &heif::GridInfo,
    heif_file: &heif::HeifFile,
) -> Result<DecodedYuv, DecodeError> {
    let out_w = grid.output_width as usize;
    let out_h = grid.output_height as usize;
    decoder.check_dims(out_w, out_h)?;
    if grid.tiles.is_empty() {
        return Err(DecodeError::Bitstream("grid has no tiles".into()));
    }
    let cols = grid.cols as usize;
    let rows = grid.rows as usize;

    let fallback_hvcc: Vec<u8> = grid
        .tiles
        .iter()
        .find(|t| !t.hvcc.is_empty())
        .map(|t| t.hvcc.clone())
        .unwrap_or_default();

    // Tile size and chroma format from the SPS (all tiles in a grid share one).
    let hvcc_ref = if !grid.tiles[0].hvcc.is_empty() {
        &grid.tiles[0].hvcc
    } else {
        &fallback_hvcc
    };
    let parsed = config::parse_hvcc_full(hvcc_ref).ok();
    let chroma_fmt = parsed
        .as_ref()
        .map(|(sps, _)| sps.chroma)
        .unwrap_or(ChromaFormat::Yuv420);
    let (tile_w, tile_h) = parsed
        .as_ref()
        .and_then(|(sps, _)| {
            let w = sps.width.saturating_sub(sps.crop_left + sps.crop_right) as usize;
            let h = sps.height.saturating_sub(sps.crop_top + sps.crop_bottom) as usize;
            if w > 0 && h > 0 { Some((w, h)) } else { None }
        })
        .unwrap_or_else(|| (out_w.div_ceil(cols), out_h.div_ceil(rows)));

    // Chroma dimensions follow the actual subsampling, not a fixed 4:2:0.
    // Monochrome has no chroma planes (zero-sized), matching the per-tile decoder.
    let sub_w = chroma_fmt.sub_w();
    let sub_h = chroma_fmt.sub_h();
    let has_chroma = !chroma_fmt.is_monochrome();
    let (cw, ch) = if has_chroma {
        (out_w.div_ceil(sub_w), out_h.div_ceil(sub_h))
    } else {
        (0, 0)
    };
    let tile_cw = tile_w.div_ceil(sub_w);
    let tile_ch = tile_h.div_ceil(sub_h);
    let tile_crop = parsed
        .as_ref()
        .map(|(sps, _)| HevcVisibleCrop {
            left: sps.crop_left as usize,
            top: sps.crop_top as usize,
        })
        .unwrap_or_default();

    // All tiles in a grid share one SPS, so the sample bit depth is fixed for
    // the whole grid; derive it once from the reference hvcC instead of reading
    // it back inside the (possibly parallel) stitch loop.
    let bit_depth = parsed
        .as_ref()
        .map(|(sps, _)| match sps.bit_depth_luma {
            10 => BitDepth::Ten,
            12 => BitDepth::Twelve,
            _ => BitDepth::Eight,
        })
        .unwrap_or(BitDepth::Eight);

    // Shared, immutable context for the band workers.
    let yctx = YuvGridCtx {
        file,
        tiles: &grid.tiles,
        fallback_hvcc: &fallback_hvcc,
        cols,
        out_w,
        out_h,
        cw,
        ch,
        tile_w,
        tile_h,
        tile_cw,
        tile_ch,
        tile_crop_left: tile_crop.left,
        tile_crop_top: tile_crop.top,
        sub_w,
        sub_h,
        has_chroma,
    };

    // Band `r` (grid-row r) owns contiguous, disjoint ranges of each plane:
    //   luma:   [r*tile_h*out_w, ..)     spanning tile_h output rows
    //   chroma: [r*tile_ch*cw, ..)       spanning tile_ch chroma rows
    let (out_y, out_cb, out_cr) = fill_grid_yuv_bands(decoder, out_w * out_h, cw * ch, rows, &yctx);

    Ok(DecodedYuv {
        y: plane_to_buf(out_y, bit_depth),
        cb: plane_to_buf(out_cb, bit_depth),
        cr: plane_to_buf(out_cr, bit_depth),
        alpha: decode_alpha_plane(file, heif_file, out_w, out_h),
        width: out_w as u32,
        height: out_h as u32,
        bit_depth,
        chroma: chroma_fmt,
        color: heif_file.primary.color.clone(),
        orientation: grid.orientation,
        clean_aperture: heif_file.primary.clap,
        pixel_aspect_ratio: heif_file.primary.pasp,
        exif: heif_file.exif.clone(),
    })
}

fn decode_hevc_item(
    sample: &[u8],
    hvcc: &[u8],
    pool: Option<&threadpool::ThreadPool>,
) -> Result<(yuv::YuvPlanes, Cicp), DecodeError> {
    use config::parse_hvcc_full;
    use decode::{FullDecoder, parse_slice_header_full};

    let (sps, pps) = parse_hvcc_full(hvcc)?;

    let vui_color = Cicp {
        primaries: Primaries::from_u8(sps.color_primaries),
        transfer: TransferFunction::from_u8(sps.transfer_characteristics),
        matrix: MatrixCoefficients::from_u8(sps.matrix_coefficients),
        full_range: sps.video_full_range,
    };

    // Collect every VCL NAL (types 0..=31) in stream order. A coded still image
    // is one access unit, which may be split into several slice segments; each
    // segment is its own VCL NAL and must be reconstructed into the same planes.
    // Non-VCL NALs (VPS/SPS/PPS/SEI, types >= 32) are skipped — parameter sets
    // come from the hvcC box.
    let mut dec: Option<FullDecoder> = None;
    let mut pos = 0;
    while pos + 4 <= sample.len() {
        let nlen = u32::from_be_bytes(sample[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + nlen > sample.len() || nlen < 2 {
            break;
        }
        let nal_bytes = &sample[pos..pos + nlen];
        pos += nlen;
        let nal_type = (nal_bytes[0] >> 1) & 0x3f;
        if nal_type > 31 {
            continue; // non-VCL
        }

        let rbsp = crate::bitreader::unescape_rbsp(nal_bytes);
        let hdr = match parse_slice_header_full(&rbsp, &sps, &pps, nal_type) {
            Ok(h) => h,
            Err(_) => continue, // skip a slice we can't parse rather than failing whole image
        };
        let cabac = &rbsp[hdr.cabac_offset.min(rbsp.len())..];

        match dec.as_mut() {
            None => {
                // The first VCL NAL must be an independent, first-in-picture
                // segment; otherwise the bitstream is malformed for a still image.
                if !hdr.first_slice_in_pic {
                    continue;
                }
                let mut d = FullDecoder::new(cabac, sps.clone(), pps.clone(), &hdr)?;
                // Try the parallel WPP wavefront first; it decodes the whole
                // picture when eligible (single independent segment, WPP, entry
                // points). Otherwise fall back to the serial per-row decode.
                let ran_wavefront = match pool {
                    Some(p) => d.try_decode_wavefront(&rbsp, nal_bytes, &hdr, p)?,
                    None => false,
                };
                if !ran_wavefront {
                    d.decode_slice(hdr.slice_segment_address)?;
                }
                dec = Some(d);
            }
            Some(d) => {
                // Subsequent segment of the same picture.
                d.decode_segment(cabac, &hdr)?;
            }
        }
    }

    match dec {
        Some(mut d) => Ok((d.finish(pool), vui_color)),
        None => Err(DecodeError::Bitstream("no VCL slice NAL found".into())),
    }
}

/// Decode a HEIF/HEIC file to display-ready pixels using a default [`Decoder`].
pub fn decode_heic(file: &[u8]) -> Result<DecodedImage, DecodeError> {
    Decoder::default().decode(file)
}

pub(crate) fn decode_heic_with(
    decoder: &Decoder,
    file: &[u8],
) -> Result<DecodedImage, DecodeError> {
    let heif = heif::parse(file, decoder.limits())?;

    if let Some(grid) = &heif.grid {
        return decode_grid(decoder, file, grid, &heif);
    }

    let start = heif.primary.data_offset as usize;
    let Some(end) = start.checked_add(heif.primary.data_length as usize) else {
        return Err(DecodeError::Bitstream(
            "image data offset/length overflows usize".into(),
        ));
    };
    if end > file.len() {
        return Err(DecodeError::Bitstream(
            "image data extends past file end".into(),
        ));
    }
    let (yuv_planes, vui_color) =
        decode_hevc_item(&file[start..end], &heif.primary.hvcc, Some(decoder.pool()))?;
    let dw = heif.primary.display_w as usize;
    let dh = heif.primary.display_h as usize;
    decoder.check_dims(dw, dh)?;

    // Color encoding for the YCbCr→RGB step.
    // The VUI matrix/range values describe how the YCbCr was encoded; prefer
    // those.  If VUI says "unspecified" (matrix==2), fall back to the HEIF
    // `colr` box, and if that is an ICC profile (no explicit CICP), default
    // to sRGB (full-range BT.709).
    // Priority: VUI (from HEVC SPS) > CICP from colr box > sRGB fallback
    let color_enc = if vui_color.matrix != MatrixCoefficients::Unspecified {
        vui_color
    } else {
        heif.primary.color.cicp.unwrap_or_else(Cicp::srgb)
    };
    let crop = visible_crop_from_hvcc(&heif.primary.hvcc);
    let rgb = yuv::yuv_to_rgb_window_with_color_pool(
        &yuv_planes,
        dw,
        dh,
        crop.left,
        crop.top,
        &color_enc,
        Some(decoder.pool()),
    );

    let alpha = decode_alpha_plane(file, &heif, dw, dh);

    let (width, height, buf2, alpha2) =
        apply_orientation(dw as u32, dh as u32, rgb, alpha, heif.primary.orientation);

    Ok(DecodedImage {
        width,
        height,
        pixels: buf2,
        alpha: alpha2,
        bit_depth: yuv_planes.bit_depth,
        color: heif.primary.color,
        orientation: heif.primary.orientation,
        content_light_level: heif.primary.cll,
        clean_aperture: heif.primary.clap,
        pixel_aspect_ratio: heif.primary.pasp,
        exif: heif.exif,
    })
}

/// Immutable per-grid context shared by every band worker. Holding only shared
/// references keeps it `Sync`, so it can be borrowed across the thread pool.
struct GridCtx<'a> {
    file: &'a [u8],
    tiles: &'a [heif::HeifItem],
    fallback_hvcc: &'a [u8],
    color_enc: &'a Cicp,
    cols: usize,
    out_w: usize,
    out_h: usize,
    tile_w: usize,
    tile_h: usize,
    tile_crop_left: usize,
    tile_crop_top: usize,
    channels: usize,
}

/// Decode every tile in grid-row `band_row` and stitch it into `band`, the
/// contiguous output slice for rows `[band_row*tile_h, ..)`. The band's local
/// row 0 is global row `band_row*tile_h`, so the vertical destination within the
/// band is simply `y`. `pick` pulls the matching-depth samples out of the
/// per-tile RGB/luma buffer. `T` is `u8` (8-bit) or `u16` (10/12-bit).
fn stitch_grid_band<T: Copy>(
    ctx: &GridCtx<'_>,
    band_row: usize,
    band: &mut [T],
    pick: &(dyn Fn(&ImageBuffer) -> Option<&[T]> + Sync),
) {
    let dst_y_base = band_row * ctx.tile_h;
    for col in 0..ctx.cols {
        let tile_idx = band_row * ctx.cols + col;
        let tile = match ctx.tiles.get(tile_idx) {
            Some(t) => t,
            None => break,
        };
        let start = tile.data_offset as usize;
        let Some(end) = start.checked_add(tile.data_length as usize) else {
            continue;
        };
        if end > ctx.file.len() {
            continue;
        }
        let hvcc = if !tile.hvcc.is_empty() {
            &tile.hvcc
        } else {
            ctx.fallback_hvcc
        };
        if hvcc.is_empty() {
            continue;
        }
        let (yuv, _) = match decode_hevc_item(&ctx.file[start..end], hvcc, None) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let dst_x = col * ctx.tile_w;
        let copy_w = ctx.tile_w.min(ctx.out_w.saturating_sub(dst_x));
        let copy_h = ctx.tile_h.min(ctx.out_h.saturating_sub(dst_y_base));
        if copy_w == 0 || copy_h == 0 {
            continue;
        }

        let tile_buf = yuv::yuv_to_rgb_window_with_color(
            &yuv,
            ctx.tile_w,
            ctx.tile_h,
            ctx.tile_crop_left,
            ctx.tile_crop_top,
            ctx.color_enc,
        );
        let src = match pick(&tile_buf) {
            Some(s) => s,
            None => continue, // depth mismatch — shouldn't happen within one grid
        };
        let ch = ctx.channels;
        for y in 0..copy_h {
            let s = y * ctx.tile_w * ch;
            let d = (y * ctx.out_w + dst_x) * ch;
            band[d..d + copy_w * ch].copy_from_slice(&src[s..s + copy_w * ch]);
        }
    }
}

/// Allocate the output plane and fill each of the `rows` horizontal bands.
fn fill_grid_bands<T: Copy + Default + Send>(
    decoder: &Decoder,
    total: usize,
    rows: usize,
    band_stride: usize,
    ctx: &GridCtx<'_>,
    pick: &(dyn Fn(&ImageBuffer) -> Option<&[T]> + Sync),
) -> Vec<T> {
    let pool = decoder.pool();
    if pool.threads() > 1 && rows > 1 {
        let dm = threadpool::DisjointMut::new(vec![T::default(); total]);
        threadpool::parallel_for(pool, rows, |r| {
            let lo = r * band_stride;
            let hi = ((r + 1) * band_stride).min(total);
            if lo >= hi {
                return;
            }
            let mut band = dm.slice_mut(lo..hi);
            stitch_grid_band(ctx, r, &mut band, pick);
        });
        return dm.into_inner();
    }

    // Serial path — used for single-threaded pools or single-band images, where
    // dispatching to the pool would only add overhead.
    let mut out = vec![T::default(); total];
    for r in 0..rows {
        let lo = r * band_stride;
        let hi = ((r + 1) * band_stride).min(total);
        if lo >= hi {
            continue;
        }
        stitch_grid_band(ctx, r, &mut out[lo..hi], pick);
    }
    out
}

/// Decode a tiled (grid) HEIC: decode each tile independently then stitch.
fn decode_grid(
    decoder: &Decoder,
    file: &[u8],
    grid: &heif::GridInfo,
    heif_file: &heif::HeifFile,
) -> Result<DecodedImage, DecodeError> {
    let out_w = grid.output_width as usize;
    let out_h = grid.output_height as usize;
    if out_w == 0 || out_h == 0 || grid.tiles.is_empty() {
        return Err(DecodeError::Bitstream(
            "grid has zero size or no tiles".into(),
        ));
    }
    decoder.check_dims(out_w, out_h)?;

    let cols = grid.cols as usize;
    let rows = grid.rows as usize;

    // Apple HEIC sometimes doesn't per-associate hvcC via ipma; all tiles share
    // the same SPS/PPS so reusing the first non-empty one is correct.
    let fallback_hvcc: Vec<u8> = grid
        .tiles
        .iter()
        .find(|t| !t.hvcc.is_empty())
        .map(|t| t.hvcc.clone())
        .unwrap_or_default();

    // Priority: (1) SPS conformance window, (2) ispe (sanity-checked),
    // (3) infer from grid layout.
    let (tile_w, tile_h) = {
        let hvcc_ref = if !grid.tiles[0].hvcc.is_empty() {
            &grid.tiles[0].hvcc
        } else {
            &fallback_hvcc
        };
        let from_sps = if !hvcc_ref.is_empty() {
            config::parse_hvcc_full(hvcc_ref).ok().and_then(|(sps, _)| {
                let w = sps.width.saturating_sub(sps.crop_left + sps.crop_right) as usize;
                let h = sps.height.saturating_sub(sps.crop_top + sps.crop_bottom) as usize;
                if w > 0 && h > 0 { Some((w, h)) } else { None }
            })
        } else {
            None
        };

        from_sps
            .or_else(|| {
                let t = &grid.tiles[0];
                if t.display_w > 0 && t.display_h > 0 {
                    // Reject if ispe equals the full output size — means the tile
                    // accidentally inherited the grid's ispe property.
                    let same_as_output = t.display_w == grid.output_width
                        && t.display_h == grid.output_height
                        && (cols > 1 || rows > 1);
                    if !same_as_output {
                        Some((t.display_w as usize, t.display_h as usize))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                let tw = out_w.div_ceil(cols);
                let th = out_h.div_ceil(rows);
                (tw, th)
            })
    };

    if tile_w == 0 || tile_h == 0 {
        return Err(DecodeError::Bitstream(
            "cannot determine tile dimensions".into(),
        ));
    }

    let color_enc = {
        let hvcc_ref = if !grid.tiles[0].hvcc.is_empty() {
            &grid.tiles[0].hvcc
        } else {
            &fallback_hvcc
        };
        if let Ok((sps, _)) = config::parse_hvcc_full(hvcc_ref) {
            Cicp {
                primaries: Primaries::from_u8(sps.color_primaries),
                transfer: TransferFunction::from_u8(sps.transfer_characteristics),
                matrix: MatrixCoefficients::from_u8(sps.matrix_coefficients),
                full_range: sps.video_full_range,
            }
        } else {
            heif_file.primary.color.cicp.unwrap_or_else(Cicp::srgb)
        }
    };

    let bit_depth = {
        let hvcc_ref = if !grid.tiles[0].hvcc.is_empty() {
            &grid.tiles[0].hvcc
        } else {
            &fallback_hvcc
        };
        config::parse_hvcc_full(hvcc_ref)
            .ok()
            .map(|(sps, _)| match sps.bit_depth_luma {
                10 => BitDepth::Ten,
                12 => BitDepth::Twelve,
                _ => BitDepth::Eight,
            })
            .unwrap_or(BitDepth::Eight)
    };

    let is_mono = {
        let hvcc_ref = if !grid.tiles[0].hvcc.is_empty() {
            &grid.tiles[0].hvcc
        } else {
            &fallback_hvcc
        };
        config::parse_hvcc_full(hvcc_ref)
            .ok()
            .map(|(sps, _)| sps.chroma.is_monochrome())
            .unwrap_or(false)
    };

    let tile_crop = {
        let hvcc_ref = if !grid.tiles[0].hvcc.is_empty() {
            &grid.tiles[0].hvcc
        } else {
            &fallback_hvcc
        };
        visible_crop_from_hvcc(hvcc_ref)
    };

    let channels = if is_mono { 1 } else { 3 };

    // Bundle everything a band needs so the (possibly parallel) worker closures
    // stay small and capture only a shared `&`.
    let ctx = GridCtx {
        file,
        tiles: &grid.tiles,
        fallback_hvcc: &fallback_hvcc,
        color_enc: &color_enc,
        cols,
        out_w,
        out_h,
        tile_w,
        tile_h,
        tile_crop_left: tile_crop.left,
        tile_crop_top: tile_crop.top,
        channels,
    };

    // Rows are split into `rows` horizontal bands; band `r` owns the contiguous
    // output range `[r*band_stride, (r+1)*band_stride)`. Bands are disjoint, so
    // they can be filled concurrently. The `pick` closure selects the source
    // samples of the matching depth from each per-tile RGB/luma buffer.
    let band_stride = tile_h * out_w * channels;

    let out_buf = if bit_depth == BitDepth::Eight {
        let v = fill_grid_bands::<u8>(
            decoder,
            out_w * out_h * channels,
            rows,
            band_stride,
            &ctx,
            &|b| match b {
                ImageBuffer::Rgb8(s) | ImageBuffer::Luma8(s) => Some(s.as_slice()),
                _ => None,
            },
        );
        if is_mono {
            ImageBuffer::Luma8(v)
        } else {
            ImageBuffer::Rgb8(v)
        }
    } else {
        let v = fill_grid_bands::<u16>(
            decoder,
            out_w * out_h * channels,
            rows,
            band_stride,
            &ctx,
            &|b| match b {
                ImageBuffer::Rgb16(s) | ImageBuffer::Luma16(s) => Some(s.as_slice()),
                _ => None,
            },
        );
        if is_mono {
            ImageBuffer::Luma16(v)
        } else {
            ImageBuffer::Rgb16(v)
        }
    };

    let alpha = decode_alpha_plane(file, heif_file, out_w, out_h);
    let (width, height, buf2, alpha2) =
        apply_orientation(out_w as u32, out_h as u32, out_buf, alpha, grid.orientation);

    Ok(DecodedImage {
        width,
        height,
        pixels: buf2,
        alpha: alpha2,
        bit_depth,
        color: heif_file.primary.color.clone(),
        // store grid.orientation so the caller knows what rotation was applied
        // (pixels have already been rotated by apply_orientation above)
        orientation: grid.orientation,
        content_light_level: heif_file.primary.cll,
        clean_aperture: heif_file.primary.clap,
        pixel_aspect_ratio: heif_file.primary.pasp,
        exif: heif_file.exif.clone(),
    })
}

/// Decode to 8-bit-per-channel RGB `Vec<u8>` (always 3 bytes/pixel) using a
/// default [`Decoder`]. Monochrome images are expanded to gray RGB. Zero-copy
/// for 8-bit color sources.
pub fn decode_heic_rgb8(file: &[u8]) -> Result<(Vec<u8>, u32, u32), DecodeError> {
    Decoder::default().decode_rgb8(file)
}

pub(crate) fn decode_heic_rgb8_with(
    decoder: &Decoder,
    file: &[u8],
) -> Result<(Vec<u8>, u32, u32), DecodeError> {
    let img = decode_heic_with(decoder, file)?;
    let shift = img.bit_depth.minus8();
    let pixels = match img.pixels {
        ImageBuffer::Rgb8(v) => v, // 8-bit color: direct move
        ImageBuffer::Rgb16(v) => v.into_iter().map(|x| (x >> shift) as u8).collect(),
        ImageBuffer::Luma8(v) => v.into_iter().flat_map(|l| [l, l, l]).collect(),
        ImageBuffer::Luma16(v) => v
            .into_iter()
            .map(|x| (x >> shift) as u8)
            .flat_map(|l| [l, l, l])
            .collect(),
    };
    Ok((pixels, img.width, img.height))
}

fn apply_orientation(
    w: u32,
    h: u32,
    buf: ImageBuffer,
    alpha: Option<SampleBuf>,
    o: Orientation,
) -> (u32, u32, ImageBuffer, Option<SampleBuf>) {
    let (nw, nh) = match o {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transverse
        | Orientation::Transpose => (h as usize, w as usize),
        _ => (w as usize, h as usize),
    };
    let buf2 = match buf {
        ImageBuffer::Luma8(px) => ImageBuffer::Luma8(rotate_luma(w as usize, h as usize, &px, o)),
        ImageBuffer::Luma16(px) => ImageBuffer::Luma16(rotate_luma(w as usize, h as usize, &px, o)),
        ImageBuffer::Rgb8(px) => ImageBuffer::Rgb8(rotate_buf(w as usize, h as usize, &px, o)),
        ImageBuffer::Rgb16(px) => ImageBuffer::Rgb16(rotate_buf(w as usize, h as usize, &px, o)),
    };
    let alpha2 = alpha.map(|a| match a {
        SampleBuf::U8(v) => SampleBuf::U8(rotate_luma(w as usize, h as usize, &v, o)),
        SampleBuf::U16(v) => SampleBuf::U16(rotate_luma(w as usize, h as usize, &v, o)),
    });
    (nw as u32, nh as u32, buf2, alpha2)
}

/// Single-channel rotation for luma-only buffers (stride = 1, not 3).
fn rotate_luma<T: Copy + Default>(w: usize, h: usize, px: &[T], o: Orientation) -> Vec<T> {
    match o {
        Orientation::Normal => px.to_vec(),
        Orientation::Rotate180 => px.iter().copied().rev().collect::<Vec<_>>(),
        Orientation::FlipH => {
            let mut out = vec![T::default(); px.len()];
            for (src_row, dst_row) in px.chunks_exact(w).zip(out.chunks_exact_mut(w)) {
                for (dst, &src) in dst_row.iter_mut().rev().zip(src_row.iter()) {
                    *dst = src;
                }
            }
            out
        }
        Orientation::FlipV => {
            let mut out = vec![T::default(); px.len()];
            for (src_row, dst_row) in px.chunks_exact(w).zip(out.chunks_exact_mut(w).rev()) {
                dst_row.copy_from_slice(src_row);
            }
            out
        }
        Orientation::Rotate90 => {
            let mut out = vec![T::default(); px.len()];
            for r in 0..h {
                for c in 0..w {
                    out[c * h + (h - 1 - r)] = px[r * w + c];
                }
            }
            out
        }
        Orientation::Rotate270 => {
            let mut out = vec![T::default(); px.len()];
            for r in 0..h {
                for c in 0..w {
                    out[(w - 1 - c) * h + r] = px[r * w + c];
                }
            }
            out
        }
        _ => px.to_vec(),
    }
}

/// Generic pixel-buffer rotation; works for both `u8` and `u16` samples.
fn rotate_buf<T: Copy + Default>(w: usize, h: usize, px: &[T], o: Orientation) -> Vec<T> {
    match o {
        Orientation::Normal => px.to_vec(),
        Orientation::Rotate180 => px
            .as_chunks::<3>()
            .0
            .iter()
            .rev()
            .flat_map(|c| c.iter().copied())
            .collect(),
        Orientation::FlipH => {
            let mut out = vec![T::default(); px.len()];
            for (src_row, dst_row) in px.chunks_exact(w * 3).zip(out.chunks_exact_mut(w * 3)) {
                for (src, dst) in src_row
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .rev()
                    .zip(dst_row.as_chunks_mut::<3>().0.iter_mut())
                {
                    dst.copy_from_slice(src);
                }
            }
            out
        }
        Orientation::FlipV => {
            let mut out = vec![T::default(); px.len()];
            for (src_row, dst_row) in px
                .chunks_exact(w * 3)
                .rev()
                .zip(out.chunks_exact_mut(w * 3))
            {
                dst_row.copy_from_slice(src_row);
            }
            out
        }
        Orientation::Rotate90 => {
            // 90° CW: (r,c) → (c, h-1-r). Blocked to keep both the source and
            // the transposed destination regions cache-resident.
            let mut out = vec![T::default(); px.len()];
            const BS: usize = 32;
            let mut rb = 0;
            while rb < h {
                let r_end = (rb + BS).min(h);
                let mut cb = 0;
                while cb < w {
                    let c_end = (cb + BS).min(w);
                    for r in rb..r_end {
                        let s_row = r * w * 3;
                        let d_col = h - 1 - r;
                        for c in cb..c_end {
                            let s = s_row + c * 3;
                            let d = (c * h + d_col) * 3;
                            out[d] = px[s];
                            out[d + 1] = px[s + 1];
                            out[d + 2] = px[s + 2];
                        }
                    }
                    cb = c_end;
                }
                rb = r_end;
            }
            out
        }
        Orientation::Rotate270 => {
            // 90° CCW: (r,c) → (w-1-c, r). Blocked transpose (see Rotate90).
            let mut out = vec![T::default(); px.len()];
            const BS: usize = 32;
            let mut rb = 0;
            while rb < h {
                let r_end = (rb + BS).min(h);
                let mut cb = 0;
                while cb < w {
                    let c_end = (cb + BS).min(w);
                    for r in rb..r_end {
                        let s_row = r * w * 3;
                        for c in cb..c_end {
                            let s = s_row + c * 3;
                            let d = ((w - 1 - c) * h + r) * 3;
                            out[d] = px[s];
                            out[d + 1] = px[s + 1];
                            out[d + 2] = px[s + 2];
                        }
                    }
                    cb = c_end;
                }
                rb = r_end;
            }
            out
        }
        _ => px.to_vec(),
    }
}
