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
mod decode;
mod error;
mod fmt;
mod heif;
mod intra;
mod metadata;
mod transform;
mod yuv;

pub use color::{ColorEncoding, ColorMetadata, MatrixCoefficients, Primaries, TransferFunction};
pub use error::DecodeError;
pub use fmt::{BitDepth, ChromaFormat, ImageBuffer, SampleBuf};
pub use metadata::{CleanAperture, ContentLightLevel, Metadata, Orientation, PixelAspectRatio};

const MAX_IMG_DIM: usize = 16_384;
const MAX_IMG_PIXELS: usize = 64 * 1024 * 1024;

/// Convert a decoded u16 YUV plane to the appropriate typed buffer.
/// 8-bit images produce `SampleBuf::U8` (direct cast, no precision loss).
fn plane_to_buf(plane: Vec<u16>, bd: BitDepth) -> SampleBuf {
    if bd == BitDepth::Eight {
        SampleBuf::U8(plane.into_iter().map(|v| v as u8).collect())
    } else {
        SampleBuf::U16(plane)
    }
}

fn check_dims(w: usize, h: usize) -> Result<(), DecodeError> {
    if w == 0
        || h == 0
        || w > MAX_IMG_DIM
        || h > MAX_IMG_DIM
        || w.saturating_mul(h) > MAX_IMG_PIXELS
    {
        Err(DecodeError::Bitstream(format!(
            "image dimensions {w}×{h} exceed maximum"
        )))
    } else {
        Ok(())
    }
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
/// cropped/limited to `dw*dh`. Returns `None` when there is no alpha item or it
/// fails to decode. Shared by the single-tile and grid YUV paths.
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
    decode_hevc_item(&file[astart..aend], &a.hvcc)
        .ok()
        .map(|(ap, _)| {
            let plane = ap.y[..(dw * dh).min(ap.y.len())].to_vec();
            plane_to_buf(plane, ap.bit_depth)
        })
}

/// Decode a HEIF/HEIC file and return raw YCbCr planes (no colour conversion).
/// For a display-ready 8-bit image use [`decode_heic_rgb8`].
pub fn decode_heic_yuv(file: &[u8]) -> Result<DecodedYuv, DecodeError> {
    let heif = heif::parse(file)?;

    if let Some(grid) = &heif.grid {
        return decode_grid_yuv(file, grid, &heif);
    }

    // Single-tile path
    let start = heif.primary.data_offset as usize;
    let end = start + heif.primary.data_length as usize;
    if end > file.len() {
        return Err(DecodeError::Bitstream(
            "image data extends past file end".into(),
        ));
    }
    let (planes, _) = decode_hevc_item(&file[start..end], &heif.primary.hvcc)?;
    let dw = heif.primary.display_w as usize;
    let dh = heif.primary.display_h as usize;
    check_dims(dw, dh)?;
    let sub_w = planes.chroma.sub_w();
    let sub_h = planes.chroma.sub_h();
    let cw = dw.div_ceil(sub_w);
    let ch = dh.div_ceil(sub_h);

    // Crop Y to display size
    let coded_cw = planes.width.div_ceil(sub_w);
    let coded_ch = planes.height.div_ceil(sub_h);
    let mono = planes.chroma.is_monochrome();
    let mut y_out = vec![0u16; dw * dh];
    // Monochrome has no chroma planes; keep the output chroma buffers empty so we
    // never index the (empty) decoded cb/cr planes below.
    let mut cb_out = vec![0u16; if mono { 0 } else { cw * ch }];
    let mut cr_out = vec![0u16; if mono { 0 } else { cw * ch }];

    {
        let src_h = planes.height;
        let src_w = planes.width;

        for (r, dst_row) in y_out.chunks_exact_mut(dw).enumerate() {
            let src_row = &planes.y[r.min(src_h - 1) * src_w..][..src_w];
            let (fill, edge) = dst_row.split_at_mut(src_w.min(dw));
            for (d, s) in fill.iter_mut().zip(src_row.iter()) {
                *d = *s;
            }
            // pad right with the last luma sample if dw > src_w
            if let (Some(&last), false) = (fill.last(), edge.is_empty()) {
                edge.fill(last);
            }
        }
    }

    if !mono {
        for (r, (cb_row, cr_row)) in cb_out
            .chunks_exact_mut(cw)
            .zip(cr_out.chunks_exact_mut(cw))
            .enumerate()
        {
            let src_r = r.min(coded_ch - 1) * coded_cw;
            let src_cb = &planes.cb[src_r..][..coded_cw];
            let src_cr = &planes.cr[src_r..][..coded_cw];

            let copy_w = coded_cw.min(cw);
            let (cb_fill, cb_edge) = cb_row.split_at_mut(copy_w);
            let (cr_fill, cr_edge) = cr_row.split_at_mut(copy_w);

            for (((d_cb, d_cr), s_cb), s_cr) in cb_fill
                .iter_mut()
                .zip(cr_fill.iter_mut())
                .zip(src_cb.iter())
                .zip(src_cr.iter())
            {
                *d_cb = *s_cb;
                *d_cr = *s_cr;
            }
            // pad right if cw > coded_cw
            if let (Some(&last_cb), Some(&last_cr), false) =
                (cb_fill.last(), cr_fill.last(), cb_edge.is_empty())
            {
                cb_edge.fill(last_cb);
                cr_edge.fill(last_cr);
            }
        }
    }

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

/// Composite a tiled grid into a single YUV image (no RGB conversion).
fn decode_grid_yuv(
    file: &[u8],
    grid: &heif::GridInfo,
    heif_file: &heif::HeifFile,
) -> Result<DecodedYuv, DecodeError> {
    let out_w = grid.output_width as usize;
    let out_h = grid.output_height as usize;
    check_dims(out_w, out_h)?;
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
    let mut chroma_fmt = parsed
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

    let mut out_y = vec![0u16; out_w * out_h];
    let mut out_cb = vec![0u16; cw * ch];
    let mut out_cr = vec![0u16; cw * ch];
    let mut bit_depth = BitDepth::Eight;

    for (tile_idx, tile) in grid.tiles.iter().enumerate() {
        let col = tile_idx % cols;
        let row = tile_idx / cols;
        if row >= rows {
            break;
        }

        let start = tile.data_offset as usize;
        let end = start + tile.data_length as usize;
        if end > file.len() {
            continue;
        }
        let hvcc = if !tile.hvcc.is_empty() {
            &tile.hvcc
        } else {
            &fallback_hvcc
        };
        if hvcc.is_empty() {
            continue;
        }

        let (planes, _) = match decode_hevc_item(&file[start..end], hvcc) {
            Ok(r) => r,
            Err(_) => continue,
        };
        bit_depth = planes.bit_depth;
        // Keep the buffers consistent with the format they were sized for; all
        // tiles in a grid share one SPS, so this normally already matches.
        chroma_fmt = planes.chroma;
        let p_cw = planes.width.div_ceil(sub_w);
        let p_ch = planes.height.div_ceil(sub_h);

        let dst_x = col * tile_w;
        let dst_y = row * tile_h;
        let copy_w = tile_w.min(out_w.saturating_sub(dst_x));
        let copy_h = tile_h.min(out_h.saturating_sub(dst_y));

        for y in 0..copy_h {
            let dst_start = (dst_y + y) * out_w + dst_x;
            let dst = &mut out_y[dst_start..dst_start + copy_w];
            let src_y = y.min(planes.height - 1);
            let src_row = &planes.y[src_y * planes.width..][..planes.width];
            let src = &src_row[..copy_w.min(planes.width)];

            let (exact, pad) = dst.split_at_mut(src.len());
            exact.copy_from_slice(src);
            if let Some(&last) = src.last() {
                pad.fill(last);
            }
        }

        // Chroma: skip entirely for monochrome (no chroma planes), and copy at
        // the format's true subsampling for 4:2:0 / 4:2:2 / 4:4:4.
        if !has_chroma || planes.cb.is_empty() {
            continue;
        }

        let c_dst_x = col * tile_cw;
        let c_dst_y = row * tile_ch;
        let c_copy_w = tile_cw.min(cw.saturating_sub(c_dst_x));
        let c_copy_h = tile_ch.min(ch.saturating_sub(c_dst_y));

        for y in 0..c_copy_h {
            let cb_start = (c_dst_y + y) * cw + c_dst_x;
            let cb_row = &mut out_cb[cb_start..cb_start + c_copy_w];
            let cr_start = (c_dst_y + y) * cw + c_dst_x;
            // out_cb and out_cr share the same geometry; index out_cr separately.
            let src_y = y.min(p_ch - 1);
            let src_cb_row = &planes.cb[src_y * p_cw..][..p_cw];
            let src_cr_row = &planes.cr[src_y * p_cw..][..p_cw];

            let copy = c_copy_w.min(p_cw);
            let (cb_exact, cb_pad) = cb_row.split_at_mut(copy);
            cb_exact.copy_from_slice(&src_cb_row[..copy]);
            if let Some(&last_cb) = src_cb_row.last() {
                cb_pad.fill(last_cb);
            }

            let cr_row = &mut out_cr[cr_start..cr_start + c_copy_w];
            let (cr_exact, cr_pad) = cr_row.split_at_mut(copy);
            cr_exact.copy_from_slice(&src_cr_row[..copy]);
            if let Some(&last_cr) = src_cr_row.last() {
                cr_pad.fill(last_cr);
            }
        }
    }

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
) -> Result<(yuv::YuvPlanes, ColorEncoding), DecodeError> {
    use bitreader::unescape_rbsp;
    use config::parse_hvcc_full;
    use decode::{FullDecoder, parse_slice_header_full};

    let (sps, pps) = parse_hvcc_full(hvcc)?;

    // Find the first IDR/CRA NAL in the length-prefixed sample.
    let mut pos = 0;
    let mut nal = Vec::new();
    let mut nal_type = 0u8;
    while pos + 4 <= sample.len() {
        let nlen = u32::from_be_bytes(sample[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + nlen > sample.len() {
            break;
        }
        let t = (sample[pos] >> 1) & 0x3f;
        if matches!(t, 19..=21) {
            nal = sample[pos..pos + nlen].to_vec();
            nal_type = t;
            break;
        }
        pos += nlen;
    }
    if nal.is_empty() {
        return Err(DecodeError::Bitstream("no IDR/CRA NAL found".into()));
    }

    let rbsp = unescape_rbsp(&nal);
    let (slice_qp, sao_luma, sao_chroma, cabac_off) =
        parse_slice_header_full(&rbsp, &sps, &pps, nal_type)?;
    let vui_color = ColorEncoding {
        primaries: Primaries::from_u8(sps.colour_primaries),
        transfer: TransferFunction::from_u8(sps.transfer_characteristics),
        matrix: MatrixCoefficients::from_u8(sps.matrix_coefficients),
        full_range: sps.video_full_range,
    };
    let mut dec = FullDecoder::new(&rbsp[cabac_off..], sps, pps, slice_qp, sao_luma, sao_chroma)?;
    Ok((dec.decode()?, vui_color))
}

pub fn decode_heic(file: &[u8]) -> Result<DecodedImage, DecodeError> {
    let heif = heif::parse(file)?;

    if let Some(grid) = &heif.grid {
        return decode_grid(file, grid, &heif);
    }

    let start = heif.primary.data_offset as usize;
    let end = start + heif.primary.data_length as usize;
    if end > file.len() {
        return Err(DecodeError::Bitstream(
            "image data extends past file end".into(),
        ));
    }
    let (yuv_planes, vui_color) = decode_hevc_item(&file[start..end], &heif.primary.hvcc)?;
    let dw = heif.primary.display_w as usize;
    let dh = heif.primary.display_h as usize;

    // Color encoding for the YCbCr→RGB step.
    // The VUI matrix/range values describe how the YCbCr was encoded; prefer
    // those.  If VUI says "unspecified" (matrix==2), fall back to the HEIF
    // `colr` box, and if that is an ICC profile (no explicit CICP), default
    // to sRGB (full-range BT.709).
    // Priority: VUI (from HEVC SPS) > CICP from colr box > sRGB fallback
    let color_enc = if vui_color.matrix != MatrixCoefficients::Unspecified {
        vui_color
    } else {
        heif.primary.color.cicp.unwrap_or_else(ColorEncoding::srgb)
    };
    let rgb = yuv::yuv_to_rgb_with_color(&yuv_planes, dw, dh, &color_enc);

    let alpha = if let Some(a) = &heif.alpha {
        if !a.hvcc.is_empty() {
            let astart = a.data_offset as usize;
            let aend = astart + a.data_length as usize;
            if aend <= file.len() {
                decode_hevc_item(&file[astart..aend], &a.hvcc)
                    .ok()
                    .map(|(ap, _)| {
                        let plane = ap.y[..(dw * dh).min(ap.y.len())].to_vec();
                        plane_to_buf(plane, ap.bit_depth)
                    })
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

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

/// Decode a tiled (grid) HEIC: decode each tile independently then stitch.
fn decode_grid(
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
            ColorEncoding {
                primaries: Primaries::from_u8(sps.colour_primaries),
                transfer: TransferFunction::from_u8(sps.transfer_characteristics),
                matrix: MatrixCoefficients::from_u8(sps.matrix_coefficients),
                full_range: sps.video_full_range,
            }
        } else {
            heif_file
                .primary
                .color
                .cicp
                .unwrap_or_else(ColorEncoding::srgb)
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
                10 => fmt::BitDepth::Ten,
                12 => fmt::BitDepth::Twelve,
                _ => fmt::BitDepth::Eight,
            })
            .unwrap_or(fmt::BitDepth::Eight)
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

    let mut out_buf = match (bit_depth == BitDepth::Eight, is_mono) {
        (true, false) => ImageBuffer::Rgb8(vec![0u8; out_w * out_h * 3]),
        (false, false) => ImageBuffer::Rgb16(vec![0u16; out_w * out_h * 3]),
        (true, true) => ImageBuffer::Luma8(vec![0u8; out_w * out_h]),
        (false, true) => ImageBuffer::Luma16(vec![0u16; out_w * out_h]),
    };

    for (tile_idx, tile) in grid.tiles.iter().enumerate() {
        let col = tile_idx % cols;
        let row = tile_idx / cols;
        if row >= rows {
            break;
        }

        let start = tile.data_offset as usize;
        let end = start + tile.data_length as usize;
        if end > file.len() {
            continue;
        }

        let hvcc = if !tile.hvcc.is_empty() {
            &tile.hvcc
        } else {
            &fallback_hvcc
        };
        if hvcc.is_empty() {
            continue;
        }

        let (yuv, _) = match decode_hevc_item(&file[start..end], hvcc) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let dst_x = col * tile_w;
        let dst_y = row * tile_h;
        let copy_w = tile_w.min(out_w.saturating_sub(dst_x));
        let copy_h = tile_h.min(out_h.saturating_sub(dst_y));
        if copy_w == 0 || copy_h == 0 {
            continue;
        }

        let tile_buf = yuv::yuv_to_rgb_with_color(&yuv, tile_w, tile_h, &color_enc);

        // Copy this tile into the output buffer.  Both tile and output share
        // the same depth (guaranteed by decoding from the same SPS), so the
        // variants always match.
        match (&tile_buf, &mut out_buf) {
            (ImageBuffer::Rgb8(src), ImageBuffer::Rgb8(dst)) => {
                for y in 0..copy_h {
                    let s = y * tile_w * 3;
                    let d = ((dst_y + y) * out_w + dst_x) * 3;
                    dst[d..d + copy_w * 3].copy_from_slice(&src[s..s + copy_w * 3]);
                }
            }
            (ImageBuffer::Rgb16(src), ImageBuffer::Rgb16(dst)) => {
                for y in 0..copy_h {
                    let s = y * tile_w * 3;
                    let d = ((dst_y + y) * out_w + dst_x) * 3;
                    dst[d..d + copy_w * 3].copy_from_slice(&src[s..s + copy_w * 3]);
                }
            }
            (ImageBuffer::Luma8(src), ImageBuffer::Luma8(dst)) => {
                for y in 0..copy_h {
                    let s = y * tile_w;
                    let d = (dst_y + y) * out_w + dst_x;
                    dst[d..d + copy_w].copy_from_slice(&src[s..s + copy_w]);
                }
            }
            (ImageBuffer::Luma16(src), ImageBuffer::Luma16(dst)) => {
                for y in 0..copy_h {
                    let s = y * tile_w;
                    let d = (dst_y + y) * out_w + dst_x;
                    dst[d..d + copy_w].copy_from_slice(&src[s..s + copy_w]);
                }
            }
            _ => {} // depth mismatch — skip (shouldn't happen within one grid)
        }
    }

    let (width, height, buf2, _alpha) =
        apply_orientation(out_w as u32, out_h as u32, out_buf, None, grid.orientation);

    Ok(DecodedImage {
        width,
        height,
        pixels: buf2,
        alpha: None,
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

/// Decode to 8-bit-per-channel RGB `Vec<u8>` (always 3 bytes/pixel).
/// Monochrome images are expanded to gray RGB. Zero-copy for 8-bit colour sources.
pub fn decode_heic_rgb8(file: &[u8]) -> Result<(Vec<u8>, u32, u32), DecodeError> {
    let img = decode_heic(file)?;
    let shift = img.bit_depth.minus8();
    let pixels = match img.pixels {
        ImageBuffer::Rgb8(v) => v, // 8-bit colour: direct move
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
        Orientation::Rotate90 | Orientation::Rotate270 => (h as usize, w as usize),
        _ => (w as usize, h as usize),
    };
    let buf2 = match buf {
        ImageBuffer::Luma8(px) => ImageBuffer::Luma8(rotate_luma(w as usize, h as usize, px, o)),
        ImageBuffer::Luma16(px) => ImageBuffer::Luma16(rotate_luma(w as usize, h as usize, px, o)),
        ImageBuffer::Rgb8(px) => ImageBuffer::Rgb8(rotate_buf(w as usize, h as usize, px, o)),
        ImageBuffer::Rgb16(px) => ImageBuffer::Rgb16(rotate_buf(w as usize, h as usize, px, o)),
    };
    let alpha2 = alpha.map(|a| match a {
        SampleBuf::U8(v) => SampleBuf::U8(rotate_luma(w as usize, h as usize, v, o)),
        SampleBuf::U16(v) => SampleBuf::U16(rotate_luma(w as usize, h as usize, v, o)),
    });
    (nw as u32, nh as u32, buf2, alpha2)
}

/// Single-channel rotation for luma-only buffers (stride = 1, not 3).
fn rotate_luma<T: Copy + Default>(w: usize, h: usize, px: Vec<T>, o: Orientation) -> Vec<T> {
    match o {
        Orientation::Normal => px,
        Orientation::Rotate180 => px.into_iter().rev().collect(),
        Orientation::FlipH => {
            let mut out = vec![T::default(); px.len()];
            for r in 0..h {
                for c in 0..w {
                    out[r * w + (w - 1 - c)] = px[r * w + c];
                }
            }
            out
        }
        Orientation::FlipV => {
            let mut out = vec![T::default(); px.len()];
            for r in 0..h {
                out[(h - 1 - r) * w..][..w].copy_from_slice(&px[r * w..][..w]);
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
        _ => px,
    }
}

/// Generic pixel-buffer rotation; works for both `u8` and `u16` samples.
fn rotate_buf<T: Copy + Default>(w: usize, h: usize, px: Vec<T>, o: Orientation) -> Vec<T> {
    match o {
        Orientation::Normal => px,
        Orientation::Rotate180 => px
            .as_chunks::<3>()
            .0
            .iter()
            .rev()
            .flat_map(|c| c.iter().copied())
            .collect(),
        Orientation::FlipH => {
            let mut out = vec![T::default(); px.len()];
            for r in 0..h {
                for c in 0..w {
                    let s = (r * w + (w - 1 - c)) * 3;
                    let d = (r * w + c) * 3;
                    out[d..d + 3].copy_from_slice(&px[s..s + 3]);
                }
            }
            out
        }
        Orientation::FlipV => {
            let mut out = vec![T::default(); px.len()];
            for r in 0..h {
                let sr = (h - 1 - r) * w * 3;
                let dr = r * w * 3;
                out[dr..dr + w * 3].copy_from_slice(&px[sr..sr + w * 3]);
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
        _ => px,
    }
}
