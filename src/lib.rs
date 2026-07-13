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

#![allow(unreachable_pub)]

/// Allocate and fill a vector without aborting the process on allocation
/// failure. Use this only for image-sized or otherwise potentially large
/// buffers; small fixed/scratch allocations should keep ordinary `vec!`.
macro_rules! try_vec {
    ($value:expr; $len:expr, $what:expr) => {{
        let len = $len;
        let value = $value;
        let element_size = core::mem::size_of_val(&value);
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| $crate::error::DecodeError::AllocationFailed {
                what: $what,
                bytes: len.saturating_mul(element_size),
            })?;
        out.resize(len, value);
        out
    }};
}
mod act;
#[cfg(all(feature = "avx", target_arch = "x86_64"))]
mod avx;
mod bitreader;
mod cabac;
mod color;
mod config;
mod deblock;
mod decode;
mod decoder;
mod demux;
mod dpb;
mod error;
mod exec;
mod fast_divide;
mod fmt;
mod heif;
mod ibc;
mod info;
mod inter;
mod intra;
mod limits;
mod mc;
mod metadata;
mod motion;
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon;
mod palette;
mod plane;
mod reconstruct;
mod rps;
mod sao;
mod settings;
#[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
mod sse;
mod threadpool;
mod tiles;
mod transform;
mod video;
mod wpp;
mod yuv;

pub use color::{Cicp, ColorMetadata, MatrixCoefficients, Primaries, TransferFunction};
pub use decoder::Decoder;
pub use error::DecodeError;
pub use fmt::{
    BitDepth, ChromaFormat, ImageBuffer, PlanarImage, PlaneBuffer, PlaneLayout, SampleBuf,
    SamplePlane, YuvBuffer,
};
pub use info::{ImageInfo, read_heic_info, read_heic_info_with_limits};
pub use limits::ParseLimits;
pub use metadata::{CleanAperture, ContentLightLevel, Metadata, Orientation, PixelAspectRatio};
pub use settings::{DecodeThreads, HeicSettings};
pub use video::{FrameYuv, VideoDecoder, VideoFrame, decode_hevc, decode_hevc_frame_at};

#[derive(Clone, Copy, Default)]
struct HevcVisibleCrop {
    left: usize,
    top: usize,
}

struct DecodedGainMapImage {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    y: SampleBuf,
    cb: Option<SampleBuf>,
    cr: Option<SampleBuf>,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    color: ColorMetadata,
    orientation: Orientation,
}

struct OrientedImage {
    width: u32,
    height: u32,
    pixels: ImageBuffer,
    alpha: Option<SampleBuf>,
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

fn plane_window<T: Copy + Default>(
    data: Vec<T>,
    src_w: usize,
    src_h: usize,
    crop_left: usize,
    crop_top: usize,
    dst_w: usize,
    dst_h: usize,
) -> Result<PlaneBuffer<T>, DecodeError> {
    if dst_w == 0 || dst_h == 0 {
        return Ok(PlaneBuffer::tight(Vec::new(), dst_w, dst_h));
    }

    #[allow(clippy::manual_checked_ops)]
    let rows = if src_w == 0 {
        0
    } else {
        (data.len() / src_w).min(src_h)
    };
    let direct = rows != 0
        && crop_left.checked_add(dst_w).is_some_and(|end| end <= src_w)
        && crop_top.checked_add(dst_h).is_some_and(|end| end <= rows)
        && crop_top
            .checked_mul(src_w)
            .and_then(|v| v.checked_add(crop_left))
            .and_then(|offset| {
                (dst_h - 1)
                    .checked_mul(src_w)
                    .and_then(|rows| offset.checked_add(rows))
                    .and_then(|v| v.checked_add(dst_w))
                    .map(|end| (offset, end))
            })
            .is_some_and(|(_, end)| end <= data.len());

    if direct {
        let offset = crop_top * src_w + crop_left;
        return Ok(PlaneBuffer::from_parts(
            data,
            PlaneLayout {
                width: dst_w,
                height: dst_h,
                stride: src_w,
                offset,
            },
        ));
    }

    let len = dst_w
        .checked_mul(dst_h)
        .ok_or_else(|| DecodeError::Bitstream("visible plane dimensions overflow usize".into()))?;
    let mut out = try_vec![T::default(); len, "visible image plane"];
    if rows == 0 || src_w == 0 || data.is_empty() {
        return Ok(PlaneBuffer::tight(out, dst_w, dst_h));
    }
    let src_y0 = crop_top.min(rows - 1);
    let src_x0 = crop_left.min(src_w - 1);
    for (row, dst) in out.chunks_exact_mut(dst_w).enumerate() {
        let src_y = (src_y0 + row).min(rows - 1);
        let src_base = src_y * src_w;
        let available = (data.len() - src_base).min(src_w);
        if available == 0 {
            continue;
        }
        let src_x = src_x0.min(available - 1);
        let copy = dst_w.min(available - src_x);
        let (exact, pad) = dst.split_at_mut(copy);
        exact.copy_from_slice(&data[src_base + src_x..src_base + src_x + copy]);
        if !pad.is_empty() {
            pad.fill(data[src_base + src_x + copy - 1]);
        }
    }
    Ok(PlaneBuffer::tight(out, dst_w, dst_h))
}

fn visible_native_planes<T: Copy + Default>(
    planes: yuv::NativePlanes<T>,
    dw: usize,
    dh: usize,
    crop: HevcVisibleCrop,
) -> Result<PlanarImage<T>, DecodeError> {
    let y = plane_window(
        planes.y,
        planes.width,
        planes.height,
        crop.left,
        crop.top,
        dw,
        dh,
    )?;
    if planes.chroma.is_monochrome() {
        return Ok(PlanarImage {
            y,
            cb: None,
            cr: None,
        });
    }

    let sub_w = planes.chroma.sub_w();
    let sub_h = planes.chroma.sub_h();
    let coded_cw = planes.width.div_ceil(sub_w);
    let coded_ch = planes.height.div_ceil(sub_h);
    let cw = dw.div_ceil(sub_w);
    let ch = dh.div_ceil(sub_h);
    let crop_cx = crop.left / sub_w;
    let crop_cy = crop.top / sub_h;
    Ok(PlanarImage {
        y,
        cb: Some(plane_window(
            planes.cb, coded_cw, coded_ch, crop_cx, crop_cy, cw, ch,
        )?),
        cr: Some(plane_window(
            planes.cr, coded_cw, coded_ch, crop_cx, crop_cy, cw, ch,
        )?),
    })
}

fn native_to_visible_buffer(
    planes: yuv::NativeYuvPlanes,
    dw: usize,
    dh: usize,
    crop: HevcVisibleCrop,
) -> Result<YuvBuffer, DecodeError> {
    match planes {
        yuv::NativeYuvPlanes::U8(planes) => {
            Ok(YuvBuffer::U8(visible_native_planes(planes, dw, dh, crop)?))
        }
        yuv::NativeYuvPlanes::U16(planes) => {
            Ok(YuvBuffer::U16(visible_native_planes(planes, dw, dh, crop)?))
        }
    }
}

/// Decoded HDR gain map carried as a HEIF auxiliary image.
#[derive(Clone, Debug)]
pub struct GainMap {
    /// Luma-plane dimensions.
    pub width: u32,
    pub height: u32,
    /// Dimensions of each chroma plane. Both are zero for monochrome maps.
    pub chroma_width: u32,
    pub chroma_height: u32,
    /// Gain-map luma samples.
    pub y: SampleBuf,
    /// Gain-map blue-difference chroma samples. `None` for monochrome maps.
    pub cb: Option<SampleBuf>,
    /// Gain-map red-difference chroma samples. `None` for monochrome maps.
    pub cr: Option<SampleBuf>,
    /// Chroma subsampling signaled by the gain-map HEVC SPS. The explicit
    /// chroma dimensions remain authoritative after display orientation; this
    /// matters for a 90°/270° rotation of a coded 4:2:2 map.
    pub chroma: ChromaFormat,
    pub bit_depth: BitDepth,
    /// Color signalling attached to the gain-map item. This matters for
    /// converting a three-channel gain map from YCbCr to RGB gains.
    pub color: ColorMetadata,
    /// Source orientation from the gain-map item. Display-ready decode applies
    /// it to all present gain-map planes; raw-YUV decode leaves the samples in
    /// coded orientation, matching the parent image planes.
    pub orientation: Orientation,
    /// Opaque metadata item directly associated with the gain-map auxiliary
    /// image. Apple files normally store an XMP packet containing fields such
    /// as `HDRGainMapVersion` and `HDRGainMapHeadroom`.
    pub metadata: Option<Vec<u8>>,
}

impl GainMap {
    /// Number of gain channels represented by the decoded auxiliary image.
    #[inline]
    pub fn channels(&self) -> usize {
        if self.chroma.is_monochrome() { 1 } else { 3 }
    }
}

/// Strided, typed gain-map planes returned by [`Decoder::decode_yuv`].
#[derive(Clone, Debug)]
pub struct GainMapFrame {
    pub planes: YuvBuffer,
    pub bit_depth: BitDepth,
    pub chroma: ChromaFormat,
    pub color: ColorMetadata,
    pub orientation: Orientation,
    pub metadata: Option<Vec<u8>>,
}

impl GainMapFrame {
    #[inline]
    pub fn width(&self) -> usize {
        self.planes.width()
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.planes.height()
    }

    #[inline]
    pub fn channels(&self) -> usize {
        if self.chroma.is_monochrome() { 1 } else { 3 }
    }

    pub fn into_packed(self) -> Result<GainMap, DecodeError> {
        let width = self.planes.width() as u32;
        let height = self.planes.height() as u32;
        let (chroma_width, chroma_height) = match &self.planes {
            YuvBuffer::U8(planes) => planes
                .cb
                .as_ref()
                .map(|plane| (plane.width() as u32, plane.height() as u32))
                .unwrap_or((0, 0)),
            YuvBuffer::U16(planes) => planes
                .cb
                .as_ref()
                .map(|plane| (plane.width() as u32, plane.height() as u32))
                .unwrap_or((0, 0)),
        };
        let (y, cb, cr) = match self.planes {
            YuvBuffer::U8(planes) => (
                SampleBuf::U8(planes.y.into_tight()?),
                planes
                    .cb
                    .map(|plane| plane.into_tight().map(SampleBuf::U8))
                    .transpose()?,
                planes
                    .cr
                    .map(|plane| plane.into_tight().map(SampleBuf::U8))
                    .transpose()?,
            ),
            YuvBuffer::U16(planes) => (
                SampleBuf::U16(planes.y.into_tight()?),
                planes
                    .cb
                    .map(|plane| plane.into_tight().map(SampleBuf::U16))
                    .transpose()?,
                planes
                    .cr
                    .map(|plane| plane.into_tight().map(SampleBuf::U16))
                    .transpose()?,
            ),
        };
        Ok(GainMap {
            width,
            height,
            chroma_width,
            chroma_height,
            y,
            cb,
            cr,
            chroma: self.chroma,
            bit_depth: self.bit_depth,
            color: self.color,
            orientation: self.orientation,
            metadata: self.metadata,
        })
    }
}

/// Zero-copy-or-minimal-copy raw YCbCr result. Single-item HEIC images retain
/// their decoder allocation and expose the visible crop through plane offsets
/// and strides. Grid images are naturally tightly packed.
#[derive(Clone, Debug)]
pub struct DecodedYuv {
    pub planes: YuvBuffer,
    pub alpha: Option<SamplePlane>,
    pub gain_map: Option<GainMapFrame>,
    pub bit_depth: BitDepth,
    pub chroma: ChromaFormat,
    pub color: ColorMetadata,
    pub orientation: Orientation,
    pub clean_aperture: Option<CleanAperture>,
    pub pixel_aspect_ratio: Option<PixelAspectRatio>,
    pub exif: Option<Vec<u8>>,
}

impl DecodedYuv {
    #[inline]
    pub fn width(&self) -> usize {
        self.planes.width()
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.planes.height()
    }
}

/// Packed display-ready 8-bit RGB output.
#[derive(Clone, Debug)]
pub struct Rgb8Image {
    /// Interleaved RGB samples, exactly three bytes per pixel.
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// A fully decoded HEIF/HEIC image.
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB pixels, typed to the source bit depth.
    pub pixels: ImageBuffer,
    pub alpha: Option<SampleBuf>,
    /// Optional Apple HDR gain-map auxiliary image.
    pub gain_map: Option<GainMap>,
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

/// Keep malformed optional auxiliary images non-fatal, while still surfacing
/// allocation failure instead of aborting the process.
fn optional_auxiliary<T>(result: Result<T, DecodeError>) -> Result<Option<T>, DecodeError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(err @ DecodeError::AllocationFailed { .. }) => Err(err),
        Err(_) => Ok(None),
    }
}

/// Decode the optional alpha auxiliary item (`auxl`) into a luma-only plane,
/// applying the alpha HEVC conformance-window offset. `clap` is intentionally
/// not applied here.
fn decode_alpha_plane_native(
    decoder: &Decoder,
    file: &[u8],
    heif: &heif::HeifFile,
    dw: usize,
    dh: usize,
) -> Result<Option<SamplePlane>, DecodeError> {
    let Some(alpha) = heif.alpha.as_ref() else {
        return Ok(None);
    };
    if alpha.hvcc.is_empty() {
        return Ok(None);
    }
    let (Ok(start), Ok(length)) = (
        usize::try_from(alpha.data_offset),
        usize::try_from(alpha.data_length),
    ) else {
        return Ok(None);
    };
    let Some(end) = start.checked_add(length) else {
        return Ok(None);
    };
    let Some(sample) = file.get(start..end) else {
        return Ok(None);
    };
    let Some((planes, _)) = optional_auxiliary(decode_hevc_item_native(
        sample,
        &alpha.hvcc,
        decoder.exec(),
        Some(decoder.pool()),
    ))?
    else {
        return Ok(None);
    };
    let crop = visible_crop_from_hvcc(&alpha.hvcc);
    let plane = match planes {
        yuv::NativeYuvPlanes::U8(planes) => SamplePlane::U8(plane_window(
            planes.y,
            planes.width,
            planes.height,
            crop.left,
            crop.top,
            dw,
            dh,
        )?),
        yuv::NativeYuvPlanes::U16(planes) => SamplePlane::U16(plane_window(
            planes.y,
            planes.width,
            planes.height,
            crop.left,
            crop.top,
            dw,
            dh,
        )?),
    };
    Ok(Some(plane))
}

/// Decode alpha for display-ready RGB output and tightly pack the visible rows.
fn decode_alpha_plane(
    decoder: &Decoder,
    file: &[u8],
    heif: &heif::HeifFile,
    dw: usize,
    dh: usize,
) -> Result<Option<SampleBuf>, DecodeError> {
    decode_alpha_plane_native(decoder, file, heif, dw, dh)?
        .map(|plane| match plane {
            SamplePlane::U8(plane) => plane.into_tight().map(SampleBuf::U8),
            SamplePlane::U16(plane) => plane.into_tight().map(SampleBuf::U16),
        })
        .transpose()
}

impl DecodedGainMapImage {
    fn apply_orientation(self) -> Result<Self, DecodeError> {
        let Self {
            width,
            height,
            chroma_width,
            chroma_height,
            y,
            cb,
            cr,
            chroma,
            bit_depth,
            color,
            orientation,
        } = self;
        let (oriented_width, oriented_height) = oriented_dimensions(width, height, orientation);
        let (oriented_chroma_width, oriented_chroma_height) = if chroma.is_monochrome() {
            (0, 0)
        } else {
            oriented_dimensions(chroma_width, chroma_height, orientation)
        };
        Ok(Self {
            width: oriented_width,
            height: oriented_height,
            chroma_width: oriented_chroma_width,
            chroma_height: oriented_chroma_height,
            y: rotate_sample_buf(y, width, height, orientation)?,
            cb: cb
                .map(|plane| rotate_sample_buf(plane, chroma_width, chroma_height, orientation))
                .transpose()?,
            cr: cr
                .map(|plane| rotate_sample_buf(plane, chroma_width, chroma_height, orientation))
                .transpose()?,
            chroma,
            bit_depth,
            color,
            orientation,
        })
    }
}

fn rotate_sample_buf(
    samples: SampleBuf,
    width: u32,
    height: u32,
    orientation: Orientation,
) -> Result<SampleBuf, DecodeError> {
    match samples {
        SampleBuf::U8(samples) => {
            rotate_luma(width as usize, height as usize, samples, orientation).map(SampleBuf::U8)
        }
        SampleBuf::U16(samples) => {
            rotate_luma(width as usize, height as usize, samples, orientation).map(SampleBuf::U16)
        }
    }
}

fn decode_gain_map_frame_item(
    decoder: &Decoder,
    file: &[u8],
    item: &heif::HeifItem,
) -> Result<GainMapFrame, DecodeError> {
    let start = usize::try_from(item.data_offset)
        .map_err(|_| DecodeError::Bitstream("gain-map data offset exceeds usize".into()))?;
    let length = usize::try_from(item.data_length)
        .map_err(|_| DecodeError::Bitstream("gain-map data length exceeds usize".into()))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| DecodeError::Bitstream("gain-map data range overflows usize".into()))?;
    let sample = file
        .get(start..end)
        .ok_or_else(|| DecodeError::Bitstream("gain-map data extends past file end".into()))?;
    let (planes, vui_color) =
        decode_hevc_item_native(sample, &item.hvcc, decoder.exec(), Some(decoder.pool()))?;
    let crop = visible_crop_from_hvcc(&item.hvcc);
    let (coded_w, coded_h) = planes.dims();
    let (fallback_w, fallback_h) = config::parse_hvcc_full(&item.hvcc)
        .ok()
        .map(|(sps, _)| {
            (
                sps.width.saturating_sub(sps.crop_left + sps.crop_right),
                sps.height.saturating_sub(sps.crop_top + sps.crop_bottom),
            )
        })
        .unwrap_or((coded_w as u32, coded_h as u32));
    let width = if item.display_w == 0 {
        fallback_w
    } else {
        item.display_w
    };
    let height = if item.display_h == 0 {
        fallback_h
    } else {
        item.display_h
    };
    decoder.check_dims(width as usize, height as usize)?;
    let bit_depth = planes.bit_depth();
    let chroma = planes.chroma();
    Ok(GainMapFrame {
        planes: native_to_visible_buffer(planes, width as usize, height as usize, crop)?,
        bit_depth,
        chroma,
        color: gain_map_color_metadata(&item.color, vui_color),
        orientation: item.orientation,
        metadata: None,
    })
}

fn decode_gain_map_frame_grid(
    decoder: &Decoder,
    file: &[u8],
    grid: &heif::GridInfo,
) -> Result<GainMapFrame, DecodeError> {
    let decoded = decode_native_grid_image(decoder, file, grid)?;
    let item_color = if grid.color.is_empty() {
        grid.tiles
            .iter()
            .find(|tile| !tile.color.is_empty())
            .map(|tile| &tile.color)
            .unwrap_or(&grid.tiles[0].color)
    } else {
        &grid.color
    };
    Ok(GainMapFrame {
        planes: decoded.planes,
        bit_depth: decoded.bit_depth,
        chroma: decoded.chroma,
        color: gain_map_color_metadata(item_color, decoded.vui_color),
        orientation: grid.orientation,
        metadata: None,
    })
}

fn decode_gain_map_frame(
    decoder: &Decoder,
    file: &[u8],
    heif: &heif::HeifFile,
) -> Result<Option<GainMapFrame>, DecodeError> {
    let Some(gain) = heif.gain_map.as_ref() else {
        return Ok(None);
    };
    let result = match &gain.image {
        heif::HeifImageSource::Item(item) => decode_gain_map_frame_item(decoder, file, item),
        heif::HeifImageSource::Grid(grid) => decode_gain_map_frame_grid(decoder, file, grid),
    };
    let Some(mut decoded) = optional_auxiliary(result)? else {
        return Ok(None);
    };
    decoded.metadata = gain.metadata.clone();
    Ok(Some(decoded))
}

fn decode_gain_map(
    decoder: &Decoder,
    file: &[u8],
    heif: &heif::HeifFile,
    apply_orientation: bool,
) -> Result<Option<GainMap>, DecodeError> {
    let Some(frame) = decode_gain_map_frame(decoder, file, heif)? else {
        return Ok(None);
    };
    let packed = frame.into_packed()?;
    if !apply_orientation {
        return Ok(Some(packed));
    }
    let decoded = DecodedGainMapImage {
        width: packed.width,
        height: packed.height,
        chroma_width: packed.chroma_width,
        chroma_height: packed.chroma_height,
        y: packed.y,
        cb: packed.cb,
        cr: packed.cr,
        chroma: packed.chroma,
        bit_depth: packed.bit_depth,
        color: packed.color,
        orientation: packed.orientation,
    }
    .apply_orientation()?;
    Ok(Some(GainMap {
        width: decoded.width,
        height: decoded.height,
        chroma_width: decoded.chroma_width,
        chroma_height: decoded.chroma_height,
        y: decoded.y,
        cb: decoded.cb,
        cr: decoded.cr,
        chroma: decoded.chroma,
        bit_depth: decoded.bit_depth,
        color: decoded.color,
        orientation: decoded.orientation,
        metadata: packed.metadata,
    }))
}

fn gain_map_color_metadata(item_color: &ColorMetadata, vui_color: Cicp) -> ColorMetadata {
    let mut color = item_color.clone();
    if vui_color.matrix != MatrixCoefficients::Unspecified {
        color.cicp = Some(vui_color);
    }
    color
}

/// Decode a HEIF/HEIC file and return typed, owning, strided raw YCbCr
/// planes using a default [`Decoder`]. Eight-bit items use `u8`; 10/12-bit
/// items use `u16`. Single-item images retain coded stride and padding.
pub fn decode_heic_yuv(file: &[u8]) -> Result<DecodedYuv, DecodeError> {
    Decoder::default().decode_yuv(file)
}

/// Decode typed, strided raw YCbCr planes with one explicit settings value.
pub fn decode_heic_yuv_with_settings(
    file: &[u8],
    settings: &HeicSettings,
) -> Result<DecodedYuv, DecodeError> {
    Decoder::from_settings(*settings).decode_yuv(file)
}

pub(crate) fn decode_heic_yuv_with(
    decoder: &Decoder,
    file: &[u8],
) -> Result<DecodedYuv, DecodeError> {
    let heif = heif::parse(
        file,
        decoder.limits(),
        heif::HeifParseOptions {
            load_alpha: decoder.decodes_alpha(),
            load_gain_map: decoder.decodes_gain_map(),
        },
    )?;

    if let Some(grid) = &heif.grid {
        return decode_grid_yuv(decoder, file, grid, &heif);
    }

    let start = usize::try_from(heif.primary.data_offset)
        .map_err(|_| DecodeError::Bitstream("image data offset exceeds usize".into()))?;
    let length = usize::try_from(heif.primary.data_length)
        .map_err(|_| DecodeError::Bitstream("image data length exceeds usize".into()))?;
    let end = start
        .checked_add(length)
        .ok_or_else(|| DecodeError::Bitstream("image data range overflows usize".into()))?;
    let sample = file
        .get(start..end)
        .ok_or_else(|| DecodeError::Bitstream("image data extends past file end".into()))?;
    let (planes, _) = decode_hevc_item_native(
        sample,
        &heif.primary.hvcc,
        decoder.exec(),
        Some(decoder.pool()),
    )?;
    let width = heif.primary.display_w as usize;
    let height = heif.primary.display_h as usize;
    decoder.check_dims(width, height)?;
    let bit_depth = planes.bit_depth();
    let chroma = planes.chroma();
    let crop = visible_crop_from_hvcc(&heif.primary.hvcc);
    let visible = native_to_visible_buffer(planes, width, height, crop)?;
    let alpha = if decoder.decodes_alpha() {
        decode_alpha_plane_native(decoder, file, &heif, width, height)?
    } else {
        None
    };
    let gain_map = if decoder.decodes_gain_map() {
        decode_gain_map_frame(decoder, file, &heif)?
    } else {
        None
    };

    Ok(DecodedYuv {
        planes: visible,
        alpha,
        gain_map,
        bit_depth,
        chroma,
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
    exec: &'a exec::ExecContext,
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

trait NativeGridSample: Copy + Default + Send {
    fn select(planes: &yuv::NativeYuvPlanes) -> Option<&yuv::NativePlanes<Self>>;
}

impl NativeGridSample for u8 {
    fn select(planes: &yuv::NativeYuvPlanes) -> Option<&yuv::NativePlanes<Self>> {
        match planes {
            yuv::NativeYuvPlanes::U8(planes) => Some(planes),
            yuv::NativeYuvPlanes::U16(_) => None,
        }
    }
}

impl NativeGridSample for u16 {
    fn select(planes: &yuv::NativeYuvPlanes) -> Option<&yuv::NativePlanes<Self>> {
        match planes {
            yuv::NativeYuvPlanes::U8(_) => None,
            yuv::NativeYuvPlanes::U16(planes) => Some(planes),
        }
    }
}

fn stitch_native_yuv_band<T: NativeGridSample>(
    ctx: &YuvGridCtx<'_>,
    band_row: usize,
    y_band: &mut [T],
    cb_band: &mut [T],
    cr_band: &mut [T],
) -> Result<(), DecodeError> {
    for col in 0..ctx.cols {
        let tile_idx = band_row * ctx.cols + col;
        let Some(tile) = ctx.tiles.get(tile_idx) else {
            break;
        };
        let Ok(start) = usize::try_from(tile.data_offset) else {
            continue;
        };
        let Ok(length) = usize::try_from(tile.data_length) else {
            continue;
        };
        let Some(end) = start.checked_add(length) else {
            continue;
        };
        let Some(sample) = ctx.file.get(start..end) else {
            continue;
        };
        let hvcc = if tile.hvcc.is_empty() {
            ctx.fallback_hvcc
        } else {
            &tile.hvcc
        };
        if hvcc.is_empty() {
            continue;
        }
        let Some((native, _)) =
            optional_auxiliary(decode_hevc_item_native(sample, hvcc, ctx.exec, None))?
        else {
            continue;
        };
        let Some(planes) = T::select(&native) else {
            continue;
        };

        let dst_x = col * ctx.tile_w;
        let dst_y_base = band_row * ctx.tile_h;
        let copy_w = ctx.tile_w.min(ctx.out_w.saturating_sub(dst_x));
        let copy_h = ctx.tile_h.min(ctx.out_h.saturating_sub(dst_y_base));
        if copy_w == 0 || copy_h == 0 || planes.width == 0 || planes.height == 0 {
            continue;
        }
        let crop_left = ctx.tile_crop_left.min(planes.width - 1);
        let crop_top = ctx.tile_crop_top.min(planes.height - 1);
        for y in 0..copy_h {
            let dst = &mut y_band[y * ctx.out_w + dst_x..][..copy_w];
            let src_y = (crop_top + y).min(planes.height - 1);
            let src_base = src_y * planes.width;
            if src_base >= planes.y.len() {
                continue;
            }
            let available = (planes.y.len() - src_base).min(planes.width);
            if available == 0 {
                continue;
            }
            let src_x = crop_left.min(available - 1);
            let exact = copy_w.min(available - src_x);
            let (copied, pad) = dst.split_at_mut(exact);
            copied.copy_from_slice(&planes.y[src_base + src_x..src_base + src_x + exact]);
            if !pad.is_empty() {
                pad.fill(planes.y[src_base + src_x + exact - 1]);
            }
        }

        if !ctx.has_chroma || planes.cb.is_empty() || planes.cr.is_empty() {
            continue;
        }
        let plane_cw = planes.width.div_ceil(ctx.sub_w);
        let plane_ch = planes.height.div_ceil(ctx.sub_h);
        if plane_cw == 0 || plane_ch == 0 {
            continue;
        }
        let dst_x = col * ctx.tile_cw;
        let copy_w = ctx.tile_cw.min(ctx.cw.saturating_sub(dst_x));
        let copy_h = ctx
            .tile_ch
            .min(ctx.ch.saturating_sub(band_row * ctx.tile_ch));
        if copy_w == 0 || copy_h == 0 {
            continue;
        }
        let crop_left = (ctx.tile_crop_left / ctx.sub_w).min(plane_cw - 1);
        let crop_top = (ctx.tile_crop_top / ctx.sub_h).min(plane_ch - 1);
        for y in 0..copy_h {
            let dst_start = y * ctx.cw + dst_x;
            let src_y = (crop_top + y).min(plane_ch - 1);
            let src_base = src_y * plane_cw;
            if src_base >= planes.cb.len() || src_base >= planes.cr.len() {
                continue;
            }
            let available = (planes.cb.len() - src_base)
                .min(plane_cw)
                .min((planes.cr.len() - src_base).min(plane_cw));
            if available == 0 {
                continue;
            }
            let src_x = crop_left.min(available - 1);
            let exact = copy_w.min(available - src_x);

            let cb_dst = &mut cb_band[dst_start..dst_start + copy_w];
            let (copied, pad) = cb_dst.split_at_mut(exact);
            copied.copy_from_slice(&planes.cb[src_base + src_x..src_base + src_x + exact]);
            if !pad.is_empty() {
                pad.fill(planes.cb[src_base + src_x + exact - 1]);
            }

            let cr_dst = &mut cr_band[dst_start..dst_start + copy_w];
            let (copied, pad) = cr_dst.split_at_mut(exact);
            copied.copy_from_slice(&planes.cr[src_base + src_x..src_base + src_x + exact]);
            if !pad.is_empty() {
                pad.fill(planes.cr[src_base + src_x + exact - 1]);
            }
        }
    }
    Ok(())
}

fn fill_native_grid_yuv<T: NativeGridSample>(
    decoder: &Decoder,
    rows: usize,
    ctx: &YuvGridCtx<'_>,
) -> Result<PlanarImage<T>, DecodeError> {
    let y_total = ctx
        .out_w
        .checked_mul(ctx.out_h)
        .ok_or_else(|| DecodeError::Bitstream("grid luma dimensions overflow usize".into()))?;
    let c_total = ctx
        .cw
        .checked_mul(ctx.ch)
        .ok_or_else(|| DecodeError::Bitstream("grid chroma dimensions overflow usize".into()))?;
    let y_stride = ctx.tile_h.saturating_mul(ctx.out_w);
    let c_stride = ctx.tile_ch.saturating_mul(ctx.cw);
    let pool = decoder.pool();

    let (y, cb, cr) = if pool.threads() > 1 && rows > 1 {
        let y = threadpool::DisjointMut::new(try_vec![T::default(); y_total, "grid luma plane"]);
        let cb = threadpool::DisjointMut::new(try_vec![T::default(); c_total, "grid Cb plane"]);
        let cr = threadpool::DisjointMut::new(try_vec![T::default(); c_total, "grid Cr plane"]);
        let error = std::sync::Mutex::new(None);
        threadpool::parallel_for(pool, rows, |row| {
            if error.lock().unwrap().is_some() {
                return;
            }
            let y_lo = row * y_stride;
            let y_hi = ((row + 1) * y_stride).min(y_total);
            if y_lo >= y_hi {
                return;
            }
            let mut y_band = y.slice_mut(y_lo..y_hi);
            let c_lo = row * c_stride;
            let c_hi = ((row + 1) * c_stride).min(c_total);
            let result = if c_lo < c_hi {
                let mut cb_band = cb.slice_mut(c_lo..c_hi);
                let mut cr_band = cr.slice_mut(c_lo..c_hi);
                stitch_native_yuv_band(ctx, row, &mut y_band, &mut cb_band, &mut cr_band)
            } else {
                stitch_native_yuv_band(ctx, row, &mut y_band, &mut [], &mut [])
            };
            if let Err(err) = result {
                *error.lock().unwrap() = Some(err);
            }
        });
        if let Some(err) = error.into_inner().unwrap() {
            return Err(err);
        }
        (y.into_inner(), cb.into_inner(), cr.into_inner())
    } else {
        let mut y = try_vec![T::default(); y_total, "grid luma plane"];
        let mut cb = try_vec![T::default(); c_total, "grid Cb plane"];
        let mut cr = try_vec![T::default(); c_total, "grid Cr plane"];
        for row in 0..rows {
            let y_lo = row * y_stride;
            let y_hi = ((row + 1) * y_stride).min(y_total);
            if y_lo >= y_hi {
                continue;
            }
            let c_lo = row * c_stride;
            let c_hi = ((row + 1) * c_stride).min(c_total);
            if c_lo < c_hi {
                stitch_native_yuv_band(
                    ctx,
                    row,
                    &mut y[y_lo..y_hi],
                    &mut cb[c_lo..c_hi],
                    &mut cr[c_lo..c_hi],
                )?;
            } else {
                stitch_native_yuv_band(ctx, row, &mut y[y_lo..y_hi], &mut [], &mut [])?;
            }
        }
        (y, cb, cr)
    };

    Ok(PlanarImage {
        y: PlaneBuffer::tight(y, ctx.out_w, ctx.out_h),
        cb: ctx
            .has_chroma
            .then(|| PlaneBuffer::tight(cb, ctx.cw, ctx.ch)),
        cr: ctx
            .has_chroma
            .then(|| PlaneBuffer::tight(cr, ctx.cw, ctx.ch)),
    })
}

struct NativeGridImage {
    planes: YuvBuffer,
    bit_depth: BitDepth,
    chroma: ChromaFormat,
    vui_color: Cicp,
}

fn decode_native_grid_image(
    decoder: &Decoder,
    file: &[u8],
    grid: &heif::GridInfo,
) -> Result<NativeGridImage, DecodeError> {
    let out_w = grid.output_width as usize;
    let out_h = grid.output_height as usize;
    decoder.check_dims(out_w, out_h)?;
    if grid.tiles.is_empty() {
        return Err(DecodeError::Bitstream("grid has no tiles".into()));
    }
    let cols = grid.cols as usize;
    let rows = grid.rows as usize;
    let fallback_hvcc = grid
        .tiles
        .iter()
        .find(|tile| !tile.hvcc.is_empty())
        .map(|tile| tile.hvcc.clone())
        .unwrap_or_default();
    let hvcc = if grid.tiles[0].hvcc.is_empty() {
        &fallback_hvcc
    } else {
        &grid.tiles[0].hvcc
    };
    if hvcc.is_empty() {
        return Err(DecodeError::MissingBox("grid hvcC property"));
    }
    let (sps, _) = config::parse_hvcc_full(hvcc)?;
    let chroma = sps.chroma;
    let bit_depth = match sps.bit_depth_luma {
        10 => BitDepth::Ten,
        12 => BitDepth::Twelve,
        _ => BitDepth::Eight,
    };
    let tile_w = sps.width.saturating_sub(sps.crop_left + sps.crop_right) as usize;
    let tile_h = sps.height.saturating_sub(sps.crop_top + sps.crop_bottom) as usize;
    let (tile_w, tile_h) = if tile_w != 0 && tile_h != 0 {
        (tile_w, tile_h)
    } else {
        (out_w.div_ceil(cols), out_h.div_ceil(rows))
    };
    let sub_w = chroma.sub_w();
    let sub_h = chroma.sub_h();
    let has_chroma = !chroma.is_monochrome();
    let (cw, ch) = if has_chroma {
        (out_w.div_ceil(sub_w), out_h.div_ceil(sub_h))
    } else {
        (0, 0)
    };
    let ctx = YuvGridCtx {
        file,
        exec: decoder.exec(),
        tiles: &grid.tiles,
        fallback_hvcc: &fallback_hvcc,
        cols,
        out_w,
        out_h,
        cw,
        ch,
        tile_w,
        tile_h,
        tile_cw: tile_w.div_ceil(sub_w),
        tile_ch: tile_h.div_ceil(sub_h),
        tile_crop_left: sps.crop_left as usize,
        tile_crop_top: sps.crop_top as usize,
        sub_w,
        sub_h,
        has_chroma,
    };
    let planes = if bit_depth == BitDepth::Eight {
        YuvBuffer::U8(fill_native_grid_yuv::<u8>(decoder, rows, &ctx)?)
    } else {
        YuvBuffer::U16(fill_native_grid_yuv::<u16>(decoder, rows, &ctx)?)
    };
    Ok(NativeGridImage {
        planes,
        bit_depth,
        chroma,
        vui_color: Cicp {
            primaries: Primaries::from_u8(sps.color_primaries),
            transfer: TransferFunction::from_u8(sps.transfer_characteristics),
            matrix: MatrixCoefficients::from_u8(sps.matrix_coefficients),
            full_range: sps.video_full_range,
        },
    })
}

/// Composite a tiled grid into a strided typed YUV frame. Grid output is
/// tightly packed because stitching writes directly into final-sized planes.
fn decode_grid_yuv(
    decoder: &Decoder,
    file: &[u8],
    grid: &heif::GridInfo,
    heif_file: &heif::HeifFile,
) -> Result<DecodedYuv, DecodeError> {
    let decoded = decode_native_grid_image(decoder, file, grid)?;
    let width = grid.output_width as usize;
    let height = grid.output_height as usize;
    let alpha = if decoder.decodes_alpha() {
        decode_alpha_plane_native(decoder, file, heif_file, width, height)?
    } else {
        None
    };
    let gain_map = if decoder.decodes_gain_map() {
        decode_gain_map_frame(decoder, file, heif_file)?
    } else {
        None
    };
    Ok(DecodedYuv {
        planes: decoded.planes,
        alpha,
        gain_map,
        bit_depth: decoded.bit_depth,
        chroma: decoded.chroma,
        color: heif_file.primary.color.clone(),
        orientation: grid.orientation,
        clean_aperture: heif_file.primary.clap,
        pixel_aspect_ratio: heif_file.primary.pasp,
        exif: heif_file.exif.clone(),
    })
}

#[inline]
fn hvcc_nal_length_size(hvcc: &[u8]) -> Result<usize, DecodeError> {
    // HEVCDecoderConfigurationRecord.lengthSizeMinusOne occupies the low two
    // bits of byte 21; samples may therefore use 1, 2, 3, or 4-byte lengths.
    let length_size_minus_one = *hvcc
        .get(21)
        .ok_or_else(|| DecodeError::ParamSet("hvcC too short for NAL length size".into()))?
        & 0x03;
    Ok(length_size_minus_one as usize + 1)
}

#[inline]
fn read_length_prefixed_nal_size(
    sample: &[u8],
    pos: &mut usize,
    length_size: usize,
) -> Result<usize, DecodeError> {
    let end = pos
        .checked_add(length_size)
        .ok_or_else(|| DecodeError::Bitstream("NAL length offset overflows usize".into()))?;
    let bytes = sample
        .get(*pos..end)
        .ok_or_else(|| DecodeError::Bitstream("truncated HEIF NAL length prefix".into()))?;
    *pos = end;

    let mut size = 0usize;
    for &byte in bytes {
        size = size
            .checked_shl(8)
            .and_then(|v| v.checked_add(byte as usize))
            .ok_or_else(|| DecodeError::Bitstream("HEIF NAL length overflows usize".into()))?;
    }
    Ok(size)
}

fn configure_still_inter_state(
    decoder: &mut decode::FullDecoder<'_>,
    sps: &config::Sps,
    pps: &config::Pps,
    hdr: &decode::SliceHeader,
) -> Result<(), DecodeError> {
    if !(sps.curr_pic_ref_enabled && pps.curr_pic_ref_enabled) {
        return Ok(());
    }

    // HEVC image items are independently decodable, so an SCC still picture
    // has no preceding DPB references. CurrPic is nevertheless a real active
    // long-term entry in RefPicList0/1. The video driver installs this entry;
    // the old HEIF path left both lists empty, causing every IBC PU to miss the
    // current-picture path and fall back to mid-grey prediction.
    let current = dpb::RefEntry {
        _dpb_index: usize::MAX,
        poc: 0,
        long_term: true,
    };
    let empty_rps = dpb::RpsPocs {
        before: Vec::new(),
        after: Vec::new(),
        lt: Vec::new(),
    };
    let (list0, list1) = dpb::Dpb::new(1).build_ref_lists(
        &empty_rps,
        hdr.num_ref_idx_l0,
        hdr.num_ref_idx_l1,
        hdr.slice_type == inter::SLICE_B,
        Some(current),
        &hdr.list_mod_l0,
        &hdr.list_mod_l1,
    )?;
    decoder.set_inter_state(0, list0, list1, Vec::new());
    Ok(())
}

trait StillPictureOutput {
    type Planes;

    fn finish(
        decoder: &mut decode::FullDecoder<'_>,
        pool: Option<&threadpool::ThreadPool>,
    ) -> Result<Self::Planes, DecodeError>;
}

struct U16StillPicture;
struct NativeStillPicture;

impl StillPictureOutput for U16StillPicture {
    type Planes = yuv::YuvPlanes;

    #[inline]
    fn finish(
        decoder: &mut decode::FullDecoder<'_>,
        pool: Option<&threadpool::ThreadPool>,
    ) -> Result<Self::Planes, DecodeError> {
        decoder.finish_with(pool, pool)
    }
}

impl StillPictureOutput for NativeStillPicture {
    type Planes = yuv::NativeYuvPlanes;

    #[inline]
    fn finish(
        decoder: &mut decode::FullDecoder<'_>,
        pool: Option<&threadpool::ThreadPool>,
    ) -> Result<Self::Planes, DecodeError> {
        decoder.finish_native_with(pool, pool)
    }
}

fn decode_hevc_item_as<O: StillPictureOutput>(
    sample: &[u8],
    hvcc: &[u8],
    exec: &exec::ExecContext,
    pool: Option<&threadpool::ThreadPool>,
) -> Result<(O::Planes, Cicp), DecodeError> {
    use config::parse_hvcc_full;
    use decode::{FullDecoder, SliceHeader, parse_slice_header_full};

    struct StillSlice {
        source: Vec<u8>,
        rbsp: Vec<u8>,
        hdr: SliceHeader,
    }

    let (sps, pps) = parse_hvcc_full(hvcc)?;
    let nal_length_size = hvcc_nal_length_size(hvcc)?;

    let vui_color = Cicp {
        primaries: Primaries::from_u8(sps.color_primaries),
        transfer: TransferFunction::from_u8(sps.transfer_characteristics),
        matrix: MatrixCoefficients::from_u8(sps.matrix_coefficients),
        full_range: sps.video_full_range,
    };

    // Collect the complete access unit before decoding. Besides making malformed
    // length prefixes fail loudly, this lets us enforce the same WPP rule as the
    // video path: the whole-picture wavefront is valid only for one independent
    // slice segment.
    let mut slices = Vec::new();
    let mut pos = 0usize;
    while pos < sample.len() {
        let nlen = read_length_prefixed_nal_size(sample, &mut pos, nal_length_size)?;
        if nlen < 2 {
            return Err(DecodeError::Bitstream(
                "HEIF sample contains an undersized NAL unit".into(),
            ));
        }
        let end = pos
            .checked_add(nlen)
            .ok_or_else(|| DecodeError::Bitstream("NAL payload offset overflows usize".into()))?;
        let nal_bytes = sample
            .get(pos..end)
            .ok_or_else(|| DecodeError::Bitstream("truncated HEIF NAL payload".into()))?;
        pos = end;

        let nal_type = (nal_bytes[0] >> 1) & 0x3f;
        if nal_type > 31 {
            continue;
        }

        let source = nal_bytes.to_vec();
        let rbsp = bitreader::unescape_rbsp(&source);
        let hdr = parse_slice_header_full(&rbsp, &sps, &pps, nal_type)?;
        slices.push(StillSlice { source, rbsp, hdr });
    }

    let first = slices
        .first()
        .ok_or_else(|| DecodeError::Bitstream("no VCL slice NAL found".into()))?;
    if !first.hdr.first_slice_in_pic || first.hdr.dependent_slice_segment {
        return Err(DecodeError::Bitstream(
            "HEIF item does not begin with an independent first slice".into(),
        ));
    }

    let first_cabac = &first.rbsp[first.hdr.cabac_offset.min(first.rbsp.len())..];
    let mut decoder = FullDecoder::new_with_exec(
        first_cabac,
        sps.clone(),
        pps.clone(),
        &first.hdr,
        exec.clone(),
    )?;
    configure_still_inter_state(&mut decoder, &sps, &pps, &first.hdr)?;

    let first_sub_starts: Vec<usize> = if first.hdr.entry_points.is_empty() {
        Vec::new()
    } else {
        let src_of = bitreader::rbsp_src_map(&first.source);
        wpp::substream_starts_rbsp_rel(
            &src_of,
            first.hdr.cabac_offset,
            &first.hdr.entry_points,
            first.rbsp.len(),
        )
    };

    // Match VideoDecoder: its row decoder currently supports only a single
    // whole-picture I-slice. Running it for the first segment of a multi-slice
    // HEIF item decodes past that segment and then decodes later segments again.
    let ran_wavefront = if slices.len() == 1 && first.hdr.slice_type == inter::SLICE_I {
        match pool {
            Some(p) => decoder.try_decode_wavefront(&first.rbsp, &first.source, &first.hdr, p)?,
            None => false,
        }
    } else {
        false
    };
    if !ran_wavefront {
        let starts = if first_sub_starts.is_empty() {
            None
        } else {
            Some((first_cabac, first_sub_starts.as_slice()))
        };
        decoder.decode_slice_ctx(first.hdr.slice_segment_address, starts)?;
    }

    for segment in slices.iter().skip(1) {
        if segment.hdr.first_slice_in_pic {
            return Err(DecodeError::Bitstream(
                "HEIF image item contains more than one coded picture".into(),
            ));
        }
        if !segment.hdr.dependent_slice_segment {
            configure_still_inter_state(&mut decoder, &sps, &pps, &segment.hdr)?;
        }

        let sub_starts: Vec<usize> = if segment.hdr.entry_points.is_empty() {
            Vec::new()
        } else {
            let src_of = bitreader::rbsp_src_map(&segment.source);
            wpp::substream_starts_rbsp_rel(
                &src_of,
                segment.hdr.cabac_offset,
                &segment.hdr.entry_points,
                segment.rbsp.len(),
            )
        };
        let cabac = &segment.rbsp[segment.hdr.cabac_offset.min(segment.rbsp.len())..];
        decoder.decode_segment(cabac, &segment.hdr, &sub_starts)?;
    }

    // Keep the still-image path bit-identical to VideoDecoder. The parallel
    // chroma deblock kernel is intentionally not selected there because it is
    // not yet bit-exact; SAO can remain parallel.
    Ok((O::finish(&mut decoder, pool)?, vui_color))
}

fn decode_hevc_item(
    sample: &[u8],
    hvcc: &[u8],
    exec: &exec::ExecContext,
    pool: Option<&threadpool::ThreadPool>,
) -> Result<(yuv::YuvPlanes, Cicp), DecodeError> {
    decode_hevc_item_as::<U16StillPicture>(sample, hvcc, exec, pool)
}

fn decode_hevc_item_native(
    sample: &[u8],
    hvcc: &[u8],
    exec: &exec::ExecContext,
    pool: Option<&threadpool::ThreadPool>,
) -> Result<(yuv::NativeYuvPlanes, Cicp), DecodeError> {
    decode_hevc_item_as::<NativeStillPicture>(sample, hvcc, exec, pool)
}

/// Decode a HEIF/HEIC file to display-ready pixels using a default [`Decoder`].
pub fn decode_heic(file: &[u8]) -> Result<DecodedImage, DecodeError> {
    Decoder::default().decode(file)
}

/// Decode a display-ready image with one explicit settings value. This creates
/// a decoder and worker pool for the call; reuse [`Decoder::from_settings`] for
/// a sequence of images.
pub fn decode_heic_with_settings(
    file: &[u8],
    settings: &HeicSettings,
) -> Result<DecodedImage, DecodeError> {
    Decoder::from_settings(*settings).decode(file)
}

pub(crate) fn decode_heic_with(
    decoder: &Decoder,
    file: &[u8],
) -> Result<DecodedImage, DecodeError> {
    decode_heic_impl(decoder, file, true)
}

fn decode_heic_impl(
    decoder: &Decoder,
    file: &[u8],
    decode_auxiliary_images: bool,
) -> Result<DecodedImage, DecodeError> {
    let heif = heif::parse(
        file,
        decoder.limits(),
        heif::HeifParseOptions {
            load_alpha: decode_auxiliary_images && decoder.decodes_alpha(),
            load_gain_map: decode_auxiliary_images && decoder.decodes_gain_map(),
        },
    )?;

    if let Some(grid) = &heif.grid {
        return decode_grid(decoder, file, grid, &heif, decode_auxiliary_images);
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
    let (yuv_planes, vui_color) = decode_hevc_item(
        &file[start..end],
        &heif.primary.hvcc,
        decoder.exec(),
        Some(decoder.pool()),
    )?;
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
    )?;

    let alpha = if decode_auxiliary_images && decoder.decodes_alpha() {
        decode_alpha_plane(decoder, file, &heif, dw, dh)?
    } else {
        None
    };
    let gain_map = if decode_auxiliary_images && decoder.decodes_gain_map() {
        decode_gain_map(decoder, file, &heif, true)?
    } else {
        None
    };

    let oriented = apply_orientation(dw as u32, dh as u32, rgb, alpha, heif.primary.orientation)?;

    Ok(DecodedImage {
        width: oriented.width,
        height: oriented.height,
        pixels: oriented.pixels,
        alpha: oriented.alpha,
        gain_map,
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
    exec: &'a exec::ExecContext,
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
) -> Result<(), DecodeError> {
    let dst_y_base = band_row * ctx.tile_h;
    for col in 0..ctx.cols {
        let tile_idx = band_row * ctx.cols + col;
        let tile = match ctx.tiles.get(tile_idx) {
            Some(tile) => tile,
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
        let Some((yuv, _)) = optional_auxiliary(decode_hevc_item(
            &ctx.file[start..end],
            hvcc,
            ctx.exec,
            None,
        ))?
        else {
            continue;
        };

        let dst_x = col * ctx.tile_w;
        let copy_w = ctx.tile_w.min(ctx.out_w.saturating_sub(dst_x));
        let copy_h = ctx.tile_h.min(ctx.out_h.saturating_sub(dst_y_base));
        if copy_w == 0 || copy_h == 0 {
            continue;
        }

        let tile_buf = yuv::yuv_to_rgb_window_with_color_pool(
            &yuv,
            ctx.tile_w,
            ctx.tile_h,
            ctx.tile_crop_left,
            ctx.tile_crop_top,
            ctx.color_enc,
            None,
        )?;
        let src = match pick(&tile_buf) {
            Some(samples) => samples,
            None => continue,
        };
        let channels = ctx.channels;
        for y in 0..copy_h {
            let src_start = y * ctx.tile_w * channels;
            let dst_start = (y * ctx.out_w + dst_x) * channels;
            band[dst_start..dst_start + copy_w * channels]
                .copy_from_slice(&src[src_start..src_start + copy_w * channels]);
        }
    }
    Ok(())
}

/// Allocate the output plane and fill each of the `rows` horizontal bands.
fn fill_grid_bands<T: Copy + Default + Send>(
    decoder: &Decoder,
    total: usize,
    rows: usize,
    band_stride: usize,
    ctx: &GridCtx<'_>,
    pick: &(dyn Fn(&ImageBuffer) -> Option<&[T]> + Sync),
) -> Result<Vec<T>, DecodeError> {
    let pool = decoder.pool();
    if pool.threads() > 1 && rows > 1 {
        let output = threadpool::DisjointMut::new(try_vec![T::default(); total, "grid RGB output"]);
        let error = std::sync::Mutex::new(None);
        threadpool::parallel_for(pool, rows, |row| {
            if error.lock().unwrap().is_some() {
                return;
            }
            let lo = row * band_stride;
            let hi = ((row + 1) * band_stride).min(total);
            if lo >= hi {
                return;
            }
            let mut band = output.slice_mut(lo..hi);
            if let Err(err) = stitch_grid_band(ctx, row, &mut band, pick) {
                *error.lock().unwrap() = Some(err);
            }
        });
        if let Some(err) = error.into_inner().unwrap() {
            return Err(err);
        }
        return Ok(output.into_inner());
    }

    let mut output = try_vec![T::default(); total, "grid RGB output"];
    for row in 0..rows {
        let lo = row * band_stride;
        let hi = ((row + 1) * band_stride).min(total);
        if lo < hi {
            stitch_grid_band(ctx, row, &mut output[lo..hi], pick)?;
        }
    }
    Ok(output)
}

/// Decode a tiled (grid) HEIC: decode each tile independently then stitch.
fn decode_grid(
    decoder: &Decoder,
    file: &[u8],
    grid: &heif::GridInfo,
    heif_file: &heif::HeifFile,
    decode_auxiliary_images: bool,
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
        exec: decoder.exec(),
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
        )?;
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
        )?;
        if is_mono {
            ImageBuffer::Luma16(v)
        } else {
            ImageBuffer::Rgb16(v)
        }
    };

    let alpha = if decode_auxiliary_images && decoder.decodes_alpha() {
        decode_alpha_plane(decoder, file, heif_file, out_w, out_h)?
    } else {
        None
    };
    let gain_map = if decode_auxiliary_images && decoder.decodes_gain_map() {
        decode_gain_map(decoder, file, heif_file, true)?
    } else {
        None
    };
    let oriented = apply_orientation(out_w as u32, out_h as u32, out_buf, alpha, grid.orientation)?;

    Ok(DecodedImage {
        width: oriented.width,
        height: oriented.height,
        pixels: oriented.pixels,
        alpha: oriented.alpha,
        gain_map,
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

/// Decode to display-ready packed 8-bit RGB using a default [`Decoder`].
/// Monochrome images are expanded to gray RGB. The returned buffer always has
/// exactly three bytes per pixel.
pub fn decode_heic_rgb8(file: &[u8]) -> Result<Rgb8Image, DecodeError> {
    Decoder::default().decode_rgb8(file)
}

/// Decode packed RGB8 with one explicit settings value. This output cannot
/// expose auxiliary planes, so alpha and gain-map payloads are never decoded.
pub fn decode_heic_rgb8_with_settings(
    file: &[u8],
    settings: &HeicSettings,
) -> Result<Rgb8Image, DecodeError> {
    Decoder::from_settings(*settings).decode_rgb8(file)
}

pub(crate) fn decode_heic_rgb8_with(
    decoder: &Decoder,
    file: &[u8],
) -> Result<Rgb8Image, DecodeError> {
    let img = decode_heic_impl(decoder, file, false)?;
    let shift = img.bit_depth.minus8();
    let pixels = match img.pixels {
        ImageBuffer::Rgb8(pixels) => pixels,
        ImageBuffer::Rgb16(samples) => {
            let mut pixels = try_vec![0u8; samples.len(), "RGB8 output"];
            for (dst, sample) in pixels.iter_mut().zip(samples) {
                *dst = (sample >> shift) as u8;
            }
            pixels
        }
        ImageBuffer::Luma8(samples) => {
            let len = samples.len().checked_mul(3).ok_or_else(|| {
                DecodeError::Bitstream("RGB8 output length overflows usize".into())
            })?;
            let mut pixels = try_vec![0u8; len, "RGB8 output"];
            for (luma, rgb) in samples.into_iter().zip(pixels.as_chunks_mut::<3>().0) {
                *rgb = [luma; 3];
            }
            pixels
        }
        ImageBuffer::Luma16(samples) => {
            let len = samples.len().checked_mul(3).ok_or_else(|| {
                DecodeError::Bitstream("RGB8 output length overflows usize".into())
            })?;
            let mut pixels = try_vec![0u8; len, "RGB8 output"];
            for (luma, rgb) in samples.into_iter().zip(pixels.as_chunks_mut::<3>().0) {
                *rgb = [(luma >> shift) as u8; 3];
            }
            pixels
        }
    };
    Ok(Rgb8Image {
        pixels,
        width: img.width,
        height: img.height,
    })
}

fn oriented_dimensions(width: u32, height: u32, orientation: Orientation) -> (u32, u32) {
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transverse
        | Orientation::Transpose => (height, width),
        _ => (width, height),
    }
}

fn apply_orientation(
    width: u32,
    height: u32,
    pixels: ImageBuffer,
    alpha: Option<SampleBuf>,
    orientation: Orientation,
) -> Result<OrientedImage, DecodeError> {
    let (oriented_width, oriented_height) = oriented_dimensions(width, height, orientation);
    let pixels = match pixels {
        ImageBuffer::Luma8(samples) => ImageBuffer::Luma8(rotate_luma(
            width as usize,
            height as usize,
            samples,
            orientation,
        )?),
        ImageBuffer::Luma16(samples) => ImageBuffer::Luma16(rotate_luma(
            width as usize,
            height as usize,
            samples,
            orientation,
        )?),
        ImageBuffer::Rgb8(samples) => ImageBuffer::Rgb8(rotate_buf(
            width as usize,
            height as usize,
            samples,
            orientation,
        )?),
        ImageBuffer::Rgb16(samples) => ImageBuffer::Rgb16(rotate_buf(
            width as usize,
            height as usize,
            samples,
            orientation,
        )?),
    };
    let alpha = alpha
        .map(|samples| rotate_sample_buf(samples, width, height, orientation))
        .transpose()?;
    Ok(OrientedImage {
        width: oriented_width,
        height: oriented_height,
        pixels,
        alpha,
    })
}

/// Single-channel rotation for luma-only buffers. Normal and 180° orientation
/// reuse the original allocation; orientations requiring a new layout allocate
/// through `try_vec!`.
fn rotate_luma<T: Copy + Default>(
    w: usize,
    h: usize,
    mut pixels: Vec<T>,
    orientation: Orientation,
) -> Result<Vec<T>, DecodeError> {
    match orientation {
        Orientation::Normal => Ok(pixels),
        Orientation::Rotate180 => {
            pixels.reverse();
            Ok(pixels)
        }
        Orientation::FlipH => {
            let mut out = try_vec![T::default(); pixels.len(), "oriented luma image"];
            for (src_row, dst_row) in pixels.chunks_exact(w).zip(out.chunks_exact_mut(w)) {
                for (dst, &src) in dst_row.iter_mut().rev().zip(src_row) {
                    *dst = src;
                }
            }
            Ok(out)
        }
        Orientation::FlipV => {
            let mut out = try_vec![T::default(); pixels.len(), "oriented luma image"];
            for (src_row, dst_row) in pixels.chunks_exact(w).zip(out.chunks_exact_mut(w).rev()) {
                dst_row.copy_from_slice(src_row);
            }
            Ok(out)
        }
        Orientation::Rotate90 => {
            let mut out = try_vec![T::default(); pixels.len(), "oriented luma image"];
            for row in 0..h {
                for col in 0..w {
                    out[col * h + (h - 1 - row)] = pixels[row * w + col];
                }
            }
            Ok(out)
        }
        Orientation::Rotate270 => {
            let mut out = try_vec![T::default(); pixels.len(), "oriented luma image"];
            for row in 0..h {
                for col in 0..w {
                    out[(w - 1 - col) * h + row] = pixels[row * w + col];
                }
            }
            Ok(out)
        }
        _ => Ok(pixels),
    }
}

/// Generic three-channel pixel-buffer rotation.
fn rotate_buf<T: Copy + Default>(
    w: usize,
    h: usize,
    mut pixels: Vec<T>,
    orientation: Orientation,
) -> Result<Vec<T>, DecodeError> {
    match orientation {
        Orientation::Normal => Ok(pixels),
        Orientation::Rotate180 => {
            let count = pixels.len() / 3;
            for left in 0..count / 2 {
                let right = count - 1 - left;
                for channel in 0..3 {
                    pixels.swap(left * 3 + channel, right * 3 + channel);
                }
            }
            Ok(pixels)
        }
        Orientation::FlipH => {
            let mut out = try_vec![T::default(); pixels.len(), "oriented RGB image"];
            for (src_row, dst_row) in pixels.chunks_exact(w * 3).zip(out.chunks_exact_mut(w * 3)) {
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
            Ok(out)
        }
        Orientation::FlipV => {
            let mut out = try_vec![T::default(); pixels.len(), "oriented RGB image"];
            for (src_row, dst_row) in pixels
                .chunks_exact(w * 3)
                .rev()
                .zip(out.chunks_exact_mut(w * 3))
            {
                dst_row.copy_from_slice(src_row);
            }
            Ok(out)
        }
        Orientation::Rotate90 => {
            let mut out = try_vec![T::default(); pixels.len(), "oriented RGB image"];
            const BLOCK: usize = 32;
            let mut row_base = 0;
            while row_base < h {
                let row_end = (row_base + BLOCK).min(h);
                let mut col_base = 0;
                while col_base < w {
                    let col_end = (col_base + BLOCK).min(w);
                    for row in row_base..row_end {
                        let src_row = row * w * 3;
                        let dst_col = h - 1 - row;
                        for col in col_base..col_end {
                            let src = src_row + col * 3;
                            let dst = (col * h + dst_col) * 3;
                            out[dst..dst + 3].copy_from_slice(&pixels[src..src + 3]);
                        }
                    }
                    col_base = col_end;
                }
                row_base = row_end;
            }
            Ok(out)
        }
        Orientation::Rotate270 => {
            let mut out = try_vec![T::default(); pixels.len(), "oriented RGB image"];
            const BLOCK: usize = 32;
            let mut row_base = 0;
            while row_base < h {
                let row_end = (row_base + BLOCK).min(h);
                let mut col_base = 0;
                while col_base < w {
                    let col_end = (col_base + BLOCK).min(w);
                    for row in row_base..row_end {
                        let src_row = row * w * 3;
                        for col in col_base..col_end {
                            let src = src_row + col * 3;
                            let dst = ((w - 1 - col) * h + row) * 3;
                            out[dst..dst + 3].copy_from_slice(&pixels[src..src + 3]);
                        }
                    }
                    col_base = col_end;
                }
                row_base = row_end;
            }
            Ok(out)
        }
        _ => Ok(pixels),
    }
}
