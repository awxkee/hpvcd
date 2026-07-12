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

use crate::color::ColorMetadata;
use crate::config;
use crate::error::DecodeError;
use crate::fmt::{BitDepth, ChromaFormat};
use crate::heif;
use crate::metadata::{CleanAperture, ContentLightLevel, Orientation, PixelAspectRatio};

struct PrimaryImageDescription<'a> {
    coded_width: u32,
    coded_height: u32,
    orientation: Orientation,
    hvcc: &'a [u8],
}

impl PrimaryImageDescription<'_> {
    fn display_dimensions(&self) -> (u32, u32) {
        match self.orientation {
            Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Transverse
            | Orientation::Transpose => (self.coded_height, self.coded_width),
            _ => (self.coded_width, self.coded_height),
        }
    }
}

/// Lightweight, decode-free description of a HEIF/HEIC image.
///
/// Obtained from [`read_heic_info`]. Every field here is read straight from the
/// container boxes and the HEVC SPS; no image data is decoded, so producing an
/// `ImageInfo` is cheap even for very large images.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageInfo {
    /// Display width in pixels, *after* the orientation transform is applied —
    /// a 90°/270° rotation (or transpose/transverse) swaps width and height, so
    /// this matches the `width` you would get from [`crate::decode_heic`].
    pub width: u32,
    /// Display height in pixels, after the orientation transform.
    pub height: u32,
    /// Stored width before orientation — the `ispe` display width for a single
    /// item, or the composed grid width for a tiled image.
    pub coded_width: u32,
    /// Stored height before orientation.
    pub coded_height: u32,
    /// Bit depth per sample (8, 10 or 12), from the SPS luma bit depth.
    pub bit_depth: BitDepth,
    /// Chroma subsampling of the coded image (4:0:0 / 4:2:0 / 4:2:2 / 4:4:4).
    pub chroma: ChromaFormat,
    /// color metadata (CICP code points and/or ICC profile) from the container.
    pub color: ColorMetadata,
    /// Display orientation (`irot` / `imir`). `Normal` means the stored pixels
    /// are already upright.
    pub orientation: Orientation,
    /// `true` when the file carries a recognized alpha auxiliary item (`auxl`)
    /// with an HEVC configuration property.
    pub has_alpha: bool,
    /// `true` when the file carries an Apple HDR gain-map auxiliary image.
    pub has_gain_map: bool,
    /// `true` when the primary item is a tiled `grid` derivation rather than a
    /// single coded image.
    pub is_grid: bool,
    /// HDR content light level (`clli`), if present.
    pub content_light_level: Option<ContentLightLevel>,
    /// Clean aperture / spatial crop (`clap`), if present.
    pub clean_aperture: Option<CleanAperture>,
    /// Pixel aspect ratio (`pasp`), if present. `None` means assume square pixels.
    pub pixel_aspect_ratio: Option<PixelAspectRatio>,
    /// `true` when the file carries an EXIF metadata item.
    pub has_exif: bool,
}

impl ImageInfo {
    pub fn channels(&self) -> u8 {
        let color = if self.chroma.is_monochrome() { 1 } else { 3 };
        color + u8::from(self.has_alpha)
    }

    /// Average bits per pixel of the displayed image
    pub fn bits_per_pixel(&self) -> f32 {
        // Luma contributes one sample per pixel. Each of the two chroma planes
        // contributes 1 / (sub_w * sub_h) samples per pixel; monochrome has none.
        let samples_per_pixel = if self.chroma.is_monochrome() {
            1.0
        } else {
            let sub = (self.chroma.sub_w() * self.chroma.sub_h()) as f32;
            1.0 + 2.0 / sub
        };
        let alpha = if self.has_alpha { 1.0 } else { 0.0 };
        (samples_per_pixel + alpha) * self.bit_depth.bits() as f32
    }

    /// Total number of pixels in the displayed image (`width * height`).
    pub fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// Read image metadata from a HEIF/HEIC file without decoding pixels, using
/// default [`crate::ParseLimits`].
pub fn read_heic_info(file: &[u8]) -> Result<ImageInfo, DecodeError> {
    read_heic_info_with_limits(file, &crate::limits::ParseLimits::default())
}

/// Read image metadata under caller-supplied parse limits.
pub fn read_heic_info_with_limits(
    file: &[u8],
    limits: &crate::limits::ParseLimits,
) -> Result<ImageInfo, DecodeError> {
    let heif = heif::parse(
        file,
        limits,
        heif::HeifParseOptions {
            load_alpha: false,
            load_gain_map: false,
        },
    )?;

    let description = if let Some(grid) = &heif.grid {
        let hvcc = grid
            .tiles
            .iter()
            .map(|tile| tile.hvcc.as_slice())
            .find(|hvcc| !hvcc.is_empty())
            .unwrap_or(&[]);
        PrimaryImageDescription {
            coded_width: grid.output_width,
            coded_height: grid.output_height,
            orientation: grid.orientation,
            hvcc,
        }
    } else {
        PrimaryImageDescription {
            coded_width: heif.primary.display_w,
            coded_height: heif.primary.display_h,
            orientation: heif.primary.orientation,
            hvcc: heif.primary.hvcc.as_slice(),
        }
    };

    let (sps, _pps) = config::parse_hvcc_full(description.hvcc)?;
    let bit_depth = sps.bit_depth()?;
    let chroma = sps.chroma;

    let (width, height) = description.display_dimensions();

    Ok(ImageInfo {
        width,
        height,
        coded_width: description.coded_width,
        coded_height: description.coded_height,
        bit_depth,
        chroma,
        color: heif.primary.color.clone(),
        orientation: description.orientation,
        has_alpha: heif.has_alpha,
        has_gain_map: heif.has_gain_map,
        is_grid: heif.grid.is_some(),
        content_light_level: heif.primary.cll,
        clean_aperture: heif.primary.clap,
        pixel_aspect_ratio: heif.primary.pasp,
        has_exif: heif.exif.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(chroma: ChromaFormat, bd: BitDepth, alpha: bool) -> ImageInfo {
        ImageInfo {
            width: 100,
            height: 50,
            coded_width: 100,
            coded_height: 50,
            bit_depth: bd,
            chroma,
            color: ColorMetadata::default(),
            orientation: Orientation::Normal,
            has_alpha: alpha,
            has_gain_map: false,
            is_grid: false,
            content_light_level: None,
            clean_aperture: None,
            pixel_aspect_ratio: None,
            has_exif: false,
        }
    }

    #[test]
    fn bpp_matches_subsampling() {
        assert_eq!(
            info(ChromaFormat::Yuv420, BitDepth::Eight, false).bits_per_pixel(),
            12.0
        );
        assert_eq!(
            info(ChromaFormat::Yuv420, BitDepth::Ten, false).bits_per_pixel(),
            15.0
        );
        assert_eq!(
            info(ChromaFormat::Yuv422, BitDepth::Eight, false).bits_per_pixel(),
            16.0
        );
        assert_eq!(
            info(ChromaFormat::Yuv444, BitDepth::Eight, false).bits_per_pixel(),
            24.0
        );
        assert_eq!(
            info(ChromaFormat::Monochrome, BitDepth::Eight, false).bits_per_pixel(),
            8.0
        );
    }

    #[test]
    fn alpha_adds_one_plane() {
        // 4:4:4 8-bit with alpha = 3 color + 1 alpha planes = 32 bpp.
        assert_eq!(
            info(ChromaFormat::Yuv444, BitDepth::Eight, true).bits_per_pixel(),
            32.0
        );
        // 4:2:0 8-bit with alpha = 12 + 8 = 20 bpp.
        assert_eq!(
            info(ChromaFormat::Yuv420, BitDepth::Eight, true).bits_per_pixel(),
            20.0
        );
    }

    #[test]
    fn channel_counts() {
        assert_eq!(
            info(ChromaFormat::Yuv420, BitDepth::Eight, false).channels(),
            3
        );
        assert_eq!(
            info(ChromaFormat::Yuv420, BitDepth::Eight, true).channels(),
            4
        );
        assert_eq!(
            info(ChromaFormat::Monochrome, BitDepth::Eight, false).channels(),
            1
        );
        assert_eq!(
            info(ChromaFormat::Monochrome, BitDepth::Eight, true).channels(),
            2
        );
    }

    #[test]
    fn pixel_count_uses_display_dims() {
        assert_eq!(
            info(ChromaFormat::Yuv420, BitDepth::Eight, false).pixel_count(),
            5000
        );
    }
}
