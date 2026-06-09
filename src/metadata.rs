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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Orientation {
    /// 1 — 0°, no transform (the stored pixels are already upright).
    #[default]
    Normal,
    /// 2 — mirrored horizontally.
    FlipH,
    /// 3 — rotated 180°.
    Rotate180,
    /// 4 — mirrored vertically.
    FlipV,
    /// 5 — mirrored horizontally then rotated 90° clockwise (transpose).
    Transpose,
    /// 6 — rotated 90° clockwise.
    Rotate90,
    /// 7 — mirrored horizontally then rotated 90° anti-clockwise (transverse).
    Transverse,
    /// 8 — rotated 90° anti-clockwise.
    Rotate270,
}

impl Orientation {
    /// Map a raw EXIF Orientation value (1..=8) to an [`Orientation`]; anything out of
    /// range (including 0) is treated as `Normal`.
    pub fn from_exif(v: u16) -> Self {
        match v {
            2 => Orientation::FlipH,
            3 => Orientation::Rotate180,
            4 => Orientation::FlipV,
            5 => Orientation::Transpose,
            6 => Orientation::Rotate90,
            7 => Orientation::Transverse,
            8 => Orientation::Rotate270,
            _ => Orientation::Normal,
        }
    }

    /// Whether this orientation needs any container transform property at all.
    pub fn is_identity(self) -> bool {
        matches!(self, Orientation::Normal)
    }

    /// The `imir` mirror axis, if mirroring is part of this orientation.
    /// `Some(true)` = mirror across the horizontal axis (vertical flip);
    /// `Some(false)` = mirror across the vertical axis (horizontal flip);
    /// `None` = no mirror. (HEIF `imir`: low bit, 1 = horizontal axis.)
    pub fn imir_axis(self) -> Option<bool> {
        match self {
            Orientation::FlipH | Orientation::Transpose | Orientation::Transverse => Some(false),
            Orientation::FlipV => Some(true),
            _ => None,
        }
    }

    /// The `irot` rotation in 90° anti-clockwise steps (0..=3), applied after mirroring.
    /// HEIF defines `irot` as anti-clockwise; we decompose each EXIF orientation into
    /// an optional mirror followed by an anti-clockwise rotation.
    pub fn irot_steps(self) -> u8 {
        match self {
            Orientation::Normal | Orientation::FlipH | Orientation::FlipV => 0,
            Orientation::Rotate180 => 2,
            // EXIF 6 = 90° CW = 270° CCW = 3 steps. EXIF 8 = 90° CCW = 1 step.
            Orientation::Rotate90 => 3,
            Orientation::Rotate270 => 1,
            // 5 (transpose) = flipH + 90°CW = flipH + 3 CCW steps.
            // 7 (transverse) = flipH + 90°CCW = flipH + 1 CCW step.
            Orientation::Transpose => 3,
            Orientation::Transverse => 1,
        }
    }
}

/// Clean aperture (spatial crop) from the ISOBMFF `clap` property
/// (ISO 14496-12 §12.1.4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CleanAperture {
    /// Numerator of the clean aperture width (pixels).
    pub width_n: u32,
    /// Denominator of the clean aperture width.
    pub width_d: u32,
    /// Numerator of the clean aperture height (pixels).
    pub height_n: u32,
    /// Denominator of the clean aperture height.
    pub height_d: u32,
    /// Numerator of the horizontal center offset (signed, pixels).
    pub horiz_off_n: i32,
    /// Denominator of the horizontal center offset.
    pub horiz_off_d: u32,
    /// Numerator of the vertical center offset (signed, pixels).
    pub vert_off_n: i32,
    /// Denominator of the vertical center offset.
    pub vert_off_d: u32,
}

impl CleanAperture {
    /// Pixel width of the clean aperture, rounded down.
    /// Returns `None` when the denominator is zero.
    pub fn width_pixels(&self) -> Option<u32> {
        self.width_n.checked_div(self.width_d)
    }

    /// Pixel height of the clean aperture, rounded down.
    /// Returns `None` when the denominator is zero.
    pub fn height_pixels(&self) -> Option<u32> {
        self.height_n.checked_div(self.height_d)
    }
}

/// Pixel aspect ratio from the ISOBMFF `pasp` property (ISO 14496-12 §12.1.4.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelAspectRatio {
    /// Horizontal spacing — numerator of pixel width relative to pixel height.
    pub h_spacing: u32,
    /// Vertical spacing — denominator.
    pub v_spacing: u32,
}

impl PixelAspectRatio {
    /// Returns `true` when pixels are square (no display rescaling required).
    pub fn is_square(self) -> bool {
        self.h_spacing == self.v_spacing
    }
}

/// HDR content light level (CTA-861.3 / ISOBMFF `clli`): the maximum content light
/// level (MaxCLL) and maximum frame-average light level (MaxFALL), both in cd/m²
/// (nits). Written as the `ContentLightLevelBox`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContentLightLevel {
    /// MaxCLL — maximum content light level, nits.
    pub max_content_light_level: u16,
    /// MaxFALL — maximum frame-average light level, nits.
    pub max_pic_average_light_level: u16,
}

impl ContentLightLevel {
    pub fn new(max_cll: u16, max_fall: u16) -> Self {
        ContentLightLevel {
            max_content_light_level: max_cll,
            max_pic_average_light_level: max_fall,
        }
    }

    /// The `clli` box payload (4 bytes, big-endian), without the box header.
    pub fn clli_payload(&self) -> [u8; 4] {
        let cll = self.max_content_light_level.to_be_bytes();
        let fall = self.max_pic_average_light_level.to_be_bytes();
        [cll[0], cll[1], fall[0], fall[1]]
    }
}

/// All optional metadata passed to the encoder, threaded through [`crate::EncodeConfig`].
#[derive(Clone, Debug, Default)]
pub struct Metadata {
    /// Display orientation (`irot`/`imir`). Default `Normal` writes no transform.
    pub orientation: Orientation,
    /// HDR content light level (`clli`). `None` writes no box.
    pub content_light_level: Option<ContentLightLevel>,
    /// Raw EXIF/TIFF payload (the bytes after the `"Exif\0\0"` identifier — i.e. the
    /// TIFF header onward). `None` writes no EXIF item. The encoder prepends the
    /// 4-byte ItemInfo offset field required by the HEIF Exif item format.
    pub exif: Option<Vec<u8>>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_orientation(mut self, o: Orientation) -> Self {
        self.orientation = o;
        self
    }

    pub fn with_content_light_level(mut self, cll: ContentLightLevel) -> Self {
        self.content_light_level = Some(cll);
        self
    }

    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.exif = Some(exif);
        self
    }

    /// True when there is nothing to write (avoids emitting empty boxes/items).
    pub fn is_empty(&self) -> bool {
        self.orientation.is_identity() && self.content_light_level.is_none() && self.exif.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_decomposition() {
        assert!(Orientation::Normal.is_identity());
        assert_eq!(Orientation::Rotate90.irot_steps(), 3); // 90°CW = 270°CCW
        assert_eq!(Orientation::Rotate270.irot_steps(), 1); // 90°CCW
        assert_eq!(Orientation::Rotate180.irot_steps(), 2);
        assert_eq!(Orientation::FlipH.imir_axis(), Some(false));
        assert_eq!(Orientation::FlipV.imir_axis(), Some(true));
        assert_eq!(Orientation::Rotate90.imir_axis(), None);
    }

    #[test]
    fn from_exif_mapping() {
        assert_eq!(Orientation::from_exif(1), Orientation::Normal);
        assert_eq!(Orientation::from_exif(6), Orientation::Rotate90);
        assert_eq!(Orientation::from_exif(8), Orientation::Rotate270);
        assert_eq!(Orientation::from_exif(0), Orientation::Normal); // out of range
        assert_eq!(Orientation::from_exif(99), Orientation::Normal);
    }

    #[test]
    fn clli_payload_layout() {
        let cll = ContentLightLevel::new(1000, 400);
        assert_eq!(cll.clli_payload(), [0x03, 0xE8, 0x01, 0x90]);
    }
}
