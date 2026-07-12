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

//! Configurable limits applied during container (ISOBMFF/HEIF) parsing.
//!
//! These bounds let an embedder cap how much work and memory a single file may
//! provoke *before* any pixels are decoded, giving granular control over
//! untrusted input. Every field has a generous default that comfortably fits
//! real-world HEIC/HEIF files while rejecting pathological or hostile ones.
//!
//! The limits fall into two groups the caller usually thinks about as
//! "image size" and "tag size":
//!
//! * **Image size** — [`max_dimension`](ParseLimits::max_dimension) and
//!   [`max_pixels`](ParseLimits::max_pixels) bound declared image geometry
//!   (from `ispe` and the grid descriptor), rejected at parse time rather than
//!   only after decode.
//! * **Tag / box size** — [`max_box_size`](ParseLimits::max_box_size),
//!   [`max_item_size`](ParseLimits::max_item_size),
//!   [`max_hvcc_size`](ParseLimits::max_hvcc_size), and
//!   [`max_exif_size`](ParseLimits::max_exif_size) bound the sizes of
//!   individual boxes and the variable-length metadata blobs copied out of
//!   them, while
//!   [`max_items`](ParseLimits::max_items),
//!   [`max_extents_per_item`](ParseLimits::max_extents_per_item), and
//!   [`max_tiles`](ParseLimits::max_tiles) bound how many of each the parser
//!   will enumerate.

/// Limits enforced while parsing the HEIF container.
#[derive(Clone, Copy, Debug)]
pub struct ParseLimits {
    /// Maximum image width or height, in samples, accepted from `ispe` / grid
    /// descriptors. Larger declared geometry is rejected during parsing.
    pub max_dimension: u32,
    /// Maximum total pixel count (`width * height`) accepted from declared
    /// geometry.
    pub max_pixels: u64,
    /// Maximum size, in bytes, of any single ISOBMFF box the parser will accept.
    /// Boxes claiming to be larger are treated as truncated/invalid.
    pub max_box_size: u64,
    /// Maximum size, in bytes, of a single item's coded data (its `iloc`
    /// extent length / total length).
    pub max_item_size: u64,
    /// Maximum size, in bytes, of an `hvcC` property blob copied out of `ipco`.
    pub max_hvcc_size: usize,
    /// Maximum size, in bytes, of an EXIF or auxiliary gain-map metadata
    /// payload copied out of the file.
    pub max_exif_size: usize,
    /// Maximum number of items enumerated from `iloc` / `iinf`.
    pub max_items: usize,
    /// Maximum number of extents enumerated per item in `iloc`.
    pub max_extents_per_item: usize,
    /// Maximum number of grid tiles enumerated for a tiled image.
    pub max_tiles: usize,
}

impl ParseLimits {
    /// The default width/height cap (16 384 samples).
    pub const DEFAULT_MAX_DIMENSION: u32 = 16_384;
    /// The default pixel-count cap (64 Mpx).
    pub const DEFAULT_MAX_PIXELS: u64 = 64 * 1024 * 1024;
    /// The default per-box size cap (256 MiB).
    pub const DEFAULT_MAX_BOX_SIZE: u64 = 256 * 1024 * 1024;
    /// The default per-item data-size cap (256 MiB).
    pub const DEFAULT_MAX_ITEM_SIZE: u64 = 256 * 1024 * 1024;
    /// The default `hvcC` blob cap (64 KiB — real hvcC boxes are a few hundred
    /// bytes).
    pub const DEFAULT_MAX_HVCC_SIZE: usize = 64 * 1024;
    /// The default copied-metadata blob cap (1 MiB).
    pub const DEFAULT_MAX_EXIF_SIZE: usize = 1024 * 1024;
    /// The default item-count cap.
    pub const DEFAULT_MAX_ITEMS: usize = 4096;
    /// The default per-item extent-count cap.
    pub const DEFAULT_MAX_EXTENTS_PER_ITEM: usize = 64;
    /// The default grid-tile-count cap (a 128×128 grid of tiles).
    pub const DEFAULT_MAX_TILES: usize = 16_384;

    /// Construct the default set of limits.
    pub const fn new() -> Self {
        ParseLimits {
            max_dimension: Self::DEFAULT_MAX_DIMENSION,
            max_pixels: Self::DEFAULT_MAX_PIXELS,
            max_box_size: Self::DEFAULT_MAX_BOX_SIZE,
            max_item_size: Self::DEFAULT_MAX_ITEM_SIZE,
            max_hvcc_size: Self::DEFAULT_MAX_HVCC_SIZE,
            max_exif_size: Self::DEFAULT_MAX_EXIF_SIZE,
            max_items: Self::DEFAULT_MAX_ITEMS,
            max_extents_per_item: Self::DEFAULT_MAX_EXTENTS_PER_ITEM,
            max_tiles: Self::DEFAULT_MAX_TILES,
        }
    }

    /// Validate declared image geometry against the image-size limits.
    pub(crate) fn check_image(&self, w: u32, h: u32) -> Result<(), crate::error::DecodeError> {
        use crate::error::DecodeError;
        if w == 0 || h == 0 {
            return Err(DecodeError::BadDimensions { w, h });
        }
        if w > self.max_dimension {
            return Err(DecodeError::LimitExceeded {
                what: "image width",
                value: w as u64,
                limit: self.max_dimension as u64,
            });
        }
        if h > self.max_dimension {
            return Err(DecodeError::LimitExceeded {
                what: "image height",
                value: h as u64,
                limit: self.max_dimension as u64,
            });
        }
        let px = w as u64 * h as u64;
        if px > self.max_pixels {
            return Err(DecodeError::LimitExceeded {
                what: "image pixels",
                value: px,
                limit: self.max_pixels,
            });
        }
        Ok(())
    }
}

impl Default for ParseLimits {
    fn default() -> Self {
        ParseLimits::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_accept_reasonable_and_reject_huge() {
        let l = ParseLimits::default();
        assert!(l.check_image(4032, 3024).is_ok()); // 12 Mpx phone photo
        assert!(l.check_image(0, 3024).is_err()); // zero dimensions are invalid
        assert!(l.check_image(17_000, 10).is_err()); // over max dimension
        assert!(l.check_image(16_000, 16_000).is_err()); // 256 Mpx > 64 Mpx
    }

    #[test]
    fn custom_limits_are_honoured() {
        let l = ParseLimits {
            max_dimension: 1000,
            max_pixels: 500_000,
            ..ParseLimits::default()
        };
        assert!(l.check_image(800, 600).is_ok());
        assert!(l.check_image(1001, 1).is_err());
        assert!(l.check_image(900, 900).is_err()); // 810k > 500k
    }
}
