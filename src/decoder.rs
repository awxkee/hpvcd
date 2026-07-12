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

//! The [`Decoder`] entry point: an explicit, reusable decoder object that owns
//! its worker pool and applies one [`DecodeSettings`](crate::HeicSettings)
//! value to every decode:
//!
//! ```ignore
//! let settings = DecodeSettings::new()
//!     .with_threads(4)
//!     .with_decode_alpha(false)
//!     .with_decode_gain_map(false);
//! let decoder = Decoder::from_settings(settings);
//! let image = decoder.decode(&bytes)?;
//! let rgb = decoder.decode_rgb8(&other_bytes)?;
//! assert_eq!(rgb.pixels.len(), rgb.width as usize * rgb.height as usize * 3);
//! ```

use crate::error::DecodeError;
use crate::limits::ParseLimits;
use crate::settings::{DecodeThreads, HeicSettings};
use crate::{DecodedImage, DecodedYuv, ImageInfo, Rgb8Image};

/// A configured, reusable HEIF/HEIC decoder.
///
/// The owned worker pool and resolved SIMD dispatch are created once and reused
/// across calls. Per-image bitstream and picture state is reset for every decode.
pub struct Decoder {
    settings: HeicSettings,
    pool: crate::threadpool::ThreadPool,
    exec: crate::exec::ExecContext,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// Create a decoder with [`HeicSettings::default`].
    pub fn new() -> Self {
        Self::from_settings(HeicSettings::default())
    }

    /// Create a reusable decoder from an explicit settings value.
    pub fn from_settings(settings: HeicSettings) -> Self {
        let pool = crate::threadpool::ThreadPool::new(settings.threads.worker_count());
        Self {
            settings,
            pool,
            exec: crate::exec::ExecContext::new(),
        }
    }

    /// Replace all settings. The worker pool is rebuilt only when the resolved
    /// worker count changes; limits and auxiliary-image options are cheap updates.
    pub fn set_settings(&mut self, settings: HeicSettings) {
        let worker_count = settings.threads.worker_count();
        if self.pool.threads() != worker_count {
            self.pool = crate::threadpool::ThreadPool::new(worker_count);
        }
        self.settings = settings;
    }

    /// Builder form of [`Decoder::set_settings`].
    pub fn with_settings(mut self, settings: HeicSettings) -> Self {
        self.set_settings(settings);
        self
    }

    /// Set a fixed worker-thread count. `1` forces serial execution; values
    /// below one are clamped to one.
    pub fn with_threads(mut self, threads: usize) -> Self {
        let settings = HeicSettings {
            threads: DecodeThreads::Fixed(threads),
            ..self.settings
        };
        self.set_settings(settings);
        self
    }

    /// Restore automatic worker-count selection from available parallelism.
    pub fn with_available_threads(mut self) -> Self {
        let settings = HeicSettings {
            threads: DecodeThreads::Available,
            ..self.settings
        };
        self.set_settings(settings);
        self
    }

    /// Replace the complete set of container and image limits.
    pub fn with_limits(mut self, limits: ParseLimits) -> Self {
        self.settings.limits = limits;
        self
    }

    /// Enable or disable alpha auxiliary-image decoding.
    pub fn with_decode_alpha(mut self, decode_alpha: bool) -> Self {
        self.settings.decode_alpha = decode_alpha;
        self
    }

    /// Enable or disable Apple HDR gain-map decoding.
    pub fn with_decode_gain_map(mut self, decode_gain_map: bool) -> Self {
        self.settings.decode_gain_map = decode_gain_map;
        self
    }

    /// Decode only the primary image, skipping alpha and gain-map HEVC payloads.
    pub fn primary_only(mut self) -> Self {
        self.settings.decode_alpha = false;
        self.settings.decode_gain_map = false;
        self
    }

    /// Set the maximum allowed width or height, in samples.
    pub fn with_max_dimension(mut self, max_dimension: u32) -> Self {
        self.settings.limits.max_dimension = max_dimension;
        self
    }

    /// Set the maximum allowed total pixel count (`width * height`).
    pub fn with_max_pixels(mut self, max_pixels: u64) -> Self {
        self.settings.limits.max_pixels = max_pixels;
        self
    }

    /// Set the maximum size, in bytes, of a single ISOBMFF box.
    pub fn with_max_box_size(mut self, max_box_size: u64) -> Self {
        self.settings.limits.max_box_size = max_box_size;
        self
    }

    /// Set the maximum size, in bytes, of a single item's coded data.
    pub fn with_max_item_size(mut self, max_item_size: u64) -> Self {
        self.settings.limits.max_item_size = max_item_size;
        self
    }

    /// The complete settings value in effect.
    pub fn settings(&self) -> &HeicSettings {
        &self.settings
    }

    /// The container and image limits in effect.
    pub fn limits(&self) -> &ParseLimits {
        &self.settings.limits
    }

    /// Number of worker threads currently owned by this decoder.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Whether alpha auxiliary images are decoded.
    pub fn decodes_alpha(&self) -> bool {
        self.settings.decode_alpha
    }

    /// Whether Apple HDR gain maps are decoded.
    pub fn decodes_gain_map(&self) -> bool {
        self.settings.decode_gain_map
    }

    /// Validate decoded dimensions against the configured limits.
    pub(crate) fn check_dims(&self, w: usize, h: usize) -> Result<(), DecodeError> {
        if w == 0 || h == 0 {
            return Err(DecodeError::Bitstream(format!(
                "image dimensions {w}×{h} are zero"
            )));
        }
        let (Ok(w32), Ok(h32)) = (u32::try_from(w), u32::try_from(h)) else {
            return Err(DecodeError::LimitExceeded {
                what: "image dimension",
                value: w.max(h) as u64,
                limit: self.settings.limits.max_dimension as u64,
            });
        };
        self.settings.limits.check_image(w32, h32)
    }

    pub(crate) fn pool(&self) -> &crate::threadpool::ThreadPool {
        &self.pool
    }

    pub(crate) fn exec(&self) -> &crate::exec::ExecContext {
        &self.exec
    }

    /// Read container/SPS information without decoding pixels.
    pub fn read_info(&self, file: &[u8]) -> Result<ImageInfo, DecodeError> {
        crate::read_heic_info_with_limits(file, self.limits())
    }

    /// Decode to display-ready RGB (or luma) pixels with color conversion.
    pub fn decode(&self, file: &[u8]) -> Result<DecodedImage, DecodeError> {
        crate::decode_heic_with(self, file)
    }

    /// Decode to raw YCbCr planes without color conversion.
    pub fn decode_yuv(&self, file: &[u8]) -> Result<DecodedYuv, DecodeError> {
        crate::decode_heic_yuv_with(self, file)
    }

    /// Decode to packed 8-bit RGB, three bytes per pixel.
    pub fn decode_rgb8(&self, file: &[u8]) -> Result<Rgb8Image, DecodeError> {
        crate::decode_heic_rgb8_with(self, file)
    }
}

impl From<HeicSettings> for Decoder {
    fn from(settings: HeicSettings) -> Self {
        Self::from_settings(settings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_sets_limits() {
        let d = Decoder::new().with_max_dimension(100).with_max_pixels(4000);
        assert!(d.check_dims(50, 50).is_ok());
        assert!(d.check_dims(0, 10).is_err());
        assert!(d.check_dims(101, 10).is_err());
        assert!(d.check_dims(80, 80).is_err());
    }

    #[test]
    fn default_limits_match_historical_constants() {
        let d = Decoder::default();
        let max = ParseLimits::DEFAULT_MAX_DIMENSION as usize;
        assert!(d.check_dims(max, 1).is_ok());
        assert!(d.check_dims(max + 1, 1).is_err());
    }

    #[test]
    fn box_and_item_size_setters() {
        let d = Decoder::new()
            .with_max_box_size(1024)
            .with_max_item_size(2048);
        assert_eq!(d.limits().max_box_size, 1024);
        assert_eq!(d.limits().max_item_size, 2048);
    }

    #[test]
    fn threads_reports_configured_count() {
        assert!(Decoder::new().threads() >= 1);
        assert_eq!(Decoder::new().with_threads(2).threads(), 2);
        assert_eq!(Decoder::new().with_threads(0).threads(), 1);
    }

    #[test]
    fn settings_control_auxiliary_decode() {
        let d = Decoder::from_settings(HeicSettings::new().primary_only());
        assert!(!d.decodes_alpha());
        assert!(!d.decodes_gain_map());
    }

    #[test]
    fn set_settings_reconfigures_reusable_decoder() {
        let mut d = Decoder::new().with_threads(1);
        d.set_settings(HeicSettings::new().with_threads(2).with_decode_alpha(false));
        assert_eq!(d.threads(), 2);
        assert!(!d.decodes_alpha());
        assert!(d.decodes_gain_map());
    }
}
