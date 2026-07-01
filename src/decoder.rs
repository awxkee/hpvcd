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

//! The [`Decoder`] entry point: an explicit, reusable decoder object that holds
//! decode settings and owns its own [`crate::threadpool::ThreadPool`] instead of
//! relying on a process-wide global. Create one, configure it, and reuse it
//! across many images:
//!
//! ```ignore
//! let decoder = Decoder::new().with_threads(4);
//! let image = decoder.decode(&bytes)?;
//! let (rgb, w, h) = decoder.decode_rgb8(&other_bytes)?;
//! ```
//!
use crate::error::DecodeError;
use crate::limits::ParseLimits;
use crate::{DecodedImage, DecodedYuv};

/// A configured, reusable HEIF/HEIC decoder.
///
/// Holds parse limits and an owned work-stealing thread pool. All decode entry
/// points are methods on this type; the crate-level free functions
/// ([`crate::decode_heic`] etc.) simply delegate to a default `Decoder`.
pub struct Decoder {
    /// Container-parse limits (image + tag/box sizes) applied before decoding.
    limits: ParseLimits,
    /// Owned work-stealing pool used for parallel grid decoding.
    pool: crate::threadpool::ThreadPool,
}

impl Default for Decoder {
    fn default() -> Self {
        Decoder::new()
    }
}

impl Decoder {
    /// Create a decoder with default limits and a pool sized to the machine's
    /// available parallelism.
    pub fn new() -> Self {
        Decoder {
            limits: ParseLimits::default(),
            pool: crate::threadpool::ThreadPool::with_available_parallelism(),
        }
    }

    /// Set the worker-thread count, rebuilding the owned pool. `1` forces a
    /// serial decode path (no pool dispatch); values are clamped to at least 1.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.pool = crate::threadpool::ThreadPool::new(threads);
        self
    }

    /// Replace the full set of container-parse limits.
    pub fn with_limits(mut self, limits: ParseLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Set the maximum allowed width or height (in samples).
    pub fn with_max_dimension(mut self, max_dimension: u32) -> Self {
        self.limits.max_dimension = max_dimension;
        self
    }

    /// Set the maximum allowed total pixel count (`width * height`).
    pub fn with_max_pixels(mut self, max_pixels: u64) -> Self {
        self.limits.max_pixels = max_pixels;
        self
    }

    /// Set the maximum size, in bytes, of a single ISOBMFF box the parser will
    /// accept.
    pub fn with_max_box_size(mut self, max_box_size: u64) -> Self {
        self.limits.max_box_size = max_box_size;
        self
    }

    /// Set the maximum size, in bytes, of a single item's coded data.
    pub fn with_max_item_size(mut self, max_item_size: u64) -> Self {
        self.limits.max_item_size = max_item_size;
        self
    }

    /// The container-parse limits in effect.
    pub fn limits(&self) -> &ParseLimits {
        &self.limits
    }

    /// Number of worker threads the decoder will use.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Validate decoded dimensions against the configured limits.
    pub(crate) fn check_dims(&self, w: usize, h: usize) -> Result<(), DecodeError> {
        if w == 0 || h == 0 {
            return Err(DecodeError::Bitstream(format!(
                "image dimensions {w}×{h} are zero"
            )));
        }
        // Anything beyond u32 range is by definition over any sane limit.
        let (Ok(w32), Ok(h32)) = (u32::try_from(w), u32::try_from(h)) else {
            return Err(DecodeError::LimitExceeded {
                what: "image dimension",
                value: w.max(h) as u64,
                limit: self.limits.max_dimension as u64,
            });
        };
        // Delegate the upper bounds to the shared parse limits so parse-time and
        // post-decode checks can never disagree.
        self.limits.check_image(w32, h32)
    }

    /// The owned work-stealing thread pool.
    pub(crate) fn pool(&self) -> &crate::threadpool::ThreadPool {
        &self.pool
    }

    /// Decode to display-ready RGB (or luma) pixels with color conversion.
    pub fn decode(&self, file: &[u8]) -> Result<DecodedImage, DecodeError> {
        crate::decode_heic_with(self, file)
    }

    /// Decode to raw YCbCr planes (no color conversion).
    pub fn decode_yuv(&self, file: &[u8]) -> Result<DecodedYuv, DecodeError> {
        crate::decode_heic_yuv_with(self, file)
    }

    /// Decode to packed 8-bit RGB (`Vec<u8>`, 3 bytes/pixel).
    pub fn decode_rgb8(&self, file: &[u8]) -> Result<(Vec<u8>, u32, u32), DecodeError> {
        crate::decode_heic_rgb8_with(self, file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_sets_limits() {
        let d = Decoder::new().with_max_dimension(100).with_max_pixels(4000);
        assert!(d.check_dims(50, 50).is_ok()); // 2500 <= 4000, within dims
        assert!(d.check_dims(0, 10).is_err()); // zero rejected
        assert!(d.check_dims(101, 10).is_err()); // over max_dimension
        assert!(d.check_dims(80, 80).is_err()); // 6400 > max_pixels
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
        // Clamped to at least one worker.
        assert_eq!(Decoder::new().with_threads(0).threads(), 1);
    }
}
