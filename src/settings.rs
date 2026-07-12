/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use crate::limits::ParseLimits;

/// Worker-thread policy used by [`HeicSettings`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeThreads {
    /// Size the decoder-owned pool from `std::thread::available_parallelism()`.
    #[default]
    Available,
    /// Use a fixed number of worker threads. Values below one are clamped to one.
    Fixed(usize),
}

impl DecodeThreads {
    pub(crate) fn worker_count(self) -> usize {
        match self {
            DecodeThreads::Available => std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1),
            DecodeThreads::Fixed(count) => count.max(1),
        }
    }
}

/// Configuration shared by all HEIF/HEIC decode entry points.
///
/// Construct a reusable [`crate::Decoder`] with [`crate::Decoder::from_settings`]
/// when decoding more than one image. The decoder keeps its worker pool and
/// resolved SIMD dispatch alive while resetting per-image picture state.
#[derive(Clone, Copy, Debug)]
pub struct HeicSettings {
    /// Worker-thread policy for primary images, grids, alpha, and gain maps.
    pub threads: DecodeThreads,
    /// Limits enforced while parsing the HEIF container and validating decoded
    /// dimensions.
    pub limits: ParseLimits,
    /// Decode a recognized alpha auxiliary image into the returned alpha plane.
    /// Detection still occurs when disabled, but its HEVC payload is not decoded.
    pub decode_alpha: bool,
    /// Decode a recognized Apple HDR gain-map auxiliary image. Detection still
    /// occurs when disabled, but its HEVC payload is not decoded.
    pub decode_gain_map: bool,
}

impl HeicSettings {
    /// Settings matching the historical decoder behavior.
    pub const fn new() -> Self {
        Self {
            threads: DecodeThreads::Available,
            limits: ParseLimits::new(),
            decode_alpha: true,
            decode_gain_map: true,
        }
    }

    /// Configure a fixed number of worker threads.
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.threads = DecodeThreads::Fixed(threads);
        self
    }

    /// Replace the complete set of container and image limits.
    pub const fn with_limits(mut self, limits: ParseLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Enable or disable alpha auxiliary-image decoding.
    pub const fn with_decode_alpha(mut self, decode_alpha: bool) -> Self {
        self.decode_alpha = decode_alpha;
        self
    }

    /// Enable or disable HDR gain-map auxiliary-image decoding.
    pub const fn with_decode_gain_map(mut self, decode_gain_map: bool) -> Self {
        self.decode_gain_map = decode_gain_map;
        self
    }

    /// Disable both alpha and gain-map decoding while retaining their container
    /// detection for metadata queries.
    pub const fn primary_only(mut self) -> Self {
        self.decode_alpha = false;
        self.decode_gain_map = false;
        self
    }
}

impl Default for HeicSettings {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_auxiliary_decoding() {
        let settings = HeicSettings::default();
        assert_eq!(settings.threads, DecodeThreads::Available);
        assert!(settings.decode_alpha);
        assert!(settings.decode_gain_map);
    }

    #[test]
    fn primary_only_disables_auxiliary_payloads() {
        let settings = HeicSettings::new().primary_only();
        assert!(!settings.decode_alpha);
        assert!(!settings.decode_gain_map);
    }

    #[test]
    fn fixed_zero_is_resolved_to_one_worker() {
        assert_eq!(DecodeThreads::Fixed(0).worker_count(), 1);
    }
}
