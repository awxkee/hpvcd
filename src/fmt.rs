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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitDepth {
    Eight,
    Ten,
    Twelve,
}

impl BitDepth {
    /// Construct from a raw bit count (8, 10, or 12). Panics on unsupported depths.
    pub fn from_bits(bits: u8) -> Self {
        match bits {
            8 => BitDepth::Eight,
            10 => BitDepth::Ten,
            12 => BitDepth::Twelve,
            other => panic!("unsupported bit depth: {other} (only 8, 10, 12 supported)"),
        }
    }

    /// Bit count (8, 10, or 12).
    pub fn bits(self) -> u8 {
        match self {
            BitDepth::Eight => 8,
            BitDepth::Ten => 10,
            BitDepth::Twelve => 12,
        }
    }

    /// `bit_depth - 8`, the value the SPS/hvcC store as `*_minus8`.
    pub fn minus8(self) -> u8 {
        self.bits() - 8
    }

    /// Maximum representable sample value: `(1 << bits) - 1` (255, 1023, or 4095).
    pub fn max_val(self) -> u16 {
        (1u16 << self.bits()) - 1
    }

    /// Neutral / midpoint sample: `1 << (bits - 1)` (128, 512, or 2048). Used as the
    /// unavailable-reference default in intra prediction.
    pub fn neutral(self) -> u16 {
        1u16 << (self.bits() - 1)
    }

    /// QpBdOffset = `6 * (bit_depth - 8)` (0, 12, or 24). The decoder dequantises at
    /// `SliceQp + QpBdOffset`, so the encoder must use the same effective QP.
    pub fn qp_bd_offset(self) -> u8 {
        6 * self.minus8()
    }
}

/// Chroma subsampling mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaFormat {
    /// 4:0:0 — monochrome (luma only, no chroma). ChromaArrayType = 0.
    Monochrome,
    /// 4:2:0 — chroma is half width, half height. ChromaArrayType = 1.
    Yuv420,
    /// 4:2:2 — chroma is half width, full height. ChromaArrayType = 2.
    Yuv422,
    /// 4:4:4 — chroma is full width, full height. ChromaArrayType = 3.
    Yuv444,
}

impl ChromaFormat {
    /// `chroma_format_idc` value written in the SPS (HEVC Table 6-1).
    pub fn idc(self) -> u32 {
        match self {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::Yuv420 => 1,
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 3,
        }
    }

    /// True when there is no chroma (4:0:0).
    pub fn is_monochrome(self) -> bool {
        matches!(self, ChromaFormat::Monochrome)
    }

    /// Horizontal subsampling factor: luma_width / chroma_width. (1 for monochrome,
    /// which has no chroma; the value is unused but kept well-defined.)
    pub fn sub_w(self) -> usize {
        match self {
            ChromaFormat::Yuv420 | ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 | ChromaFormat::Monochrome => 1,
        }
    }

    /// Vertical subsampling factor: luma_height / chroma_height.
    pub fn sub_h(self) -> usize {
        match self {
            ChromaFormat::Yuv420 => 2,
            ChromaFormat::Yuv422 | ChromaFormat::Yuv444 | ChromaFormat::Monochrome => 1,
        }
    }

    /// Side length of each chroma transform block for an 8×8 luma CU.
    pub fn chroma_tb_size(self) -> usize {
        match self {
            ChromaFormat::Yuv420 | ChromaFormat::Yuv422 => 4,
            ChromaFormat::Yuv444 => 8,
            ChromaFormat::Monochrome => 0,
        }
    }

    /// Number of chroma transform blocks stacked vertically per 8×8 luma CU.
    pub fn chroma_tbs_per_cu(self) -> usize {
        match self {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::Yuv420 => 1,
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 1,
        }
    }
}

/// Pixel buffer returned by [`crate::decode_heic`].
///
/// Variants encode both bit depth and channel count:
/// - `Luma8` / `Luma16` — monochrome (4:0:0); one sample per pixel.
/// - `U8` / `U16`       — colour (4:2:0/2:2/4:4:4); R₀G₀B₀ R₁G₁B₁ … interleaved.
#[derive(Clone, Debug)]
pub enum ImageBuffer {
    Luma8(Vec<u8>),   // 8-bit grayscale
    Luma16(Vec<u16>), // 10/12-bit grayscale
    Rgb8(Vec<u8>),    // 8-bit RGB interleaved
    Rgb16(Vec<u16>),  // 10/12-bit RGB interleaved
}

impl ImageBuffer {
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::Luma8(v) | Self::Rgb8(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_u16(&self) -> Option<&[u16]> {
        match self {
            Self::Luma16(v) | Self::Rgb16(v) => Some(v),
            _ => None,
        }
    }
    /// Number of channels: 1 for luma-only, 3 for RGB.
    pub fn channels(&self) -> usize {
        match self {
            Self::Luma8(_) | Self::Luma16(_) => 1,
            _ => 3,
        }
    }
    /// Total sample count (= `width × height × channels()`).
    pub fn len(&self) -> usize {
        match self {
            Self::Luma8(v) => v.len(),
            Self::Luma16(v) => v.len(),
            Self::Rgb8(v) => v.len(),
            Self::Rgb16(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn is_luma(&self) -> bool {
        matches!(self, Self::Luma8(_) | Self::Luma16(_))
    }
}

/// A typed single-plane sample buffer — `u8` for 8-bit, `u16` for 10/12-bit.
///
/// Used for the individual Y, Cb, Cr planes of [`crate::DecodedYuv`].
#[derive(Clone, Debug)]
pub enum SampleBuf {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl SampleBuf {
    pub fn as_u8(&self) -> Option<&[u8]> {
        if let Self::U8(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_u16(&self) -> Option<&[u16]> {
        if let Self::U16(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
//
// /// Full pixel-format description: chroma subsampling and sample bit depth.
// #[derive(Clone, Copy, Debug)]
// pub struct PixelFormat {
//     pub chroma: ChromaFormat,
//     pub bit_depth: BitDepth,
// }
//
// impl PixelFormat {
//     pub fn new(chroma: ChromaFormat, bit_depth: BitDepth) -> Self {
//         PixelFormat { chroma, bit_depth }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitdepth_derived_quantities() {
        assert_eq!(BitDepth::Eight.bits(), 8);
        assert_eq!(BitDepth::Ten.bits(), 10);
        assert_eq!(BitDepth::Twelve.bits(), 12);
        assert_eq!(BitDepth::Eight.minus8(), 0);
        assert_eq!(BitDepth::Ten.minus8(), 2);
        assert_eq!(BitDepth::Twelve.minus8(), 4);
        assert_eq!(BitDepth::Eight.max_val(), 255);
        assert_eq!(BitDepth::Ten.max_val(), 1023);
        assert_eq!(BitDepth::Twelve.max_val(), 4095);
        assert_eq!(BitDepth::Eight.neutral(), 128);
        assert_eq!(BitDepth::Ten.neutral(), 512);
        assert_eq!(BitDepth::Twelve.neutral(), 2048);
        assert_eq!(BitDepth::Eight.qp_bd_offset(), 0);
        assert_eq!(BitDepth::Ten.qp_bd_offset(), 12);
        assert_eq!(BitDepth::Twelve.qp_bd_offset(), 24);
    }

    #[test]
    fn bitdepth_from_bits_roundtrip() {
        assert_eq!(BitDepth::from_bits(8), BitDepth::Eight);
        assert_eq!(BitDepth::from_bits(10), BitDepth::Ten);
        assert_eq!(BitDepth::from_bits(12), BitDepth::Twelve);
    }

    #[test]
    #[should_panic]
    fn bitdepth_rejects_unsupported() {
        let _ = BitDepth::from_bits(16);
    }
}
