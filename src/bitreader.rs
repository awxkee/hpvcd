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

use crate::error::DecodeError;

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // bits consumed in data[byte_pos], 0..=7 (0 = fresh byte)
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read one bit (0 or 1).
    #[inline]
    pub(crate) fn read_bit(&mut self) -> Result<u32, DecodeError> {
        if self.byte_pos >= self.data.len() {
            return Err(DecodeError::Bitstream("unexpected end of stream".into()));
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
        Ok(bit as u32)
    }

    /// Read `n` bits (0..=32) MSB-first.
    #[inline]
    pub(crate) fn read_bits(&mut self, n: u32) -> Result<u32, DecodeError> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()?;
        }
        Ok(v)
    }

    /// Read a flag (1 bit as bool).
    #[inline]
    pub(crate) fn read_flag(&mut self) -> Result<bool, DecodeError> {
        Ok(self.read_bit()? != 0)
    }

    /// Read an unsigned Exp-Golomb coded value ue(v) (HEVC §9.1.1).
    pub(crate) fn read_ue(&mut self) -> Result<u32, DecodeError> {
        let mut leading_zeros = 0u32;
        while self.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(DecodeError::Bitstream(
                    "ue(v) leading zeros exceed 31".into(),
                ));
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let suffix = self.read_bits(leading_zeros)?;
        Ok((1 << leading_zeros) - 1 + suffix)
    }

    /// Read a signed Exp-Golomb coded value se(v) (HEVC §9.1.1).
    pub(crate) fn read_se(&mut self) -> Result<i32, DecodeError> {
        let ue = self.read_ue()?;
        let v = if ue & 1 == 1 {
            ((ue + 1) >> 1) as i32
        } else {
            -((ue >> 1) as i32)
        };
        Ok(v)
    }

    /// Byte position (for alignment checks).
    pub(crate) fn bit_pos(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }
}

/// Remove HEVC emulation-prevention bytes from an RBSP byte sequence.
/// The encoder inserts 0x03 after 0x00 0x00 whenever the next byte would be ≤ 0x03;
/// the decoder must strip these (HEVC §B.2).
pub(crate) fn unescape_rbsp(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    let mut i = 0usize;
    while i < src.len() {
        if i + 2 < src.len() && src[i] == 0x00 && src[i + 1] == 0x00 && src[i + 2] == 0x03 {
            out.push(0x00);
            out.push(0x00);
            i += 3; // skip the 0x03 emulation-prevention byte
        } else {
            out.push(src[i]);
            i += 1;
        }
    }
    out
}

/// Like [`unescape_rbsp`], but also returns, for each RBSP (output) byte, the
/// index of the NAL (source) byte it came from. This dense map is used by
/// tests as a reference for [`RbspOffsetMap`].
#[cfg(test)]
pub(crate) fn unescape_rbsp_with_map(src: &[u8]) -> (Vec<u8>, Vec<usize>) {
    let mut out = Vec::with_capacity(src.len());
    let mut src_of = Vec::with_capacity(src.len());
    let mut i = 0usize;
    while i < src.len() {
        if i + 2 < src.len() && src[i] == 0x00 && src[i + 1] == 0x00 && src[i + 2] == 0x03 {
            out.push(0x00);
            src_of.push(i);
            out.push(0x00);
            src_of.push(i + 1);
            i += 3; // skip the 0x03 emulation-prevention byte
        } else {
            out.push(src[i]);
            src_of.push(i);
            i += 1;
        }
    }
    (out, src_of)
}

/// Translates byte offsets between a NAL and its unescaped RBSP.
pub(crate) struct RbspOffsetMap {
    nal_len: usize,
    // NAL and RBSP offsets at each skipped emulation-prevention byte.
    skipped: Vec<(usize, usize)>,
}

impl RbspOffsetMap {
    pub(crate) fn from_nal(nal: &[u8]) -> Self {
        let mut skipped = Vec::new();
        // The 00 00 03 pattern cannot overlap itself, so inspecting every
        // window finds the same positions as advancing past each match.
        for (start, bytes) in nal.windows(3).enumerate() {
            if bytes == [0, 0, 3] {
                let nal_offset = start + 2;
                skipped.push((nal_offset, nal_offset - skipped.len()));
            }
        }
        Self {
            nal_len: nal.len(),
            skipped,
        }
    }

    /// NAL index of the byte at an RBSP offset, or `None` at or beyond the end.
    pub(crate) fn to_nal(&self, rbsp_offset: usize) -> Option<usize> {
        if rbsp_offset >= self.nal_len - self.skipped.len() {
            return None;
        }
        let skipped = self
            .skipped
            .partition_point(|&(_, rbsp)| rbsp <= rbsp_offset);
        Some(rbsp_offset + skipped)
    }

    /// First RBSP index whose source index is at least `nal_offset`.
    pub(crate) fn to_rbsp(&self, nal_offset: usize) -> usize {
        let nal_offset = nal_offset.min(self.nal_len);
        let skipped = self.skipped.partition_point(|&(nal, _)| nal < nal_offset);
        nal_offset - skipped
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, RbspOffsetMap, unescape_rbsp, unescape_rbsp_with_map};

    #[test]
    fn sparse_map_matches_dense_at_all_boundaries() {
        let cases: &[&[u8]] = &[
            &[],
            &[0, 3],
            &[0, 0, 3],
            &[0xAA, 0, 0, 3, 0, 0, 3, 0xBB, 0, 0, 3, 1, 0xCC],
            &[0, 0, 0, 3, 0xFF, 0, 0, 3, 3, 0, 0],
        ];
        for &nal in cases {
            let (_, dense) = unescape_rbsp_with_map(nal);
            let sparse = RbspOffsetMap::from_nal(nal);

            for nal_offset in (0..=nal.len()).chain([usize::MAX]) {
                let expected = dense.partition_point(|&source| source < nal_offset);
                assert_eq!(sparse.to_rbsp(nal_offset), expected, "NAL {nal:?}");
            }
            for rbsp_offset in (0..=dense.len()).chain([usize::MAX]) {
                assert_eq!(
                    sparse.to_nal(rbsp_offset),
                    dense.get(rbsp_offset).copied(),
                    "NAL {nal:?}"
                );
            }
        }
    }

    #[test]
    fn round_trip_ue() {
        // Build a hand-crafted bit sequence and read it back.
        // ue(0) = "1", ue(1) = "010", ue(2) = "011", ue(5) = "00110"
        let bits: &[u8] = &[
            0b1_010_011_0, // ue(0), ue(1), ue(2), partial
            0b0110_0000,   // rest of ue(5)
        ];
        let mut r = BitReader::new(bits);
        assert_eq!(r.read_ue().unwrap(), 0);
        assert_eq!(r.read_ue().unwrap(), 1);
        assert_eq!(r.read_ue().unwrap(), 2);
        assert_eq!(r.read_ue().unwrap(), 5);
    }

    #[test]
    fn unescape_removes_emulation_prevention() {
        let input = vec![0x00, 0x00, 0x03, 0x01, 0xFF];
        let out = unescape_rbsp(&input);
        assert_eq!(out, vec![0x00, 0x00, 0x01, 0xFF]);
    }
}
