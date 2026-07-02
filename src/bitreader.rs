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
/// index of the NAL (source) byte it came from. Only used by tests now; the
/// production wavefront path builds just the map via [`rbsp_src_map`] (it
/// already holds the RBSP).
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

pub(crate) fn rbsp_src_map(src: &[u8]) -> Vec<usize> {
    let mut src_of = Vec::with_capacity(src.len());
    let mut i = 0usize;
    while i < src.len() {
        if i + 2 < src.len() && src[i] == 0x00 && src[i + 1] == 0x00 && src[i + 2] == 0x03 {
            src_of.push(i);
            src_of.push(i + 1);
            i += 3;
        } else {
            src_of.push(i);
            i += 1;
        }
    }
    src_of
}

/// Translate a NAL-relative byte offset into an RBSP byte offset using the
/// `src_of` map from [`unescape_rbsp_with_map`]. Returns the first RBSP index
/// whose source index is ≥ `nal_off`; if none (the offset is past the last kept
/// byte), returns `src_of.len()` (i.e. the RBSP length).
pub(crate) fn nal_to_rbsp_offset(src_of: &[usize], nal_off: usize) -> usize {
    // `src_of` is strictly increasing, so binary-search the partition point.
    src_of.partition_point(|&s| s < nal_off)
}

#[cfg(test)]
mod tests {
    use crate::bitreader::{BitReader, rbsp_src_map, unescape_rbsp, unescape_rbsp_with_map};

    #[test]
    fn src_map_matches_with_map_variant() {
        // rbsp_src_map must produce the same offset map as the paired variant,
        // including across emulation-prevention bytes.
        let nal = [
            0x00u8, 0x00, 0x03, 0x00, 0xAA, 0x00, 0x00, 0x03, 0x01, 0xBB, 0xCC,
        ];
        let (_rbsp, map_ref) = unescape_rbsp_with_map(&nal);
        let map = rbsp_src_map(&nal);
        assert_eq!(map, map_ref);
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
