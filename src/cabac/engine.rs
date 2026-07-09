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

pub(crate) use crate::cabac::contexts::CtxModel;

use std::borrow::Cow;

#[rustfmt::skip]
pub(crate) static RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128,176,208,240],[128,167,197,227],[128,158,187,216],[123,150,178,205],
    [116,142,169,195],[111,135,160,185],[105,128,152,175],[100,122,144,166],
    [95,116,137,158],[90,110,130,150],[85,104,123,142],[81,99,117,135],
    [77,94,111,128],[73,89,105,122],[69,85,100,116],[66,80,95,110],
    [62,76,90,104],[59,72,86,99],[56,69,81,94],[53,65,77,89],
    [51,62,73,85],[48,59,69,80],[46,56,66,76],[43,53,63,72],
    [41,50,59,69],[39,48,56,65],[37,45,54,62],[35,43,51,59],
    [33,41,48,56],[32,39,46,53],[30,37,43,50],[29,35,41,48],
    [27,33,39,45],[26,31,37,43],[24,30,35,41],[23,28,33,39],
    [22,27,32,37],[21,26,30,35],[20,24,29,33],[19,23,27,31],
    [18,22,26,30],[17,21,25,28],[16,20,23,27],[15,19,22,25],
    [14,18,21,24],[14,17,20,23],[13,16,19,22],[12,15,18,21],
    [12,14,17,20],[11,14,16,19],[11,13,15,18],[10,12,15,17],
    [10,12,14,16],[9,11,13,15],[9,11,12,14],[8,10,12,14],
    [8,9,11,13],[7,9,11,12],[7,9,10,12],[7,8,10,11],
    [6,8,9,11],[6,7,9,10],[6,7,8,9],[2,2,2,2],
];

#[rustfmt::skip]
pub(crate) static TRANS_IDX_MPS: [u8; 64] = [
     1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,
    17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    49,50,51,52,53,54,55,56,57,58,59,60,61,62,62,63,
];

#[rustfmt::skip]
pub(crate) static TRANS_IDX_LPS: [u8; 64] = [
     0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9,11,11,12,
    13,13,15,15,16,16,18,18,19,19,21,21,22,22,23,24,
    24,25,26,26,27,27,28,29,29,30,30,30,31,32,32,33,
    33,33,34,34,35,35,35,36,36,36,37,37,37,38,38,63,
];

/// CABAC decoder. Feed it EBSP-unescaped slice payload bytes.
/// Initialises by consuming 9 seed bits.
pub(crate) struct CabacDecoder<'a> {
    pub(crate) range: u32,
    pub(crate) offset: u32,
    data: Cow<'a, [u8]>,
    /// Index of the next byte in `data` not yet loaded into `bitbuf`.
    pub(crate) byte_pos: usize,
    /// Bit reservoir, left-aligned: the next bit to consume is bit 63.
    bitbuf: u64,
    /// Number of valid (unconsumed) bits currently in `bitbuf`.
    bitcnt: u32,
}

impl CabacDecoder<'static> {
    pub(crate) fn new(data: &[u8]) -> Result<Self, crate::error::DecodeError> {
        Self::from_cow(Cow::Owned(data.to_vec()))
    }
}

impl<'a> CabacDecoder<'a> {
    pub(crate) fn new_borrowed(data: &'a [u8]) -> Result<Self, crate::error::DecodeError> {
        CabacDecoder::from_cow(Cow::Borrowed(data))
    }

    fn from_cow(data: Cow<'a, [u8]>) -> Result<Self, crate::error::DecodeError> {
        if data.len() < 2 {
            return Err(crate::error::DecodeError::Bitstream(
                "CABAC payload too short to initialise".into(),
            ));
        }
        let mut dec = CabacDecoder {
            range: 510,
            offset: 0,
            data,
            byte_pos: 0,
            bitbuf: 0,
            bitcnt: 0,
        };
        dec.offset = dec.read_bits(9);
        Ok(dec)
    }

    /// Reset the engine onto a new owned byte buffer (e.g. the next slice
    /// segment's CABAC data) and prime it, reusing this decoder instance.
    pub(crate) fn reset_with(&mut self, data: &[u8]) -> Result<(), crate::error::DecodeError> {
        if data.len() < 2 {
            return Err(crate::error::DecodeError::Bitstream(
                "CABAC payload too short to initialise".into(),
            ));
        }
        self.data = Cow::Owned(data.to_vec());
        self.range = 510;
        self.offset = 0;
        self.byte_pos = 0;
        self.bitbuf = 0;
        self.bitcnt = 0;
        self.offset = self.read_bits(9);
        Ok(())
    }

    /// Refill the reservoir with whole bytes until at least 56 bits are buffered
    /// (or the input is exhausted).
    #[inline(always)]
    fn refill(&mut self) {
        while self.bitcnt <= 56 && self.byte_pos < self.data.len() {
            self.bitbuf |= (self.data[self.byte_pos] as u64) << (56 - self.bitcnt);
            self.bitcnt += 8;
            self.byte_pos += 1;
        }
    }

    /// Read the next input bit (MSB-first). Returns stuffed 1s past end-of-input.
    #[inline(always)]
    fn next_bit(&mut self) -> u32 {
        if self.bitcnt == 0 {
            self.refill();
            if self.bitcnt == 0 {
                return 1; // past end of input: bitstream stuffing
            }
        }
        let bit = (self.bitbuf >> 63) as u32;
        self.bitbuf <<= 1;
        self.bitcnt -= 1;
        bit
    }

    /// Read `n` (≤ 32) bits MSB-first, padding past end-of-input with 1s.
    #[inline]
    fn read_bits(&mut self, n: u32) -> u32 {
        if self.bitcnt < n {
            self.refill();
        }
        if self.bitcnt >= n {
            // Fast path: all n bits are in the reservoir.
            let v = (self.bitbuf >> (64 - n)) as u32;
            self.bitbuf <<= n;
            self.bitcnt -= n;
            v
        } else {
            // Slow path near end of input: take what we have, stuff the rest.
            let mut v = 0u32;
            for _ in 0..n {
                v = (v << 1) | self.next_bit();
            }
            v
        }
    }

    #[inline]
    fn renorm(&mut self) {
        // range ∈ [1, 510]; shift left until bit 8 (value 256) is set, then pull
        // the same number of fresh bits into offset in one batched read.
        if self.range < 256 {
            let shift = self.range.leading_zeros() - 23; // = 8 - floor(log2(range))
            self.range <<= shift;
            self.offset = (self.offset << shift) | self.read_bits(shift);
        }
    }

    /// Decode one context-coded bin (HEVC Table 9-15 DecodeDecision).
    #[inline]
    pub(crate) fn decode_bin(&mut self, ctx: &mut CtxModel) -> u8 {
        let ctx_state = ctx.state;
        let state = (ctx_state & 63) as usize;
        let mps = ctx_state >> 6;
        let lps = RANGE_TAB_LPS[state][(self.range >> 6) as usize & 3] as u32;
        self.range -= lps;
        if self.offset >= self.range {
            let bin_val = mps ^ 1;
            self.offset -= self.range;
            self.range = lps;
            let next_mps = if state == 0 { mps ^ 1 } else { mps };
            ctx.state = TRANS_IDX_LPS[state] | (next_mps << 6);
            self.renorm();
            bin_val
        } else {
            ctx.state = TRANS_IDX_MPS[state] | (mps << 6);
            self.renorm();
            mps
        }
    }

    /// Decode one bypass bin (HEVC §9.3.4.2 DecodeBypass).
    #[inline]
    pub(crate) fn decode_bypass(&mut self) -> u8 {
        self.offset = (self.offset << 1) | self.next_bit();
        if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        }
    }

    /// Decode terminate bin (end_of_slice / end_of_sub_stream).
    #[inline]
    pub(crate) fn decode_terminate(&mut self) -> u8 {
        self.range -= 2;
        if self.offset >= self.range {
            1
        } else {
            self.renorm();
            0
        }
    }
}

impl<'a> CabacDecoder<'a> {
    /// Byte-align: discard buffered bits so the next read starts on a byte
    /// boundary (WPP). The raw bit-reader position (bits consumed from `data`)
    /// is `byte_pos*8 - bitcnt`; rounding that up to the next byte boundary
    /// gives `byte_pos - bitcnt/8` regardless of whether we are mid-byte.
    pub(crate) fn byte_align(&mut self) {
        self.byte_pos -= (self.bitcnt / 8) as usize;
        self.bitbuf = 0;
        self.bitcnt = 0;
    }

    /// Re-prime the engine from the current byte position (WPP row start).
    pub(crate) fn reinit_engine(&mut self) {
        self.range = 510;
        self.offset = 0;
        self.bitbuf = 0;
        self.bitcnt = 0;
        self.offset = self.read_bits(9);
    }

    /// Read `n` (≤ 32) raw bits from the stream, MSB-first. Used for PCM sample
    /// data (§7.3.8.5 / §9.3.1), which is coded as fixed-length uncompressed bits
    /// between an engine byte-alignment and a subsequent re-initialization.
    pub(crate) fn read_pcm_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        self.read_bits(n)
    }
}
