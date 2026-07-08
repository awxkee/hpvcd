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
use std::ops::Div;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FastDivU32 {
    magic: u32,
    more: u8,
}

const ADD_MARKER: u8 = 0x40;
const SHIFT_MASK: u8 = 0x1f;

impl FastDivU32 {
    pub(crate) fn new(divisor: u32) -> Self {
        assert_ne!(divisor, 0, "division by zero");

        let floor_log_2_d = 31 - divisor.leading_zeros();

        // Powers of two, including divisor == 1.
        if divisor.is_power_of_two() {
            return Self {
                magic: 0,
                more: floor_log_2_d as u8,
            };
        }

        // Compute floor(2^(32 + floor_log_2_d) / divisor).
        let numerator = 1u64 << (32 + floor_log_2_d);
        let mut proposed_m = numerator / divisor as u64;
        let rem = (numerator % divisor as u64) as u32;

        let e = divisor - rem;
        let more: u8;

        if e < (1u32 << floor_log_2_d) {
            // Smaller magic, no correction add.
            more = floor_log_2_d as u8;
        } else {
            // Larger magic, requires correction add during division.
            proposed_m += proposed_m;

            let twice_rem = rem.wrapping_add(rem);
            if twice_rem >= divisor || twice_rem < rem {
                proposed_m += 1;
            }

            more = floor_log_2_d as u8 | ADD_MARKER;
        }

        Self {
            magic: proposed_m.wrapping_add(1) as u32,
            more,
        }
    }

    #[inline(always)]
    fn mul_hi_u32(a: u32, b: u32) -> u32 {
        (((a as u64) * (b as u64)) >> 32) as u32
    }

    #[inline(always)]
    pub(crate) fn div_fast(self, n: u32) -> u32 {
        if self.magic == 0 {
            return n >> self.more;
        }

        let q = Self::mul_hi_u32(n, self.magic);

        if (self.more & ADD_MARKER) != 0 {
            let shift = self.more & SHIFT_MASK;

            // q = (((n - q) >> 1) + q) >> shift
            let t = ((n.wrapping_sub(q)) >> 1).wrapping_add(q);
            t >> shift
        } else {
            q >> self.more
        }
    }
}

impl Div<FastDivU32> for u32 {
    type Output = u32;

    fn div(self, rhs: FastDivU32) -> Self::Output {
        rhs.div_fast(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_divisors() {
        let divisors = [
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            16,
            31,
            32,
            63,
            64,
            127,
            128,
            255,
            256,
            1023,
            1024,
            65535,
            65536,
            u32::MAX - 1,
            u32::MAX,
        ];

        let numerators = [
            0,
            1,
            2,
            3,
            4,
            5,
            7,
            15,
            16,
            31,
            32,
            63,
            64,
            127,
            128,
            255,
            256,
            1023,
            1024,
            123456789,
            u32::MAX - 1,
            u32::MAX,
        ];

        for &d in &divisors {
            let fd = FastDivU32::new(d);

            for &n in &numerators {
                assert_eq!(fd.div_fast(n), n / d, "bad div: {n} / {d}");
            }
        }
    }

    #[test]
    fn test_small_exhaustive() {
        for d in 1u32..10_000 {
            let fd = FastDivU32::new(d);

            for n in 0u32..10_000 {
                assert_eq!(fd.div_fast(n), n / d, "bad div: {n} / {d}");
            }
        }
    }

    #[test]
    fn test_pseudo_random_full_range() {
        let mut x = 0x1234_5678u32;

        for _ in 0..1_000_000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = x;

            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let d = x | 1; // nonzero, mostly odd divisors

            let fd = FastDivU32::new(d);

            assert_eq!(fd.div_fast(n), n / d, "bad div: {n} / {d}");
        }
    }
}
