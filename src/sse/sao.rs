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

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sao::apply_sao_plane_scalar;

#[inline]
#[target_feature(enable = "sse4.1")]
fn shr_epi32_sse41(v: __m128i, shift: u8) -> __m128i {
    let cnt = _mm_cvtsi32_si128(shift.min(11) as i32);
    _mm_srl_epi32(v, cnt)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn band_offset4_sse41(
    samples: __m128i,
    offsets: &[i32; 4],
    band_pos: __m128i,
    shift: u8,
    zero: __m128i,
    max: __m128i,
) -> (__m128i, __m128i) {
    let band = shr_epi32_sse41(samples, shift);
    let rel = _mm_sub_epi32(band, band_pos);
    let mut off = zero;

    let m0 = _mm_cmpeq_epi32(rel, _mm_setzero_si128());
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[0]), m0);
    let m1 = _mm_cmpeq_epi32(rel, _mm_set1_epi32(1));
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[1]), m1);
    let m2 = _mm_cmpeq_epi32(rel, _mm_set1_epi32(2));
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[2]), m2);
    let m3 = _mm_cmpeq_epi32(rel, _mm_set1_epi32(3));
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[3]), m3);

    let active = _mm_or_si128(_mm_or_si128(m0, m1), _mm_or_si128(m2, m3));
    let v = _mm_add_epi32(samples, off);
    (_mm_min_epi32(_mm_max_epi32(v, zero), max), active)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn band_offset8_sse41(
    dst: &[u16],
    src: &[u16],
    offsets: &[i32; 4],
    band_pos: __m128i,
    shift: u8,
    zero: __m128i,
    max: __m128i,
) -> __m128i {
    debug_assert!(dst.len() >= 8);
    debug_assert!(src.len() >= 8);
    let old = unsafe { _mm_loadu_si128(dst.as_ptr().cast::<__m128i>()) };
    let s = unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) };
    let lo = _mm_cvtepu16_epi32(s);
    let hi = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(s));
    let (lo, mlo) = band_offset4_sse41(lo, offsets, band_pos, shift, zero, max);
    let (hi, mhi) = band_offset4_sse41(hi, offsets, band_pos, shift, zero, max);
    let out = _mm_packus_epi32(lo, hi);
    let mask = _mm_packs_epi32(mlo, mhi);
    _mm_blendv_epi8(old, out, mask)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_sao_band_offset_sse41_impl(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    band_pos: u8,
    bd: u8,
) {
    let max_val = ((1u32 << bd) - 1) as i32;
    let max = _mm_set1_epi32(max_val);
    let zero = _mm_setzero_si128();
    let band_pos_v = _mm_set1_epi32(band_pos as i32);
    let shift = bd - 5;

    for y in y0..y_end {
        let row = y * w;
        let mut x = x0;

        while x + 8 <= x_end {
            let out = band_offset8_sse41(
                &dst[row + x..],
                &src[row + x..],
                offsets,
                band_pos_v,
                shift,
                zero,
                max,
            );
            unsafe { _mm_storeu_si128(dst[row + x..].as_mut_ptr().cast::<__m128i>(), out) };
            x += 8;
        }

        for x in x..x_end {
            let s = src[row + x] as i32;
            let band = (s >> shift) as u8;
            let rel = band.wrapping_sub(band_pos);
            if rel < 4 {
                dst[row + x] = (s + offsets[rel as usize]).clamp(0, max_val) as u16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_plane_sse41(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    type_idx: u8,
    offsets: &[i32; 4],
    band_pos: u8,
    eo_class: u8,
    bd: u8,
) {
    if type_idx != 1 || x_end <= x0 || y_end <= y0 {
        apply_sao_plane_scalar(
            dst, src, w, h, x0, y0, x_end, y_end, type_idx, offsets, band_pos, eo_class, bd,
        );
        return;
    }

    unsafe {
        apply_sao_band_offset_sse41_impl(dst, src, w, x0, y0, x_end, y_end, offsets, band_pos, bd)
    }
}
