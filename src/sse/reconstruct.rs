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

use crate::reconstruct::add_residual_into_scalar;

#[inline]
fn supported_n(n: usize) -> bool {
    matches!(n, 2 | 4 | 8 | 16 | 32)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u16x2(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 2);
    unsafe { _mm_castps_si128(_mm_load_ss(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u16x4(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u16x8(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i32x2(src: &[i32]) -> __m128i {
    debug_assert!(src.len() >= 2);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i32x4(src: &[i32]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x2(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 2);
    unsafe {
        _mm_store_ss(dst.as_mut_ptr().cast(), _mm_castsi128_ps(v));
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x4(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 4);
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x8(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 8);
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn add_clip4_sse41(pred: __m128i, res: __m128i, zero: __m128i, max: __m128i) -> __m128i {
    let pred = _mm_cvtepu16_epi32(pred);
    let sum = _mm_add_epi32(pred, res);
    _mm_min_epi32(_mm_max_epi32(sum, zero), max)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn add_clip8_sse41(pred: __m128i, res: &[i32], zero: __m128i, max: __m128i) -> __m128i {
    let lo = add_clip4_sse41(pred, load_i32x4(res), zero, max);
    let hi = add_clip4_sse41(_mm_srli_si128::<8>(pred), load_i32x4(&res[4..]), zero, max);
    _mm_packus_epi32(lo, hi)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn add_clip_row_sse41(dst: &mut [u16], pred: &[u16], res: &[i32], n: usize, max: __m128i) {
    let zero = _mm_setzero_si128();

    if n == 2 {
        let pred = load_u16x2(pred);
        let sum = add_clip4_sse41(pred, load_i32x2(res), zero, max);
        store_u16x2(dst, _mm_packus_epi32(sum, zero));
        return;
    }

    if n == 4 {
        let pred = load_u16x4(pred);
        let sum = add_clip4_sse41(pred, load_i32x4(res), zero, max);
        store_u16x4(dst, _mm_packus_epi32(sum, zero));
        return;
    }

    let mut x = 0usize;
    while x < n {
        let pred = load_u16x8(&pred[x..]);
        let out = add_clip8_sse41(pred, &res[x..], zero, max);
        store_u16x8(&mut dst[x..], out);
        x += 8;
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn add_residual_into_sse41_impl(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i32],
    n: usize,
    bit_depth: u8,
) {
    debug_assert!(supported_n(n));
    let pred = &pred[..n * n];
    let res = &res[..n * n];
    let max = _mm_set1_epi32((1i32 << bit_depth) - 1);

    for y in 0..n {
        let row_off = y * n;
        let dst_off = y * stride;
        add_clip_row_sse41(
            &mut dst[dst_off..dst_off + n],
            &pred[row_off..row_off + n],
            &res[row_off..row_off + n],
            n,
            max,
        );
    }
}

pub(crate) fn add_residual_into_sse41(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i32],
    n: usize,
    bit_depth: u8,
) {
    if !supported_n(n) {
        add_residual_into_scalar(dst, stride, pred, res, n, bit_depth);
        return;
    }

    unsafe { add_residual_into_sse41_impl(dst, stride, pred, res, n, bit_depth) }
}
