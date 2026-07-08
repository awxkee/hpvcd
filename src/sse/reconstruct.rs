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

use crate::reconstruct::{
    add_residual_into_scalar, add_residual_into_scalar16, can_reconstruct_full_block, sample_max,
};

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
fn add_clip8_sse41(pred: __m128i, res: &[i32; 8], zero: __m128i, max: __m128i) -> __m128i {
    let (res4, _) = res.as_chunks::<4>();
    let lo = add_clip4_sse41(pred, load_i32x4(&res4[0]), zero, max);
    let hi = add_clip4_sse41(_mm_srli_si128::<8>(pred), load_i32x4(&res4[1]), zero, max);
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

    let (pred8, _) = pred[..n].as_chunks::<8>();
    let (res8, _) = res[..n].as_chunks::<8>();
    let (dst8, _) = dst[..n].as_chunks_mut::<8>();

    for ((pred, res), dst) in pred8.iter().zip(res8.iter()).zip(dst8.iter_mut()) {
        let out = add_clip8_sse41(load_u16x8(pred), res, zero, max);
        store_u16x8(dst, out);
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
    let Some(n2) = n.checked_mul(n) else {
        return;
    };
    let Some(pred) = pred.get(..n2) else {
        return;
    };
    let Some(res) = res.get(..n2) else {
        return;
    };
    let max = _mm_set1_epi32(sample_max(bit_depth));

    let dst_rows = dst.chunks_mut(stride).take(n);
    let pred_rows = pred.chunks_exact(n);
    let res_rows = res.chunks_exact(n);
    for ((dst_row, pred_row), res_row) in dst_rows.zip(pred_rows).zip(res_rows) {
        add_clip_row_sse41(&mut dst_row[..n], pred_row, res_row, n, max);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_residual_into_sse41(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i32],
    n: usize,
    valid_w: usize,
    valid_h: usize,
    bit_depth: u8,
) {
    if !supported_n(n)
        || !can_reconstruct_full_block(dst, stride, pred, res, n, valid_w, valid_h, bit_depth)
    {
        add_residual_into_scalar(dst, stride, pred, res, n, valid_w, valid_h, bit_depth);
        return;
    }

    unsafe { add_residual_into_sse41_impl(dst, stride, pred, res, n, bit_depth) }
}

// 8-bit i16-residual path: saturating i16 adds, 8 px per op.

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i16x2(src: &[i16]) -> __m128i {
    debug_assert!(src.len() >= 2);
    unsafe { _mm_castps_si128(_mm_load_ss(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i16x4(src: &[i16]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i16x8(src: &[i16]) -> __m128i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

/// Saturating add equals widen+clamp because the result is clamped to max <= 32767.
#[inline]
#[target_feature(enable = "sse4.1")]
fn add_clip_i16(pred: __m128i, res: __m128i, zero: __m128i, max: __m128i) -> __m128i {
    _mm_min_epi16(_mm_max_epi16(_mm_adds_epi16(pred, res), zero), max)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn add_clip_row_sse41_16(dst: &mut [u16], pred: &[u16], res: &[i16], n: usize, max: __m128i) {
    let zero = _mm_setzero_si128();

    if n == 2 {
        let s = add_clip_i16(load_u16x2(pred), load_i16x2(res), zero, max);
        store_u16x2(dst, s);
        return;
    }
    if n == 4 {
        let s = add_clip_i16(load_u16x4(pred), load_i16x4(res), zero, max);
        store_u16x4(dst, s);
        return;
    }

    let (pred8, _) = pred[..n].as_chunks::<8>();
    let (res8, _) = res[..n].as_chunks::<8>();
    let (dst8, _) = dst[..n].as_chunks_mut::<8>();

    for ((pred, res), dst) in pred8.iter().zip(res8.iter()).zip(dst8.iter_mut()) {
        let s = add_clip_i16(load_u16x8(pred), load_i16x8(res), zero, max);
        store_u16x8(dst, s);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn add_residual_into_sse41_impl_16(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i16],
    n: usize,
    bit_depth: u8,
) {
    debug_assert!(supported_n(n));
    let Some(n2) = n.checked_mul(n) else {
        return;
    };
    let Some(pred) = pred.get(..n2) else {
        return;
    };
    let Some(res) = res.get(..n2) else {
        return;
    };
    let max = _mm_set1_epi16(sample_max(bit_depth) as i16);
    let dst_rows = dst.chunks_mut(stride).take(n);
    let pred_rows = pred.chunks_exact(n);
    let res_rows = res.chunks_exact(n);
    for ((dst_row, pred_row), res_row) in dst_rows.zip(pred_rows).zip(res_rows) {
        add_clip_row_sse41_16(&mut dst_row[..n], pred_row, res_row, n, max);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_residual_into_sse41_16(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i16],
    n: usize,
    valid_w: usize,
    valid_h: usize,
    bit_depth: u8,
) {
    if !supported_n(n)
        || sample_max(bit_depth) > 32767
        || !can_reconstruct_full_block(dst, stride, pred, res, n, valid_w, valid_h, bit_depth)
    {
        add_residual_into_scalar16(dst, stride, pred, res, n, valid_w, valid_h, bit_depth);
        return;
    }
    unsafe { add_residual_into_sse41_impl_16(dst, stride, pred, res, n, bit_depth) }
}
