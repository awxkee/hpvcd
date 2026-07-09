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

use core::arch::x86_64::*;

use crate::reconstruct::{can_reconstruct_full_block, sample_max};

#[inline(always)]
fn supported_n(n: usize) -> bool {
    matches!(n, 2 | 4 | 8 | 16 | 32)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x2(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 2);
    unsafe { _mm_castps_si128(_mm_load_ss(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x16(src: &[u16]) -> __m256i {
    debug_assert!(src.len() >= 16);
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16x2(src: &[i16]) -> __m128i {
    debug_assert!(src.len() >= 2);
    unsafe { _mm_castps_si128(_mm_load_ss(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16x4(src: &[i16]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16x8(src: &[i16]) -> __m128i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16x16(src: &[i16]) -> __m256i {
    debug_assert!(src.len() >= 16);
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x2(src: &[i32]) -> __m128i {
    debug_assert!(src.len() >= 2);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x4(src: &[i32]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x8(src: &[i32]) -> __m256i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn zext128(v: __m128i) -> __m256i {
    _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x2(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 2);
    unsafe { _mm_store_ss(dst.as_mut_ptr().cast(), _mm_castsi128_ps(v)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x4(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 4);
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x8(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 8);
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x16(dst: &mut [u16], v: __m256i) {
    debug_assert!(dst.len() >= 16);
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast::<__m256i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_u16x8(sum: __m256i) -> __m128i {
    let packed = _mm256_packus_epi32(sum, _mm256_setzero_si256());
    let packed = _mm256_permute4x64_epi64::<0xD8>(packed);
    _mm256_castsi256_si128(packed)
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_u16x16(lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_clip8_i32(pred: __m128i, res: __m256i, zero: __m256i, max: __m256i) -> __m256i {
    let pred = _mm256_cvtepu16_epi32(pred);
    let sum = _mm256_add_epi32(pred, res);
    _mm256_min_epi32(_mm256_max_epi32(sum, zero), max)
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_clip16_i32(
    pred: __m256i,
    res_lo: __m256i,
    res_hi: __m256i,
    zero: __m256i,
    max: __m256i,
) -> (__m256i, __m256i) {
    let pred_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(pred));
    let pred_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(pred));
    let sum_lo = _mm256_add_epi32(pred_lo, res_lo);
    let sum_hi = _mm256_add_epi32(pred_hi, res_hi);
    (
        _mm256_min_epi32(_mm256_max_epi32(sum_lo, zero), max),
        _mm256_min_epi32(_mm256_max_epi32(sum_hi, zero), max),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_clip_row_avx2(dst: &mut [u16], pred: &[u16], res: &[i32], n: usize, max: __m256i) {
    let zero = _mm256_setzero_si256();

    if n == 2 {
        let sum = add_clip8_i32(load_u16x2(pred), zext128(load_i32x2(res)), zero, max);
        store_u16x2(dst, pack_u16x8(sum));
        return;
    }

    if n == 4 {
        let sum = add_clip8_i32(load_u16x4(pred), zext128(load_i32x4(res)), zero, max);
        store_u16x4(dst, pack_u16x8(sum));
        return;
    }

    if n == 8 {
        let sum = add_clip8_i32(load_u16x8(pred), load_i32x8(res), zero, max);
        store_u16x8(dst, pack_u16x8(sum));
        return;
    }

    let (pred16, _) = pred[..n].as_chunks::<16>();
    let (res16, _) = res[..n].as_chunks::<16>();
    let (dst16, _) = dst[..n].as_chunks_mut::<16>();

    for ((pred, res), dst) in pred16.iter().zip(res16.iter()).zip(dst16.iter_mut()) {
        let pred = load_u16x16(pred);
        let (res_lo, res_hi) = res.split_at(8);
        let (sum_lo, sum_hi) =
            add_clip16_i32(pred, load_i32x8(res_lo), load_i32x8(res_hi), zero, max);
        store_u16x16(dst, pack_u16x16(sum_lo, sum_hi));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_clip_row_avx2_16(dst: &mut [u16], pred: &[u16], res: &[i16], n: usize, max: __m256i) {
    let zero = _mm256_setzero_si256();

    if n == 2 {
        let pred = zext128(load_u16x2(pred));
        let res = zext128(load_i16x2(res));
        let sum = _mm256_min_epi16(_mm256_max_epi16(_mm256_adds_epi16(pred, res), zero), max);
        store_u16x2(dst, _mm256_castsi256_si128(sum));
        return;
    }

    if n == 4 {
        let pred = zext128(load_u16x4(pred));
        let res = zext128(load_i16x4(res));
        let sum = _mm256_min_epi16(_mm256_max_epi16(_mm256_adds_epi16(pred, res), zero), max);
        store_u16x4(dst, _mm256_castsi256_si128(sum));
        return;
    }

    if n == 8 {
        let pred = zext128(load_u16x8(pred));
        let res = zext128(load_i16x8(res));
        let sum = _mm256_min_epi16(_mm256_max_epi16(_mm256_adds_epi16(pred, res), zero), max);
        store_u16x8(dst, _mm256_castsi256_si128(sum));
        return;
    }

    let (pred16, _) = pred[..n].as_chunks::<16>();
    let (res16, _) = res[..n].as_chunks::<16>();
    let (dst16, _) = dst[..n].as_chunks_mut::<16>();

    for ((pred, res), dst) in pred16.iter().zip(res16.iter()).zip(dst16.iter_mut()) {
        let pred = load_u16x16(pred);
        let res = load_i16x16(res);
        let sum = _mm256_min_epi16(_mm256_max_epi16(_mm256_adds_epi16(pred, res), zero), max);
        store_u16x16(dst, sum);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_residual_into_avx2_impl(
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

    let max = _mm256_set1_epi32(sample_max(bit_depth));
    let dst_rows = dst.chunks_mut(stride).take(n);
    let pred_rows = pred.chunks_exact(n);
    let res_rows = res.chunks_exact(n);

    for ((dst_row, pred_row), res_row) in dst_rows.zip(pred_rows).zip(res_rows) {
        add_clip_row_avx2(&mut dst_row[..n], pred_row, res_row, n, max);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_residual_into_avx2_impl_16(
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

    let max = _mm256_set1_epi16(sample_max(bit_depth) as i16);
    let dst_rows = dst.chunks_mut(stride).take(n);
    let pred_rows = pred.chunks_exact(n);
    let res_rows = res.chunks_exact(n);

    for ((dst_row, pred_row), res_row) in dst_rows.zip(pred_rows).zip(res_rows) {
        add_clip_row_avx2_16(&mut dst_row[..n], pred_row, res_row, n, max);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_residual_into_avx2(
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
        return;
    }

    unsafe { add_residual_into_avx2_impl(dst, stride, pred, res, n, bit_depth) }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_residual_into_avx2_16(
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
        return;
    }

    unsafe { add_residual_into_avx2_impl_16(dst, stride, pred, res, n, bit_depth) }
}
