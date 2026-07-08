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

use crate::transform::{
    DequantParams, ScalingMatrix, TransformSkipParams, dequantize_into_scalar_i16,
    dequantize_into_scalar_i32, dequantize_scaled_into_scalar_i16,
    dequantize_scaled_into_scalar_i32, dequantize_transform_skip_into_scalar_i16,
    dequantize_transform_skip_into_scalar_i32, dequantize_transform_skip_scaled_into_scalar_i16,
    dequantize_transform_skip_scaled_into_scalar_i32,
};

#[inline]
fn supported_n(n: usize) -> bool {
    matches!(n, 4 | 8 | 16 | 32)
}

#[inline]
fn mullo_safe(levels: &[i32], n: usize, params: DequantParams) -> bool {
    let Some(count) = n.checked_mul(n) else {
        return false;
    };
    let Some(levels) = levels.get(..count) else {
        return false;
    };
    if params.factor <= 0 || params.factor > i32::MAX as i64 || params.add > i32::MAX as i64 {
        return false;
    }
    let limit = ((i32::MAX as i64 - params.add) / params.factor).max(0);
    levels.iter().all(|&v| (v as i64).abs() <= limit)
}

#[inline]
fn scaling_factors4(base_factor: i64, scaling: ScalingMatrix<'_>, idx: usize) -> [i32; 4] {
    std::array::from_fn(|lane| (base_factor * scaling.coeff(idx + lane)) as i32)
}

#[inline]
fn scaled_mullo_safe(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
) -> bool {
    let Some(count) = n.checked_mul(n) else {
        return false;
    };
    let Some(levels) = levels.get(..count) else {
        return false;
    };
    let base_factor = params.factor / 16;
    if base_factor <= 0 || params.add > i32::MAX as i64 {
        return false;
    }

    levels.iter().enumerate().all(|(idx, &level)| {
        let factor = base_factor * scaling.coeff(idx);
        if factor <= 0 || factor > i32::MAX as i64 {
            return false;
        }
        let limit = ((i32::MAX as i64 - params.add) / factor).max(0);
        (level as i64).abs() <= limit
    })
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i32x4(src: &[i32; 4]) -> __m128i {
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i32x4(dst: &mut [i32; 4], v: __m128i) {
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i16x8(dst: &mut [i16; 8], v: __m128i) {
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sra_epi32(v: __m128i, shift: i32) -> __m128i {
    _mm_sra_epi32(v, _mm_cvtsi32_si128(shift))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn shl_epi32(v: __m128i, shift: i32) -> __m128i {
    _mm_sll_epi32(v, _mm_cvtsi32_si128(shift))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn clip_i16_s32x4(v: __m128i) -> __m128i {
    _mm_max_epi32(
        _mm_min_epi32(v, _mm_set1_epi32(32767)),
        _mm_set1_epi32(-32768),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn dequant4_sse41(levels: &[i32; 4], params: DequantParams) -> __m128i {
    let v = load_i32x4(levels);
    let v = _mm_mullo_epi32(v, _mm_set1_epi32(params.factor as i32));
    let v = _mm_add_epi32(v, _mm_set1_epi32(params.add as i32));
    clip_i16_s32x4(sra_epi32(v, params.shift))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn dequant4_scaled_sse41(levels: &[i32; 4], factors: &[i32; 4], params: DequantParams) -> __m128i {
    let v = load_i32x4(levels);
    let f = load_i32x4(factors);
    let v = _mm_mullo_epi32(v, f);
    let v = _mm_add_epi32(v, _mm_set1_epi32(params.add as i32));
    clip_i16_s32x4(sra_epi32(v, params.shift))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn transform_skip4_sse41(levels: &[i32; 4], params: TransformSkipParams) -> __m128i {
    let deq = dequant4_sse41(levels, params.dequant);
    let shifted = if params.tr_shift >= 0 {
        let v = _mm_add_epi32(deq, _mm_set1_epi32(params.tr_add));
        sra_epi32(v, params.tr_shift)
    } else {
        shl_epi32(deq, -params.tr_shift)
    };
    clip_i16_s32x4(shifted)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn transform_skip4_scaled_sse41(
    levels: &[i32; 4],
    factors: &[i32; 4],
    params: TransformSkipParams,
) -> __m128i {
    let deq = dequant4_scaled_sse41(levels, factors, params.dequant);
    let shifted = if params.tr_shift >= 0 {
        let v = _mm_add_epi32(deq, _mm_set1_epi32(params.tr_add));
        sra_epi32(v, params.tr_shift)
    } else {
        shl_epi32(deq, -params.tr_shift)
    };
    clip_i16_s32x4(shifted)
}

#[target_feature(enable = "sse4.1")]
fn dequantize_into_sse41_impl(levels: &[i32], n: usize, params: DequantParams, out: &mut [i32]) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<4>();
    let (out, _) = out[..count].as_chunks_mut::<4>();

    for (src, dst) in levels.iter().zip(out.iter_mut()) {
        store_i32x4(dst, dequant4_sse41(src, params));
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_into_sse41_16_impl(levels: &[i32], n: usize, params: DequantParams, out: &mut [i16]) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    for (src, dst) in levels.iter().zip(out.iter_mut()) {
        let (src4, _) = src.as_chunks::<4>();
        let lo = dequant4_sse41(&src4[0], params);
        let hi = dequant4_sse41(&src4[1], params);
        store_i16x8(dst, _mm_packs_epi32(lo, hi));
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_transform_skip_into_sse41_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<4>();
    let (out, _) = out[..16].as_chunks_mut::<4>();

    for (src, dst) in levels.iter().zip(out.iter_mut()) {
        store_i32x4(dst, transform_skip4_sse41(src, params));
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_transform_skip_into_sse41_16_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<8>();

    for (src, dst) in levels.iter().zip(out.iter_mut()) {
        let (src4, _) = src.as_chunks::<4>();
        let lo = transform_skip4_sse41(&src4[0], params);
        let hi = transform_skip4_sse41(&src4[1], params);
        store_i16x8(dst, _mm_packs_epi32(lo, hi));
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_scaled_into_sse41_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    let count = n * n;
    let base_factor = params.factor / 16;
    let (levels, _) = levels[..count].as_chunks::<4>();
    let (out, _) = out[..count].as_chunks_mut::<4>();

    for (block_idx, (src, dst)) in levels.iter().zip(out.iter_mut()).enumerate() {
        let factors = scaling_factors4(base_factor, scaling, block_idx * 4);
        store_i32x4(dst, dequant4_scaled_sse41(src, &factors, params));
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_scaled_into_sse41_16_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    let count = n * n;
    let base_factor = params.factor / 16;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    for (block_idx, (src, dst)) in levels.iter().zip(out.iter_mut()).enumerate() {
        let (src4, _) = src.as_chunks::<4>();
        let idx = block_idx * 8;
        let factors_lo = scaling_factors4(base_factor, scaling, idx);
        let factors_hi = scaling_factors4(base_factor, scaling, idx + 4);
        let lo = dequant4_scaled_sse41(&src4[0], &factors_lo, params);
        let hi = dequant4_scaled_sse41(&src4[1], &factors_hi, params);
        store_i16x8(dst, _mm_packs_epi32(lo, hi));
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_transform_skip_scaled_into_sse41_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    let base_factor = params.dequant.factor / 16;
    let (levels, _) = levels[..16].as_chunks::<4>();
    let (out, _) = out[..16].as_chunks_mut::<4>();

    for (block_idx, (src, dst)) in levels.iter().zip(out.iter_mut()).enumerate() {
        let factors = scaling_factors4(base_factor, scaling, block_idx * 4);
        store_i32x4(dst, transform_skip4_scaled_sse41(src, &factors, params));
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_transform_skip_scaled_into_sse41_16_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    let base_factor = params.dequant.factor / 16;
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<8>();

    for (block_idx, (src, dst)) in levels.iter().zip(out.iter_mut()).enumerate() {
        let (src4, _) = src.as_chunks::<4>();
        let idx = block_idx * 8;
        let factors_lo = scaling_factors4(base_factor, scaling, idx);
        let factors_hi = scaling_factors4(base_factor, scaling, idx + 4);
        let lo = transform_skip4_scaled_sse41(&src4[0], &factors_lo, params);
        let hi = transform_skip4_scaled_sse41(&src4[1], &factors_hi, params);
        store_i16x8(dst, _mm_packs_epi32(lo, hi));
    }
}

pub(crate) fn dequantize_into_sse41(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i32],
) {
    if !supported_n(n) || !mullo_safe(levels, n, params) {
        dequantize_into_scalar_i32(levels, n, params, out);
        return;
    }
    unsafe { dequantize_into_sse41_impl(levels, n, params, out) }
}

pub(crate) fn dequantize_into_sse41_16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i16],
) {
    if !supported_n(n) || !mullo_safe(levels, n, params) {
        dequantize_into_scalar_i16(levels, n, params, out);
        return;
    }
    unsafe { dequantize_into_sse41_16_impl(levels, n, params, out) }
}

pub(crate) fn dequantize_transform_skip_into_sse41(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i32],
) {
    if n != 4 || !mullo_safe(levels, n, params.dequant) {
        dequantize_transform_skip_into_scalar_i32(levels, n, params, out);
        return;
    }
    unsafe { dequantize_transform_skip_into_sse41_impl(levels, n, params, out) }
}

pub(crate) fn dequantize_transform_skip_into_sse41_16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i16],
) {
    if n != 4 || !mullo_safe(levels, n, params.dequant) {
        dequantize_transform_skip_into_scalar_i16(levels, n, params, out);
        return;
    }
    unsafe { dequantize_transform_skip_into_sse41_16_impl(levels, n, params, out) }
}

pub(crate) fn dequantize_scaled_into_sse41(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    if !supported_n(n) || !scaled_mullo_safe(levels, n, params, scaling) {
        dequantize_scaled_into_scalar_i32(levels, n, params, scaling, out);
        return;
    }
    unsafe { dequantize_scaled_into_sse41_impl(levels, n, params, scaling, out) }
}

pub(crate) fn dequantize_scaled_into_sse41_16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    if !supported_n(n) || !scaled_mullo_safe(levels, n, params, scaling) {
        dequantize_scaled_into_scalar_i16(levels, n, params, scaling, out);
        return;
    }
    unsafe { dequantize_scaled_into_sse41_16_impl(levels, n, params, scaling, out) }
}

pub(crate) fn dequantize_transform_skip_scaled_into_sse41(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    if n != 4 || !scaled_mullo_safe(levels, n, params.dequant, scaling) {
        dequantize_transform_skip_scaled_into_scalar_i32(levels, n, params, scaling, out);
        return;
    }
    unsafe { dequantize_transform_skip_scaled_into_sse41_impl(levels, n, params, scaling, out) }
}

pub(crate) fn dequantize_transform_skip_scaled_into_sse41_16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    if n != 4 || !scaled_mullo_safe(levels, n, params.dequant, scaling) {
        dequantize_transform_skip_scaled_into_scalar_i16(levels, n, params, scaling, out);
        return;
    }
    unsafe { dequantize_transform_skip_scaled_into_sse41_16_impl(levels, n, params, scaling, out) }
}
