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
fn mullo_safe_from_max(max_abs_level: i32, params: DequantParams) -> bool {
    if params.factor <= 0 || params.factor > i32::MAX as i64 || params.add > i32::MAX as i64 {
        return false;
    }
    let limit = ((i32::MAX as i64 - params.add) / params.factor).max(0);
    params.clipped_max_abs(max_abs_level) <= limit
}

#[inline]
fn scaled_mullo_safe_from_max(max_abs_level: i32, params: DequantParams, max_coeff: i64) -> bool {
    let base_factor = params.factor / 16;
    if base_factor <= 0 || params.add > i32::MAX as i64 {
        return false;
    }
    let max_factor = base_factor * max_coeff;
    if max_factor <= 0 || max_factor > i32::MAX as i64 {
        return false;
    }
    let limit = ((i32::MAX as i64 - params.add) / max_factor).max(0);
    params.clipped_max_abs(max_abs_level) <= limit
}

#[inline]
fn sse_i64_factor_ok(params: DequantParams) -> bool {
    params.factor > 0 && params.factor <= i32::MAX as i64 && params.add <= i32::MAX as i64
}

#[inline]
fn sse_scaled_i64_factor_ok(params: DequantParams, max_coeff: i64) -> bool {
    let base_factor = params.factor / 16;
    base_factor > 0 && base_factor * max_coeff <= i32::MAX as i64 && params.add <= i32::MAX as i64
}

#[inline]
fn scaling_factors4(base_factor: i64, scaling: ScalingMatrix<'_>, idx: usize) -> [i32; 4] {
    std::array::from_fn(|lane| (base_factor * scaling.coeff(idx + lane)) as i32)
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
fn sra_epi32_count(v: __m128i, shift: __m128i) -> __m128i {
    _mm_sra_epi32(v, shift)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn shl_epi32_count(v: __m128i, shift: __m128i) -> __m128i {
    _mm_sll_epi32(v, shift)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn clip_i16_s32x4_with(v: __m128i, lo: __m128i, hi: __m128i) -> __m128i {
    _mm_max_epi32(_mm_min_epi32(v, hi), lo)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_dequant_levels4(src: &[i32; 4], clip_lo: __m128i, clip_hi: __m128i) -> __m128i {
    // HEVC clips the coefficient level before inverse quantisation. This is
    // observable with scaling-list coefficients below the neutral value 16.
    clip_i16_s32x4_with(load_i32x4(src), clip_lo, clip_hi)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn dequant4_sse41_const(
    levels: &[i32; 4],
    factor: __m128i,
    add: __m128i,
    shift: __m128i,
    clip_lo: __m128i,
    clip_hi: __m128i,
) -> __m128i {
    let v = load_dequant_levels4(levels, clip_lo, clip_hi);
    let v = _mm_mullo_epi32(v, factor);
    let v = _mm_add_epi32(v, add);
    clip_i16_s32x4_with(sra_epi32_count(v, shift), clip_lo, clip_hi)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn dequant4_scaled_sse41_const(
    levels: &[i32; 4],
    factors: &[i32; 4],
    add: __m128i,
    shift: __m128i,
    clip_lo: __m128i,
    clip_hi: __m128i,
) -> __m128i {
    let v = load_dequant_levels4(levels, clip_lo, clip_hi);
    let f = load_i32x4(factors);
    let v = _mm_mullo_epi32(v, f);
    let v = _mm_add_epi32(v, add);
    clip_i16_s32x4_with(sra_epi32_count(v, shift), clip_lo, clip_hi)
}

#[inline]
fn floor_div_i64(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    let q = n / d;
    let r = n % d;
    if r < 0 { q - 1 } else { q }
}

#[inline]
fn ceil_div_i64(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    -floor_div_i64(-n, d)
}

#[inline]
fn dequant_pos_cut_minus_one(factor: i64, add: i64, shift: i32) -> i32 {
    if factor <= 0 {
        return i32::MAX;
    }
    let threshold = (32767i64 << shift) - add;
    (ceil_div_i64(threshold, factor) - 1).clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

#[inline]
fn dequant_neg_keep_min(factor: i64, add: i64, shift: i32) -> i32 {
    if factor <= 0 {
        return i32::MIN;
    }
    let threshold = ((-32767i64) << shift) - 1 - add;
    (floor_div_i64(threshold, factor) + 1).clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

#[inline]
fn dequant_sat_bounds(factor: i64, add: i64, shift: i32) -> (i32, i32) {
    (
        dequant_pos_cut_minus_one(factor, add, shift),
        dequant_neg_keep_min(factor, add, shift),
    )
}

#[inline]
fn dequant_sat_bounds4(factors: &[i32; 4], add: i64, shift: i32) -> ([i32; 4], [i32; 4]) {
    let mut pos_cut_minus_one = [0i32; 4];
    let mut neg_keep_min = [0i32; 4];
    for lane in 0..4 {
        let factor = factors[lane] as i64;
        pos_cut_minus_one[lane] = dequant_pos_cut_minus_one(factor, add, shift);
        neg_keep_min[lane] = dequant_neg_keep_min(factor, add, shift);
    }
    (pos_cut_minus_one, neg_keep_min)
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
fn dequant4_sse41_saturating_const(
    levels: &[i32; 4],
    factor: __m128i,
    add: __m128i,
    shift: __m128i,
    pos_cut_minus_one: __m128i,
    neg_keep_min: __m128i,
    clip_lo: __m128i,
    clip_hi: __m128i,
) -> __m128i {
    let v = load_dequant_levels4(levels, clip_lo, clip_hi);
    let over_hi = _mm_cmpgt_epi32(v, pos_cut_minus_one);
    let over_lo = _mm_cmpgt_epi32(neg_keep_min, v);
    let safe = _mm_max_epi32(_mm_min_epi32(v, pos_cut_minus_one), neg_keep_min);
    let deq = _mm_mullo_epi32(safe, factor);
    let deq = _mm_add_epi32(deq, add);
    let deq = sra_epi32_count(deq, shift);
    let deq = _mm_blendv_epi8(deq, clip_hi, over_hi);
    _mm_blendv_epi8(deq, clip_lo, over_lo)
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
fn dequant4_scaled_sse41_saturating_const(
    levels: &[i32; 4],
    factors: &[i32; 4],
    pos_cut_minus_one: &[i32; 4],
    neg_keep_min: &[i32; 4],
    add: __m128i,
    shift: __m128i,
    clip_lo: __m128i,
    clip_hi: __m128i,
) -> __m128i {
    let v = load_dequant_levels4(levels, clip_lo, clip_hi);
    let f = load_i32x4(factors);
    let pos_cut_minus_one = load_i32x4(pos_cut_minus_one);
    let neg_keep_min = load_i32x4(neg_keep_min);
    let over_hi = _mm_cmpgt_epi32(v, pos_cut_minus_one);
    let over_lo = _mm_cmpgt_epi32(neg_keep_min, v);
    let safe = _mm_max_epi32(_mm_min_epi32(v, pos_cut_minus_one), neg_keep_min);
    let deq = _mm_mullo_epi32(safe, f);
    let deq = _mm_add_epi32(deq, add);
    let deq = sra_epi32_count(deq, shift);
    let deq = _mm_blendv_epi8(deq, clip_hi, over_hi);
    _mm_blendv_epi8(deq, clip_lo, over_lo)
}

#[target_feature(enable = "sse4.1")]
fn dequantize_into_sse41_impl(levels: &[i32], n: usize, params: DequantParams, out: &mut [i32]) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<4>();
    let (out, _) = out[..count].as_chunks_mut::<4>();

    let factor = _mm_set1_epi32(params.factor as i32);
    let add = _mm_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let v0 = dequant4_sse41_const(&src[0], factor, add, shift, clip_lo, clip_hi);
        let v1 = dequant4_sse41_const(&src[1], factor, add, shift, clip_lo, clip_hi);
        let v2 = dequant4_sse41_const(&src[2], factor, add, shift, clip_lo, clip_hi);
        let v3 = dequant4_sse41_const(&src[3], factor, add, shift, clip_lo, clip_hi);
        store_i32x4(&mut dst[0], v0);
        store_i32x4(&mut dst[1], v1);
        store_i32x4(&mut dst[2], v2);
        store_i32x4(&mut dst[3], v3);
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        store_i32x4(
            dst,
            dequant4_sse41_const(src, factor, add, shift, clip_lo, clip_hi),
        );
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_into_sse41_16_impl(levels: &[i32], n: usize, params: DequantParams, out: &mut [i16]) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    let factor = _mm_set1_epi32(params.factor as i32);
    let add = _mm_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<2>();
    let (out_groups, out_tail) = out.as_chunks_mut::<2>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let (src0, _) = src[0].as_chunks::<4>();
        let (src1, _) = src[1].as_chunks::<4>();
        let lo0 = dequant4_sse41_const(&src0[0], factor, add, shift, clip_lo, clip_hi);
        let hi0 = dequant4_sse41_const(&src0[1], factor, add, shift, clip_lo, clip_hi);
        let lo1 = dequant4_sse41_const(&src1[0], factor, add, shift, clip_lo, clip_hi);
        let hi1 = dequant4_sse41_const(&src1[1], factor, add, shift, clip_lo, clip_hi);
        store_i16x8(&mut dst[0], _mm_packs_epi32(lo0, hi0));
        store_i16x8(&mut dst[1], _mm_packs_epi32(lo1, hi1));
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        let (src4, _) = src.as_chunks::<4>();
        let lo = dequant4_sse41_const(&src4[0], factor, add, shift, clip_lo, clip_hi);
        let hi = dequant4_sse41_const(&src4[1], factor, add, shift, clip_lo, clip_hi);
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

    let factor = _mm_set1_epi32(params.dequant.factor as i32);
    let add = _mm_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let tr_add = _mm_set1_epi32(params.tr_add);
    let tr_shift = _mm_cvtsi32_si128(params.tr_shift.abs());

    let d0 = dequant4_sse41_const(&levels[0], factor, add, shift, clip_lo, clip_hi);
    let d1 = dequant4_sse41_const(&levels[1], factor, add, shift, clip_lo, clip_hi);
    let d2 = dequant4_sse41_const(&levels[2], factor, add, shift, clip_lo, clip_hi);
    let d3 = dequant4_sse41_const(&levels[3], factor, add, shift, clip_lo, clip_hi);

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift),
        )
    } else {
        (
            shl_epi32_count(d0, tr_shift),
            shl_epi32_count(d1, tr_shift),
            shl_epi32_count(d2, tr_shift),
            shl_epi32_count(d3, tr_shift),
        )
    };

    store_i32x4(&mut out[0], clip_i16_s32x4_with(v0, clip_lo, clip_hi));
    store_i32x4(&mut out[1], clip_i16_s32x4_with(v1, clip_lo, clip_hi));
    store_i32x4(&mut out[2], clip_i16_s32x4_with(v2, clip_lo, clip_hi));
    store_i32x4(&mut out[3], clip_i16_s32x4_with(v3, clip_lo, clip_hi));
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

    let factor = _mm_set1_epi32(params.dequant.factor as i32);
    let add = _mm_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let tr_add = _mm_set1_epi32(params.tr_add);
    let tr_shift = _mm_cvtsi32_si128(params.tr_shift.abs());

    let (src0, _) = levels[0].as_chunks::<4>();
    let (src1, _) = levels[1].as_chunks::<4>();
    let d0 = dequant4_sse41_const(&src0[0], factor, add, shift, clip_lo, clip_hi);
    let d1 = dequant4_sse41_const(&src0[1], factor, add, shift, clip_lo, clip_hi);
    let d2 = dequant4_sse41_const(&src1[0], factor, add, shift, clip_lo, clip_hi);
    let d3 = dequant4_sse41_const(&src1[1], factor, add, shift, clip_lo, clip_hi);

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift),
        )
    } else {
        (
            shl_epi32_count(d0, tr_shift),
            shl_epi32_count(d1, tr_shift),
            shl_epi32_count(d2, tr_shift),
            shl_epi32_count(d3, tr_shift),
        )
    };

    let lo = clip_i16_s32x4_with(v0, clip_lo, clip_hi);
    let hi = clip_i16_s32x4_with(v1, clip_lo, clip_hi);
    store_i16x8(&mut out[0], _mm_packs_epi32(lo, hi));
    let lo = clip_i16_s32x4_with(v2, clip_lo, clip_hi);
    let hi = clip_i16_s32x4_with(v3, clip_lo, clip_hi);
    store_i16x8(&mut out[1], _mm_packs_epi32(lo, hi));
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

    let add = _mm_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (group_idx, (src, dst)) in level_groups.iter().zip(out_groups.iter_mut()).enumerate() {
        let idx = group_idx * 16;
        let f0 = scaling_factors4(base_factor, scaling, idx);
        let f1 = scaling_factors4(base_factor, scaling, idx + 4);
        let f2 = scaling_factors4(base_factor, scaling, idx + 8);
        let f3 = scaling_factors4(base_factor, scaling, idx + 12);
        let v0 = dequant4_scaled_sse41_const(&src[0], &f0, add, shift, clip_lo, clip_hi);
        let v1 = dequant4_scaled_sse41_const(&src[1], &f1, add, shift, clip_lo, clip_hi);
        let v2 = dequant4_scaled_sse41_const(&src[2], &f2, add, shift, clip_lo, clip_hi);
        let v3 = dequant4_scaled_sse41_const(&src[3], &f3, add, shift, clip_lo, clip_hi);
        store_i32x4(&mut dst[0], v0);
        store_i32x4(&mut dst[1], v1);
        store_i32x4(&mut dst[2], v2);
        store_i32x4(&mut dst[3], v3);
    }

    let tail_base = level_groups.len() * 16;
    for (block_idx, (src, dst)) in level_tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let factors = scaling_factors4(base_factor, scaling, tail_base + block_idx * 4);
        store_i32x4(
            dst,
            dequant4_scaled_sse41_const(src, &factors, add, shift, clip_lo, clip_hi),
        );
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

    let add = _mm_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<2>();
    let (out_groups, out_tail) = out.as_chunks_mut::<2>();

    for (group_idx, (src, dst)) in level_groups.iter().zip(out_groups.iter_mut()).enumerate() {
        let idx = group_idx * 16;
        let (src0, _) = src[0].as_chunks::<4>();
        let (src1, _) = src[1].as_chunks::<4>();
        let f0 = scaling_factors4(base_factor, scaling, idx);
        let f1 = scaling_factors4(base_factor, scaling, idx + 4);
        let f2 = scaling_factors4(base_factor, scaling, idx + 8);
        let f3 = scaling_factors4(base_factor, scaling, idx + 12);
        let lo0 = dequant4_scaled_sse41_const(&src0[0], &f0, add, shift, clip_lo, clip_hi);
        let hi0 = dequant4_scaled_sse41_const(&src0[1], &f1, add, shift, clip_lo, clip_hi);
        let lo1 = dequant4_scaled_sse41_const(&src1[0], &f2, add, shift, clip_lo, clip_hi);
        let hi1 = dequant4_scaled_sse41_const(&src1[1], &f3, add, shift, clip_lo, clip_hi);
        store_i16x8(&mut dst[0], _mm_packs_epi32(lo0, hi0));
        store_i16x8(&mut dst[1], _mm_packs_epi32(lo1, hi1));
    }

    let tail_base = level_groups.len() * 16;
    for (block_idx, (src, dst)) in level_tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let (src4, _) = src.as_chunks::<4>();
        let idx = tail_base + block_idx * 8;
        let factors_lo = scaling_factors4(base_factor, scaling, idx);
        let factors_hi = scaling_factors4(base_factor, scaling, idx + 4);
        let lo = dequant4_scaled_sse41_const(&src4[0], &factors_lo, add, shift, clip_lo, clip_hi);
        let hi = dequant4_scaled_sse41_const(&src4[1], &factors_hi, add, shift, clip_lo, clip_hi);
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

    let add = _mm_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let tr_add = _mm_set1_epi32(params.tr_add);
    let tr_shift = _mm_cvtsi32_si128(params.tr_shift.abs());

    let f0 = scaling_factors4(base_factor, scaling, 0);
    let f1 = scaling_factors4(base_factor, scaling, 4);
    let f2 = scaling_factors4(base_factor, scaling, 8);
    let f3 = scaling_factors4(base_factor, scaling, 12);
    let d0 = dequant4_scaled_sse41_const(&levels[0], &f0, add, shift, clip_lo, clip_hi);
    let d1 = dequant4_scaled_sse41_const(&levels[1], &f1, add, shift, clip_lo, clip_hi);
    let d2 = dequant4_scaled_sse41_const(&levels[2], &f2, add, shift, clip_lo, clip_hi);
    let d3 = dequant4_scaled_sse41_const(&levels[3], &f3, add, shift, clip_lo, clip_hi);

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift),
        )
    } else {
        (
            shl_epi32_count(d0, tr_shift),
            shl_epi32_count(d1, tr_shift),
            shl_epi32_count(d2, tr_shift),
            shl_epi32_count(d3, tr_shift),
        )
    };

    store_i32x4(&mut out[0], clip_i16_s32x4_with(v0, clip_lo, clip_hi));
    store_i32x4(&mut out[1], clip_i16_s32x4_with(v1, clip_lo, clip_hi));
    store_i32x4(&mut out[2], clip_i16_s32x4_with(v2, clip_lo, clip_hi));
    store_i32x4(&mut out[3], clip_i16_s32x4_with(v3, clip_lo, clip_hi));
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

    let add = _mm_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let tr_add = _mm_set1_epi32(params.tr_add);
    let tr_shift = _mm_cvtsi32_si128(params.tr_shift.abs());

    let (src0, _) = levels[0].as_chunks::<4>();
    let (src1, _) = levels[1].as_chunks::<4>();
    let f0 = scaling_factors4(base_factor, scaling, 0);
    let f1 = scaling_factors4(base_factor, scaling, 4);
    let f2 = scaling_factors4(base_factor, scaling, 8);
    let f3 = scaling_factors4(base_factor, scaling, 12);
    let d0 = dequant4_scaled_sse41_const(&src0[0], &f0, add, shift, clip_lo, clip_hi);
    let d1 = dequant4_scaled_sse41_const(&src0[1], &f1, add, shift, clip_lo, clip_hi);
    let d2 = dequant4_scaled_sse41_const(&src1[0], &f2, add, shift, clip_lo, clip_hi);
    let d3 = dequant4_scaled_sse41_const(&src1[1], &f3, add, shift, clip_lo, clip_hi);

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift),
        )
    } else {
        (
            shl_epi32_count(d0, tr_shift),
            shl_epi32_count(d1, tr_shift),
            shl_epi32_count(d2, tr_shift),
            shl_epi32_count(d3, tr_shift),
        )
    };

    let lo = clip_i16_s32x4_with(v0, clip_lo, clip_hi);
    let hi = clip_i16_s32x4_with(v1, clip_lo, clip_hi);
    store_i16x8(&mut out[0], _mm_packs_epi32(lo, hi));
    let lo = clip_i16_s32x4_with(v2, clip_lo, clip_hi);
    let hi = clip_i16_s32x4_with(v3, clip_lo, clip_hi);
    store_i16x8(&mut out[1], _mm_packs_epi32(lo, hi));
}

#[target_feature(enable = "sse4.1")]
fn dequantize_into_sse41_i64_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i32],
) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<4>();
    let (out, _) = out[..count].as_chunks_mut::<4>();

    let factor = _mm_set1_epi32(params.factor as i32);
    let add = _mm_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let (pos_cut_minus_one, neg_keep_min) =
        dequant_sat_bounds(params.factor, params.add, params.shift);
    let pos_cut_minus_one = _mm_set1_epi32(pos_cut_minus_one);
    let neg_keep_min = _mm_set1_epi32(neg_keep_min);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let v0 = dequant4_sse41_saturating_const(
            &src[0],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let v1 = dequant4_sse41_saturating_const(
            &src[1],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let v2 = dequant4_sse41_saturating_const(
            &src[2],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let v3 = dequant4_sse41_saturating_const(
            &src[3],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        store_i32x4(&mut dst[0], v0);
        store_i32x4(&mut dst[1], v1);
        store_i32x4(&mut dst[2], v2);
        store_i32x4(&mut dst[3], v3);
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        store_i32x4(
            dst,
            dequant4_sse41_saturating_const(
                src,
                factor,
                add,
                shift,
                pos_cut_minus_one,
                neg_keep_min,
                clip_lo,
                clip_hi,
            ),
        );
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_into_sse41_16_i64_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i16],
) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    let factor = _mm_set1_epi32(params.factor as i32);
    let add = _mm_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let (pos_cut_minus_one, neg_keep_min) =
        dequant_sat_bounds(params.factor, params.add, params.shift);
    let pos_cut_minus_one = _mm_set1_epi32(pos_cut_minus_one);
    let neg_keep_min = _mm_set1_epi32(neg_keep_min);

    let (level_groups, level_tail) = levels.as_chunks::<2>();
    let (out_groups, out_tail) = out.as_chunks_mut::<2>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let (src0, _) = src[0].as_chunks::<4>();
        let (src1, _) = src[1].as_chunks::<4>();
        let lo0 = dequant4_sse41_saturating_const(
            &src0[0],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let hi0 = dequant4_sse41_saturating_const(
            &src0[1],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let lo1 = dequant4_sse41_saturating_const(
            &src1[0],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let hi1 = dequant4_sse41_saturating_const(
            &src1[1],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        store_i16x8(&mut dst[0], _mm_packs_epi32(lo0, hi0));
        store_i16x8(&mut dst[1], _mm_packs_epi32(lo1, hi1));
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        let (src4, _) = src.as_chunks::<4>();
        let lo = dequant4_sse41_saturating_const(
            &src4[0],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let hi = dequant4_sse41_saturating_const(
            &src4[1],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        store_i16x8(dst, _mm_packs_epi32(lo, hi));
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_transform_skip_into_sse41_i64_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<4>();
    let (out, _) = out[..16].as_chunks_mut::<4>();

    let factor = _mm_set1_epi32(params.dequant.factor as i32);
    let add = _mm_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let (pos_cut_minus_one, neg_keep_min) = dequant_sat_bounds(
        params.dequant.factor,
        params.dequant.add,
        params.dequant.shift,
    );
    let pos_cut_minus_one = _mm_set1_epi32(pos_cut_minus_one);
    let neg_keep_min = _mm_set1_epi32(neg_keep_min);
    let tr_add = _mm_set1_epi32(params.tr_add);
    let tr_shift = _mm_cvtsi32_si128(params.tr_shift.abs());

    let d0 = dequant4_sse41_saturating_const(
        &levels[0],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    let d1 = dequant4_sse41_saturating_const(
        &levels[1],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    let d2 = dequant4_sse41_saturating_const(
        &levels[2],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    let d3 = dequant4_sse41_saturating_const(
        &levels[3],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift),
        )
    } else {
        (
            shl_epi32_count(d0, tr_shift),
            shl_epi32_count(d1, tr_shift),
            shl_epi32_count(d2, tr_shift),
            shl_epi32_count(d3, tr_shift),
        )
    };

    store_i32x4(&mut out[0], clip_i16_s32x4_with(v0, clip_lo, clip_hi));
    store_i32x4(&mut out[1], clip_i16_s32x4_with(v1, clip_lo, clip_hi));
    store_i32x4(&mut out[2], clip_i16_s32x4_with(v2, clip_lo, clip_hi));
    store_i32x4(&mut out[3], clip_i16_s32x4_with(v3, clip_lo, clip_hi));
}

#[target_feature(enable = "sse4.1")]
fn dequantize_transform_skip_into_sse41_16_i64_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<8>();

    let factor = _mm_set1_epi32(params.dequant.factor as i32);
    let add = _mm_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let (pos_cut_minus_one, neg_keep_min) = dequant_sat_bounds(
        params.dequant.factor,
        params.dequant.add,
        params.dequant.shift,
    );
    let pos_cut_minus_one = _mm_set1_epi32(pos_cut_minus_one);
    let neg_keep_min = _mm_set1_epi32(neg_keep_min);
    let tr_add = _mm_set1_epi32(params.tr_add);
    let tr_shift = _mm_cvtsi32_si128(params.tr_shift.abs());

    let (src0, _) = levels[0].as_chunks::<4>();
    let (src1, _) = levels[1].as_chunks::<4>();
    let d0 = dequant4_sse41_saturating_const(
        &src0[0],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    let d1 = dequant4_sse41_saturating_const(
        &src0[1],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    let d2 = dequant4_sse41_saturating_const(
        &src1[0],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    let d3 = dequant4_sse41_saturating_const(
        &src1[1],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift),
        )
    } else {
        (
            shl_epi32_count(d0, tr_shift),
            shl_epi32_count(d1, tr_shift),
            shl_epi32_count(d2, tr_shift),
            shl_epi32_count(d3, tr_shift),
        )
    };

    let lo = clip_i16_s32x4_with(v0, clip_lo, clip_hi);
    let hi = clip_i16_s32x4_with(v1, clip_lo, clip_hi);
    store_i16x8(&mut out[0], _mm_packs_epi32(lo, hi));
    let lo = clip_i16_s32x4_with(v2, clip_lo, clip_hi);
    let hi = clip_i16_s32x4_with(v3, clip_lo, clip_hi);
    store_i16x8(&mut out[1], _mm_packs_epi32(lo, hi));
}

#[target_feature(enable = "sse4.1")]
fn dequantize_scaled_into_sse41_i64_impl(
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

    let add = _mm_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (group_idx, (src, dst)) in level_groups.iter().zip(out_groups.iter_mut()).enumerate() {
        let idx = group_idx * 16;
        let f0 = scaling_factors4(base_factor, scaling, idx);
        let f1 = scaling_factors4(base_factor, scaling, idx + 4);
        let f2 = scaling_factors4(base_factor, scaling, idx + 8);
        let f3 = scaling_factors4(base_factor, scaling, idx + 12);
        let (p0, n0) = dequant_sat_bounds4(&f0, params.add, params.shift);
        let (p1, n1) = dequant_sat_bounds4(&f1, params.add, params.shift);
        let (p2, n2) = dequant_sat_bounds4(&f2, params.add, params.shift);
        let (p3, n3) = dequant_sat_bounds4(&f3, params.add, params.shift);
        let v0 = dequant4_scaled_sse41_saturating_const(
            &src[0], &f0, &p0, &n0, add, shift, clip_lo, clip_hi,
        );
        let v1 = dequant4_scaled_sse41_saturating_const(
            &src[1], &f1, &p1, &n1, add, shift, clip_lo, clip_hi,
        );
        let v2 = dequant4_scaled_sse41_saturating_const(
            &src[2], &f2, &p2, &n2, add, shift, clip_lo, clip_hi,
        );
        let v3 = dequant4_scaled_sse41_saturating_const(
            &src[3], &f3, &p3, &n3, add, shift, clip_lo, clip_hi,
        );
        store_i32x4(&mut dst[0], v0);
        store_i32x4(&mut dst[1], v1);
        store_i32x4(&mut dst[2], v2);
        store_i32x4(&mut dst[3], v3);
    }

    let tail_base = level_groups.len() * 16;
    for (block_idx, (src, dst)) in level_tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let factors = scaling_factors4(base_factor, scaling, tail_base + block_idx * 4);
        let (pos_cut_minus_one, neg_keep_min) =
            dequant_sat_bounds4(&factors, params.add, params.shift);
        store_i32x4(
            dst,
            dequant4_scaled_sse41_saturating_const(
                src,
                &factors,
                &pos_cut_minus_one,
                &neg_keep_min,
                add,
                shift,
                clip_lo,
                clip_hi,
            ),
        );
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_scaled_into_sse41_16_i64_impl(
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

    let add = _mm_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<2>();
    let (out_groups, out_tail) = out.as_chunks_mut::<2>();

    for (group_idx, (src, dst)) in level_groups.iter().zip(out_groups.iter_mut()).enumerate() {
        let idx = group_idx * 16;
        let (src0, _) = src[0].as_chunks::<4>();
        let (src1, _) = src[1].as_chunks::<4>();
        let f0 = scaling_factors4(base_factor, scaling, idx);
        let f1 = scaling_factors4(base_factor, scaling, idx + 4);
        let f2 = scaling_factors4(base_factor, scaling, idx + 8);
        let f3 = scaling_factors4(base_factor, scaling, idx + 12);
        let (p0, n0) = dequant_sat_bounds4(&f0, params.add, params.shift);
        let (p1, n1) = dequant_sat_bounds4(&f1, params.add, params.shift);
        let (p2, n2) = dequant_sat_bounds4(&f2, params.add, params.shift);
        let (p3, n3) = dequant_sat_bounds4(&f3, params.add, params.shift);
        let lo0 = dequant4_scaled_sse41_saturating_const(
            &src0[0], &f0, &p0, &n0, add, shift, clip_lo, clip_hi,
        );
        let hi0 = dequant4_scaled_sse41_saturating_const(
            &src0[1], &f1, &p1, &n1, add, shift, clip_lo, clip_hi,
        );
        let lo1 = dequant4_scaled_sse41_saturating_const(
            &src1[0], &f2, &p2, &n2, add, shift, clip_lo, clip_hi,
        );
        let hi1 = dequant4_scaled_sse41_saturating_const(
            &src1[1], &f3, &p3, &n3, add, shift, clip_lo, clip_hi,
        );
        store_i16x8(&mut dst[0], _mm_packs_epi32(lo0, hi0));
        store_i16x8(&mut dst[1], _mm_packs_epi32(lo1, hi1));
    }

    let tail_base = level_groups.len() * 16;
    for (block_idx, (src, dst)) in level_tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let (src4, _) = src.as_chunks::<4>();
        let idx = tail_base + block_idx * 8;
        let factors_lo = scaling_factors4(base_factor, scaling, idx);
        let factors_hi = scaling_factors4(base_factor, scaling, idx + 4);
        let (pos_lo, neg_lo) = dequant_sat_bounds4(&factors_lo, params.add, params.shift);
        let (pos_hi, neg_hi) = dequant_sat_bounds4(&factors_hi, params.add, params.shift);
        let lo = dequant4_scaled_sse41_saturating_const(
            &src4[0],
            &factors_lo,
            &pos_lo,
            &neg_lo,
            add,
            shift,
            clip_lo,
            clip_hi,
        );
        let hi = dequant4_scaled_sse41_saturating_const(
            &src4[1],
            &factors_hi,
            &pos_hi,
            &neg_hi,
            add,
            shift,
            clip_lo,
            clip_hi,
        );
        store_i16x8(dst, _mm_packs_epi32(lo, hi));
    }
}

#[target_feature(enable = "sse4.1")]
fn dequantize_transform_skip_scaled_into_sse41_i64_impl(
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
    let add = _mm_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let tr_add = _mm_set1_epi32(params.tr_add);
    let tr_shift = _mm_cvtsi32_si128(params.tr_shift.abs());

    let f0 = scaling_factors4(base_factor, scaling, 0);
    let f1 = scaling_factors4(base_factor, scaling, 4);
    let f2 = scaling_factors4(base_factor, scaling, 8);
    let f3 = scaling_factors4(base_factor, scaling, 12);
    let (p0, n0) = dequant_sat_bounds4(&f0, params.dequant.add, params.dequant.shift);
    let (p1, n1) = dequant_sat_bounds4(&f1, params.dequant.add, params.dequant.shift);
    let (p2, n2) = dequant_sat_bounds4(&f2, params.dequant.add, params.dequant.shift);
    let (p3, n3) = dequant_sat_bounds4(&f3, params.dequant.add, params.dequant.shift);
    let d0 = dequant4_scaled_sse41_saturating_const(
        &levels[0], &f0, &p0, &n0, add, shift, clip_lo, clip_hi,
    );
    let d1 = dequant4_scaled_sse41_saturating_const(
        &levels[1], &f1, &p1, &n1, add, shift, clip_lo, clip_hi,
    );
    let d2 = dequant4_scaled_sse41_saturating_const(
        &levels[2], &f2, &p2, &n2, add, shift, clip_lo, clip_hi,
    );
    let d3 = dequant4_scaled_sse41_saturating_const(
        &levels[3], &f3, &p3, &n3, add, shift, clip_lo, clip_hi,
    );

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift),
        )
    } else {
        (
            shl_epi32_count(d0, tr_shift),
            shl_epi32_count(d1, tr_shift),
            shl_epi32_count(d2, tr_shift),
            shl_epi32_count(d3, tr_shift),
        )
    };

    store_i32x4(&mut out[0], clip_i16_s32x4_with(v0, clip_lo, clip_hi));
    store_i32x4(&mut out[1], clip_i16_s32x4_with(v1, clip_lo, clip_hi));
    store_i32x4(&mut out[2], clip_i16_s32x4_with(v2, clip_lo, clip_hi));
    store_i32x4(&mut out[3], clip_i16_s32x4_with(v3, clip_lo, clip_hi));
}

#[target_feature(enable = "sse4.1")]
fn dequantize_transform_skip_scaled_into_sse41_16_i64_impl(
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
    let add = _mm_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm_set1_epi32(-32768);
    let clip_hi = _mm_set1_epi32(32767);
    let tr_add = _mm_set1_epi32(params.tr_add);
    let tr_shift = _mm_cvtsi32_si128(params.tr_shift.abs());

    let (src0, _) = levels[0].as_chunks::<4>();
    let (src1, _) = levels[1].as_chunks::<4>();
    let f0 = scaling_factors4(base_factor, scaling, 0);
    let f1 = scaling_factors4(base_factor, scaling, 4);
    let f2 = scaling_factors4(base_factor, scaling, 8);
    let f3 = scaling_factors4(base_factor, scaling, 12);
    let (p0, n0) = dequant_sat_bounds4(&f0, params.dequant.add, params.dequant.shift);
    let (p1, n1) = dequant_sat_bounds4(&f1, params.dequant.add, params.dequant.shift);
    let (p2, n2) = dequant_sat_bounds4(&f2, params.dequant.add, params.dequant.shift);
    let (p3, n3) = dequant_sat_bounds4(&f3, params.dequant.add, params.dequant.shift);
    let d0 = dequant4_scaled_sse41_saturating_const(
        &src0[0], &f0, &p0, &n0, add, shift, clip_lo, clip_hi,
    );
    let d1 = dequant4_scaled_sse41_saturating_const(
        &src0[1], &f1, &p1, &n1, add, shift, clip_lo, clip_hi,
    );
    let d2 = dequant4_scaled_sse41_saturating_const(
        &src1[0], &f2, &p2, &n2, add, shift, clip_lo, clip_hi,
    );
    let d3 = dequant4_scaled_sse41_saturating_const(
        &src1[1], &f3, &p3, &n3, add, shift, clip_lo, clip_hi,
    );

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift),
            sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift),
        )
    } else {
        (
            shl_epi32_count(d0, tr_shift),
            shl_epi32_count(d1, tr_shift),
            shl_epi32_count(d2, tr_shift),
            shl_epi32_count(d3, tr_shift),
        )
    };

    let lo0 = clip_i16_s32x4_with(v0, clip_lo, clip_hi);
    let hi0 = clip_i16_s32x4_with(v1, clip_lo, clip_hi);
    let lo1 = clip_i16_s32x4_with(v2, clip_lo, clip_hi);
    let hi1 = clip_i16_s32x4_with(v3, clip_lo, clip_hi);
    store_i16x8(&mut out[0], _mm_packs_epi32(lo0, hi0));
    store_i16x8(&mut out[1], _mm_packs_epi32(lo1, hi1));
}

pub(crate) fn dequantize_into_sse41(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    max_abs_level: i32,
    out: &mut [i32],
) {
    if !supported_n(n) || !params.simd_i16_range() {
        dequantize_into_scalar_i32(levels, n, params, max_abs_level, out);
        return;
    }
    if mullo_safe_from_max(max_abs_level, params) {
        unsafe { dequantize_into_sse41_impl(levels, n, params, out) }
    } else if sse_i64_factor_ok(params) {
        unsafe { dequantize_into_sse41_i64_impl(levels, n, params, out) }
    } else {
        dequantize_into_scalar_i32(levels, n, params, max_abs_level, out);
    }
}

pub(crate) fn dequantize_into_sse41_16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    max_abs_level: i32,
    out: &mut [i16],
) {
    if !supported_n(n) || !params.simd_i16_range() {
        dequantize_into_scalar_i16(levels, n, params, max_abs_level, out);
        return;
    }
    if mullo_safe_from_max(max_abs_level, params) {
        unsafe { dequantize_into_sse41_16_impl(levels, n, params, out) }
    } else if sse_i64_factor_ok(params) {
        unsafe { dequantize_into_sse41_16_i64_impl(levels, n, params, out) }
    } else {
        dequantize_into_scalar_i16(levels, n, params, max_abs_level, out);
    }
}

pub(crate) fn dequantize_transform_skip_into_sse41(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    max_abs_level: i32,
    out: &mut [i32],
) {
    if n != 4 || !params.dequant.simd_i16_range() {
        dequantize_transform_skip_into_scalar_i32(levels, n, params, max_abs_level, out);
        return;
    }
    if mullo_safe_from_max(max_abs_level, params.dequant) {
        unsafe { dequantize_transform_skip_into_sse41_impl(levels, n, params, out) }
    } else if sse_i64_factor_ok(params.dequant) {
        unsafe { dequantize_transform_skip_into_sse41_i64_impl(levels, n, params, out) }
    } else {
        dequantize_transform_skip_into_scalar_i32(levels, n, params, max_abs_level, out);
    }
}

pub(crate) fn dequantize_transform_skip_into_sse41_16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    max_abs_level: i32,
    out: &mut [i16],
) {
    if n != 4 || !params.dequant.simd_i16_range() {
        dequantize_transform_skip_into_scalar_i16(levels, n, params, max_abs_level, out);
        return;
    }
    if mullo_safe_from_max(max_abs_level, params.dequant) {
        unsafe { dequantize_transform_skip_into_sse41_16_impl(levels, n, params, out) }
    } else if sse_i64_factor_ok(params.dequant) {
        unsafe { dequantize_transform_skip_into_sse41_16_i64_impl(levels, n, params, out) }
    } else {
        dequantize_transform_skip_into_scalar_i16(levels, n, params, max_abs_level, out);
    }
}

pub(crate) fn dequantize_scaled_into_sse41(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    max_abs_level: i32,
    out: &mut [i32],
) {
    if !supported_n(n) || !params.simd_i16_range() {
        dequantize_scaled_into_scalar_i32(levels, n, params, scaling, max_abs_level, out);
        return;
    }
    let max_coeff = scaling.max_coeff();
    if scaled_mullo_safe_from_max(max_abs_level, params, max_coeff) {
        unsafe { dequantize_scaled_into_sse41_impl(levels, n, params, scaling, out) }
    } else if sse_scaled_i64_factor_ok(params, max_coeff) {
        unsafe { dequantize_scaled_into_sse41_i64_impl(levels, n, params, scaling, out) }
    } else {
        dequantize_scaled_into_scalar_i32(levels, n, params, scaling, max_abs_level, out);
    }
}

pub(crate) fn dequantize_scaled_into_sse41_16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    max_abs_level: i32,
    out: &mut [i16],
) {
    if !supported_n(n) || !params.simd_i16_range() {
        dequantize_scaled_into_scalar_i16(levels, n, params, scaling, max_abs_level, out);
        return;
    }
    let max_coeff = scaling.max_coeff();
    if scaled_mullo_safe_from_max(max_abs_level, params, max_coeff) {
        unsafe { dequantize_scaled_into_sse41_16_impl(levels, n, params, scaling, out) }
    } else if sse_scaled_i64_factor_ok(params, max_coeff) {
        unsafe { dequantize_scaled_into_sse41_16_i64_impl(levels, n, params, scaling, out) }
    } else {
        dequantize_scaled_into_scalar_i16(levels, n, params, scaling, max_abs_level, out);
    }
}

pub(crate) fn dequantize_transform_skip_scaled_into_sse41(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    max_abs_level: i32,
    out: &mut [i32],
) {
    if n != 4 || !params.dequant.simd_i16_range() {
        dequantize_transform_skip_scaled_into_scalar_i32(
            levels,
            n,
            params,
            scaling,
            max_abs_level,
            out,
        );
        return;
    }
    let max_coeff = scaling.max_coeff();
    if scaled_mullo_safe_from_max(max_abs_level, params.dequant, max_coeff) {
        unsafe { dequantize_transform_skip_scaled_into_sse41_impl(levels, n, params, scaling, out) }
    } else if sse_scaled_i64_factor_ok(params.dequant, max_coeff) {
        unsafe {
            dequantize_transform_skip_scaled_into_sse41_i64_impl(levels, n, params, scaling, out)
        }
    } else {
        dequantize_transform_skip_scaled_into_scalar_i32(
            levels,
            n,
            params,
            scaling,
            max_abs_level,
            out,
        );
    }
}

pub(crate) fn dequantize_transform_skip_scaled_into_sse41_16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    max_abs_level: i32,
    out: &mut [i16],
) {
    if n != 4 || !params.dequant.simd_i16_range() {
        dequantize_transform_skip_scaled_into_scalar_i16(
            levels,
            n,
            params,
            scaling,
            max_abs_level,
            out,
        );
        return;
    }
    let max_coeff = scaling.max_coeff();
    if scaled_mullo_safe_from_max(max_abs_level, params.dequant, max_coeff) {
        unsafe {
            dequantize_transform_skip_scaled_into_sse41_16_impl(levels, n, params, scaling, out)
        }
    } else if sse_scaled_i64_factor_ok(params.dequant, max_coeff) {
        unsafe {
            dequantize_transform_skip_scaled_into_sse41_16_i64_impl(levels, n, params, scaling, out)
        }
    } else {
        dequantize_transform_skip_scaled_into_scalar_i16(
            levels,
            n,
            params,
            scaling,
            max_abs_level,
            out,
        );
    }
}
