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
fn dequant4_sse41_const(
    levels: &[i32; 4],
    factor: __m128i,
    add: __m128i,
    shift: __m128i,
    clip_lo: __m128i,
    clip_hi: __m128i,
) -> __m128i {
    let v = load_i32x4(levels);
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
    let v = load_i32x4(levels);
    let f = load_i32x4(factors);
    let v = _mm_mullo_epi32(v, f);
    let v = _mm_add_epi32(v, add);
    clip_i16_s32x4_with(sra_epi32_count(v, shift), clip_lo, clip_hi)
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

    let v0;
    let v1;
    let v2;
    let v3;
    if params.tr_shift >= 0 {
        v0 = sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift);
        v1 = sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift);
        v2 = sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift);
        v3 = sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift);
    } else {
        v0 = shl_epi32_count(d0, tr_shift);
        v1 = shl_epi32_count(d1, tr_shift);
        v2 = shl_epi32_count(d2, tr_shift);
        v3 = shl_epi32_count(d3, tr_shift);
    }

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

    let v0;
    let v1;
    let v2;
    let v3;
    if params.tr_shift >= 0 {
        v0 = sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift);
        v1 = sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift);
        v2 = sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift);
        v3 = sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift);
    } else {
        v0 = shl_epi32_count(d0, tr_shift);
        v1 = shl_epi32_count(d1, tr_shift);
        v2 = shl_epi32_count(d2, tr_shift);
        v3 = shl_epi32_count(d3, tr_shift);
    }

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

    let v0;
    let v1;
    let v2;
    let v3;
    if params.tr_shift >= 0 {
        v0 = sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift);
        v1 = sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift);
        v2 = sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift);
        v3 = sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift);
    } else {
        v0 = shl_epi32_count(d0, tr_shift);
        v1 = shl_epi32_count(d1, tr_shift);
        v2 = shl_epi32_count(d2, tr_shift);
        v3 = shl_epi32_count(d3, tr_shift);
    }

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

    let v0;
    let v1;
    let v2;
    let v3;
    if params.tr_shift >= 0 {
        v0 = sra_epi32_count(_mm_add_epi32(d0, tr_add), tr_shift);
        v1 = sra_epi32_count(_mm_add_epi32(d1, tr_add), tr_shift);
        v2 = sra_epi32_count(_mm_add_epi32(d2, tr_add), tr_shift);
        v3 = sra_epi32_count(_mm_add_epi32(d3, tr_add), tr_shift);
    } else {
        v0 = shl_epi32_count(d0, tr_shift);
        v1 = shl_epi32_count(d1, tr_shift);
        v2 = shl_epi32_count(d2, tr_shift);
        v3 = shl_epi32_count(d3, tr_shift);
    }

    let lo = clip_i16_s32x4_with(v0, clip_lo, clip_hi);
    let hi = clip_i16_s32x4_with(v1, clip_lo, clip_hi);
    store_i16x8(&mut out[0], _mm_packs_epi32(lo, hi));
    let lo = clip_i16_s32x4_with(v2, clip_lo, clip_hi);
    let hi = clip_i16_s32x4_with(v3, clip_lo, clip_hi);
    store_i16x8(&mut out[1], _mm_packs_epi32(lo, hi));
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
