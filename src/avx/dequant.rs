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

use crate::transform::{DequantParams, ScalingMatrix, TransformSkipParams};

#[inline(always)]
fn supported_n(n: usize) -> bool {
    matches!(n, 4 | 8 | 16 | 32)
}

#[inline(always)]
fn mullo_safe_from_max(max_abs_level: i32, params: DequantParams) -> bool {
    if params.factor <= 0 || params.factor > i32::MAX as i64 || params.add > i32::MAX as i64 {
        return false;
    }
    let limit = ((i32::MAX as i64 - params.add) / params.factor).max(0);
    i64::from(max_abs_level) <= limit
}

#[inline(always)]
fn scaled_mullo_safe_from_max(
    max_abs_level: i32,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
) -> bool {
    let base_factor = params.factor / 16;
    if base_factor <= 0 || params.add > i32::MAX as i64 {
        return false;
    }
    let max_factor = base_factor * scaling.max_coeff();
    if max_factor <= 0 || max_factor > i32::MAX as i64 {
        return false;
    }
    let limit = ((i32::MAX as i64 - params.add) / max_factor).max(0);
    i64::from(max_abs_level) <= limit
}

#[inline(always)]
fn avx2_factor_ok(params: DequantParams) -> bool {
    params.factor > 0 && params.factor <= i32::MAX as i64 && params.add <= i32::MAX as i64
}

#[inline(always)]
fn avx2_scaled_factor_ok(params: DequantParams, scaling: ScalingMatrix<'_>) -> bool {
    let base_factor = params.factor / 16;
    base_factor > 0
        && base_factor * scaling.max_coeff() <= i32::MAX as i64
        && params.add <= i32::MAX as i64
}

#[inline(always)]
fn scaling_factors8(base_factor: i64, scaling: ScalingMatrix<'_>, idx: usize) -> [i32; 8] {
    std::array::from_fn(|lane| (base_factor * scaling.coeff(idx + lane)) as i32)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x8(src: &[i32; 8]) -> __m256i {
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8(dst: &mut [i32; 8], v: __m256i) {
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast::<__m256i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x16(dst: &mut [i16; 16], v: __m256i) {
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast::<__m256i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn sra_epi32_count(v: __m256i, shift: __m128i) -> __m256i {
    _mm256_sra_epi32(v, shift)
}

#[inline]
#[target_feature(enable = "avx2")]
fn shl_epi32_count(v: __m256i, shift: __m128i) -> __m256i {
    _mm256_sll_epi32(v, shift)
}

#[inline]
#[target_feature(enable = "avx2")]
fn clip_i16_s32x8_with(v: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_max_epi32(_mm256_min_epi32(v, hi), lo)
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_i32x8_to_i16x16(lo: __m256i, hi: __m256i) -> __m256i {
    // vpackssdw is lane-local: [lo0..3, hi0..3 | lo4..7, hi4..7].
    // Reorder qwords into the natural output order: [lo0..7, hi0..7].
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequant8_avx2_const(
    levels: &[i32; 8],
    factor: __m256i,
    add: __m256i,
    shift: __m128i,
    clip_lo: __m256i,
    clip_hi: __m256i,
) -> __m256i {
    let v = load_i32x8(levels);
    let v = _mm256_mullo_epi32(v, factor);
    let v = _mm256_add_epi32(v, add);
    clip_i16_s32x8_with(sra_epi32_count(v, shift), clip_lo, clip_hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequant8_scaled_avx2_const(
    levels: &[i32; 8],
    factors: &[i32; 8],
    add: __m256i,
    shift: __m128i,
    clip_lo: __m256i,
    clip_hi: __m256i,
) -> __m256i {
    let v = load_i32x8(levels);
    let f = load_i32x8(factors);
    let v = _mm256_mullo_epi32(v, f);
    let v = _mm256_add_epi32(v, add);
    clip_i16_s32x8_with(sra_epi32_count(v, shift), clip_lo, clip_hi)
}

#[inline(always)]
fn floor_div_i64(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    let q = n / d;
    let r = n % d;
    if r < 0 { q - 1 } else { q }
}

#[inline(always)]
fn ceil_div_i64(n: i64, d: i64) -> i64 {
    debug_assert!(d > 0);
    -floor_div_i64(-n, d)
}

#[inline(always)]
fn dequant_pos_cut_minus_one(factor: i64, add: i64, shift: i32) -> i32 {
    if factor <= 0 {
        return i32::MAX;
    }
    let threshold = (32767i64 << shift) - add;
    (ceil_div_i64(threshold, factor) - 1).clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

#[inline(always)]
fn dequant_neg_keep_min(factor: i64, add: i64, shift: i32) -> i32 {
    if factor <= 0 {
        return i32::MIN;
    }
    let threshold = ((-32767i64) << shift) - 1 - add;
    (floor_div_i64(threshold, factor) + 1).clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

#[inline(always)]
fn dequant_sat_bounds(factor: i64, add: i64, shift: i32) -> (i32, i32) {
    (
        dequant_pos_cut_minus_one(factor, add, shift),
        dequant_neg_keep_min(factor, add, shift),
    )
}

#[inline(always)]
fn dequant_sat_bounds8(factors: &[i32; 8], add: i64, shift: i32) -> ([i32; 8], [i32; 8]) {
    let mut pos_cut_minus_one = [0i32; 8];
    let mut neg_keep_min = [0i32; 8];
    for lane in 0..8 {
        let factor = factors[lane] as i64;
        pos_cut_minus_one[lane] = dequant_pos_cut_minus_one(factor, add, shift);
        neg_keep_min[lane] = dequant_neg_keep_min(factor, add, shift);
    }
    (pos_cut_minus_one, neg_keep_min)
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn dequant8_avx2_saturating_const(
    levels: &[i32; 8],
    factor: __m256i,
    add: __m256i,
    shift: __m128i,
    pos_cut_minus_one: __m256i,
    neg_keep_min: __m256i,
    clip_lo: __m256i,
    clip_hi: __m256i,
) -> __m256i {
    let v = load_i32x8(levels);
    let over_hi = _mm256_cmpgt_epi32(v, pos_cut_minus_one);
    let over_lo = _mm256_cmpgt_epi32(neg_keep_min, v);
    let safe = _mm256_max_epi32(_mm256_min_epi32(v, pos_cut_minus_one), neg_keep_min);
    let deq = _mm256_mullo_epi32(safe, factor);
    let deq = _mm256_add_epi32(deq, add);
    let deq = sra_epi32_count(deq, shift);
    let deq = _mm256_blendv_epi8(deq, clip_hi, over_hi);
    _mm256_blendv_epi8(deq, clip_lo, over_lo)
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn dequant8_scaled_avx2_saturating_const(
    levels: &[i32; 8],
    factors: &[i32; 8],
    pos_cut_minus_one: &[i32; 8],
    neg_keep_min: &[i32; 8],
    add: __m256i,
    shift: __m128i,
    clip_lo: __m256i,
    clip_hi: __m256i,
) -> __m256i {
    let v = load_i32x8(levels);
    let f = load_i32x8(factors);
    let pos_cut_minus_one = load_i32x8(pos_cut_minus_one);
    let neg_keep_min = load_i32x8(neg_keep_min);
    let over_hi = _mm256_cmpgt_epi32(v, pos_cut_minus_one);
    let over_lo = _mm256_cmpgt_epi32(neg_keep_min, v);
    let safe = _mm256_max_epi32(_mm256_min_epi32(v, pos_cut_minus_one), neg_keep_min);
    let deq = _mm256_mullo_epi32(safe, f);
    let deq = _mm256_add_epi32(deq, add);
    let deq = sra_epi32_count(deq, shift);
    let deq = _mm256_blendv_epi8(deq, clip_hi, over_hi);
    _mm256_blendv_epi8(deq, clip_lo, over_lo)
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply_transform_skip_shift8(
    v: __m256i,
    params: TransformSkipParams,
    clip_lo: __m256i,
    clip_hi: __m256i,
) -> __m256i {
    let shift = _mm_cvtsi32_si128(params.tr_shift.abs());
    let v = if params.tr_shift >= 0 {
        sra_epi32_count(_mm256_add_epi32(v, _mm256_set1_epi32(params.tr_add)), shift)
    } else {
        shl_epi32_count(v, shift)
    };
    clip_i16_s32x8_with(v, clip_lo, clip_hi)
}

#[target_feature(enable = "avx2")]
fn dequantize_into_avx2_mullo_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i32],
) {
    debug_assert!(supported_n(n));
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    let factor = _mm256_set1_epi32(params.factor as i32);
    let add = _mm256_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let v0 = dequant8_avx2_const(&src[0], factor, add, shift, clip_lo, clip_hi);
        let v1 = dequant8_avx2_const(&src[1], factor, add, shift, clip_lo, clip_hi);
        let v2 = dequant8_avx2_const(&src[2], factor, add, shift, clip_lo, clip_hi);
        let v3 = dequant8_avx2_const(&src[3], factor, add, shift, clip_lo, clip_hi);
        store_i32x8(&mut dst[0], v0);
        store_i32x8(&mut dst[1], v1);
        store_i32x8(&mut dst[2], v2);
        store_i32x8(&mut dst[3], v3);
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        store_i32x8(
            dst,
            dequant8_avx2_const(src, factor, add, shift, clip_lo, clip_hi),
        );
    }
}

#[target_feature(enable = "avx2")]
fn dequantize_into_avx2_16_mullo_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i16],
) {
    debug_assert!(supported_n(n));
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<16>();
    let (out, _) = out[..count].as_chunks_mut::<16>();

    let factor = _mm256_set1_epi32(params.factor as i32);
    let add = _mm256_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<2>();
    let (out_groups, out_tail) = out.as_chunks_mut::<2>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let (src0, _) = src[0].as_chunks::<8>();
        let (src1, _) = src[1].as_chunks::<8>();
        let lo0 = dequant8_avx2_const(&src0[0], factor, add, shift, clip_lo, clip_hi);
        let hi0 = dequant8_avx2_const(&src0[1], factor, add, shift, clip_lo, clip_hi);
        let lo1 = dequant8_avx2_const(&src1[0], factor, add, shift, clip_lo, clip_hi);
        let hi1 = dequant8_avx2_const(&src1[1], factor, add, shift, clip_lo, clip_hi);
        store_i16x16(&mut dst[0], pack_i32x8_to_i16x16(lo0, hi0));
        store_i16x16(&mut dst[1], pack_i32x8_to_i16x16(lo1, hi1));
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        let (src8, _) = src.as_chunks::<8>();
        let lo = dequant8_avx2_const(&src8[0], factor, add, shift, clip_lo, clip_hi);
        let hi = dequant8_avx2_const(&src8[1], factor, add, shift, clip_lo, clip_hi);
        store_i16x16(dst, pack_i32x8_to_i16x16(lo, hi));
    }
}

#[target_feature(enable = "avx2")]
fn dequantize_into_avx2_saturating_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i32],
) {
    debug_assert!(supported_n(n));
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    let factor = _mm256_set1_epi32(params.factor as i32);
    let add = _mm256_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);
    let (pos_cut_minus_one, neg_keep_min) =
        dequant_sat_bounds(params.factor, params.add, params.shift);
    let pos_cut_minus_one = _mm256_set1_epi32(pos_cut_minus_one);
    let neg_keep_min = _mm256_set1_epi32(neg_keep_min);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let v0 = dequant8_avx2_saturating_const(
            &src[0],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let v1 = dequant8_avx2_saturating_const(
            &src[1],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let v2 = dequant8_avx2_saturating_const(
            &src[2],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let v3 = dequant8_avx2_saturating_const(
            &src[3],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        store_i32x8(&mut dst[0], v0);
        store_i32x8(&mut dst[1], v1);
        store_i32x8(&mut dst[2], v2);
        store_i32x8(&mut dst[3], v3);
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        store_i32x8(
            dst,
            dequant8_avx2_saturating_const(
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

#[target_feature(enable = "avx2")]
fn dequantize_into_avx2_16_saturating_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i16],
) {
    debug_assert!(supported_n(n));
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<16>();
    let (out, _) = out[..count].as_chunks_mut::<16>();

    let factor = _mm256_set1_epi32(params.factor as i32);
    let add = _mm256_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);
    let (pos_cut_minus_one, neg_keep_min) =
        dequant_sat_bounds(params.factor, params.add, params.shift);
    let pos_cut_minus_one = _mm256_set1_epi32(pos_cut_minus_one);
    let neg_keep_min = _mm256_set1_epi32(neg_keep_min);

    let (level_groups, level_tail) = levels.as_chunks::<2>();
    let (out_groups, out_tail) = out.as_chunks_mut::<2>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let (src0, _) = src[0].as_chunks::<8>();
        let (src1, _) = src[1].as_chunks::<8>();
        let lo0 = dequant8_avx2_saturating_const(
            &src0[0],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let hi0 = dequant8_avx2_saturating_const(
            &src0[1],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let lo1 = dequant8_avx2_saturating_const(
            &src1[0],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let hi1 = dequant8_avx2_saturating_const(
            &src1[1],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        store_i16x16(&mut dst[0], pack_i32x8_to_i16x16(lo0, hi0));
        store_i16x16(&mut dst[1], pack_i32x8_to_i16x16(lo1, hi1));
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        let (src8, _) = src.as_chunks::<8>();
        let lo = dequant8_avx2_saturating_const(
            &src8[0],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        let hi = dequant8_avx2_saturating_const(
            &src8[1],
            factor,
            add,
            shift,
            pos_cut_minus_one,
            neg_keep_min,
            clip_lo,
            clip_hi,
        );
        store_i16x16(dst, pack_i32x8_to_i16x16(lo, hi));
    }
}

#[target_feature(enable = "avx2")]
fn dequantize_transform_skip_into_avx2_mullo_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<8>();

    let factor = _mm256_set1_epi32(params.dequant.factor as i32);
    let add = _mm256_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let d0 = dequant8_avx2_const(&levels[0], factor, add, shift, clip_lo, clip_hi);
    let d1 = dequant8_avx2_const(&levels[1], factor, add, shift, clip_lo, clip_hi);
    store_i32x8(
        &mut out[0],
        apply_transform_skip_shift8(d0, params, clip_lo, clip_hi),
    );
    store_i32x8(
        &mut out[1],
        apply_transform_skip_shift8(d1, params, clip_lo, clip_hi),
    );
}

#[target_feature(enable = "avx2")]
fn dequantize_transform_skip_into_avx2_16_mullo_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<16>();

    let factor = _mm256_set1_epi32(params.dequant.factor as i32);
    let add = _mm256_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let d0 = dequant8_avx2_const(&levels[0], factor, add, shift, clip_lo, clip_hi);
    let d1 = dequant8_avx2_const(&levels[1], factor, add, shift, clip_lo, clip_hi);
    let v0 = apply_transform_skip_shift8(d0, params, clip_lo, clip_hi);
    let v1 = apply_transform_skip_shift8(d1, params, clip_lo, clip_hi);
    store_i16x16(&mut out[0], pack_i32x8_to_i16x16(v0, v1));
}

#[target_feature(enable = "avx2")]
fn dequantize_transform_skip_into_avx2_saturating_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<8>();

    let factor = _mm256_set1_epi32(params.dequant.factor as i32);
    let add = _mm256_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);
    let (pos_cut_minus_one, neg_keep_min) = dequant_sat_bounds(
        params.dequant.factor,
        params.dequant.add,
        params.dequant.shift,
    );
    let pos_cut_minus_one = _mm256_set1_epi32(pos_cut_minus_one);
    let neg_keep_min = _mm256_set1_epi32(neg_keep_min);

    let d0 = dequant8_avx2_saturating_const(
        &levels[0],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    let d1 = dequant8_avx2_saturating_const(
        &levels[1],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    store_i32x8(
        &mut out[0],
        apply_transform_skip_shift8(d0, params, clip_lo, clip_hi),
    );
    store_i32x8(
        &mut out[1],
        apply_transform_skip_shift8(d1, params, clip_lo, clip_hi),
    );
}

#[target_feature(enable = "avx2")]
fn dequantize_transform_skip_into_avx2_16_saturating_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<16>();

    let factor = _mm256_set1_epi32(params.dequant.factor as i32);
    let add = _mm256_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);
    let (pos_cut_minus_one, neg_keep_min) = dequant_sat_bounds(
        params.dequant.factor,
        params.dequant.add,
        params.dequant.shift,
    );
    let pos_cut_minus_one = _mm256_set1_epi32(pos_cut_minus_one);
    let neg_keep_min = _mm256_set1_epi32(neg_keep_min);

    let d0 = dequant8_avx2_saturating_const(
        &levels[0],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    let d1 = dequant8_avx2_saturating_const(
        &levels[1],
        factor,
        add,
        shift,
        pos_cut_minus_one,
        neg_keep_min,
        clip_lo,
        clip_hi,
    );
    let v0 = apply_transform_skip_shift8(d0, params, clip_lo, clip_hi);
    let v1 = apply_transform_skip_shift8(d1, params, clip_lo, clip_hi);
    store_i16x16(&mut out[0], pack_i32x8_to_i16x16(v0, v1));
}

#[target_feature(enable = "avx2")]
fn dequantize_scaled_into_avx2_mullo_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    debug_assert!(supported_n(n));
    let count = n * n;
    let base_factor = params.factor / 16;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    let add = _mm256_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (group_idx, (src, dst)) in level_groups.iter().zip(out_groups.iter_mut()).enumerate() {
        let idx = group_idx * 32;
        let f0 = scaling_factors8(base_factor, scaling, idx);
        let f1 = scaling_factors8(base_factor, scaling, idx + 8);
        let f2 = scaling_factors8(base_factor, scaling, idx + 16);
        let f3 = scaling_factors8(base_factor, scaling, idx + 24);
        let v0 = dequant8_scaled_avx2_const(&src[0], &f0, add, shift, clip_lo, clip_hi);
        let v1 = dequant8_scaled_avx2_const(&src[1], &f1, add, shift, clip_lo, clip_hi);
        let v2 = dequant8_scaled_avx2_const(&src[2], &f2, add, shift, clip_lo, clip_hi);
        let v3 = dequant8_scaled_avx2_const(&src[3], &f3, add, shift, clip_lo, clip_hi);
        store_i32x8(&mut dst[0], v0);
        store_i32x8(&mut dst[1], v1);
        store_i32x8(&mut dst[2], v2);
        store_i32x8(&mut dst[3], v3);
    }

    let tail_base = level_groups.len() * 32;
    for (block_idx, (src, dst)) in level_tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let factors = scaling_factors8(base_factor, scaling, tail_base + block_idx * 8);
        store_i32x8(
            dst,
            dequant8_scaled_avx2_const(src, &factors, add, shift, clip_lo, clip_hi),
        );
    }
}

#[target_feature(enable = "avx2")]
fn dequantize_scaled_into_avx2_16_mullo_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    debug_assert!(supported_n(n));
    let count = n * n;
    let base_factor = params.factor / 16;
    let (levels, _) = levels[..count].as_chunks::<16>();
    let (out, _) = out[..count].as_chunks_mut::<16>();

    let add = _mm256_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<2>();
    let (out_groups, out_tail) = out.as_chunks_mut::<2>();

    for (group_idx, (src, dst)) in level_groups.iter().zip(out_groups.iter_mut()).enumerate() {
        let idx = group_idx * 32;
        let (src0, _) = src[0].as_chunks::<8>();
        let (src1, _) = src[1].as_chunks::<8>();
        let f0 = scaling_factors8(base_factor, scaling, idx);
        let f1 = scaling_factors8(base_factor, scaling, idx + 8);
        let f2 = scaling_factors8(base_factor, scaling, idx + 16);
        let f3 = scaling_factors8(base_factor, scaling, idx + 24);
        let lo0 = dequant8_scaled_avx2_const(&src0[0], &f0, add, shift, clip_lo, clip_hi);
        let hi0 = dequant8_scaled_avx2_const(&src0[1], &f1, add, shift, clip_lo, clip_hi);
        let lo1 = dequant8_scaled_avx2_const(&src1[0], &f2, add, shift, clip_lo, clip_hi);
        let hi1 = dequant8_scaled_avx2_const(&src1[1], &f3, add, shift, clip_lo, clip_hi);
        store_i16x16(&mut dst[0], pack_i32x8_to_i16x16(lo0, hi0));
        store_i16x16(&mut dst[1], pack_i32x8_to_i16x16(lo1, hi1));
    }

    let tail_base = level_groups.len() * 32;
    for (block_idx, (src, dst)) in level_tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let (src8, _) = src.as_chunks::<8>();
        let idx = tail_base + block_idx * 16;
        let f0 = scaling_factors8(base_factor, scaling, idx);
        let f1 = scaling_factors8(base_factor, scaling, idx + 8);
        let lo = dequant8_scaled_avx2_const(&src8[0], &f0, add, shift, clip_lo, clip_hi);
        let hi = dequant8_scaled_avx2_const(&src8[1], &f1, add, shift, clip_lo, clip_hi);
        store_i16x16(dst, pack_i32x8_to_i16x16(lo, hi));
    }
}

#[target_feature(enable = "avx2")]
fn dequantize_scaled_into_avx2_saturating_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    debug_assert!(supported_n(n));
    let count = n * n;
    let base_factor = params.factor / 16;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    let add = _mm256_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (group_idx, (src, dst)) in level_groups.iter().zip(out_groups.iter_mut()).enumerate() {
        let idx = group_idx * 32;
        let f0 = scaling_factors8(base_factor, scaling, idx);
        let f1 = scaling_factors8(base_factor, scaling, idx + 8);
        let f2 = scaling_factors8(base_factor, scaling, idx + 16);
        let f3 = scaling_factors8(base_factor, scaling, idx + 24);
        let (p0, n0) = dequant_sat_bounds8(&f0, params.add, params.shift);
        let (p1, n1) = dequant_sat_bounds8(&f1, params.add, params.shift);
        let (p2, n2) = dequant_sat_bounds8(&f2, params.add, params.shift);
        let (p3, n3) = dequant_sat_bounds8(&f3, params.add, params.shift);
        let v0 = dequant8_scaled_avx2_saturating_const(
            &src[0], &f0, &p0, &n0, add, shift, clip_lo, clip_hi,
        );
        let v1 = dequant8_scaled_avx2_saturating_const(
            &src[1], &f1, &p1, &n1, add, shift, clip_lo, clip_hi,
        );
        let v2 = dequant8_scaled_avx2_saturating_const(
            &src[2], &f2, &p2, &n2, add, shift, clip_lo, clip_hi,
        );
        let v3 = dequant8_scaled_avx2_saturating_const(
            &src[3], &f3, &p3, &n3, add, shift, clip_lo, clip_hi,
        );
        store_i32x8(&mut dst[0], v0);
        store_i32x8(&mut dst[1], v1);
        store_i32x8(&mut dst[2], v2);
        store_i32x8(&mut dst[3], v3);
    }

    let tail_base = level_groups.len() * 32;
    for (block_idx, (src, dst)) in level_tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let factors = scaling_factors8(base_factor, scaling, tail_base + block_idx * 8);
        let (pos, neg) = dequant_sat_bounds8(&factors, params.add, params.shift);
        store_i32x8(
            dst,
            dequant8_scaled_avx2_saturating_const(
                src, &factors, &pos, &neg, add, shift, clip_lo, clip_hi,
            ),
        );
    }
}

#[target_feature(enable = "avx2")]
fn dequantize_scaled_into_avx2_16_saturating_impl(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    debug_assert!(supported_n(n));
    let count = n * n;
    let base_factor = params.factor / 16;
    let (levels, _) = levels[..count].as_chunks::<16>();
    let (out, _) = out[..count].as_chunks_mut::<16>();

    let add = _mm256_set1_epi32(params.add as i32);
    let shift = _mm_cvtsi32_si128(params.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<2>();
    let (out_groups, out_tail) = out.as_chunks_mut::<2>();

    for (group_idx, (src, dst)) in level_groups.iter().zip(out_groups.iter_mut()).enumerate() {
        let idx = group_idx * 32;
        let (src0, _) = src[0].as_chunks::<8>();
        let (src1, _) = src[1].as_chunks::<8>();
        let f0 = scaling_factors8(base_factor, scaling, idx);
        let f1 = scaling_factors8(base_factor, scaling, idx + 8);
        let f2 = scaling_factors8(base_factor, scaling, idx + 16);
        let f3 = scaling_factors8(base_factor, scaling, idx + 24);
        let (p0, n0) = dequant_sat_bounds8(&f0, params.add, params.shift);
        let (p1, n1) = dequant_sat_bounds8(&f1, params.add, params.shift);
        let (p2, n2) = dequant_sat_bounds8(&f2, params.add, params.shift);
        let (p3, n3) = dequant_sat_bounds8(&f3, params.add, params.shift);
        let lo0 = dequant8_scaled_avx2_saturating_const(
            &src0[0], &f0, &p0, &n0, add, shift, clip_lo, clip_hi,
        );
        let hi0 = dequant8_scaled_avx2_saturating_const(
            &src0[1], &f1, &p1, &n1, add, shift, clip_lo, clip_hi,
        );
        let lo1 = dequant8_scaled_avx2_saturating_const(
            &src1[0], &f2, &p2, &n2, add, shift, clip_lo, clip_hi,
        );
        let hi1 = dequant8_scaled_avx2_saturating_const(
            &src1[1], &f3, &p3, &n3, add, shift, clip_lo, clip_hi,
        );
        store_i16x16(&mut dst[0], pack_i32x8_to_i16x16(lo0, hi0));
        store_i16x16(&mut dst[1], pack_i32x8_to_i16x16(lo1, hi1));
    }

    let tail_base = level_groups.len() * 32;
    for (block_idx, (src, dst)) in level_tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let (src8, _) = src.as_chunks::<8>();
        let idx = tail_base + block_idx * 16;
        let f0 = scaling_factors8(base_factor, scaling, idx);
        let f1 = scaling_factors8(base_factor, scaling, idx + 8);
        let (p0, n0) = dequant_sat_bounds8(&f0, params.add, params.shift);
        let (p1, n1) = dequant_sat_bounds8(&f1, params.add, params.shift);
        let lo = dequant8_scaled_avx2_saturating_const(
            &src8[0], &f0, &p0, &n0, add, shift, clip_lo, clip_hi,
        );
        let hi = dequant8_scaled_avx2_saturating_const(
            &src8[1], &f1, &p1, &n1, add, shift, clip_lo, clip_hi,
        );
        store_i16x16(dst, pack_i32x8_to_i16x16(lo, hi));
    }
}

#[target_feature(enable = "avx2")]
fn dequantize_transform_skip_scaled_into_avx2_mullo_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    let base_factor = params.dequant.factor / 16;
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<8>();

    let add = _mm256_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let f0 = scaling_factors8(base_factor, scaling, 0);
    let f1 = scaling_factors8(base_factor, scaling, 8);
    let d0 = dequant8_scaled_avx2_const(&levels[0], &f0, add, shift, clip_lo, clip_hi);
    let d1 = dequant8_scaled_avx2_const(&levels[1], &f1, add, shift, clip_lo, clip_hi);
    store_i32x8(
        &mut out[0],
        apply_transform_skip_shift8(d0, params, clip_lo, clip_hi),
    );
    store_i32x8(
        &mut out[1],
        apply_transform_skip_shift8(d1, params, clip_lo, clip_hi),
    );
}

#[target_feature(enable = "avx2")]
fn dequantize_transform_skip_scaled_into_avx2_16_mullo_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    let base_factor = params.dequant.factor / 16;
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<16>();

    let add = _mm256_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let f0 = scaling_factors8(base_factor, scaling, 0);
    let f1 = scaling_factors8(base_factor, scaling, 8);
    let d0 = dequant8_scaled_avx2_const(&levels[0], &f0, add, shift, clip_lo, clip_hi);
    let d1 = dequant8_scaled_avx2_const(&levels[1], &f1, add, shift, clip_lo, clip_hi);
    let v0 = apply_transform_skip_shift8(d0, params, clip_lo, clip_hi);
    let v1 = apply_transform_skip_shift8(d1, params, clip_lo, clip_hi);
    store_i16x16(&mut out[0], pack_i32x8_to_i16x16(v0, v1));
}

#[target_feature(enable = "avx2")]
fn dequantize_transform_skip_scaled_into_avx2_saturating_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    let base_factor = params.dequant.factor / 16;
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<8>();

    let add = _mm256_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let f0 = scaling_factors8(base_factor, scaling, 0);
    let f1 = scaling_factors8(base_factor, scaling, 8);
    let (p0, n0) = dequant_sat_bounds8(&f0, params.dequant.add, params.dequant.shift);
    let (p1, n1) = dequant_sat_bounds8(&f1, params.dequant.add, params.dequant.shift);
    let d0 = dequant8_scaled_avx2_saturating_const(
        &levels[0], &f0, &p0, &n0, add, shift, clip_lo, clip_hi,
    );
    let d1 = dequant8_scaled_avx2_saturating_const(
        &levels[1], &f1, &p1, &n1, add, shift, clip_lo, clip_hi,
    );
    store_i32x8(
        &mut out[0],
        apply_transform_skip_shift8(d0, params, clip_lo, clip_hi),
    );
    store_i32x8(
        &mut out[1],
        apply_transform_skip_shift8(d1, params, clip_lo, clip_hi),
    );
}

#[target_feature(enable = "avx2")]
fn dequantize_transform_skip_scaled_into_avx2_16_saturating_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    let base_factor = params.dequant.factor / 16;
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<16>();

    let add = _mm256_set1_epi32(params.dequant.add as i32);
    let shift = _mm_cvtsi32_si128(params.dequant.shift);
    let clip_lo = _mm256_set1_epi32(-32768);
    let clip_hi = _mm256_set1_epi32(32767);

    let f0 = scaling_factors8(base_factor, scaling, 0);
    let f1 = scaling_factors8(base_factor, scaling, 8);
    let (p0, n0) = dequant_sat_bounds8(&f0, params.dequant.add, params.dequant.shift);
    let (p1, n1) = dequant_sat_bounds8(&f1, params.dequant.add, params.dequant.shift);
    let d0 = dequant8_scaled_avx2_saturating_const(
        &levels[0], &f0, &p0, &n0, add, shift, clip_lo, clip_hi,
    );
    let d1 = dequant8_scaled_avx2_saturating_const(
        &levels[1], &f1, &p1, &n1, add, shift, clip_lo, clip_hi,
    );
    let v0 = apply_transform_skip_shift8(d0, params, clip_lo, clip_hi);
    let v1 = apply_transform_skip_shift8(d1, params, clip_lo, clip_hi);
    store_i16x16(&mut out[0], pack_i32x8_to_i16x16(v0, v1));
}

pub(crate) fn dequantize_into_avx2(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    max_abs_level: i32,
    out: &mut [i32],
) {
    debug_assert!(supported_n(n));
    debug_assert!(avx2_factor_ok(params));
    if mullo_safe_from_max(max_abs_level, params) {
        unsafe { dequantize_into_avx2_mullo_impl(levels, n, params, out) }
    } else {
        unsafe { dequantize_into_avx2_saturating_impl(levels, n, params, out) }
    }
}

pub(crate) fn dequantize_into_avx2_16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    max_abs_level: i32,
    out: &mut [i16],
) {
    debug_assert!(supported_n(n));
    debug_assert!(avx2_factor_ok(params));
    if mullo_safe_from_max(max_abs_level, params) {
        unsafe { dequantize_into_avx2_16_mullo_impl(levels, n, params, out) }
    } else {
        unsafe { dequantize_into_avx2_16_saturating_impl(levels, n, params, out) }
    }
}

pub(crate) fn dequantize_transform_skip_into_avx2(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    max_abs_level: i32,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    debug_assert!(avx2_factor_ok(params.dequant));
    if mullo_safe_from_max(max_abs_level, params.dequant) {
        unsafe { dequantize_transform_skip_into_avx2_mullo_impl(levels, n, params, out) }
    } else {
        unsafe { dequantize_transform_skip_into_avx2_saturating_impl(levels, n, params, out) }
    }
}

pub(crate) fn dequantize_transform_skip_into_avx2_16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    max_abs_level: i32,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    debug_assert!(avx2_factor_ok(params.dequant));
    if mullo_safe_from_max(max_abs_level, params.dequant) {
        unsafe { dequantize_transform_skip_into_avx2_16_mullo_impl(levels, n, params, out) }
    } else {
        unsafe { dequantize_transform_skip_into_avx2_16_saturating_impl(levels, n, params, out) }
    }
}

pub(crate) fn dequantize_scaled_into_avx2(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    max_abs_level: i32,
    out: &mut [i32],
) {
    debug_assert!(supported_n(n));
    debug_assert!(avx2_scaled_factor_ok(params, scaling));
    if scaled_mullo_safe_from_max(max_abs_level, params, scaling) {
        unsafe { dequantize_scaled_into_avx2_mullo_impl(levels, n, params, scaling, out) }
    } else {
        unsafe { dequantize_scaled_into_avx2_saturating_impl(levels, n, params, scaling, out) }
    }
}

pub(crate) fn dequantize_scaled_into_avx2_16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    max_abs_level: i32,
    out: &mut [i16],
) {
    debug_assert!(supported_n(n));
    debug_assert!(avx2_scaled_factor_ok(params, scaling));
    if scaled_mullo_safe_from_max(max_abs_level, params, scaling) {
        unsafe { dequantize_scaled_into_avx2_16_mullo_impl(levels, n, params, scaling, out) }
    } else {
        unsafe { dequantize_scaled_into_avx2_16_saturating_impl(levels, n, params, scaling, out) }
    }
}

pub(crate) fn dequantize_transform_skip_scaled_into_avx2(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    max_abs_level: i32,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    debug_assert!(avx2_scaled_factor_ok(params.dequant, scaling));
    if scaled_mullo_safe_from_max(max_abs_level, params.dequant, scaling) {
        unsafe {
            dequantize_transform_skip_scaled_into_avx2_mullo_impl(levels, n, params, scaling, out)
        }
    } else {
        unsafe {
            dequantize_transform_skip_scaled_into_avx2_saturating_impl(
                levels, n, params, scaling, out,
            )
        }
    }
}

pub(crate) fn dequantize_transform_skip_scaled_into_avx2_16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    max_abs_level: i32,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    debug_assert!(avx2_scaled_factor_ok(params.dequant, scaling));
    if scaled_mullo_safe_from_max(max_abs_level, params.dequant, scaling) {
        unsafe {
            dequantize_transform_skip_scaled_into_avx2_16_mullo_impl(
                levels, n, params, scaling, out,
            )
        }
    } else {
        unsafe {
            dequantize_transform_skip_scaled_into_avx2_16_saturating_impl(
                levels, n, params, scaling, out,
            )
        }
    }
}
