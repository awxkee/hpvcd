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

use core::arch::aarch64::*;

use crate::transform::{
    DequantParams, ScalingMatrix, TransformSkipParams, dequantize_scaled_into_scalar_i16,
    dequantize_scaled_into_scalar_i32, dequantize_transform_skip_scaled_into_scalar_i16,
    dequantize_transform_skip_scaled_into_scalar_i32,
};

#[inline]
fn supported_n(n: usize) -> bool {
    matches!(n, 4 | 8 | 16 | 32)
}

#[inline]
fn scaling_factors4(base_factor: i64, scaling: ScalingMatrix<'_>, idx: usize) -> [i32; 4] {
    std::array::from_fn(|lane| (base_factor * scaling.coeff(idx + lane)) as i32)
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x4(src: &[i32; 4]) -> int32x4_t {
    unsafe { vld1q_s32(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4(dst: &mut [i32; 4], v: int32x4_t) {
    unsafe { vst1q_s32(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x8(dst: &mut [i16; 8], v: int16x8_t) {
    unsafe { vst1q_s16(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn clip_i16_s32x4_with(v: int32x4_t, lo: int32x4_t, hi: int32x4_t) -> int32x4_t {
    vminq_s32(vmaxq_s32(v, lo), hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn dequant4_neon_const(
    levels: &[i32; 4],
    factor: i32,
    add: int64x2_t,
    shift: int64x2_t,
    clip_lo: int32x4_t,
    clip_hi: int32x4_t,
) -> int32x4_t {
    let v = load_i32x4(levels);

    let lo = vshlq_s64(vaddq_s64(vmull_n_s32(vget_low_s32(v), factor), add), shift);
    let hi = vshlq_s64(vaddq_s64(vmull_n_s32(vget_high_s32(v), factor), add), shift);
    clip_i16_s32x4_with(
        vcombine_s32(vqmovn_s64(lo), vqmovn_s64(hi)),
        clip_lo,
        clip_hi,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn dequant4_scaled_neon_const(
    levels: &[i32; 4],
    factors: &[i32; 4],
    add: int64x2_t,
    shift: int64x2_t,
    clip_lo: int32x4_t,
    clip_hi: int32x4_t,
) -> int32x4_t {
    let v = load_i32x4(levels);
    let f = load_i32x4(factors);

    let lo = vshlq_s64(
        vaddq_s64(vmull_s32(vget_low_s32(v), vget_low_s32(f)), add),
        shift,
    );
    let hi = vshlq_s64(
        vaddq_s64(vmull_s32(vget_high_s32(v), vget_high_s32(f)), add),
        shift,
    );
    clip_i16_s32x4_with(
        vcombine_s32(vqmovn_s64(lo), vqmovn_s64(hi)),
        clip_lo,
        clip_hi,
    )
}

#[target_feature(enable = "neon")]
fn dequantize_into_neon_impl(levels: &[i32], n: usize, params: DequantParams, out: &mut [i32]) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<4>();
    let (out, _) = out[..count].as_chunks_mut::<4>();

    let factor = params.factor as i32;
    let add = vdupq_n_s64(params.add);
    let shift = vdupq_n_s64(-(params.shift as i64));
    let clip_lo = vdupq_n_s32(-32768);
    let clip_hi = vdupq_n_s32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let v0 = dequant4_neon_const(&src[0], factor, add, shift, clip_lo, clip_hi);
        let v1 = dequant4_neon_const(&src[1], factor, add, shift, clip_lo, clip_hi);
        let v2 = dequant4_neon_const(&src[2], factor, add, shift, clip_lo, clip_hi);
        let v3 = dequant4_neon_const(&src[3], factor, add, shift, clip_lo, clip_hi);
        store_i32x4(&mut dst[0], v0);
        store_i32x4(&mut dst[1], v1);
        store_i32x4(&mut dst[2], v2);
        store_i32x4(&mut dst[3], v3);
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        store_i32x4(
            dst,
            dequant4_neon_const(src, factor, add, shift, clip_lo, clip_hi),
        );
    }
}

#[target_feature(enable = "neon")]
fn dequantize_into_neon16_impl(levels: &[i32], n: usize, params: DequantParams, out: &mut [i16]) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    let factor = params.factor as i32;
    let add = vdupq_n_s64(params.add);
    let shift = vdupq_n_s64(-(params.shift as i64));
    let clip_lo = vdupq_n_s32(-32768);
    let clip_hi = vdupq_n_s32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<2>();
    let (out_groups, out_tail) = out.as_chunks_mut::<2>();

    for (src, dst) in level_groups.iter().zip(out_groups.iter_mut()) {
        let (src0, _) = src[0].as_chunks::<4>();
        let (src1, _) = src[1].as_chunks::<4>();
        let lo0 = vqmovn_s32(dequant4_neon_const(
            &src0[0], factor, add, shift, clip_lo, clip_hi,
        ));
        let hi0 = vqmovn_s32(dequant4_neon_const(
            &src0[1], factor, add, shift, clip_lo, clip_hi,
        ));
        let lo1 = vqmovn_s32(dequant4_neon_const(
            &src1[0], factor, add, shift, clip_lo, clip_hi,
        ));
        let hi1 = vqmovn_s32(dequant4_neon_const(
            &src1[1], factor, add, shift, clip_lo, clip_hi,
        ));
        store_i16x8(&mut dst[0], vcombine_s16(lo0, hi0));
        store_i16x8(&mut dst[1], vcombine_s16(lo1, hi1));
    }

    for (src, dst) in level_tail.iter().zip(out_tail.iter_mut()) {
        let (src4, _) = src.as_chunks::<4>();
        let lo = vqmovn_s32(dequant4_neon_const(
            &src4[0], factor, add, shift, clip_lo, clip_hi,
        ));
        let hi = vqmovn_s32(dequant4_neon_const(
            &src4[1], factor, add, shift, clip_lo, clip_hi,
        ));
        store_i16x8(dst, vcombine_s16(lo, hi));
    }
}

#[target_feature(enable = "neon")]
fn dequantize_transform_skip_into_neon_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i32],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<4>();
    let (out, _) = out[..16].as_chunks_mut::<4>();

    let factor = params.dequant.factor as i32;
    let add = vdupq_n_s64(params.dequant.add);
    let shift = vdupq_n_s64(-(params.dequant.shift as i64));
    let clip_lo = vdupq_n_s32(-32768);
    let clip_hi = vdupq_n_s32(32767);
    let tr_add = vdupq_n_s32(params.tr_add);
    let tr_shift = vdupq_n_s32(-params.tr_shift);

    let d0 = dequant4_neon_const(&levels[0], factor, add, shift, clip_lo, clip_hi);
    let d1 = dequant4_neon_const(&levels[1], factor, add, shift, clip_lo, clip_hi);
    let d2 = dequant4_neon_const(&levels[2], factor, add, shift, clip_lo, clip_hi);
    let d3 = dequant4_neon_const(&levels[3], factor, add, shift, clip_lo, clip_hi);

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            vshlq_s32(vaddq_s32(d0, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d1, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d2, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d3, tr_add), tr_shift),
        )
    } else {
        (
            vshlq_s32(d0, tr_shift),
            vshlq_s32(d1, tr_shift),
            vshlq_s32(d2, tr_shift),
            vshlq_s32(d3, tr_shift),
        )
    };

    store_i32x4(&mut out[0], clip_i16_s32x4_with(v0, clip_lo, clip_hi));
    store_i32x4(&mut out[1], clip_i16_s32x4_with(v1, clip_lo, clip_hi));
    store_i32x4(&mut out[2], clip_i16_s32x4_with(v2, clip_lo, clip_hi));
    store_i32x4(&mut out[3], clip_i16_s32x4_with(v3, clip_lo, clip_hi));
}

#[target_feature(enable = "neon")]
fn dequantize_transform_skip_into_neon16_impl(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i16],
) {
    debug_assert_eq!(n, 4);
    let (levels, _) = levels[..16].as_chunks::<8>();
    let (out, _) = out[..16].as_chunks_mut::<8>();

    let factor = params.dequant.factor as i32;
    let add = vdupq_n_s64(params.dequant.add);
    let shift = vdupq_n_s64(-(params.dequant.shift as i64));
    let clip_lo = vdupq_n_s32(-32768);
    let clip_hi = vdupq_n_s32(32767);
    let tr_add = vdupq_n_s32(params.tr_add);
    let tr_shift = vdupq_n_s32(-params.tr_shift);

    let (src0, _) = levels[0].as_chunks::<4>();
    let (src1, _) = levels[1].as_chunks::<4>();
    let d0 = dequant4_neon_const(&src0[0], factor, add, shift, clip_lo, clip_hi);
    let d1 = dequant4_neon_const(&src0[1], factor, add, shift, clip_lo, clip_hi);
    let d2 = dequant4_neon_const(&src1[0], factor, add, shift, clip_lo, clip_hi);
    let d3 = dequant4_neon_const(&src1[1], factor, add, shift, clip_lo, clip_hi);

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            vshlq_s32(vaddq_s32(d0, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d1, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d2, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d3, tr_add), tr_shift),
        )
    } else {
        (
            vshlq_s32(d0, tr_shift),
            vshlq_s32(d1, tr_shift),
            vshlq_s32(d2, tr_shift),
            vshlq_s32(d3, tr_shift),
        )
    };

    let lo0 = vqmovn_s32(clip_i16_s32x4_with(v0, clip_lo, clip_hi));
    let hi0 = vqmovn_s32(clip_i16_s32x4_with(v1, clip_lo, clip_hi));
    let lo1 = vqmovn_s32(clip_i16_s32x4_with(v2, clip_lo, clip_hi));
    let hi1 = vqmovn_s32(clip_i16_s32x4_with(v3, clip_lo, clip_hi));
    store_i16x8(&mut out[0], vcombine_s16(lo0, hi0));
    store_i16x8(&mut out[1], vcombine_s16(lo1, hi1));
}

#[target_feature(enable = "neon")]
fn dequantize_scaled_into_neon_impl(
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

    let add = vdupq_n_s64(params.add);
    let shift = vdupq_n_s64(-(params.shift as i64));
    let clip_lo = vdupq_n_s32(-32768);
    let clip_hi = vdupq_n_s32(32767);

    let (level_groups, level_tail) = levels.as_chunks::<4>();
    let (out_groups, out_tail) = out.as_chunks_mut::<4>();

    for (group_idx, (src, dst)) in level_groups.iter().zip(out_groups.iter_mut()).enumerate() {
        let idx = group_idx * 16;
        let f0 = scaling_factors4(base_factor, scaling, idx);
        let f1 = scaling_factors4(base_factor, scaling, idx + 4);
        let f2 = scaling_factors4(base_factor, scaling, idx + 8);
        let f3 = scaling_factors4(base_factor, scaling, idx + 12);
        let v0 = dequant4_scaled_neon_const(&src[0], &f0, add, shift, clip_lo, clip_hi);
        let v1 = dequant4_scaled_neon_const(&src[1], &f1, add, shift, clip_lo, clip_hi);
        let v2 = dequant4_scaled_neon_const(&src[2], &f2, add, shift, clip_lo, clip_hi);
        let v3 = dequant4_scaled_neon_const(&src[3], &f3, add, shift, clip_lo, clip_hi);
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
            dequant4_scaled_neon_const(src, &factors, add, shift, clip_lo, clip_hi),
        );
    }
}

#[target_feature(enable = "neon")]
fn dequantize_scaled_into_neon16_impl(
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

    let add = vdupq_n_s64(params.add);
    let shift = vdupq_n_s64(-(params.shift as i64));
    let clip_lo = vdupq_n_s32(-32768);
    let clip_hi = vdupq_n_s32(32767);

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
        let lo0 = vqmovn_s32(dequant4_scaled_neon_const(
            &src0[0], &f0, add, shift, clip_lo, clip_hi,
        ));
        let hi0 = vqmovn_s32(dequant4_scaled_neon_const(
            &src0[1], &f1, add, shift, clip_lo, clip_hi,
        ));
        let lo1 = vqmovn_s32(dequant4_scaled_neon_const(
            &src1[0], &f2, add, shift, clip_lo, clip_hi,
        ));
        let hi1 = vqmovn_s32(dequant4_scaled_neon_const(
            &src1[1], &f3, add, shift, clip_lo, clip_hi,
        ));
        store_i16x8(&mut dst[0], vcombine_s16(lo0, hi0));
        store_i16x8(&mut dst[1], vcombine_s16(lo1, hi1));
    }

    let tail_base = level_groups.len() * 16;
    for (block_idx, (src, dst)) in level_tail.iter().zip(out_tail.iter_mut()).enumerate() {
        let (src4, _) = src.as_chunks::<4>();
        let idx = tail_base + block_idx * 8;
        let factors_lo = scaling_factors4(base_factor, scaling, idx);
        let factors_hi = scaling_factors4(base_factor, scaling, idx + 4);
        let lo = vqmovn_s32(dequant4_scaled_neon_const(
            &src4[0],
            &factors_lo,
            add,
            shift,
            clip_lo,
            clip_hi,
        ));
        let hi = vqmovn_s32(dequant4_scaled_neon_const(
            &src4[1],
            &factors_hi,
            add,
            shift,
            clip_lo,
            clip_hi,
        ));
        store_i16x8(dst, vcombine_s16(lo, hi));
    }
}

#[target_feature(enable = "neon")]
fn dequantize_transform_skip_scaled_into_neon_impl(
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

    let add = vdupq_n_s64(params.dequant.add);
    let shift = vdupq_n_s64(-(params.dequant.shift as i64));
    let clip_lo = vdupq_n_s32(-32768);
    let clip_hi = vdupq_n_s32(32767);
    let tr_add = vdupq_n_s32(params.tr_add);
    let tr_shift = vdupq_n_s32(-params.tr_shift);

    let f0 = scaling_factors4(base_factor, scaling, 0);
    let f1 = scaling_factors4(base_factor, scaling, 4);
    let f2 = scaling_factors4(base_factor, scaling, 8);
    let f3 = scaling_factors4(base_factor, scaling, 12);
    let d0 = dequant4_scaled_neon_const(&levels[0], &f0, add, shift, clip_lo, clip_hi);
    let d1 = dequant4_scaled_neon_const(&levels[1], &f1, add, shift, clip_lo, clip_hi);
    let d2 = dequant4_scaled_neon_const(&levels[2], &f2, add, shift, clip_lo, clip_hi);
    let d3 = dequant4_scaled_neon_const(&levels[3], &f3, add, shift, clip_lo, clip_hi);

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            vshlq_s32(vaddq_s32(d0, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d1, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d2, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d3, tr_add), tr_shift),
        )
    } else {
        (
            vshlq_s32(d0, tr_shift),
            vshlq_s32(d1, tr_shift),
            vshlq_s32(d2, tr_shift),
            vshlq_s32(d3, tr_shift),
        )
    };

    store_i32x4(&mut out[0], clip_i16_s32x4_with(v0, clip_lo, clip_hi));
    store_i32x4(&mut out[1], clip_i16_s32x4_with(v1, clip_lo, clip_hi));
    store_i32x4(&mut out[2], clip_i16_s32x4_with(v2, clip_lo, clip_hi));
    store_i32x4(&mut out[3], clip_i16_s32x4_with(v3, clip_lo, clip_hi));
}

#[target_feature(enable = "neon")]
fn dequantize_transform_skip_scaled_into_neon16_impl(
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

    let add = vdupq_n_s64(params.dequant.add);
    let shift = vdupq_n_s64(-(params.dequant.shift as i64));
    let clip_lo = vdupq_n_s32(-32768);
    let clip_hi = vdupq_n_s32(32767);
    let tr_add = vdupq_n_s32(params.tr_add);
    let tr_shift = vdupq_n_s32(-params.tr_shift);

    let (src0, _) = levels[0].as_chunks::<4>();
    let (src1, _) = levels[1].as_chunks::<4>();
    let f0 = scaling_factors4(base_factor, scaling, 0);
    let f1 = scaling_factors4(base_factor, scaling, 4);
    let f2 = scaling_factors4(base_factor, scaling, 8);
    let f3 = scaling_factors4(base_factor, scaling, 12);
    let d0 = dequant4_scaled_neon_const(&src0[0], &f0, add, shift, clip_lo, clip_hi);
    let d1 = dequant4_scaled_neon_const(&src0[1], &f1, add, shift, clip_lo, clip_hi);
    let d2 = dequant4_scaled_neon_const(&src1[0], &f2, add, shift, clip_lo, clip_hi);
    let d3 = dequant4_scaled_neon_const(&src1[1], &f3, add, shift, clip_lo, clip_hi);

    let (v0, v1, v2, v3) = if params.tr_shift >= 0 {
        (
            vshlq_s32(vaddq_s32(d0, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d1, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d2, tr_add), tr_shift),
            vshlq_s32(vaddq_s32(d3, tr_add), tr_shift),
        )
    } else {
        (
            vshlq_s32(d0, tr_shift),
            vshlq_s32(d1, tr_shift),
            vshlq_s32(d2, tr_shift),
            vshlq_s32(d3, tr_shift),
        )
    };

    let lo0 = vqmovn_s32(clip_i16_s32x4_with(v0, clip_lo, clip_hi));
    let hi0 = vqmovn_s32(clip_i16_s32x4_with(v1, clip_lo, clip_hi));
    let lo1 = vqmovn_s32(clip_i16_s32x4_with(v2, clip_lo, clip_hi));
    let hi1 = vqmovn_s32(clip_i16_s32x4_with(v3, clip_lo, clip_hi));
    store_i16x8(&mut out[0], vcombine_s16(lo0, hi0));
    store_i16x8(&mut out[1], vcombine_s16(lo1, hi1));
}

pub(crate) fn dequantize_into_neon(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i32],
) {
    if !supported_n(n) {
        crate::transform::dequantize_into_scalar_i32(levels, n, params, out);
        return;
    }
    unsafe { dequantize_into_neon_impl(levels, n, params, out) }
}

pub(crate) fn dequantize_into_neon16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i16],
) {
    if !supported_n(n) {
        crate::transform::dequantize_into_scalar_i16(levels, n, params, out);
        return;
    }
    unsafe { dequantize_into_neon16_impl(levels, n, params, out) }
}

pub(crate) fn dequantize_transform_skip_into_neon(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i32],
) {
    if n != 4 {
        crate::transform::dequantize_transform_skip_into_scalar_i32(levels, n, params, out);
        return;
    }
    unsafe { dequantize_transform_skip_into_neon_impl(levels, n, params, out) }
}

pub(crate) fn dequantize_transform_skip_into_neon16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i16],
) {
    if n != 4 {
        crate::transform::dequantize_transform_skip_into_scalar_i16(levels, n, params, out);
        return;
    }
    unsafe { dequantize_transform_skip_into_neon16_impl(levels, n, params, out) }
}

pub(crate) fn dequantize_scaled_into_neon(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    if !supported_n(n) {
        dequantize_scaled_into_scalar_i32(levels, n, params, scaling, out);
        return;
    }
    unsafe { dequantize_scaled_into_neon_impl(levels, n, params, scaling, out) }
}

pub(crate) fn dequantize_scaled_into_neon16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    if !supported_n(n) {
        dequantize_scaled_into_scalar_i16(levels, n, params, scaling, out);
        return;
    }
    unsafe { dequantize_scaled_into_neon16_impl(levels, n, params, scaling, out) }
}

pub(crate) fn dequantize_transform_skip_scaled_into_neon(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    if n != 4 {
        dequantize_transform_skip_scaled_into_scalar_i32(levels, n, params, scaling, out);
        return;
    }
    unsafe { dequantize_transform_skip_scaled_into_neon_impl(levels, n, params, scaling, out) }
}

pub(crate) fn dequantize_transform_skip_scaled_into_neon16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    if n != 4 {
        dequantize_transform_skip_scaled_into_scalar_i16(levels, n, params, scaling, out);
        return;
    }
    unsafe { dequantize_transform_skip_scaled_into_neon16_impl(levels, n, params, scaling, out) }
}
