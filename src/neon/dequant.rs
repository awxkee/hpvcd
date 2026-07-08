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
fn clip_i16_s32x4(v: int32x4_t) -> int32x4_t {
    vminq_s32(vmaxq_s32(v, vdupq_n_s32(-32768)), vdupq_n_s32(32767))
}

#[inline]
#[target_feature(enable = "neon")]
fn dequant4_neon(levels: &[i32; 4], params: DequantParams) -> int32x4_t {
    let v = load_i32x4(levels);
    let factor = params.factor as i32;
    let add = vdupq_n_s64(params.add);
    let shift = vdupq_n_s64(-(params.shift as i64));

    let lo = vshlq_s64(vaddq_s64(vmull_n_s32(vget_low_s32(v), factor), add), shift);
    let hi = vshlq_s64(vaddq_s64(vmull_n_s32(vget_high_s32(v), factor), add), shift);
    clip_i16_s32x4(vcombine_s32(vqmovn_s64(lo), vqmovn_s64(hi)))
}

#[inline]
#[target_feature(enable = "neon")]
fn dequant4_scaled_neon(levels: &[i32; 4], factors: &[i32; 4], params: DequantParams) -> int32x4_t {
    let v = load_i32x4(levels);
    let f = load_i32x4(factors);
    let add = vdupq_n_s64(params.add);
    let shift = vdupq_n_s64(-(params.shift as i64));

    let lo = vshlq_s64(
        vaddq_s64(vmull_s32(vget_low_s32(v), vget_low_s32(f)), add),
        shift,
    );
    let hi = vshlq_s64(
        vaddq_s64(vmull_s32(vget_high_s32(v), vget_high_s32(f)), add),
        shift,
    );
    clip_i16_s32x4(vcombine_s32(vqmovn_s64(lo), vqmovn_s64(hi)))
}

#[inline]
#[target_feature(enable = "neon")]
fn transform_skip4_neon(levels: &[i32; 4], params: TransformSkipParams) -> int32x4_t {
    let deq = dequant4_neon(levels, params.dequant);
    let shifted = if params.tr_shift >= 0 {
        let v = vaddq_s32(deq, vdupq_n_s32(params.tr_add));
        vshlq_s32(v, vdupq_n_s32(-params.tr_shift))
    } else {
        vshlq_s32(deq, vdupq_n_s32(-params.tr_shift))
    };
    clip_i16_s32x4(shifted)
}

#[inline]
#[target_feature(enable = "neon")]
fn transform_skip4_scaled_neon(
    levels: &[i32; 4],
    factors: &[i32; 4],
    params: TransformSkipParams,
) -> int32x4_t {
    let deq = dequant4_scaled_neon(levels, factors, params.dequant);
    let shifted = if params.tr_shift >= 0 {
        let v = vaddq_s32(deq, vdupq_n_s32(params.tr_add));
        vshlq_s32(v, vdupq_n_s32(-params.tr_shift))
    } else {
        vshlq_s32(deq, vdupq_n_s32(-params.tr_shift))
    };
    clip_i16_s32x4(shifted)
}

#[target_feature(enable = "neon")]
fn dequantize_into_neon_impl(levels: &[i32], n: usize, params: DequantParams, out: &mut [i32]) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<4>();
    let (out, _) = out[..count].as_chunks_mut::<4>();

    for (src, dst) in levels.iter().zip(out.iter_mut()) {
        store_i32x4(dst, dequant4_neon(src, params));
    }
}

#[target_feature(enable = "neon")]
fn dequantize_into_neon16_impl(levels: &[i32], n: usize, params: DequantParams, out: &mut [i16]) {
    let count = n * n;
    let (levels, _) = levels[..count].as_chunks::<8>();
    let (out, _) = out[..count].as_chunks_mut::<8>();

    for (src, dst) in levels.iter().zip(out.iter_mut()) {
        let (src4, _) = src.as_chunks::<4>();
        let lo = vqmovn_s32(dequant4_neon(&src4[0], params));
        let hi = vqmovn_s32(dequant4_neon(&src4[1], params));
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

    for (src, dst) in levels.iter().zip(out.iter_mut()) {
        store_i32x4(dst, transform_skip4_neon(src, params));
    }
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

    for (src, dst) in levels.iter().zip(out.iter_mut()) {
        let (src4, _) = src.as_chunks::<4>();
        let lo = vqmovn_s32(transform_skip4_neon(&src4[0], params));
        let hi = vqmovn_s32(transform_skip4_neon(&src4[1], params));
        store_i16x8(dst, vcombine_s16(lo, hi));
    }
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

    for (block_idx, (src, dst)) in levels.iter().zip(out.iter_mut()).enumerate() {
        let factors = scaling_factors4(base_factor, scaling, block_idx * 4);
        store_i32x4(dst, dequant4_scaled_neon(src, &factors, params));
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

    for (block_idx, (src, dst)) in levels.iter().zip(out.iter_mut()).enumerate() {
        let (src4, _) = src.as_chunks::<4>();
        let idx = block_idx * 8;
        let factors_lo = scaling_factors4(base_factor, scaling, idx);
        let factors_hi = scaling_factors4(base_factor, scaling, idx + 4);
        let lo = vqmovn_s32(dequant4_scaled_neon(&src4[0], &factors_lo, params));
        let hi = vqmovn_s32(dequant4_scaled_neon(&src4[1], &factors_hi, params));
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

    for (block_idx, (src, dst)) in levels.iter().zip(out.iter_mut()).enumerate() {
        let factors = scaling_factors4(base_factor, scaling, block_idx * 4);
        store_i32x4(dst, transform_skip4_scaled_neon(src, &factors, params));
    }
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

    for (block_idx, (src, dst)) in levels.iter().zip(out.iter_mut()).enumerate() {
        let (src4, _) = src.as_chunks::<4>();
        let idx = block_idx * 8;
        let factors_lo = scaling_factors4(base_factor, scaling, idx);
        let factors_hi = scaling_factors4(base_factor, scaling, idx + 4);
        let lo = vqmovn_s32(transform_skip4_scaled_neon(&src4[0], &factors_lo, params));
        let hi = vqmovn_s32(transform_skip4_scaled_neon(&src4[1], &factors_hi, params));
        store_i16x8(dst, vcombine_s16(lo, hi));
    }
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
