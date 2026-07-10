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

use crate::reconstruct::{
    add_residual_into_scalar, add_residual_into_scalar16, can_reconstruct_full_block,
    narrow_i32_to_i16_scalar, sample_max,
};

#[inline]
fn supported_n(n: usize) -> bool {
    matches!(n, 2 | 4 | 8 | 16 | 32)
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x2(src: &[u16]) -> uint16x8_t {
    debug_assert!(src.len() >= 2);
    unsafe {
        let lane = vld1q_lane_u32::<0>(src.as_ptr().cast(), vdupq_n_u32(0));
        vreinterpretq_u16_u32(lane)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x4(src: &[u16]) -> uint16x8_t {
    debug_assert!(src.len() >= 4);
    unsafe { vcombine_u16(vld1_u16(src.as_ptr()), vdup_n_u16(0)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x8(src: &[u16]) -> uint16x8_t {
    debug_assert!(src.len() >= 8);
    unsafe { vld1q_u16(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x2(src: &[i32]) -> int32x4_t {
    debug_assert!(src.len() >= 2);
    unsafe { vcombine_s32(vld1_s32(src.as_ptr()), vdup_n_s32(0)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x4(src: &[i32]) -> int32x4_t {
    debug_assert!(src.len() >= 4);
    unsafe { vld1q_s32(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x2(dst: &mut [u16], v: uint16x8_t) {
    debug_assert!(dst.len() >= 2);
    unsafe {
        vst1q_lane_u32::<0>(dst.as_mut_ptr().cast(), vreinterpretq_u32_u16(v));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x4(dst: &mut [u16], v: uint16x8_t) {
    debug_assert!(dst.len() >= 4);
    unsafe { vst1_u16(dst.as_mut_ptr(), vget_low_u16(v)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x8(dst: &mut [u16], v: uint16x8_t) {
    debug_assert!(dst.len() >= 8);
    unsafe { vst1q_u16(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn widen_lo_u16x4(v: uint16x8_t) -> int32x4_t {
    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v)))
}

#[inline]
#[target_feature(enable = "neon")]
fn widen_hi_u16x4(v: uint16x8_t) -> int32x4_t {
    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v)))
}

#[inline]
#[target_feature(enable = "neon")]
fn pack_u16x8(lo: int32x4_t, hi: int32x4_t) -> uint16x8_t {
    vcombine_u16(
        vqmovn_u32(vreinterpretq_u32_s32(lo)),
        vqmovn_u32(vreinterpretq_u32_s32(hi)),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn add_clip4_neon(pred: int32x4_t, res: int32x4_t, zero: int32x4_t, max: int32x4_t) -> int32x4_t {
    let sum = vaddq_s32(pred, res);
    vminq_s32(vmaxq_s32(sum, zero), max)
}

#[inline]
#[target_feature(enable = "neon")]
fn add_clip8_neon(pred: uint16x8_t, res: &[i32; 8], zero: int32x4_t, max: int32x4_t) -> uint16x8_t {
    let (res4, _) = res.as_chunks::<4>();
    let lo = add_clip4_neon(widen_lo_u16x4(pred), load_i32x4(&res4[0]), zero, max);
    let hi = add_clip4_neon(widen_hi_u16x4(pred), load_i32x4(&res4[1]), zero, max);
    pack_u16x8(lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn add_clip_row_neon(dst: &mut [u16], pred: &[u16], res: &[i32], n: usize, max: int32x4_t) {
    let zero = vdupq_n_s32(0);

    if n == 2 {
        let pred = widen_lo_u16x4(load_u16x2(pred));
        let out = add_clip4_neon(pred, load_i32x2(res), zero, max);
        store_u16x2(dst, pack_u16x8(out, zero));
        return;
    }

    if n == 4 {
        let pred = widen_lo_u16x4(load_u16x4(pred));
        let out = add_clip4_neon(pred, load_i32x4(res), zero, max);
        store_u16x4(dst, pack_u16x8(out, zero));
        return;
    }

    let (pred8, _) = pred[..n].as_chunks::<8>();
    let (res8, _) = res[..n].as_chunks::<8>();
    let (dst8, _) = dst[..n].as_chunks_mut::<8>();

    for ((pred, res), dst) in pred8.iter().zip(res8.iter()).zip(dst8.iter_mut()) {
        let out = add_clip8_neon(load_u16x8(pred), res, zero, max);
        store_u16x8(dst, out);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn add_residual_into_neon_impl(
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
    let max = vdupq_n_s32(sample_max(bit_depth));

    let dst_rows = dst.chunks_mut(stride).take(n);
    let pred_rows = pred.chunks_exact(n);
    let res_rows = res.chunks_exact(n);
    for ((dst_row, pred_row), res_row) in dst_rows.zip(pred_rows).zip(res_rows) {
        add_clip_row_neon(&mut dst_row[..n], pred_row, res_row, n, max);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_residual_into_neon(
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

    unsafe { add_residual_into_neon_impl(dst, stride, pred, res, n, bit_depth) }
}

// 8-bit i16-residual path: saturating i16 adds, 8 px per op.

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x2(src: &[i16]) -> int16x4_t {
    let (src2, _) = src.as_chunks::<2>();
    debug_assert!(!src2.is_empty());
    let src = src2[0];
    let v = vset_lane_s16::<0>(src[0], vdup_n_s16(0));
    vset_lane_s16::<1>(src[1], v)
}

/// Saturating add equals widen+clamp because the result is clamped to max <= 32767.
#[inline]
#[target_feature(enable = "neon")]
fn add_clip_i16q(pred: int16x8_t, res: int16x8_t, zero: int16x8_t, max: int16x8_t) -> int16x8_t {
    vminq_s16(vmaxq_s16(vqaddq_s16(pred, res), zero), max)
}

#[inline]
#[target_feature(enable = "neon")]
fn add_clip_i16(pred: int16x4_t, res: int16x4_t, max: int16x4_t) -> int16x4_t {
    vmin_s16(vmax_s16(vqadd_s16(pred, res), vdup_n_s16(0)), max)
}

#[inline]
#[target_feature(enable = "neon")]
fn add_clip_row_neon_16(dst: &mut [u16], pred: &[u16], res: &[i16], n: usize, max: int16x8_t) {
    let zero = vdupq_n_s16(0);

    if n == 2 {
        let p = vget_low_s16(vreinterpretq_s16_u16(load_u16x2(pred)));
        let s = add_clip_i16(p, load_i16x2(res), vget_low_s16(max));
        store_u16x2(dst, vreinterpretq_u16_s16(vcombine_s16(s, vdup_n_s16(0))));
        return;
    }
    if n == 4 {
        let p = vget_low_s16(vreinterpretq_s16_u16(load_u16x4(pred)));
        let s = add_clip_i16(p, unsafe { vld1_s16(res.as_ptr()) }, vget_low_s16(max));
        unsafe { vst1_u16(dst.as_mut_ptr(), vreinterpret_u16_s16(s)) };
        return;
    }

    let (pred8, _) = pred[..n].as_chunks::<8>();
    let (res8, _) = res[..n].as_chunks::<8>();
    let (dst8, _) = dst[..n].as_chunks_mut::<8>();

    for ((pred, res), dst) in pred8.iter().zip(res8.iter()).zip(dst8.iter_mut()) {
        let p = vreinterpretq_s16_u16(load_u16x8(pred));
        let r = unsafe { vld1q_s16(res.as_ptr()) };
        let s = add_clip_i16q(p, r, zero, max);
        store_u16x8(dst, vreinterpretq_u16_s16(s));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn add_residual_into_neon_impl_16(
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
    let max = vdupq_n_s16(sample_max(bit_depth) as i16);

    let dst_rows = dst.chunks_mut(stride).take(n);
    let pred_rows = pred.chunks_exact(n);
    let res_rows = res.chunks_exact(n);
    for ((dst_row, pred_row), res_row) in dst_rows.zip(pred_rows).zip(res_rows) {
        add_clip_row_neon_16(&mut dst_row[..n], pred_row, res_row, n, max);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_residual_into_neon16(
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

    unsafe { add_residual_into_neon_impl_16(dst, stride, pred, res, n, bit_depth) }
}

#[inline]
#[target_feature(enable = "neon")]
fn narrow_i32x8_neon(src: &[i32; 8]) -> int16x8_t {
    let (src4, _) = src.as_chunks::<4>();
    vcombine_s16(
        vqmovn_s32(load_i32x4(&src4[0])),
        vqmovn_s32(load_i32x4(&src4[1])),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn narrow_i32_to_i16_neon_impl(src: &[i32], dst: &mut [i16]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() & 7, 0);

    let (src32, src_rem) = src.as_chunks::<32>();
    let (dst32, dst_rem) = dst.as_chunks_mut::<32>();
    for (src, dst) in src32.iter().zip(dst32.iter_mut()) {
        let (src8, _) = src.as_chunks::<8>();
        let (dst8, _) = dst.as_chunks_mut::<8>();

        let p0 = narrow_i32x8_neon(&src8[0]);
        let p1 = narrow_i32x8_neon(&src8[1]);
        let p2 = narrow_i32x8_neon(&src8[2]);
        let p3 = narrow_i32x8_neon(&src8[3]);
        unsafe {
            vst1q_s16(dst8[0].as_mut_ptr(), p0);
            vst1q_s16(dst8[1].as_mut_ptr(), p1);
            vst1q_s16(dst8[2].as_mut_ptr(), p2);
            vst1q_s16(dst8[3].as_mut_ptr(), p3);
        }
    }

    let (src8, src_tail) = src_rem.as_chunks::<8>();
    let (dst8, dst_tail) = dst_rem.as_chunks_mut::<8>();
    debug_assert!(src_tail.is_empty());
    debug_assert!(dst_tail.is_empty());
    for (src, dst) in src8.iter().zip(dst8.iter_mut()) {
        let packed = narrow_i32x8_neon(src);
        unsafe { vst1q_s16(dst.as_mut_ptr(), packed) };
    }
}

pub(crate) fn narrow_i32_to_i16_neon(src: &[i32], dst: &mut [i16]) {
    let len = src.len().min(dst.len());
    let simd_len = len & !7;
    let (src_simd, src_tail) = src[..len].split_at(simd_len);
    let (dst_simd, dst_tail) = dst[..len].split_at_mut(simd_len);

    if simd_len != 0 {
        unsafe { narrow_i32_to_i16_neon_impl(src_simd, dst_simd) };
    }
    narrow_i32_to_i16_scalar(src_tail, dst_tail);
}
