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

use crate::reconstruct::add_residual_into_scalar;

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
fn add_clip8_neon(pred: uint16x8_t, res: &[i32], zero: int32x4_t, max: int32x4_t) -> uint16x8_t {
    let lo = add_clip4_neon(widen_lo_u16x4(pred), load_i32x4(res), zero, max);
    let hi = add_clip4_neon(widen_hi_u16x4(pred), load_i32x4(&res[4..]), zero, max);
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

    let mut x = 0usize;
    while x < n {
        let pred = load_u16x8(&pred[x..]);
        let out = add_clip8_neon(pred, &res[x..], zero, max);
        store_u16x8(&mut dst[x..], out);
        x += 8;
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
    let pred = &pred[..n * n];
    let res = &res[..n * n];
    let max = vdupq_n_s32((1i32 << bit_depth) - 1);

    for y in 0..n {
        let row_off = y * n;
        let dst_off = y * stride;
        add_clip_row_neon(
            &mut dst[dst_off..dst_off + n],
            &pred[row_off..row_off + n],
            &res[row_off..row_off + n],
            n,
            max,
        );
    }
}

pub(crate) fn add_residual_into_neon(
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

    unsafe { add_residual_into_neon_impl(dst, stride, pred, res, n, bit_depth) }
}
