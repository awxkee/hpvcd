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

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn load_s32x4(src: &[i32]) -> int32x4_t {
    debug_assert!(src.len() >= 4);
    unsafe { vld1q_s32(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn store_s32x4(dst: &mut [i32], v: int32x4_t) {
    debug_assert!(dst.len() >= 4);
    unsafe { vst1q_s32(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn ld_i16x4(src: &[i16]) -> int16x4_t {
    debug_assert!(src.len() >= 4);
    unsafe { vld1_s16(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn st_i16x4(dst: &mut [i16], v: int16x4_t) {
    debug_assert!(dst.len() >= 4);
    unsafe { vst1_s16(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn zero() -> int32x4_t {
    vdupq_n_s32(0)
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn add(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vaddq_s32(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn sub(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vsubq_s32(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn mul_const(v: int32x4_t, c: i32) -> int32x4_t {
    vmulq_s32(v, vdupq_n_s32(c))
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn madd_const(acc: int32x4_t, v: int32x4_t, c: i32) -> int32x4_t {
    add(acc, mul_const(v, c))
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn round_shift_s32x4(v: int32x4_t, add: i32, shift: i32) -> int32x4_t {
    vshlq_s32(vaddq_s32(v, vdupq_n_s32(add)), vdupq_n_s32(-shift))
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn round_shift_clip_i16_s32x4(v: int32x4_t, add: i32, shift: i32) -> int32x4_t {
    let v = round_shift_s32x4(v, add, shift);
    vmaxq_s32(vminq_s32(v, vdupq_n_s32(32767)), vdupq_n_s32(-32768))
}
