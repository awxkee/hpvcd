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

use crate::intra::{DC, INTRA_PRED_ANGLE, INV_ANGLE, PLANAR, predict_into_scalar};

#[inline]
fn supported_n(n: usize) -> bool {
    matches!(n, 2 | 4 | 8 | 16 | 32)
}

#[inline]
fn supported_predictor(mode: u8, n: usize) -> bool {
    supported_n(n) && (mode == PLANAR || mode == DC || (2..=34).contains(&mode))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x8(src: &[u16]) -> uint16x8_t {
    debug_assert!(src.len() >= 8);
    unsafe { vld1q_u16(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x4(src: &[u16]) -> uint16x8_t {
    debug_assert!(src.len() >= 4);
    unsafe { vcombine_u16(vld1_u16(src.as_ptr()), vdup_n_u16(0)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x8(dst: &mut [u16], v: uint16x8_t) {
    debug_assert!(dst.len() >= 8);
    unsafe { vst1q_u16(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x4(dst: &mut [u16], v: uint16x8_t) {
    debug_assert!(dst.len() >= 4);
    unsafe { vst1_u16(dst.as_mut_ptr(), vget_low_u16(v)) }
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
fn mul_const(v: int32x4_t, c: i32) -> int32x4_t {
    vmulq_s32(v, vdupq_n_s32(c))
}

#[inline]
#[target_feature(enable = "neon")]
fn round_shift_s32x4(v: int32x4_t, add: i32, shift: i32) -> int32x4_t {
    vshlq_s32(vaddq_s32(v, vdupq_n_s32(add)), vdupq_n_s32(-shift))
}

#[inline]
#[target_feature(enable = "neon")]
fn pack_u16x8(lo: int32x4_t, hi: int32x4_t) -> uint16x8_t {
    vcombine_u16(
        vqmovn_u32(vreinterpretq_u32_s32(lo)),
        vqmovn_u32(vreinterpretq_u32_s32(hi)),
    )
}

const X_LANE_OFFSETS: [i32; 4] = [1, 2, 3, 4];
const NX_LANE_OFFSETS: [i32; 4] = [0, 1, 2, 3];

#[inline]
#[target_feature(enable = "neon")]
fn x_lane_offsets() -> int32x4_t {
    unsafe { vld1q_s32(X_LANE_OFFSETS.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn nx_lane_offsets() -> int32x4_t {
    unsafe { vld1q_s32(NX_LANE_OFFSETS.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn x_lanes(base: usize) -> int32x4_t {
    vaddq_s32(vdupq_n_s32(base as i32), x_lane_offsets())
}

#[inline]
#[target_feature(enable = "neon")]
fn nx_lanes(n: usize, base: usize) -> int32x4_t {
    vsubq_s32(vdupq_n_s32((n - 1 - base) as i32), nx_lane_offsets())
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
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn planar4_neon(
    above: int32x4_t,
    x: usize,
    n: usize,
    left_y: i32,
    tr: i32,
    bl_part: i32,
    ny: i32,
    add: i32,
    shift: i32,
) -> int32x4_t {
    let h = vaddq_s32(
        vmulq_s32(nx_lanes(n, x), vdupq_n_s32(left_y)),
        mul_const(x_lanes(x), tr),
    );
    let v = vaddq_s32(mul_const(above, ny), vdupq_n_s32(bl_part));
    round_shift_s32x4(vaddq_s32(h, v), add, shift)
}

#[inline]
#[target_feature(enable = "neon")]
fn store_planar_row_neon(
    out: &mut [u16],
    above: &[u16],
    left_y: i32,
    tr: i32,
    bl: i32,
    y: usize,
    n: usize,
) {
    let shift = n.trailing_zeros() as i32 + 1;
    let add = n as i32;
    let ny = (n - 1 - y) as i32;
    let bl_part = (y + 1) as i32 * bl;

    if n == 2 {
        let a = load_u16x4(&above[1..]);
        let lo = planar4_neon(widen_lo_u16x4(a), 0, n, left_y, tr, bl_part, ny, add, shift);
        store_u16x2(out, pack_u16x8(lo, vdupq_n_s32(0)));
        return;
    }

    let above_row = &above[1..1 + n];
    let (above16, above_rem) = above_row.as_chunks::<16>();
    let (out16, out_rem) = out[..n].as_chunks_mut::<16>();
    for (block, (a, dst)) in above16.iter().zip(out16.iter_mut()).enumerate() {
        let x = block * 16;
        let a0 = load_u16x8(&a[..]);
        let a1 = load_u16x8(&a[8..]);
        let lo0 = planar4_neon(
            widen_lo_u16x4(a0),
            x,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        let hi0 = planar4_neon(
            widen_hi_u16x4(a0),
            x + 4,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        let lo1 = planar4_neon(
            widen_lo_u16x4(a1),
            x + 8,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        let hi1 = planar4_neon(
            widen_hi_u16x4(a1),
            x + 12,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        store_u16x8(&mut dst[..], pack_u16x8(lo0, hi0));
        store_u16x8(&mut dst[8..], pack_u16x8(lo1, hi1));
    }

    let x8_base = above16.len() * 16;
    let (above8, above_tail) = above_rem.as_chunks::<8>();
    let (out8, out_tail) = out_rem.as_chunks_mut::<8>();
    for (block, (a, dst)) in above8.iter().zip(out8.iter_mut()).enumerate() {
        let x = x8_base + block * 8;
        let a = load_u16x8(&a[..]);
        let lo = planar4_neon(widen_lo_u16x4(a), x, n, left_y, tr, bl_part, ny, add, shift);
        let hi = planar4_neon(
            widen_hi_u16x4(a),
            x + 4,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        store_u16x8(&mut dst[..], pack_u16x8(lo, hi));
    }

    if !above_tail.is_empty() {
        let x = x8_base + above8.len() * 8;
        let a = load_u16x8(&above[1 + x..]);
        let lo = planar4_neon(widen_lo_u16x4(a), x, n, left_y, tr, bl_part, ny, add, shift);
        store_u16x4(out_tail, pack_u16x8(lo, vdupq_n_s32(0)));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn fill_row_neon(dst: &mut [u16], n: usize, v: u16) {
    let v = vdupq_n_u16(v);
    if n == 2 {
        store_u16x2(dst, v);
    } else if n == 4 {
        store_u16x4(dst, v);
    } else {
        let (dst16, dst_rem) = dst[..n].as_chunks_mut::<16>();
        for block in dst16 {
            store_u16x8(&mut block[..], v);
            store_u16x8(&mut block[8..], v);
        }
        let (dst8, _) = dst_rem.as_chunks_mut::<8>();
        for block in dst8 {
            store_u16x8(&mut block[..], v);
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn copy_row_neon(dst: &mut [u16], src: &[u16], n: usize) {
    if n == 2 {
        let v = load_u16x4(src);
        store_u16x2(dst, v);
    } else if n == 4 {
        let v = load_u16x8(src);
        store_u16x4(dst, v);
    } else {
        let (src16, src_rem) = src[..n].as_chunks::<16>();
        let (dst16, dst_rem) = dst[..n].as_chunks_mut::<16>();
        for (src, dst) in src16.iter().zip(dst16.iter_mut()) {
            let v0 = load_u16x8(&src[..]);
            let v1 = load_u16x8(&src[8..]);
            store_u16x8(&mut dst[..], v0);
            store_u16x8(&mut dst[8..], v1);
        }
        let (src8, _) = src_rem.as_chunks::<8>();
        let (dst8, _) = dst_rem.as_chunks_mut::<8>();
        for (src, dst) in src8.iter().zip(dst8.iter_mut()) {
            let v = load_u16x8(&src[..]);
            store_u16x8(&mut dst[..], v);
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x4(src: &[i32]) -> int32x4_t {
    debug_assert!(src.len() >= 4);
    unsafe { vld1q_s32(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4_s32(v: [int32x4_t; 4]) -> [int32x4_t; 4] {
    let ab = vtrnq_s32(v[0], v[1]);
    let cd = vtrnq_s32(v[2], v[3]);
    [
        vcombine_s32(vget_low_s32(ab.0), vget_low_s32(cd.0)),
        vcombine_s32(vget_low_s32(ab.1), vget_low_s32(cd.1)),
        vcombine_s32(vget_high_s32(ab.0), vget_high_s32(cd.0)),
        vcombine_s32(vget_high_s32(ab.1), vget_high_s32(cd.1)),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn clip_s32x4(v: int32x4_t, max_v: int32x4_t) -> int32x4_t {
    vminq_s32(vmaxq_s32(v, vdupq_n_s32(0)), max_v)
}

#[inline]
#[target_feature(enable = "neon")]
fn round_weighted_s32x4(r0: int32x4_t, r1: int32x4_t, frac: i32) -> int32x4_t {
    let a = vmulq_s32(r0, vdupq_n_s32(32 - frac));
    let b = vmulq_s32(r1, vdupq_n_s32(frac));
    vshrq_n_s32::<5>(vaddq_s32(vaddq_s32(a, b), vdupq_n_s32(16)))
}

#[inline]
#[target_feature(enable = "neon")]
fn round_weighted_var_s32x4(r0: int32x4_t, r1: int32x4_t, frac: int32x4_t) -> int32x4_t {
    let inv_frac = vsubq_s32(vdupq_n_s32(32), frac);
    let a = vmulq_s32(r0, inv_frac);
    let b = vmulq_s32(r1, frac);
    vshrq_n_s32::<5>(vaddq_s32(vaddq_s32(a, b), vdupq_n_s32(16)))
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4_as_u16(dst: &mut [u16], v: int32x4_t, max_v: int32x4_t) {
    let v = clip_s32x4(v, max_v);
    store_u16x4(dst, pack_u16x8(v, vdupq_n_s32(0)));
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x2_as_u16(dst: &mut [u16], v: int32x4_t, max_v: int32x4_t) {
    let v = clip_s32x4(v, max_v);
    store_u16x2(dst, pack_u16x8(v, vdupq_n_s32(0)));
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x8_as_u16(dst: &mut [u16], lo: int32x4_t, hi: int32x4_t, max_v: int32x4_t) {
    let lo = clip_s32x4(lo, max_v);
    let hi = clip_s32x4(hi, max_v);
    store_u16x8(dst, pack_u16x8(lo, hi));
}

fn prepare_angular_refs_neon(
    mode: u8,
    above: &[u16],
    left: &[u16],
    n: usize,
    refs_ang: &mut [i32],
) -> (i32, bool) {
    let angle = INTRA_PRED_ANGLE[mode as usize - 2];
    let vertical = mode >= 18;
    let main = if vertical { above } else { left };
    let side = if vertical { left } else { above };
    let refs = &mut refs_ang[..3 * n + 1];

    for (dst, &main) in refs[n..=3 * n].iter_mut().zip(main[..=2 * n].iter()) {
        *dst = main as i32;
    }
    if angle < 0 {
        let inv = INV_ANGLE[mode as usize - 11];
        let lim = (n as i32 * angle) >> 5;
        let mut k = -1i32;
        while k >= lim {
            let idx = ((k * inv + 128) >> 8).min(2 * n as i32);
            refs[(k + n as i32) as usize] = side[idx.max(0) as usize] as i32;
            k -= 1;
        }
    }

    (angle, vertical)
}

#[inline]
#[target_feature(enable = "neon")]
fn set2_i32x4_neon(a: i32, b: i32) -> int32x4_t {
    let v = vdupq_n_s32(0);
    let v = vsetq_lane_s32::<0>(a, v);
    vsetq_lane_s32::<1>(b, v)
}

#[inline]
#[target_feature(enable = "neon")]
fn predict_angular_vertical2_row_neon(
    dst: &mut [u16],
    refs: &[i32],
    n: usize,
    row: usize,
    angle: i32,
    max_v: int32x4_t,
) {
    debug_assert_eq!(n, 2);
    let base = n as i32;
    let pos = (row as i32 + 1) * angle;
    let i_idx = pos >> 5;
    let frac = pos & 31;
    let idx = (i_idx + 1 + base) as usize;
    let r0 = set2_i32x4_neon(refs[idx], refs[idx + 1]);
    let v = if frac == 0 {
        r0
    } else {
        let r1 = set2_i32x4_neon(refs[idx + 1], refs[idx + 2]);
        round_weighted_s32x4(r0, r1, frac)
    };
    store_i32x2_as_u16(dst, v, max_v);
}

#[inline]
#[target_feature(enable = "neon")]
fn predict_angular_vertical_row_neon(
    dst: &mut [u16],
    refs: &[i32],
    n: usize,
    row: usize,
    angle: i32,
    max_v: int32x4_t,
) {
    let base = n as i32;
    let pos = (row as i32 + 1) * angle;
    let i_idx = pos >> 5;
    let frac = pos & 31;
    let (dst16, dst_rem) = dst[..n].as_chunks_mut::<16>();
    for (block, dst) in dst16.iter_mut().enumerate() {
        let x = block * 16;
        let idx = (x as i32 + i_idx + 1 + base) as usize;
        let r0_0 = load_i32x4(&refs[idx..]);
        let r0_1 = load_i32x4(&refs[idx + 4..]);
        let r0_2 = load_i32x4(&refs[idx + 8..]);
        let r0_3 = load_i32x4(&refs[idx + 12..]);
        let (v0, v1, v2, v3) = if frac == 0 {
            (r0_0, r0_1, r0_2, r0_3)
        } else {
            let r1_0 = load_i32x4(&refs[idx + 1..]);
            let r1_1 = load_i32x4(&refs[idx + 5..]);
            let r1_2 = load_i32x4(&refs[idx + 9..]);
            let r1_3 = load_i32x4(&refs[idx + 13..]);
            (
                round_weighted_s32x4(r0_0, r1_0, frac),
                round_weighted_s32x4(r0_1, r1_1, frac),
                round_weighted_s32x4(r0_2, r1_2, frac),
                round_weighted_s32x4(r0_3, r1_3, frac),
            )
        };
        store_i32x8_as_u16(&mut dst[..], v0, v1, max_v);
        store_i32x8_as_u16(&mut dst[8..], v2, v3, max_v);
    }

    let x8_base = dst16.len() * 16;
    let (dst8, dst_tail) = dst_rem.as_chunks_mut::<8>();
    for (block, dst) in dst8.iter_mut().enumerate() {
        let x = x8_base + block * 8;
        let idx = (x as i32 + i_idx + 1 + base) as usize;
        let r0_lo = load_i32x4(&refs[idx..]);
        let r0_hi = load_i32x4(&refs[idx + 4..]);
        let (lo, hi) = if frac == 0 {
            (r0_lo, r0_hi)
        } else {
            let r1_lo = load_i32x4(&refs[idx + 1..]);
            let r1_hi = load_i32x4(&refs[idx + 5..]);
            (
                round_weighted_s32x4(r0_lo, r1_lo, frac),
                round_weighted_s32x4(r0_hi, r1_hi, frac),
            )
        };
        store_i32x8_as_u16(&mut dst[..], lo, hi, max_v);
    }

    if !dst_tail.is_empty() {
        let x = x8_base + dst8.len() * 8;
        let idx = (x as i32 + i_idx + 1 + base) as usize;
        let r0 = load_i32x4(&refs[idx..]);
        let v = if frac == 0 {
            r0
        } else {
            let r1 = load_i32x4(&refs[idx + 1..]);
            round_weighted_s32x4(r0, r1, frac)
        };
        store_i32x4_as_u16(dst_tail, v, max_v);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn angular_horizontal_col4_neon(
    refs: &[i32],
    n: usize,
    y: usize,
    x: usize,
    angle: i32,
) -> int32x4_t {
    let base = n as i32;
    let pos = (x as i32 + 1) * angle;
    let i_idx = pos >> 5;
    let frac = pos & 31;
    let idx = (y as i32 + i_idx + 1 + base) as usize;
    let r0 = load_i32x4(&refs[idx..]);
    if frac == 0 {
        r0
    } else {
        let r1 = load_i32x4(&refs[idx + 1..]);
        round_weighted_s32x4(r0, r1, frac)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_angular_horizontal4x4_neon(
    rows: &mut [u16],
    refs: &[i32],
    n: usize,
    y: usize,
    x: usize,
    angle: i32,
    max_v: int32x4_t,
) {
    debug_assert!(rows.len() >= 4 * n);
    debug_assert!(x + 4 <= n);
    let c0 = angular_horizontal_col4_neon(refs, n, y, x, angle);
    let c1 = angular_horizontal_col4_neon(refs, n, y, x + 1, angle);
    let c2 = angular_horizontal_col4_neon(refs, n, y, x + 2, angle);
    let c3 = angular_horizontal_col4_neon(refs, n, y, x + 3, angle);
    let r = transpose_4x4_s32([c0, c1, c2, c3]);
    store_i32x4_as_u16(&mut rows[x..], r[0], max_v);
    store_i32x4_as_u16(&mut rows[n + x..], r[1], max_v);
    store_i32x4_as_u16(&mut rows[2 * n + x..], r[2], max_v);
    store_i32x4_as_u16(&mut rows[3 * n + x..], r[3], max_v);
}

#[inline]
#[target_feature(enable = "neon")]
fn angular_horizontal2_neon(refs: &[i32], n: usize, y: usize, angle: i32) -> int32x4_t {
    debug_assert_eq!(n, 2);
    let base = n as i32;
    let pos0 = angle;
    let i0 = pos0 >> 5;
    let f0 = pos0 & 31;
    let idx0 = (y as i32 + i0 + 1 + base) as usize;

    let pos1 = angle << 1;
    let i1 = pos1 >> 5;
    let f1 = pos1 & 31;
    let idx1 = (y as i32 + i1 + 1 + base) as usize;

    let r0 = set2_i32x4_neon(refs[idx0], refs[idx1]);
    let r1 = set2_i32x4_neon(
        if f0 == 0 { 0 } else { refs[idx0 + 1] },
        if f1 == 0 { 0 } else { refs[idx1 + 1] },
    );
    let frac = set2_i32x4_neon(f0, f1);
    round_weighted_var_s32x4(r0, r1, frac)
}

#[target_feature(enable = "neon")]
fn predict_angular_neon(
    mode: u8,
    above: &[u16],
    left: &[u16],
    n: usize,
    bit_depth: u8,
    out: &mut [u16],
    refs_ang: &mut [i32],
) {
    debug_assert!((2..=34).contains(&mode));
    debug_assert!(supported_n(n));
    debug_assert!(above.len() > 2 * n);
    debug_assert!(left.len() > 2 * n);
    debug_assert!(out.len() >= n * n);

    let (angle, vertical) = prepare_angular_refs_neon(mode, above, left, n, refs_ang);
    let refs = &refs_ang[..3 * n + 1];
    let max_v = vdupq_n_s32(((1i32 << bit_depth) - 1).max(0));
    let out = &mut out[..n * n];

    if vertical {
        if n == 2 {
            for (y, row) in out.chunks_exact_mut(n).enumerate() {
                predict_angular_vertical2_row_neon(row, refs, n, y, angle, max_v);
            }
        } else {
            for (y, row) in out.chunks_exact_mut(n).enumerate() {
                predict_angular_vertical_row_neon(row, refs, n, y, angle, max_v);
            }
        }
    } else {
        if n == 2 {
            for (y, row) in out.chunks_exact_mut(n).enumerate() {
                let v = angular_horizontal2_neon(refs, n, y, angle);
                store_i32x2_as_u16(row, v, max_v);
            }
            return;
        }
        for (y_block, rows) in out.chunks_exact_mut(4 * n).enumerate() {
            let y = y_block * 4;
            let (tiles8, has_tail) = {
                let (first_row, _) = rows.split_at(n);
                let (tiles8, tail) = first_row.as_chunks::<8>();
                (tiles8.len(), !tail.is_empty())
            };
            for tile in 0..tiles8 {
                let x = tile * 8;
                store_angular_horizontal4x4_neon(rows, refs, n, y, x, angle, max_v);
                store_angular_horizontal4x4_neon(rows, refs, n, y, x + 4, angle, max_v);
            }
            if has_tail {
                store_angular_horizontal4x4_neon(rows, refs, n, y, tiles8 * 8, angle, max_v);
            }
        }
    }
}

#[target_feature(enable = "neon")]
fn predict_planar_neon(above: &[u16], left: &[u16], n: usize, out: &mut [u16]) {
    debug_assert!(supported_n(n));
    debug_assert!(above.len() > 2 * n);
    debug_assert!(left.len() > 2 * n);
    debug_assert!(out.len() >= n * n);

    let tr = above[n + 1] as i32;
    let bl = left[n + 1] as i32;
    for (y, row) in out[..n * n].chunks_exact_mut(n).enumerate() {
        store_planar_row_neon(row, above, left[y + 1] as i32, tr, bl, y, n);
    }
}

#[target_feature(enable = "neon")]
fn predict_dc_neon(above: &[u16], left: &[u16], n: usize, is_luma: bool, out: &mut [u16]) {
    debug_assert!(supported_n(n));
    debug_assert!(above.len() > n);
    debug_assert!(left.len() > n);
    debug_assert!(out.len() >= n * n);

    let mut sum = 0i32;
    for (&above, &left) in above[1..=n].iter().zip(left[1..=n].iter()) {
        sum += above as i32 + left as i32;
    }
    let dc = (sum + n as i32) >> ((n as u32).trailing_zeros() + 1);
    let dc_u16 = dc as u16;
    for row in out.chunks_exact_mut(n).take(n) {
        fill_row_neon(row, n, dc_u16);
    }

    if is_luma && n < 32 {
        out[0] = ((left[1] as i32 + 2 * dc + above[1] as i32 + 2) >> 2) as u16;
        for (dst, &above) in out[1..n].iter_mut().zip(above[2..n + 1].iter()) {
            *dst = ((above as i32 + 3 * dc + 2) >> 2) as u16;
        }
        for (y, &left) in (1..n).zip(left[2..n + 1].iter()) {
            out[y * n] = ((left as i32 + 3 * dc + 2) >> 2) as u16;
        }
    }
}

#[target_feature(enable = "neon")]
fn predict_mode26_neon(
    above: &[u16],
    left: &[u16],
    n: usize,
    is_luma: bool,
    bit_depth: u8,
    out: &mut [u16],
) {
    debug_assert!(supported_n(n));
    debug_assert!(above.len() > n);
    debug_assert!(left.len() > n);
    debug_assert!(out.len() >= n * n);

    for row in out.chunks_exact_mut(n).take(n) {
        copy_row_neon(row, &above[1..], n);
    }

    if is_luma && n < 32 {
        let max = ((1i32 << bit_depth) - 1).max(0);
        for (y, &left) in left[1..n + 1].iter().enumerate() {
            let v = above[1] as i32 + ((left as i32 - above[0] as i32) >> 1);
            out[y * n] = v.clamp(0, max) as u16;
        }
    }
}

#[target_feature(enable = "neon")]
fn predict_mode10_neon(
    above: &[u16],
    left: &[u16],
    n: usize,
    is_luma: bool,
    bit_depth: u8,
    out: &mut [u16],
) {
    debug_assert!(supported_n(n));
    debug_assert!(above.len() > n);
    debug_assert!(left.len() > n);
    debug_assert!(out.len() >= n * n);

    for (y, row) in out.chunks_exact_mut(n).take(n).enumerate() {
        fill_row_neon(row, n, left[y + 1]);
    }

    if is_luma && n < 32 {
        let max = ((1i32 << bit_depth) - 1).max(0);
        for (dst, &above_p1) in out[..n].iter_mut().zip(above[1..n + 1].iter()) {
            let v = left[1] as i32 + ((above_p1 as i32 - above[0] as i32) >> 1);
            *dst = v.clamp(0, max) as u16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_into_neon(
    mode: u8,
    above: &[u16],
    left: &[u16],
    n: usize,
    is_luma: bool,
    bit_depth: u8,
    out: &mut [u16],
    refs_ang: &mut [i32],
) {
    if !supported_predictor(mode, n) {
        predict_into_scalar(mode, above, left, n, is_luma, bit_depth, out, refs_ang);
        return;
    }

    unsafe {
        match mode {
            PLANAR => predict_planar_neon(above, left, n, out),
            DC => predict_dc_neon(above, left, n, is_luma, out),
            10 => predict_mode10_neon(above, left, n, is_luma, bit_depth, out),
            26 => predict_mode26_neon(above, left, n, is_luma, bit_depth, out),
            2..=34 => predict_angular_neon(mode, above, left, n, bit_depth, out, refs_ang),
            _ => predict_into_scalar(mode, above, left, n, is_luma, bit_depth, out, refs_ang),
        }
    }
}
