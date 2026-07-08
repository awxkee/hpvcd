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
#[target_feature(enable = "sse4.1")]
fn load_u16x8(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u16x4(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x8(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 8);
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x4(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 4);
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x2(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 2);
    unsafe {
        _mm_store_ss(dst.as_mut_ptr().cast(), _mm_castsi128_ps(v));
    };
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn mul_const(v: __m128i, c: i32) -> __m128i {
    _mm_mullo_epi32(v, _mm_set1_epi32(c))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn round_shift_s32x4(v: __m128i, add: i32, shift: i32) -> __m128i {
    _mm_sra_epi32(
        _mm_add_epi32(v, _mm_set1_epi32(add)),
        _mm_cvtsi32_si128(shift),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn pack_u16x8(lo: __m128i, hi: __m128i) -> __m128i {
    _mm_packus_epi32(lo, hi)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn x_lanes(base: usize) -> __m128i {
    _mm_add_epi32(_mm_set1_epi32(base as i32), _mm_setr_epi32(1, 2, 3, 4))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn nx_lanes(n: usize, base: usize) -> __m128i {
    _mm_sub_epi32(
        _mm_set1_epi32((n - 1 - base) as i32),
        _mm_setr_epi32(0, 1, 2, 3),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn u16x8_lo_to_i32x4(v: __m128i, zero: __m128i) -> __m128i {
    _mm_unpacklo_epi16(v, zero)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn u16x8_hi_to_i32x4(v: __m128i, zero: __m128i) -> __m128i {
    _mm_unpackhi_epi16(v, zero)
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
fn planar4_wide_sse41(
    above: __m128i,
    x: usize,
    n: usize,
    left_y: i32,
    tr: i32,
    bl_part: i32,
    ny: i32,
    add: i32,
    shift: i32,
) -> __m128i {
    let h = _mm_add_epi32(
        _mm_mullo_epi32(nx_lanes(n, x), _mm_set1_epi32(left_y)),
        mul_const(x_lanes(x), tr),
    );
    let v = _mm_add_epi32(mul_const(above, ny), _mm_set1_epi32(bl_part));
    round_shift_s32x4(_mm_add_epi32(h, v), add, shift)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_planar_row_sse41(
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
    let zero = _mm_setzero_si128();

    if n == 2 {
        let a = u16x8_lo_to_i32x4(load_u16x4(&above[1..]), zero);
        let lo = planar4_wide_sse41(a, 0, n, left_y, tr, bl_part, ny, add, shift);
        store_u16x2(out, pack_u16x8(lo, zero));
        return;
    }

    let above_row = &above[1..1 + n];
    let (above16, above_rem) = above_row.as_chunks::<16>();
    let (out16, out_rem) = out[..n].as_chunks_mut::<16>();
    for (block, (a, dst)) in above16.iter().zip(out16.iter_mut()).enumerate() {
        let x = block * 16;
        let a0 = load_u16x8(&a[..]);
        let a1 = load_u16x8(&a[8..]);
        let lo0 = planar4_wide_sse41(
            u16x8_lo_to_i32x4(a0, zero),
            x,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        let hi0 = planar4_wide_sse41(
            u16x8_hi_to_i32x4(a0, zero),
            x + 4,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        let lo1 = planar4_wide_sse41(
            u16x8_lo_to_i32x4(a1, zero),
            x + 8,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        let hi1 = planar4_wide_sse41(
            u16x8_hi_to_i32x4(a1, zero),
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
        let lo = planar4_wide_sse41(
            u16x8_lo_to_i32x4(a, zero),
            x,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        let hi = planar4_wide_sse41(
            u16x8_hi_to_i32x4(a, zero),
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
        let a = u16x8_lo_to_i32x4(load_u16x8(&above[1 + x..]), zero);
        let lo = planar4_wide_sse41(a, x, n, left_y, tr, bl_part, ny, add, shift);
        store_u16x4(out_tail, pack_u16x8(lo, zero));
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn fill_row_sse41(dst: &mut [u16], n: usize, v: u16) {
    let v = _mm_set1_epi16(v as i16);
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
#[target_feature(enable = "sse4.1")]
fn copy_row_sse41(dst: &mut [u16], src: &[u16], n: usize) {
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
#[target_feature(enable = "sse4.1")]
fn load_i32x4(src: &[i32]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn transpose_4x4_s32(v: [__m128i; 4]) -> [__m128i; 4] {
    let t0 = _mm_unpacklo_epi32(v[0], v[1]);
    let t1 = _mm_unpackhi_epi32(v[0], v[1]);
    let t2 = _mm_unpacklo_epi32(v[2], v[3]);
    let t3 = _mm_unpackhi_epi32(v[2], v[3]);
    [
        _mm_unpacklo_epi64(t0, t2),
        _mm_unpackhi_epi64(t0, t2),
        _mm_unpacklo_epi64(t1, t3),
        _mm_unpackhi_epi64(t1, t3),
    ]
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn clip_s32x4(v: __m128i, max_v: __m128i) -> __m128i {
    _mm_min_epi32(_mm_max_epi32(v, _mm_setzero_si128()), max_v)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn round_weighted_s32x4(r0: __m128i, r1: __m128i, frac: i32) -> __m128i {
    let a = _mm_mullo_epi32(r0, _mm_set1_epi32(32 - frac));
    let b = _mm_mullo_epi32(r1, _mm_set1_epi32(frac));
    _mm_srai_epi32::<5>(_mm_add_epi32(_mm_add_epi32(a, b), _mm_set1_epi32(16)))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn round_weighted_var_s32x4(r0: __m128i, r1: __m128i, frac: __m128i) -> __m128i {
    let inv_frac = _mm_sub_epi32(_mm_set1_epi32(32), frac);
    let a = _mm_mullo_epi32(r0, inv_frac);
    let b = _mm_mullo_epi32(r1, frac);
    _mm_srai_epi32::<5>(_mm_add_epi32(_mm_add_epi32(a, b), _mm_set1_epi32(16)))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i32x4_as_u16(dst: &mut [u16], v: __m128i, max_v: __m128i) {
    let v = clip_s32x4(v, max_v);
    store_u16x4(dst, pack_u16x8(v, _mm_setzero_si128()));
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i32x2_as_u16(dst: &mut [u16], v: __m128i, max_v: __m128i) {
    let v = clip_s32x4(v, max_v);
    store_u16x2(dst, pack_u16x8(v, _mm_setzero_si128()));
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i32x8_as_u16(dst: &mut [u16], lo: __m128i, hi: __m128i, max_v: __m128i) {
    let lo = clip_s32x4(lo, max_v);
    let hi = clip_s32x4(hi, max_v);
    store_u16x8(dst, pack_u16x8(lo, hi));
}

fn prepare_angular_refs_sse41(
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
#[target_feature(enable = "sse4.1")]
fn set2_i32x4_sse41(a: i32, b: i32) -> __m128i {
    _mm_setr_epi32(a, b, 0, 0)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn predict_angular_vertical2_row_sse41(
    dst: &mut [u16],
    refs: &[i32],
    n: usize,
    row: usize,
    angle: i32,
    max_v: __m128i,
) {
    debug_assert_eq!(n, 2);
    let base = n as i32;
    let pos = (row as i32 + 1) * angle;
    let i_idx = pos >> 5;
    let frac = pos & 31;
    let idx = (i_idx + 1 + base) as usize;
    let r0 = set2_i32x4_sse41(refs[idx], refs[idx + 1]);
    let v = if frac == 0 {
        r0
    } else {
        let r1 = set2_i32x4_sse41(refs[idx + 1], refs[idx + 2]);
        round_weighted_s32x4(r0, r1, frac)
    };
    store_i32x2_as_u16(dst, v, max_v);
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn predict_angular_vertical_row_sse41(
    dst: &mut [u16],
    refs: &[i32],
    n: usize,
    row: usize,
    angle: i32,
    max_v: __m128i,
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
#[target_feature(enable = "sse4.1")]
fn angular_horizontal_col4_sse41(
    refs: &[i32],
    n: usize,
    y: usize,
    x: usize,
    angle: i32,
) -> __m128i {
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
#[target_feature(enable = "sse4.1")]
fn store_angular_horizontal4x4_sse41(
    rows: &mut [u16],
    refs: &[i32],
    n: usize,
    y: usize,
    x: usize,
    angle: i32,
    max_v: __m128i,
) {
    debug_assert!(rows.len() >= 4 * n);
    debug_assert!(x + 4 <= n);
    let c0 = angular_horizontal_col4_sse41(refs, n, y, x, angle);
    let c1 = angular_horizontal_col4_sse41(refs, n, y, x + 1, angle);
    let c2 = angular_horizontal_col4_sse41(refs, n, y, x + 2, angle);
    let c3 = angular_horizontal_col4_sse41(refs, n, y, x + 3, angle);
    let r = transpose_4x4_s32([c0, c1, c2, c3]);
    store_i32x4_as_u16(&mut rows[x..], r[0], max_v);
    store_i32x4_as_u16(&mut rows[n + x..], r[1], max_v);
    store_i32x4_as_u16(&mut rows[2 * n + x..], r[2], max_v);
    store_i32x4_as_u16(&mut rows[3 * n + x..], r[3], max_v);
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn angular_horizontal2_sse41(refs: &[i32], n: usize, y: usize, angle: i32) -> __m128i {
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

    let r0 = set2_i32x4_sse41(refs[idx0], refs[idx1]);
    let r1 = set2_i32x4_sse41(
        if f0 == 0 { 0 } else { refs[idx0 + 1] },
        if f1 == 0 { 0 } else { refs[idx1 + 1] },
    );
    let frac = set2_i32x4_sse41(f0, f1);
    round_weighted_var_s32x4(r0, r1, frac)
}

#[target_feature(enable = "sse4.1")]
fn predict_angular_sse41(
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

    let (angle, vertical) = prepare_angular_refs_sse41(mode, above, left, n, refs_ang);
    let refs = &refs_ang[..3 * n + 1];
    let max_v = _mm_set1_epi32(((1i32 << bit_depth) - 1).max(0));
    let out = &mut out[..n * n];

    if vertical {
        if n == 2 {
            for (y, row) in out.chunks_exact_mut(n).enumerate() {
                predict_angular_vertical2_row_sse41(row, refs, n, y, angle, max_v);
            }
        } else {
            for (y, row) in out.chunks_exact_mut(n).enumerate() {
                predict_angular_vertical_row_sse41(row, refs, n, y, angle, max_v);
            }
        }
    } else {
        if n == 2 {
            for (y, row) in out.chunks_exact_mut(n).enumerate() {
                let v = angular_horizontal2_sse41(refs, n, y, angle);
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
                store_angular_horizontal4x4_sse41(rows, refs, n, y, x, angle, max_v);
                store_angular_horizontal4x4_sse41(rows, refs, n, y, x + 4, angle, max_v);
            }
            if has_tail {
                store_angular_horizontal4x4_sse41(rows, refs, n, y, tiles8 * 8, angle, max_v);
            }
        }
    }
}

#[target_feature(enable = "sse4.1")]
fn predict_planar_sse41(above: &[u16], left: &[u16], n: usize, out: &mut [u16]) {
    debug_assert!(supported_n(n));
    debug_assert!(above.len() > 2 * n);
    debug_assert!(left.len() > 2 * n);
    debug_assert!(out.len() >= n * n);

    let tr = above[n + 1] as i32;
    let bl = left[n + 1] as i32;
    for (y, row) in out[..n * n].chunks_exact_mut(n).enumerate() {
        store_planar_row_sse41(row, above, left[y + 1] as i32, tr, bl, y, n);
    }
}

#[target_feature(enable = "sse4.1")]
fn predict_dc_sse41(above: &[u16], left: &[u16], n: usize, is_luma: bool, out: &mut [u16]) {
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
        fill_row_sse41(row, n, dc_u16);
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

#[target_feature(enable = "sse4.1")]
fn predict_mode26_sse41(
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
        copy_row_sse41(row, &above[1..], n);
    }

    if is_luma && n < 32 {
        let max = ((1i32 << bit_depth) - 1).max(0);
        for (y, &left) in left[1..n + 1].iter().enumerate() {
            let v = above[1] as i32 + ((left as i32 - above[0] as i32) >> 1);
            out[y * n] = v.clamp(0, max) as u16;
        }
    }
}

#[target_feature(enable = "sse4.1")]
fn predict_mode10_sse41(
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
        fill_row_sse41(row, n, left[y + 1]);
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
pub(crate) fn predict_into_sse41(
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
            PLANAR => predict_planar_sse41(above, left, n, out),
            DC => predict_dc_sse41(above, left, n, is_luma, out),
            10 => predict_mode10_sse41(above, left, n, is_luma, bit_depth, out),
            26 => predict_mode26_sse41(above, left, n, is_luma, bit_depth, out),
            2..=34 => predict_angular_sse41(mode, above, left, n, bit_depth, out, refs_ang),
            _ => predict_into_scalar(mode, above, left, n, is_luma, bit_depth, out, refs_ang),
        }
    }
}
