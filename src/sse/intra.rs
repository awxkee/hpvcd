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
#[allow(clippy::too_many_arguments)]
fn planar4_sse41(
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
    let above = _mm_cvtepu16_epi32(above);
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

    if n == 2 {
        let a = load_u16x4(&above[1..]);
        let lo = planar4_sse41(a, 0, n, left_y, tr, bl_part, ny, add, shift);
        store_u16x2(out, pack_u16x8(lo, _mm_setzero_si128()));
        return;
    }

    let mut x = 0usize;
    while x + 8 <= n {
        let a = load_u16x8(&above[1 + x..]);
        let lo = planar4_sse41(a, x, n, left_y, tr, bl_part, ny, add, shift);
        let hi = planar4_sse41(
            _mm_srli_si128::<8>(a),
            x + 4,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        store_u16x8(&mut out[x..], pack_u16x8(lo, hi));
        x += 8;
    }
    if x < n {
        let a = load_u16x8(&above[1 + x..]);
        let lo = planar4_sse41(a, x, n, left_y, tr, bl_part, ny, add, shift);
        store_u16x4(&mut out[x..], pack_u16x8(lo, _mm_setzero_si128()));
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
        let mut x = 0usize;
        while x < n {
            store_u16x8(&mut dst[x..], v);
            x += 8;
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
        let mut x = 0usize;
        while x < n {
            let v = load_u16x8(&src[x..]);
            store_u16x8(&mut dst[x..], v);
            x += 8;
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
    let mut x = 0usize;

    while x + 8 <= n {
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
        store_i32x8_as_u16(&mut dst[x..], lo, hi, max_v);
        x += 8;
    }

    if x < n {
        let idx = (x as i32 + i_idx + 1 + base) as usize;
        let r0 = load_i32x4(&refs[idx..]);
        let v = if frac == 0 {
            r0
        } else {
            let r1 = load_i32x4(&refs[idx + 1..]);
            round_weighted_s32x4(r0, r1, frac)
        };
        store_i32x4_as_u16(&mut dst[x..], v, max_v);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn angular_horizontal4_sse41(refs: &[i32], n: usize, y: usize, x: usize, angle: i32) -> __m128i {
    let base = n as i32;
    let mut r0 = [0i32; 4];
    let mut r1 = [0i32; 4];
    let mut frac = [0i32; 4];

    for lane in 0..4 {
        let pos = (x as i32 + lane as i32 + 1) * angle;
        let i_idx = pos >> 5;
        let f = pos & 31;
        let idx = (y as i32 + i_idx + 1 + base) as usize;
        r0[lane] = refs[idx];
        r1[lane] = if f == 0 { 0 } else { refs[idx + 1] };
        frac[lane] = f;
    }

    let r0 = _mm_setr_epi32(r0[0], r0[1], r0[2], r0[3]);
    let r1 = _mm_setr_epi32(r1[0], r1[1], r1[2], r1[3]);
    let frac = _mm_setr_epi32(frac[0], frac[1], frac[2], frac[3]);
    round_weighted_var_s32x4(r0, r1, frac)
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
        for (y, row) in out.chunks_exact_mut(n).enumerate() {
            let mut x = 0usize;
            while x + 8 <= n {
                let lo = angular_horizontal4_sse41(refs, n, y, x, angle);
                let hi = angular_horizontal4_sse41(refs, n, y, x + 4, angle);
                store_i32x8_as_u16(&mut row[x..], lo, hi, max_v);
                x += 8;
            }
            if x < n {
                let v = angular_horizontal4_sse41(refs, n, y, x, angle);
                store_i32x4_as_u16(&mut row[x..], v, max_v);
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
    for y in 0..n {
        let row = &mut out[y * n..y * n + n];
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
