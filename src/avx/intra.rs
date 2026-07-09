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

use crate::intra::{DC, INTRA_PRED_ANGLE, INV_ANGLE, PLANAR};

#[inline]
fn supported_n(n: usize) -> bool {
    matches!(n, 2 | 4 | 8 | 16 | 32)
}

#[inline]
fn supported_predictor(mode: u8, n: usize) -> bool {
    supported_n(n) && (mode == PLANAR || mode == DC || (2..=34).contains(&mode))
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x16(src: &[u16]) -> __m256i {
    debug_assert!(src.len() >= 16);
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8_as_i32x8(src: &[u16]) -> __m256i {
    _mm256_cvtepu16_epi32(load_u16x8(src))
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4_as_i32x8(src: &[u16]) -> __m256i {
    _mm256_cvtepu16_epi32(load_u16x4(src))
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x8(src: &[i32]) -> __m256i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x4_as_i32x8(src: &[i32]) -> __m256i {
    debug_assert!(src.len() >= 4);
    let lo = unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) };
    _mm256_castsi128_si256(lo)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x16(dst: &mut [u16], v: __m256i) {
    debug_assert!(dst.len() >= 16);
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast::<__m256i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x8(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 8);
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x4(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 4);
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x2(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 2);
    unsafe { _mm_store_ss(dst.as_mut_ptr().cast(), _mm_castsi128_ps(v)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn x_lanes8(base: usize) -> __m256i {
    _mm256_add_epi32(
        _mm256_set1_epi32(base as i32),
        _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn nx_lanes8(n: usize, base: usize) -> __m256i {
    _mm256_sub_epi32(
        _mm256_set1_epi32((n - 1 - base) as i32),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn mul_const(v: __m256i, c: i32) -> __m256i {
    _mm256_mullo_epi32(v, _mm256_set1_epi32(c))
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_shift_s32x8(v: __m256i, add: i32, shift: i32) -> __m256i {
    _mm256_sra_epi32(
        _mm256_add_epi32(v, _mm256_set1_epi32(add)),
        _mm_cvtsi32_si128(shift),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn clip_s32x8(v: __m256i, max_v: __m256i) -> __m256i {
    _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max_v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_i32x8_to_u16x8(v: __m256i) -> __m128i {
    let packed = _mm256_packus_epi32(v, _mm256_setzero_si256());
    let packed = _mm256_permute4x64_epi64::<0xd8>(packed);
    _mm256_castsi256_si128(packed)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8_as_u16(dst: &mut [u16], v: __m256i, max_v: __m256i) {
    let v = clip_s32x8(v, max_v);
    store_u16x8(dst, pack_i32x8_to_u16x8(v));
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x4_as_u16(dst: &mut [u16], v: __m256i, max_v: __m256i) {
    let v = clip_s32x8(v, max_v);
    store_u16x4(dst, pack_i32x8_to_u16x8(v));
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x2_as_u16(dst: &mut [u16], v: __m256i, max_v: __m256i) {
    let v = clip_s32x8(v, max_v);
    store_u16x2(dst, pack_i32x8_to_u16x8(v));
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_i32x16_to_u16x16(lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x16_as_u16(dst: &mut [u16], lo: __m256i, hi: __m256i, max_v: __m256i) {
    let lo = clip_s32x8(lo, max_v);
    let hi = clip_s32x8(hi, max_v);
    store_u16x16(dst, pack_i32x16_to_u16x16(lo, hi));
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn planar8_avx2(
    above: __m256i,
    x: usize,
    n: usize,
    left_y: i32,
    tr: i32,
    bl_part: i32,
    ny: i32,
    add: i32,
    shift: i32,
) -> __m256i {
    let h = _mm256_add_epi32(
        _mm256_mullo_epi32(nx_lanes8(n, x), _mm256_set1_epi32(left_y)),
        mul_const(x_lanes8(x), tr),
    );
    let v = _mm256_add_epi32(mul_const(above, ny), _mm256_set1_epi32(bl_part));
    round_shift_s32x8(_mm256_add_epi32(h, v), add, shift)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_planar_row_avx2(
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
        let a = load_u16x4_as_i32x8(&above[1..]);
        let v = planar8_avx2(a, 0, n, left_y, tr, bl_part, ny, add, shift);
        store_i32x2_as_u16(out, v, _mm256_set1_epi32(i32::MAX));
        return;
    }

    if n == 4 {
        let a = load_u16x8_as_i32x8(&above[1..]);
        let v = planar8_avx2(a, 0, n, left_y, tr, bl_part, ny, add, shift);
        store_i32x4_as_u16(out, v, _mm256_set1_epi32(i32::MAX));
        return;
    }

    let above_row = &above[1..1 + n];
    let (above16, above_rem) = above_row.as_chunks::<16>();
    let (out16, out_rem) = out[..n].as_chunks_mut::<16>();
    for (block, (a, dst)) in above16.iter().zip(out16.iter_mut()).enumerate() {
        let x = block * 16;
        let lo = planar8_avx2(
            load_u16x8_as_i32x8(&a[..]),
            x,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        let hi = planar8_avx2(
            load_u16x8_as_i32x8(&a[8..]),
            x + 8,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        store_i32x16_as_u16(dst, lo, hi, _mm256_set1_epi32(i32::MAX));
    }

    let x8_base = above16.len() * 16;
    let (above8, _) = above_rem.as_chunks::<8>();
    let (out8, _) = out_rem.as_chunks_mut::<8>();
    for (block, (a, dst)) in above8.iter().zip(out8.iter_mut()).enumerate() {
        let x = x8_base + block * 8;
        let v = planar8_avx2(
            load_u16x8_as_i32x8(&a[..]),
            x,
            n,
            left_y,
            tr,
            bl_part,
            ny,
            add,
            shift,
        );
        store_i32x8_as_u16(dst, v, _mm256_set1_epi32(i32::MAX));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn fill_row_avx2(dst: &mut [u16], n: usize, v: u16) {
    let v128 = _mm_set1_epi16(v as i16);
    if n == 2 {
        store_u16x2(dst, v128);
    } else if n == 4 {
        store_u16x4(dst, v128);
    } else if n == 8 {
        store_u16x8(dst, v128);
    } else {
        let v256 = _mm256_set1_epi16(v as i16);
        let (dst16, _) = dst[..n].as_chunks_mut::<16>();
        for block in dst16 {
            store_u16x16(block, v256);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn copy_row_avx2(dst: &mut [u16], src: &[u16], n: usize) {
    if n == 2 {
        let v = load_u16x4(src);
        store_u16x2(dst, v);
    } else if n == 4 {
        let v = load_u16x4(src);
        store_u16x4(dst, v);
    } else if n == 8 {
        let v = load_u16x8(src);
        store_u16x8(dst, v);
    } else {
        let (src16, _) = src[..n].as_chunks::<16>();
        let (dst16, _) = dst[..n].as_chunks_mut::<16>();
        for (src, dst) in src16.iter().zip(dst16.iter_mut()) {
            store_u16x16(dst, load_u16x16(src));
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn sum_refs_avx2(above: &[u16], left: &[u16], n: usize) -> i32 {
    if n == 2 {
        return above[1] as i32 + above[2] as i32 + left[1] as i32 + left[2] as i32;
    }

    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 8 <= n {
        let a = load_u16x8_as_i32x8(&above[1 + i..]);
        let l = load_u16x8_as_i32x8(&left[1 + i..]);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(a, l));
        i += 8;
    }

    if n == 4 {
        let a = load_u16x4_as_i32x8(&above[1..]);
        let l = load_u16x4_as_i32x8(&left[1..]);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(a, l));
    }

    let hi = _mm256_extracti128_si256::<1>(acc);
    let lo = _mm256_castsi256_si128(acc);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_hadd_epi32(s, s);
    let s = _mm_hadd_epi32(s, s);
    _mm_cvtsi128_si32(s)
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_weighted_s32x8(r0: __m256i, r1: __m256i, frac: i32) -> __m256i {
    let a = _mm256_mullo_epi32(r0, _mm256_set1_epi32(32 - frac));
    let b = _mm256_mullo_epi32(r1, _mm256_set1_epi32(frac));
    _mm256_srai_epi32::<5>(_mm256_add_epi32(
        _mm256_add_epi32(a, b),
        _mm256_set1_epi32(16),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_weighted_var_s32x8(r0: __m256i, r1: __m256i, frac: __m256i) -> __m256i {
    let inv_frac = _mm256_sub_epi32(_mm256_set1_epi32(32), frac);
    let a = _mm256_mullo_epi32(r0, inv_frac);
    let b = _mm256_mullo_epi32(r1, frac);
    _mm256_srai_epi32::<5>(_mm256_add_epi32(
        _mm256_add_epi32(a, b),
        _mm256_set1_epi32(16),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
fn set2_i32x8(a: i32, b: i32) -> __m256i {
    _mm256_setr_epi32(a, b, 0, 0, 0, 0, 0, 0)
}

fn prepare_angular_refs_avx2(
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
#[target_feature(enable = "avx2")]
fn transpose_4x4_s32(v: [__m256i; 4]) -> [__m256i; 4] {
    let a0 = _mm256_castsi256_si128(v[0]);
    let a1 = _mm256_castsi256_si128(v[1]);
    let a2 = _mm256_castsi256_si128(v[2]);
    let a3 = _mm256_castsi256_si128(v[3]);
    let t0 = _mm_unpacklo_epi32(a0, a1);
    let t1 = _mm_unpackhi_epi32(a0, a1);
    let t2 = _mm_unpacklo_epi32(a2, a3);
    let t3 = _mm_unpackhi_epi32(a2, a3);
    [
        _mm256_castsi128_si256(_mm_unpacklo_epi64(t0, t2)),
        _mm256_castsi128_si256(_mm_unpackhi_epi64(t0, t2)),
        _mm256_castsi128_si256(_mm_unpacklo_epi64(t1, t3)),
        _mm256_castsi128_si256(_mm_unpackhi_epi64(t1, t3)),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8_s32(c: [__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi32(c[0], c[1]);
    let t1 = _mm256_unpackhi_epi32(c[0], c[1]);
    let t2 = _mm256_unpacklo_epi32(c[2], c[3]);
    let t3 = _mm256_unpackhi_epi32(c[2], c[3]);
    let t4 = _mm256_unpacklo_epi32(c[4], c[5]);
    let t5 = _mm256_unpackhi_epi32(c[4], c[5]);
    let t6 = _mm256_unpacklo_epi32(c[6], c[7]);
    let t7 = _mm256_unpackhi_epi32(c[6], c[7]);

    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);

    [
        _mm256_permute2x128_si256::<0x20>(u0, u4),
        _mm256_permute2x128_si256::<0x20>(u1, u5),
        _mm256_permute2x128_si256::<0x20>(u2, u6),
        _mm256_permute2x128_si256::<0x20>(u3, u7),
        _mm256_permute2x128_si256::<0x31>(u0, u4),
        _mm256_permute2x128_si256::<0x31>(u1, u5),
        _mm256_permute2x128_si256::<0x31>(u2, u6),
        _mm256_permute2x128_si256::<0x31>(u3, u7),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
fn predict_angular_vertical2_row_avx2(
    dst: &mut [u16],
    refs: &[i32],
    n: usize,
    row: usize,
    angle: i32,
    max_v: __m256i,
) {
    debug_assert_eq!(n, 2);
    let base = n as i32;
    let pos = (row as i32 + 1) * angle;
    let i_idx = pos >> 5;
    let frac = pos & 31;
    let idx = (i_idx + 1 + base) as usize;
    let r0 = set2_i32x8(refs[idx], refs[idx + 1]);
    let v = if frac == 0 {
        r0
    } else {
        let r1 = set2_i32x8(refs[idx + 1], refs[idx + 2]);
        round_weighted_s32x8(r0, r1, frac)
    };
    store_i32x2_as_u16(dst, v, max_v);
}

#[inline]
#[target_feature(enable = "avx2")]
fn predict_angular_vertical4_row_avx2(
    dst: &mut [u16],
    refs: &[i32],
    n: usize,
    row: usize,
    angle: i32,
    max_v: __m256i,
) {
    debug_assert_eq!(n, 4);
    let base = n as i32;
    let pos = (row as i32 + 1) * angle;
    let i_idx = pos >> 5;
    let frac = pos & 31;
    let idx = (i_idx + 1 + base) as usize;
    let r0 = load_i32x4_as_i32x8(&refs[idx..]);
    let v = if frac == 0 {
        r0
    } else {
        let r1 = load_i32x4_as_i32x8(&refs[idx + 1..]);
        round_weighted_s32x8(r0, r1, frac)
    };
    store_i32x4_as_u16(dst, v, max_v);
}

#[inline]
#[target_feature(enable = "avx2")]
fn predict_angular_vertical_row_avx2(
    dst: &mut [u16],
    refs: &[i32],
    n: usize,
    row: usize,
    angle: i32,
    max_v: __m256i,
) {
    debug_assert!(n >= 8);
    let base = n as i32;
    let pos = (row as i32 + 1) * angle;
    let i_idx = pos >> 5;
    let frac = pos & 31;
    let (dst8, _) = dst[..n].as_chunks_mut::<8>();
    for (block, dst) in dst8.iter_mut().enumerate() {
        let x = block * 8;
        let idx = (x as i32 + i_idx + 1 + base) as usize;
        let r0 = load_i32x8(&refs[idx..]);
        let v = if frac == 0 {
            r0
        } else {
            let r1 = load_i32x8(&refs[idx + 1..]);
            round_weighted_s32x8(r0, r1, frac)
        };
        store_i32x8_as_u16(dst, v, max_v);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn angular_horizontal_col4_avx2(refs: &[i32], n: usize, y: usize, x: usize, angle: i32) -> __m256i {
    let base = n as i32;
    let pos = (x as i32 + 1) * angle;
    let i_idx = pos >> 5;
    let frac = pos & 31;
    let idx = (y as i32 + i_idx + 1 + base) as usize;
    let r0 = load_i32x4_as_i32x8(&refs[idx..]);
    if frac == 0 {
        r0
    } else {
        let r1 = load_i32x4_as_i32x8(&refs[idx + 1..]);
        round_weighted_s32x8(r0, r1, frac)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn angular_horizontal_col8_avx2(refs: &[i32], n: usize, y: usize, x: usize, angle: i32) -> __m256i {
    let base = n as i32;
    let pos = (x as i32 + 1) * angle;
    let i_idx = pos >> 5;
    let frac = pos & 31;
    let idx = (y as i32 + i_idx + 1 + base) as usize;
    let r0 = load_i32x8(&refs[idx..]);
    if frac == 0 {
        r0
    } else {
        let r1 = load_i32x8(&refs[idx + 1..]);
        round_weighted_s32x8(r0, r1, frac)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_angular_horizontal4x4_avx2(
    rows: &mut [u16],
    refs: &[i32],
    n: usize,
    y: usize,
    x: usize,
    angle: i32,
    max_v: __m256i,
) {
    debug_assert!(rows.len() >= 4 * n);
    debug_assert!(x + 4 <= n);
    let c0 = angular_horizontal_col4_avx2(refs, n, y, x, angle);
    let c1 = angular_horizontal_col4_avx2(refs, n, y, x + 1, angle);
    let c2 = angular_horizontal_col4_avx2(refs, n, y, x + 2, angle);
    let c3 = angular_horizontal_col4_avx2(refs, n, y, x + 3, angle);
    let r = transpose_4x4_s32([c0, c1, c2, c3]);
    store_i32x4_as_u16(&mut rows[x..], r[0], max_v);
    store_i32x4_as_u16(&mut rows[n + x..], r[1], max_v);
    store_i32x4_as_u16(&mut rows[2 * n + x..], r[2], max_v);
    store_i32x4_as_u16(&mut rows[3 * n + x..], r[3], max_v);
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_angular_horizontal8x8_avx2(
    rows: &mut [u16],
    refs: &[i32],
    n: usize,
    y: usize,
    x: usize,
    angle: i32,
    max_v: __m256i,
) {
    debug_assert!(rows.len() >= 8 * n);
    debug_assert!(x + 8 <= n);
    let c = [
        angular_horizontal_col8_avx2(refs, n, y, x, angle),
        angular_horizontal_col8_avx2(refs, n, y, x + 1, angle),
        angular_horizontal_col8_avx2(refs, n, y, x + 2, angle),
        angular_horizontal_col8_avx2(refs, n, y, x + 3, angle),
        angular_horizontal_col8_avx2(refs, n, y, x + 4, angle),
        angular_horizontal_col8_avx2(refs, n, y, x + 5, angle),
        angular_horizontal_col8_avx2(refs, n, y, x + 6, angle),
        angular_horizontal_col8_avx2(refs, n, y, x + 7, angle),
    ];
    let r = transpose_8x8_s32(c);
    store_i32x8_as_u16(&mut rows[x..], r[0], max_v);
    store_i32x8_as_u16(&mut rows[n + x..], r[1], max_v);
    store_i32x8_as_u16(&mut rows[2 * n + x..], r[2], max_v);
    store_i32x8_as_u16(&mut rows[3 * n + x..], r[3], max_v);
    store_i32x8_as_u16(&mut rows[4 * n + x..], r[4], max_v);
    store_i32x8_as_u16(&mut rows[5 * n + x..], r[5], max_v);
    store_i32x8_as_u16(&mut rows[6 * n + x..], r[6], max_v);
    store_i32x8_as_u16(&mut rows[7 * n + x..], r[7], max_v);
}

#[inline]
#[target_feature(enable = "avx2")]
fn angular_horizontal2_avx2(refs: &[i32], n: usize, y: usize, angle: i32) -> __m256i {
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

    let r0 = set2_i32x8(refs[idx0], refs[idx1]);
    let r1 = set2_i32x8(
        if f0 == 0 { 0 } else { refs[idx0 + 1] },
        if f1 == 0 { 0 } else { refs[idx1 + 1] },
    );
    let frac = set2_i32x8(f0, f1);
    round_weighted_var_s32x8(r0, r1, frac)
}

#[target_feature(enable = "avx2")]
fn predict_angular_avx2(
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

    let (angle, vertical) = prepare_angular_refs_avx2(mode, above, left, n, refs_ang);
    let refs = &refs_ang[..3 * n + 1];
    let max_v = _mm256_set1_epi32(((1i32 << bit_depth) - 1).max(0));
    let out = &mut out[..n * n];

    if vertical {
        match n {
            2 => {
                for (y, row) in out.chunks_exact_mut(n).enumerate() {
                    predict_angular_vertical2_row_avx2(row, refs, n, y, angle, max_v);
                }
            }
            4 => {
                for (y, row) in out.chunks_exact_mut(n).enumerate() {
                    predict_angular_vertical4_row_avx2(row, refs, n, y, angle, max_v);
                }
            }
            8 | 16 | 32 => {
                for (y, row) in out.chunks_exact_mut(n).enumerate() {
                    predict_angular_vertical_row_avx2(row, refs, n, y, angle, max_v);
                }
            }
            _ => unreachable!(),
        }
    } else {
        match n {
            2 => {
                for (y, row) in out.chunks_exact_mut(n).enumerate() {
                    let v = angular_horizontal2_avx2(refs, n, y, angle);
                    store_i32x2_as_u16(row, v, max_v);
                }
            }
            4 => {
                store_angular_horizontal4x4_avx2(out, refs, n, 0, 0, angle, max_v);
            }
            8 | 16 | 32 => {
                for (y_block, rows) in out.chunks_exact_mut(8 * n).enumerate() {
                    let y = y_block * 8;
                    for x in (0..n).step_by(8) {
                        store_angular_horizontal8x8_avx2(rows, refs, n, y, x, angle, max_v);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}

#[target_feature(enable = "avx2")]
fn predict_planar_avx2(above: &[u16], left: &[u16], n: usize, out: &mut [u16]) {
    debug_assert!(supported_n(n));
    debug_assert!(above.len() > 2 * n);
    debug_assert!(left.len() > 2 * n);
    debug_assert!(out.len() >= n * n);

    let tr = above[n + 1] as i32;
    let bl = left[n + 1] as i32;
    for (y, row) in out[..n * n].chunks_exact_mut(n).enumerate() {
        store_planar_row_avx2(row, above, left[y + 1] as i32, tr, bl, y, n);
    }
}

#[target_feature(enable = "avx2")]
fn predict_dc_avx2(above: &[u16], left: &[u16], n: usize, is_luma: bool, out: &mut [u16]) {
    debug_assert!(supported_n(n));
    debug_assert!(above.len() > n);
    debug_assert!(left.len() > n);
    debug_assert!(out.len() >= n * n);

    let sum = sum_refs_avx2(above, left, n);
    let dc = (sum + n as i32) >> ((n as u32).trailing_zeros() + 1);
    let dc_u16 = dc as u16;
    for row in out.chunks_exact_mut(n).take(n) {
        fill_row_avx2(row, n, dc_u16);
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

#[target_feature(enable = "avx2")]
fn predict_mode26_avx2(
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
        copy_row_avx2(row, &above[1..], n);
    }

    if is_luma && n < 32 {
        let max = ((1i32 << bit_depth) - 1).max(0);
        for (y, &left) in left[1..n + 1].iter().enumerate() {
            let v = above[1] as i32 + ((left as i32 - above[0] as i32) >> 1);
            out[y * n] = v.clamp(0, max) as u16;
        }
    }
}

#[target_feature(enable = "avx2")]
fn predict_mode10_avx2(
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
        fill_row_avx2(row, n, left[y + 1]);
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
pub(crate) fn predict_into_avx2(
    mode: u8,
    above: &[u16],
    left: &[u16],
    n: usize,
    is_luma: bool,
    bit_depth: u8,
    out: &mut [u16],
    refs_ang: &mut [i32],
) {
    debug_assert!(supported_predictor(mode, n));

    unsafe {
        match mode {
            PLANAR => predict_planar_avx2(above, left, n, out),
            DC => predict_dc_avx2(above, left, n, is_luma, out),
            10 => predict_mode10_avx2(above, left, n, is_luma, bit_depth, out),
            26 => predict_mode26_avx2(above, left, n, is_luma, bit_depth, out),
            2..=34 => predict_angular_avx2(mode, above, left, n, bit_depth, out, refs_ang),
            _ => unreachable!(),
        }
    }
}
