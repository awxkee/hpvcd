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

use super::common::*;
use crate::transform::{DST4, inv_transform_into_scalar};

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
fn tr_store_4x4_s32(
    dst: &mut [i32],
    stride: usize,
    lane_base: usize,
    elem_base: usize,
    v: [int32x4_t; 4],
) {
    debug_assert!(dst.len() >= (lane_base + 4) * stride);
    debug_assert!(elem_base + 4 <= stride);
    let t = transpose_4x4_s32(v);
    store_s32x4(&mut dst[(lane_base + 0) * stride + elem_base..], t[0]);
    store_s32x4(&mut dst[(lane_base + 1) * stride + elem_base..], t[1]);
    store_s32x4(&mut dst[(lane_base + 2) * stride + elem_base..], t[2]);
    store_s32x4(&mut dst[(lane_base + 3) * stride + elem_base..], t[3]);
}

macro_rules! lin_s32x4 {
    ($v0:expr, $k0:expr $(, $v:expr, $k:expr)+ $(,)?) => {{
        let mut acc = mul_const($v0, $k0);
        $(
            acc = madd_const(acc, $v, $k);
        )+
        acc
    }};
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_odd_16_s32x4(c: [int32x4_t; 16]) -> [int32x4_t; 8] {
    let c1 = c[1];
    let c3 = c[3];
    let c5 = c[5];
    let c7 = c[7];
    let c9 = c[9];
    let c11 = c[11];
    let c13 = c[13];
    let c15 = c[15];
    [
        lin_s32x4!(
            c1, 90, c3, 87, c5, 80, c7, 70, c9, 57, c11, 43, c13, 25, c15, 9
        ),
        lin_s32x4!(
            c1, 87, c3, 57, c5, 9, c7, -43, c9, -80, c11, -90, c13, -70, c15, -25
        ),
        lin_s32x4!(
            c1, 80, c3, 9, c5, -70, c7, -87, c9, -25, c11, 57, c13, 90, c15, 43
        ),
        lin_s32x4!(
            c1, 70, c3, -43, c5, -87, c7, 9, c9, 90, c11, 25, c13, -80, c15, -57
        ),
        lin_s32x4!(
            c1, 57, c3, -80, c5, -25, c7, 90, c9, -9, c11, -87, c13, 43, c15, 70
        ),
        lin_s32x4!(
            c1, 43, c3, -90, c5, 57, c7, 25, c9, -87, c11, 70, c13, 9, c15, -80
        ),
        lin_s32x4!(
            c1, 25, c3, -70, c5, 90, c7, -80, c9, 43, c11, 9, c13, -57, c15, 87
        ),
        lin_s32x4!(
            c1, 9, c3, -25, c5, 43, c7, -57, c9, 70, c11, -80, c13, 87, c15, -90
        ),
    ]
}
#[inline]
#[target_feature(enable = "neon")]
fn idct_odd_32_s32x4(c: [int32x4_t; 32]) -> [int32x4_t; 16] {
    let c1 = c[1];
    let c3 = c[3];
    let c5 = c[5];
    let c7 = c[7];
    let c9 = c[9];
    let c11 = c[11];
    let c13 = c[13];
    let c15 = c[15];
    let c17 = c[17];
    let c19 = c[19];
    let c21 = c[21];
    let c23 = c[23];
    let c25 = c[25];
    let c27 = c[27];
    let c29 = c[29];
    let c31 = c[31];
    [
        lin_s32x4!(
            c1, 90, c3, 90, c5, 88, c7, 85, c9, 82, c11, 78, c13, 73, c15, 67, c17, 61, c19, 54,
            c21, 46, c23, 38, c25, 31, c27, 22, c29, 13, c31, 4
        ),
        lin_s32x4!(
            c1, 90, c3, 82, c5, 67, c7, 46, c9, 22, c11, -4, c13, -31, c15, -54, c17, -73, c19,
            -85, c21, -90, c23, -88, c25, -78, c27, -61, c29, -38, c31, -13
        ),
        lin_s32x4!(
            c1, 88, c3, 67, c5, 31, c7, -13, c9, -54, c11, -82, c13, -90, c15, -78, c17, -46, c19,
            -4, c21, 38, c23, 73, c25, 90, c27, 85, c29, 61, c31, 22
        ),
        lin_s32x4!(
            c1, 85, c3, 46, c5, -13, c7, -67, c9, -90, c11, -73, c13, -22, c15, 38, c17, 82, c19,
            88, c21, 54, c23, -4, c25, -61, c27, -90, c29, -78, c31, -31
        ),
        lin_s32x4!(
            c1, 82, c3, 22, c5, -54, c7, -90, c9, -61, c11, 13, c13, 78, c15, 85, c17, 31, c19,
            -46, c21, -90, c23, -67, c25, 4, c27, 73, c29, 88, c31, 38
        ),
        lin_s32x4!(
            c1, 78, c3, -4, c5, -82, c7, -73, c9, 13, c11, 85, c13, 67, c15, -22, c17, -88, c19,
            -61, c21, 31, c23, 90, c25, 54, c27, -38, c29, -90, c31, -46
        ),
        lin_s32x4!(
            c1, 73, c3, -31, c5, -90, c7, -22, c9, 78, c11, 67, c13, -38, c15, -90, c17, -13, c19,
            82, c21, 61, c23, -46, c25, -88, c27, -4, c29, 85, c31, 54
        ),
        lin_s32x4!(
            c1, 67, c3, -54, c5, -78, c7, 38, c9, 85, c11, -22, c13, -90, c15, 4, c17, 90, c19, 13,
            c21, -88, c23, -31, c25, 82, c27, 46, c29, -73, c31, -61
        ),
        lin_s32x4!(
            c1, 61, c3, -73, c5, -46, c7, 82, c9, 31, c11, -88, c13, -13, c15, 90, c17, -4, c19,
            -90, c21, 22, c23, 85, c25, -38, c27, -78, c29, 54, c31, 67
        ),
        lin_s32x4!(
            c1, 54, c3, -85, c5, -4, c7, 88, c9, -46, c11, -61, c13, 82, c15, 13, c17, -90, c19,
            38, c21, 67, c23, -78, c25, -22, c27, 90, c29, -31, c31, -73
        ),
        lin_s32x4!(
            c1, 46, c3, -90, c5, 38, c7, 54, c9, -90, c11, 31, c13, 61, c15, -88, c17, 22, c19, 67,
            c21, -85, c23, 13, c25, 73, c27, -82, c29, 4, c31, 78
        ),
        lin_s32x4!(
            c1, 38, c3, -88, c5, 73, c7, -4, c9, -67, c11, 90, c13, -46, c15, -31, c17, 85, c19,
            -78, c21, 13, c23, 61, c25, -90, c27, 54, c29, 22, c31, -82
        ),
        lin_s32x4!(
            c1, 31, c3, -78, c5, 90, c7, -61, c9, 4, c11, 54, c13, -88, c15, 82, c17, -38, c19,
            -22, c21, 73, c23, -90, c25, 67, c27, -13, c29, -46, c31, 85
        ),
        lin_s32x4!(
            c1, 22, c3, -61, c5, 85, c7, -90, c9, 73, c11, -38, c13, -4, c15, 46, c17, -78, c19,
            90, c21, -82, c23, 54, c25, -13, c27, -31, c29, 67, c31, -88
        ),
        lin_s32x4!(
            c1, 13, c3, -38, c5, 61, c7, -78, c9, 88, c11, -90, c13, 85, c15, -73, c17, 54, c19,
            -31, c21, 4, c23, 22, c25, -46, c27, 67, c29, -82, c31, 90
        ),
        lin_s32x4!(
            c1, 4, c3, -13, c5, 22, c7, -31, c9, 38, c11, -46, c13, 54, c15, -61, c17, 67, c19,
            -73, c21, 78, c23, -82, c25, 85, c27, -88, c29, 90, c31, -90
        ),
    ]
}
#[inline]
#[target_feature(enable = "neon")]
fn idct_raw_4_s32x4(c: [int32x4_t; 4]) -> [int32x4_t; 4] {
    let e0 = mul_const(add(c[0], c[2]), 64);
    let e1 = mul_const(sub(c[0], c[2]), 64);
    let o0 = add(mul_const(c[1], 83), mul_const(c[3], 36));
    let o1 = sub(mul_const(c[1], 36), mul_const(c[3], 83));

    [add(e0, o0), add(e1, o1), sub(e1, o1), sub(e0, o0)]
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_raw_8_s32x4(c: [int32x4_t; 8]) -> [int32x4_t; 8] {
    let ee = idct_raw_4_s32x4([c[0], c[2], c[4], c[6]]);

    let c1 = c[1];
    let c3 = c[3];
    let c5 = c[5];
    let c7 = c[7];
    let o0 = add(
        add(mul_const(c1, 89), mul_const(c3, 75)),
        add(mul_const(c5, 50), mul_const(c7, 18)),
    );
    let o1 = sub(
        sub(mul_const(c1, 75), mul_const(c3, 18)),
        add(mul_const(c5, 89), mul_const(c7, 50)),
    );
    let o2 = add(
        sub(mul_const(c1, 50), mul_const(c3, 89)),
        add(mul_const(c5, 18), mul_const(c7, 75)),
    );
    let o3 = sub(
        add(mul_const(c1, 18), mul_const(c5, 75)),
        add(mul_const(c3, 50), mul_const(c7, 89)),
    );
    let oo = [o0, o1, o2, o3];

    let mut out = [zero(); 8];
    for k in 0..4 {
        out[k] = add(ee[k], oo[k]);
        out[7 - k] = sub(ee[k], oo[k]);
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_raw_16_s32x4(c: [int32x4_t; 16]) -> [int32x4_t; 16] {
    let ee = idct_raw_8_s32x4(std::array::from_fn(|j| c[2 * j]));
    let oo = idct_odd_16_s32x4(c);

    let mut out = [zero(); 16];
    for k in 0..8 {
        out[k] = add(ee[k], oo[k]);
        out[15 - k] = sub(ee[k], oo[k]);
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_raw_32_s32x4(c: [int32x4_t; 32]) -> [int32x4_t; 32] {
    let ee = idct_raw_16_s32x4(std::array::from_fn(|j| c[2 * j]));
    let oo = idct_odd_32_s32x4(c);

    let mut out = [zero(); 32];
    for k in 0..16 {
        out[k] = add(ee[k], oo[k]);
        out[31 - k] = sub(ee[k], oo[k]);
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_raw_s32x4<const N: usize>(c: [int32x4_t; N]) -> [int32x4_t; N] {
    debug_assert!(N == 4 || N == 8 || N == 16 || N == 32);

    match N {
        4 => {
            let src = [c[0], c[1], c[2], c[3]];
            let r = idct_raw_4_s32x4(src);
            std::array::from_fn(|i| r[i])
        }
        8 => {
            let src = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
            let r = idct_raw_8_s32x4(src);
            std::array::from_fn(|i| r[i])
        }
        16 => {
            let src = [
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12],
                c[13], c[14], c[15],
            ];
            let r = idct_raw_16_s32x4(src);
            std::array::from_fn(|i| r[i])
        }
        32 => {
            let src = [
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12],
                c[13], c[14], c[15], c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23], c[24],
                c[25], c[26], c[27], c[28], c[29], c[30], c[31],
            ];
            let r = idct_raw_32_s32x4(src);
            std::array::from_fn(|i| r[i])
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "neon")]
fn inv_dct_n_into_neon<const N: usize>(coeff: &[i32], bit_depth: u8, nx: usize, out: &mut [i32]) {
    debug_assert!(N == 4 || N == 8 || N == 16 || N == 32);
    debug_assert!(coeff.len() >= N * N);
    debug_assert!(out.len() >= N * N);

    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    let mut tmp = [0i32; 32 * 32];

    // Columns >= nx are zero on input; skip them. Stage 1 writes a
    // transposed scratch: tmp[input_column][stage1_row]. That lets stage 2
    // use contiguous loads and normal 4x4 transpose stores instead of scalar
    // row gather/scatter.
    let ncol = ((nx.min(N) + 3) & !3).max(4);
    for c in (0..ncol).step_by(4) {
        let src: [int32x4_t; N] = std::array::from_fn(|k| load_s32x4(&coeff[k * N + c..]));
        let raw = idct_raw_s32x4::<N>(src);
        for m in (0..N).step_by(4) {
            let v = std::array::from_fn(|j| round_shift_clip_i16_s32x4(raw[m + j], add1, shift1));
            tr_store_4x4_s32(&mut tmp, N, c, m, v);
        }
    }

    for r in (0..N).step_by(4) {
        let src: [int32x4_t; N] = std::array::from_fn(|k| load_s32x4(&tmp[k * N + r..]));
        let raw = idct_raw_s32x4::<N>(src);
        for x in (0..N).step_by(4) {
            let v = std::array::from_fn(|j| round_shift_s32x4(raw[x + j], add2, shift2));
            tr_store_4x4_s32(out, N, r, x, v);
        }
    }
}

#[target_feature(enable = "neon")]
fn inv_transform_dst4_into_neon(coeff: &[i32], bit_depth: u8, out: &mut [i32]) {
    debug_assert!(coeff.len() >= 16);
    debug_assert!(out.len() >= 16);

    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    let mut tmp = [0i32; 16];

    let mut acc = [zero(); 4];
    for k in 0..4 {
        let ck = load_s32x4(&coeff[k * 4..]);
        for (acc, &tm) in acc.iter_mut().zip(DST4[k].iter()) {
            *acc = madd_const(*acc, ck, tm);
        }
    }
    let v = std::array::from_fn(|m| round_shift_clip_i16_s32x4(acc[m], add1, shift1));
    tr_store_4x4_s32(&mut tmp, 4, 0, 0, v);

    let src: [int32x4_t; 4] = std::array::from_fn(|k| load_s32x4(&tmp[k * 4..]));
    let mut v = [zero(); 4];
    for x in 0..4 {
        let mut acc = zero();
        for (rk, trow) in src.iter().copied().zip(DST4.iter()) {
            acc = madd_const(acc, rk, trow[x]);
        }
        v[x] = round_shift_s32x4(acc, add2, shift2);
    }
    tr_store_4x4_s32(out, 4, 0, 0, v);
}

pub(crate) fn inv_transform_into_neon(
    coeff: &[i32],
    n: usize,
    bit_depth: u8,
    nx: usize,
    out: &mut [i32],
) {
    unsafe {
        match n {
            4 => inv_dct_n_into_neon::<4>(coeff, bit_depth, nx, out),
            8 => inv_dct_n_into_neon::<8>(coeff, bit_depth, nx, out),
            16 => inv_dct_n_into_neon::<16>(coeff, bit_depth, nx, out),
            32 => inv_dct_n_into_neon::<32>(coeff, bit_depth, nx, out),
            _ => inv_transform_into_scalar(coeff, n, bit_depth, nx, out),
        }
    }
}

pub(crate) fn inv_transform_dst_into_neon(coeff: &[i32], bit_depth: u8, out: &mut [i32]) {
    unsafe { inv_transform_dst4_into_neon(coeff, bit_depth, out) }
}
