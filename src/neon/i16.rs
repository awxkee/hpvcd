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
use crate::transform::{DST4, inv_transform_into_scalar16};

macro_rules! lin_i16x4 {
    ($c:expr; $i0:expr, $k0:expr $(, $i:expr, $k:expr)+ $(,)?) => {{
        let mut acc = vmull_n_s16($c[$i0], $k0);
        $(
            acc = vmlal_n_s16(acc, $c[$i], $k);
        )+
        acc
    }};
}

#[inline]
#[target_feature(enable = "neon")]
fn odd8_p(c: &[int16x4_t; 8]) -> [int32x4_t; 4] {
    [
        lin_i16x4!(
            c;
            1, 89, 3, 75, 5, 50, 7, 18
        ),
        lin_i16x4!(
            c;
            1, 75, 3, -18, 5, -89, 7, -50
        ),
        lin_i16x4!(
            c;
            1, 50, 3, -89, 5, 18, 7, 75
        ),
        lin_i16x4!(
            c;
            1, 18, 3, -50, 5, 75, 7, -89
        ),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn odd16_p(c: &[int16x4_t; 16]) -> [int32x4_t; 8] {
    [
        lin_i16x4!(
            c;
            1, 90, 3, 87, 5, 80, 7, 70,
            9, 57, 11, 43, 13, 25, 15, 9
        ),
        lin_i16x4!(
            c;
            1, 87, 3, 57, 5, 9, 7, -43,
            9, -80, 11, -90, 13, -70, 15, -25
        ),
        lin_i16x4!(
            c;
            1, 80, 3, 9, 5, -70, 7, -87,
            9, -25, 11, 57, 13, 90, 15, 43
        ),
        lin_i16x4!(
            c;
            1, 70, 3, -43, 5, -87, 7, 9,
            9, 90, 11, 25, 13, -80, 15, -57
        ),
        lin_i16x4!(
            c;
            1, 57, 3, -80, 5, -25, 7, 90,
            9, -9, 11, -87, 13, 43, 15, 70
        ),
        lin_i16x4!(
            c;
            1, 43, 3, -90, 5, 57, 7, 25,
            9, -87, 11, 70, 13, 9, 15, -80
        ),
        lin_i16x4!(
            c;
            1, 25, 3, -70, 5, 90, 7, -80,
            9, 43, 11, 9, 13, -57, 15, 87
        ),
        lin_i16x4!(
            c;
            1, 9, 3, -25, 5, 43, 7, -57,
            9, 70, 11, -80, 13, 87, 15, -90
        ),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn odd32_p(c: &[int16x4_t; 32]) -> [int32x4_t; 16] {
    [
        lin_i16x4!(
            c;
            1, 90, 3, 90, 5, 88, 7, 85,
            9, 82, 11, 78, 13, 73, 15, 67,
            17, 61, 19, 54, 21, 46, 23, 38,
            25, 31, 27, 22, 29, 13, 31, 4
        ),
        lin_i16x4!(
            c;
            1, 90, 3, 82, 5, 67, 7, 46,
            9, 22, 11, -4, 13, -31, 15, -54,
            17, -73, 19, -85, 21, -90, 23, -88,
            25, -78, 27, -61, 29, -38, 31, -13
        ),
        lin_i16x4!(
            c;
            1, 88, 3, 67, 5, 31, 7, -13,
            9, -54, 11, -82, 13, -90, 15, -78,
            17, -46, 19, -4, 21, 38, 23, 73,
            25, 90, 27, 85, 29, 61, 31, 22
        ),
        lin_i16x4!(
            c;
            1, 85, 3, 46, 5, -13, 7, -67,
            9, -90, 11, -73, 13, -22, 15, 38,
            17, 82, 19, 88, 21, 54, 23, -4,
            25, -61, 27, -90, 29, -78, 31, -31
        ),
        lin_i16x4!(
            c;
            1, 82, 3, 22, 5, -54, 7, -90,
            9, -61, 11, 13, 13, 78, 15, 85,
            17, 31, 19, -46, 21, -90, 23, -67,
            25, 4, 27, 73, 29, 88, 31, 38
        ),
        lin_i16x4!(
            c;
            1, 78, 3, -4, 5, -82, 7, -73,
            9, 13, 11, 85, 13, 67, 15, -22,
            17, -88, 19, -61, 21, 31, 23, 90,
            25, 54, 27, -38, 29, -90, 31, -46
        ),
        lin_i16x4!(
            c;
            1, 73, 3, -31, 5, -90, 7, -22,
            9, 78, 11, 67, 13, -38, 15, -90,
            17, -13, 19, 82, 21, 61, 23, -46,
            25, -88, 27, -4, 29, 85, 31, 54
        ),
        lin_i16x4!(
            c;
            1, 67, 3, -54, 5, -78, 7, 38,
            9, 85, 11, -22, 13, -90, 15, 4,
            17, 90, 19, 13, 21, -88, 23, -31,
            25, 82, 27, 46, 29, -73, 31, -61
        ),
        lin_i16x4!(
            c;
            1, 61, 3, -73, 5, -46, 7, 82,
            9, 31, 11, -88, 13, -13, 15, 90,
            17, -4, 19, -90, 21, 22, 23, 85,
            25, -38, 27, -78, 29, 54, 31, 67
        ),
        lin_i16x4!(
            c;
            1, 54, 3, -85, 5, -4, 7, 88,
            9, -46, 11, -61, 13, 82, 15, 13,
            17, -90, 19, 38, 21, 67, 23, -78,
            25, -22, 27, 90, 29, -31, 31, -73
        ),
        lin_i16x4!(
            c;
            1, 46, 3, -90, 5, 38, 7, 54,
            9, -90, 11, 31, 13, 61, 15, -88,
            17, 22, 19, 67, 21, -85, 23, 13,
            25, 73, 27, -82, 29, 4, 31, 78
        ),
        lin_i16x4!(
            c;
            1, 38, 3, -88, 5, 73, 7, -4,
            9, -67, 11, 90, 13, -46, 15, -31,
            17, 85, 19, -78, 21, 13, 23, 61,
            25, -90, 27, 54, 29, 22, 31, -82
        ),
        lin_i16x4!(
            c;
            1, 31, 3, -78, 5, 90, 7, -61,
            9, 4, 11, 54, 13, -88, 15, 82,
            17, -38, 19, -22, 21, 73, 23, -90,
            25, 67, 27, -13, 29, -46, 31, 85
        ),
        lin_i16x4!(
            c;
            1, 22, 3, -61, 5, 85, 7, -90,
            9, 73, 11, -38, 13, -4, 15, 46,
            17, -78, 19, 90, 21, -82, 23, 54,
            25, -13, 27, -31, 29, 67, 31, -88
        ),
        lin_i16x4!(
            c;
            1, 13, 3, -38, 5, 61, 7, -78,
            9, 88, 11, -90, 13, 85, 15, -73,
            17, 54, 19, -31, 21, 4, 23, 22,
            25, -46, 27, 67, 29, -82, 31, 90
        ),
        lin_i16x4!(
            c;
            1, 4, 3, -13, 5, 22, 7, -31,
            9, 38, 11, -46, 13, 54, 15, -61,
            17, 67, 19, -73, 21, 78, 23, -82,
            25, 85, 27, -88, 29, 90, 31, -90
        ),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn idct4_p(c: [int16x4_t; 4]) -> [int32x4_t; 4] {
    let e0 = vshlq_n_s32::<6>(vaddl_s16(c[0], c[2]));
    let e1 = vshlq_n_s32::<6>(vsubl_s16(c[0], c[2]));
    let o0 = vmlal_n_s16(vmull_n_s16(c[1], 83), c[3], 36);
    let o1 = vmlal_n_s16(vmull_n_s16(c[1], 36), c[3], -83);
    [add(e0, o0), add(e1, o1), sub(e1, o1), sub(e0, o0)]
}

#[inline]
#[target_feature(enable = "neon")]
fn idct8_p(c: &[int16x4_t; 8]) -> [int32x4_t; 8] {
    let ee = idct4_p([c[0], c[2], c[4], c[6]]);
    let oo = odd8_p(c);
    let mut out = [zero(); 8];
    for k in 0..4 {
        out[k] = add(ee[k], oo[k]);
        out[7 - k] = sub(ee[k], oo[k]);
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn idct16_p(c: &[int16x4_t; 16]) -> [int32x4_t; 16] {
    let ee = idct8_p(&std::array::from_fn(|j| c[2 * j]));
    let oo = odd16_p(c);
    let mut out = [zero(); 16];
    for k in 0..8 {
        out[k] = add(ee[k], oo[k]);
        out[15 - k] = sub(ee[k], oo[k]);
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn idct32_p(c: &[int16x4_t; 32]) -> [int32x4_t; 32] {
    let ee = idct16_p(&std::array::from_fn(|j| c[2 * j]));
    let oo = odd32_p(c);
    let mut out = [zero(); 32];
    for k in 0..16 {
        out[k] = add(ee[k], oo[k]);
        out[31 - k] = sub(ee[k], oo[k]);
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_p<const N: usize>(c: &[int16x4_t; N]) -> [int32x4_t; N] {
    debug_assert!(N == 4 || N == 8 || N == 16 || N == 32);
    match N {
        4 => {
            let r = idct4_p([c[0], c[1], c[2], c[3]]);
            std::array::from_fn(|i| r[i])
        }
        8 => {
            let src = std::array::from_fn(|i| c[i]);
            let r = idct8_p(&src);
            std::array::from_fn(|i| r[i])
        }
        16 => {
            let src = std::array::from_fn(|i| c[i]);
            let r = idct16_p(&src);
            std::array::from_fn(|i| r[i])
        }
        32 => {
            let src = std::array::from_fn(|i| c[i]);
            let r = idct32_p(&src);
            std::array::from_fn(|i| r[i])
        }
        _ => unreachable!(),
    }
}

/// Narrow 4 shifted i32x4 (lanes = 4 blocks) to i16 and store transposed 4x4.
#[inline]
#[target_feature(enable = "neon")]
fn tr_store_4x4(
    dst: &mut [i16],
    stride: usize,
    lane_base: usize,
    elem_base: usize,
    v: [int32x4_t; 4],
) {
    let n: [int16x4_t; 4] = std::array::from_fn(|j| vqmovn_s32(v[j]));
    let t01 = vtrn_s16(n[0], n[1]);
    let t23 = vtrn_s16(n[2], n[3]);
    let r0 = vtrn_s32(vreinterpret_s32_s16(t01.0), vreinterpret_s32_s16(t23.0));
    let r1 = vtrn_s32(vreinterpret_s32_s16(t01.1), vreinterpret_s32_s16(t23.1));
    st_i16x4(
        &mut dst[lane_base * stride + elem_base..],
        vreinterpret_s16_s32(r0.0),
    );
    st_i16x4(
        &mut dst[(lane_base + 1) * stride + elem_base..],
        vreinterpret_s16_s32(r1.0),
    );
    st_i16x4(
        &mut dst[(lane_base + 2) * stride + elem_base..],
        vreinterpret_s16_s32(r0.1),
    );
    st_i16x4(
        &mut dst[(lane_base + 3) * stride + elem_base..],
        vreinterpret_s16_s32(r1.1),
    );
}

#[target_feature(enable = "neon")]
fn inv_dct_n_into_neon_16<const N: usize>(
    coeff: &[i16],
    bit_depth: u8,
    nx: usize,
    out: &mut [i16],
) {
    debug_assert!(N == 4 || N == 8 || N == 16 || N == 32);
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    let mut tmp = [0i16; 32 * 32];

    // Columns >= nx are zero on input; skip them (tmp stays zero there).
    let ncol = ((nx.min(N) + 3) & !3).max(4);
    for c in (0..ncol).step_by(4) {
        let src: [int16x4_t; N] = std::array::from_fn(|k| ld_i16x4(&coeff[k * N + c..]));
        let raw = idct_p::<N>(&src);
        for m in (0..N).step_by(4) {
            let v = std::array::from_fn(|j| round_shift_s32x4(raw[m + j], add1, shift1));
            tr_store_4x4(&mut tmp, N, c, m, v);
        }
    }

    for r in (0..N).step_by(4) {
        let src: [int16x4_t; N] = std::array::from_fn(|k| ld_i16x4(&tmp[k * N + r..]));
        let raw = idct_p::<N>(&src);
        for x in (0..N).step_by(4) {
            let v = std::array::from_fn(|j| round_shift_s32x4(raw[x + j], add2, shift2));
            tr_store_4x4(out, N, r, x, v);
        }
    }
}

#[target_feature(enable = "neon")]
fn inv_transform_dst4_into_neon_16(coeff: &[i16], bit_depth: u8, out: &mut [i16]) {
    debug_assert!(coeff.len() >= 16);
    debug_assert!(out.len() >= 16);
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    let mut tmp = [0i16; 16];

    let c: [int16x4_t; 4] = std::array::from_fn(|k| ld_i16x4(&coeff[k * 4..]));
    let v = std::array::from_fn(|m| {
        let acc = vmlal_n_s16(
            vmlal_n_s16(
                vmlal_n_s16(
                    vmull_n_s16(c[0], DST4[0][m] as i16),
                    c[1],
                    DST4[1][m] as i16,
                ),
                c[2],
                DST4[2][m] as i16,
            ),
            c[3],
            DST4[3][m] as i16,
        );
        round_shift_s32x4(acc, add1, shift1)
    });
    tr_store_4x4(&mut tmp, 4, 0, 0, v);

    let c: [int16x4_t; 4] = std::array::from_fn(|k| ld_i16x4(&tmp[k * 4..]));
    let v = std::array::from_fn(|m| {
        let acc = vmlal_n_s16(
            vmlal_n_s16(
                vmlal_n_s16(
                    vmull_n_s16(c[0], DST4[0][m] as i16),
                    c[1],
                    DST4[1][m] as i16,
                ),
                c[2],
                DST4[2][m] as i16,
            ),
            c[3],
            DST4[3][m] as i16,
        );
        round_shift_s32x4(acc, add2, shift2)
    });
    tr_store_4x4(out, 4, 0, 0, v);
}

pub(crate) fn inv_transform_into_neon16(
    coeff: &[i16],
    n: usize,
    bit_depth: u8,
    nx: usize,
    out: &mut [i16],
) {
    unsafe {
        match n {
            4 => inv_dct_n_into_neon_16::<4>(coeff, bit_depth, nx, out),
            8 => inv_dct_n_into_neon_16::<8>(coeff, bit_depth, nx, out),
            16 => inv_dct_n_into_neon_16::<16>(coeff, bit_depth, nx, out),
            32 => inv_dct_n_into_neon_16::<32>(coeff, bit_depth, nx, out),
            _ => inv_transform_into_scalar16(coeff, n, bit_depth, nx, out),
        }
    }
}

pub(crate) fn inv_transform_dst_into_neon16(coeff: &[i16], bit_depth: u8, out: &mut [i16]) {
    unsafe { inv_transform_dst4_into_neon_16(coeff, bit_depth, out) }
}
