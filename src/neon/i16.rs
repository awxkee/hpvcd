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
    ($c:expr; $active:expr; $i0:expr, $k0:expr $(, $i:expr, $k:expr)+ $(,)?) => {{
        let mut acc = vmull_n_s16($c[$i0], $k0);
        $(
            if $i < $active {
                acc = vmlal_n_s16(acc, $c[$i], $k);
            }
        )+
        acc
    }};
}

#[inline]
#[target_feature(enable = "neon")]
fn odd8_p<const ACTIVE: usize>(c: &[int16x4_t; ACTIVE]) -> [int32x4_t; 4] {
    [
        lin_i16x4!(
            c; ACTIVE;
            1, 89, 3, 75, 5, 50, 7, 18
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 75, 3, -18, 5, -89, 7, -50
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 50, 3, -89, 5, 18, 7, 75
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 18, 3, -50, 5, 75, 7, -89
        ),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn odd16_p<const ACTIVE: usize>(c: &[int16x4_t; ACTIVE]) -> [int32x4_t; 8] {
    [
        lin_i16x4!(
            c; ACTIVE;
            1, 90, 3, 87, 5, 80, 7, 70,
            9, 57, 11, 43, 13, 25, 15, 9
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 87, 3, 57, 5, 9, 7, -43,
            9, -80, 11, -90, 13, -70, 15, -25
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 80, 3, 9, 5, -70, 7, -87,
            9, -25, 11, 57, 13, 90, 15, 43
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 70, 3, -43, 5, -87, 7, 9,
            9, 90, 11, 25, 13, -80, 15, -57
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 57, 3, -80, 5, -25, 7, 90,
            9, -9, 11, -87, 13, 43, 15, 70
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 43, 3, -90, 5, 57, 7, 25,
            9, -87, 11, 70, 13, 9, 15, -80
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 25, 3, -70, 5, 90, 7, -80,
            9, 43, 11, 9, 13, -57, 15, 87
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 9, 3, -25, 5, 43, 7, -57,
            9, 70, 11, -80, 13, 87, 15, -90
        ),
    ]
}

#[inline]
#[target_feature(enable = "neon")]
fn odd32_p<const ACTIVE: usize>(c: &[int16x4_t; ACTIVE]) -> [int32x4_t; 16] {
    [
        lin_i16x4!(
            c; ACTIVE;
            1, 90, 3, 90, 5, 88, 7, 85,
            9, 82, 11, 78, 13, 73, 15, 67,
            17, 61, 19, 54, 21, 46, 23, 38,
            25, 31, 27, 22, 29, 13, 31, 4
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 90, 3, 82, 5, 67, 7, 46,
            9, 22, 11, -4, 13, -31, 15, -54,
            17, -73, 19, -85, 21, -90, 23, -88,
            25, -78, 27, -61, 29, -38, 31, -13
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 88, 3, 67, 5, 31, 7, -13,
            9, -54, 11, -82, 13, -90, 15, -78,
            17, -46, 19, -4, 21, 38, 23, 73,
            25, 90, 27, 85, 29, 61, 31, 22
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 85, 3, 46, 5, -13, 7, -67,
            9, -90, 11, -73, 13, -22, 15, 38,
            17, 82, 19, 88, 21, 54, 23, -4,
            25, -61, 27, -90, 29, -78, 31, -31
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 82, 3, 22, 5, -54, 7, -90,
            9, -61, 11, 13, 13, 78, 15, 85,
            17, 31, 19, -46, 21, -90, 23, -67,
            25, 4, 27, 73, 29, 88, 31, 38
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 78, 3, -4, 5, -82, 7, -73,
            9, 13, 11, 85, 13, 67, 15, -22,
            17, -88, 19, -61, 21, 31, 23, 90,
            25, 54, 27, -38, 29, -90, 31, -46
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 73, 3, -31, 5, -90, 7, -22,
            9, 78, 11, 67, 13, -38, 15, -90,
            17, -13, 19, 82, 21, 61, 23, -46,
            25, -88, 27, -4, 29, 85, 31, 54
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 67, 3, -54, 5, -78, 7, 38,
            9, 85, 11, -22, 13, -90, 15, 4,
            17, 90, 19, 13, 21, -88, 23, -31,
            25, 82, 27, 46, 29, -73, 31, -61
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 61, 3, -73, 5, -46, 7, 82,
            9, 31, 11, -88, 13, -13, 15, 90,
            17, -4, 19, -90, 21, 22, 23, 85,
            25, -38, 27, -78, 29, 54, 31, 67
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 54, 3, -85, 5, -4, 7, 88,
            9, -46, 11, -61, 13, 82, 15, 13,
            17, -90, 19, 38, 21, 67, 23, -78,
            25, -22, 27, 90, 29, -31, 31, -73
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 46, 3, -90, 5, 38, 7, 54,
            9, -90, 11, 31, 13, 61, 15, -88,
            17, 22, 19, 67, 21, -85, 23, 13,
            25, 73, 27, -82, 29, 4, 31, 78
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 38, 3, -88, 5, 73, 7, -4,
            9, -67, 11, 90, 13, -46, 15, -31,
            17, 85, 19, -78, 21, 13, 23, 61,
            25, -90, 27, 54, 29, 22, 31, -82
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 31, 3, -78, 5, 90, 7, -61,
            9, 4, 11, 54, 13, -88, 15, 82,
            17, -38, 19, -22, 21, 73, 23, -90,
            25, 67, 27, -13, 29, -46, 31, 85
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 22, 3, -61, 5, 85, 7, -90,
            9, 73, 11, -38, 13, -4, 15, 46,
            17, -78, 19, 90, 21, -82, 23, 54,
            25, -13, 27, -31, 29, 67, 31, -88
        ),
        lin_i16x4!(
            c; ACTIVE;
            1, 13, 3, -38, 5, 61, 7, -78,
            9, 88, 11, -90, 13, 85, 15, -73,
            17, 54, 19, -31, 21, 4, 23, 22,
            25, -46, 27, 67, 29, -82, 31, 90
        ),
        lin_i16x4!(
            c; ACTIVE;
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
fn idct8_p<const ACTIVE: usize>(c: &[int16x4_t; ACTIVE]) -> [int32x4_t; 8] {
    let ee = if ACTIVE == 2 {
        [vshll_n_s16::<6>(c[0]); 4]
    } else if ACTIVE == 4 {
        idct4_p([c[0], c[2], vdup_n_s16(0), vdup_n_s16(0)])
    } else {
        idct4_p([c[0], c[2], c[4], c[6]])
    };
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
fn idct16_p<const ACTIVE: usize>(c: &[int16x4_t; ACTIVE]) -> [int32x4_t; 16] {
    let ee = if ACTIVE == 2 {
        [vshll_n_s16::<6>(c[0]); 8]
    } else if ACTIVE == 4 {
        idct8_p(&[c[0], c[2]])
    } else if ACTIVE == 8 {
        idct8_p::<4>(&std::array::from_fn(|j| c[2 * j]))
    } else {
        idct8_p::<8>(&std::array::from_fn(|j| c[2 * j]))
    };
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
fn idct32_p<const ACTIVE: usize>(c: &[int16x4_t; ACTIVE]) -> [int32x4_t; 32] {
    let ee = match ACTIVE {
        4 => idct16_p::<2>(&std::array::from_fn(|j| c[2 * j])),
        8 => idct16_p::<4>(&std::array::from_fn(|j| c[2 * j])),
        16 => idct16_p::<8>(&std::array::from_fn(|j| c[2 * j])),
        _ => idct16_p::<16>(&std::array::from_fn(|j| c[2 * j])),
    };
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
            let r = idct8_p::<8>(&src);
            std::array::from_fn(|i| r[i])
        }
        16 => {
            let src = std::array::from_fn(|i| c[i]);
            let r = idct16_p::<16>(&src);
            std::array::from_fn(|i| r[i])
        }
        32 => {
            let src = std::array::from_fn(|i| c[i]);
            let r = idct32_p::<32>(&src);
            std::array::from_fn(|i| r[i])
        }
        _ => unreachable!(),
    }
}

/// Store four i16x4 vectors as a transposed 4x4 block.
#[inline]
#[target_feature(enable = "neon")]
fn tr_store_4x4(
    dst: &mut [i16],
    stride: usize,
    lane_base: usize,
    elem_base: usize,
    v: [int16x4_t; 4],
) {
    let t01 = vtrn_s16(v[0], v[1]);
    let t23 = vtrn_s16(v[2], v[3]);
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

#[inline(never)]
#[target_feature(enable = "neon")]
fn inv_dct32_pruned_second_pass<const ACTIVE: usize>(
    tmp: &[i16; 32 * 32],
    shift: int32x4_t,
    out: &mut [i16],
) {
    for r in (0..32).step_by(4) {
        let src = std::array::from_fn(|k| ld_i16x4(&tmp[k * 32 + r..]));
        let raw = idct32_p::<ACTIVE>(&src);
        for x in (0..32).step_by(4) {
            let v = std::array::from_fn(|j| vqmovn_s32(vrshlq_s32(raw[x + j], shift)));
            tr_store_4x4(out, 32, r, x, v);
        }
    }
}

#[target_feature(enable = "neon")]
fn inv_dct_n_into_neon_16<const N: usize>(
    coeff: &[i16],
    bit_depth: u8,
    nx: usize,
    coded_groups: u64,
    out: &mut [i16],
) {
    debug_assert!(N == 4 || N == 8 || N == 16 || N == 32);
    let shift2 = vdupq_n_s32(i32::from(bit_depth) - 20);
    let mut tmp = [0i16; 32 * 32];

    // Columns >= nx are zero on input; skip them (tmp stays zero there).
    let ncol = ((nx.min(N) + 3) & !3).max(4);
    for c in (0..ncol).step_by(4) {
        let active_groups = if N == 32 && coded_groups != u64::MAX {
            // One bit per group row in this strip of four input columns.
            let rows = coded_groups & (0x0101_0101_0101_0101 << (c / 4));
            (u64::BITS - rows.leading_zeros()).div_ceil(8)
        } else {
            (N / 4) as u32
        };
        let raw = if N == 32 && active_groups <= 4 {
            let raw = match active_groups {
                0 => continue,
                1 => idct32_p::<4>(&std::array::from_fn(|k| ld_i16x4(&coeff[k * N + c..]))),
                2 => idct32_p::<8>(&std::array::from_fn(|k| ld_i16x4(&coeff[k * N + c..]))),
                _ => idct32_p::<16>(&std::array::from_fn(|k| ld_i16x4(&coeff[k * N + c..]))),
            };
            std::array::from_fn(|j| raw[j])
        } else {
            let src = std::array::from_fn(|k| ld_i16x4(&coeff[k * N + c..]));
            idct_p::<N>(&src)
        };
        for m in (0..N).step_by(4) {
            let v = std::array::from_fn(|j| vqrshrn_n_s32::<7>(raw[m + j]));
            tr_store_4x4(&mut tmp, N, c, m, v);
        }
    }

    if N == 32 && coded_groups != u64::MAX && ncol <= 16 {
        // The first pass left every transposed row after ncol at zero.
        match ncol {
            4 => inv_dct32_pruned_second_pass::<4>(&tmp, shift2, out),
            8 => inv_dct32_pruned_second_pass::<8>(&tmp, shift2, out),
            _ => inv_dct32_pruned_second_pass::<16>(&tmp, shift2, out),
        }
        return;
    }

    for r in (0..N).step_by(4) {
        let src: [int16x4_t; N] = std::array::from_fn(|k| ld_i16x4(&tmp[k * N + r..]));
        let raw = idct_p::<N>(&src);
        for x in (0..N).step_by(4) {
            let v = std::array::from_fn(|j| vqmovn_s32(vrshlq_s32(raw[x + j], shift2)));
            tr_store_4x4(out, N, r, x, v);
        }
    }
}

#[target_feature(enable = "neon")]
fn inv_transform_dst4_into_neon_16(coeff: &[i16], bit_depth: u8, out: &mut [i16]) {
    debug_assert!(coeff.len() >= 16);
    debug_assert!(out.len() >= 16);
    let shift2 = vdupq_n_s32(i32::from(bit_depth) - 20);
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
        vqrshrn_n_s32::<7>(acc)
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
        vqmovn_s32(vrshlq_s32(acc, shift2))
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
            4 => inv_dct_n_into_neon_16::<4>(coeff, bit_depth, nx, u64::MAX, out),
            8 => inv_dct_n_into_neon_16::<8>(coeff, bit_depth, nx, u64::MAX, out),
            16 => inv_dct_n_into_neon_16::<16>(coeff, bit_depth, nx, u64::MAX, out),
            32 => inv_dct_n_into_neon_16::<32>(coeff, bit_depth, nx, u64::MAX, out),
            _ => inv_transform_into_scalar16(coeff, n, bit_depth, nx, out),
        }
    }
}

/// Inverse DCT with one coded flag per 4×4 input group, in row-major order.
pub(crate) fn inv_transform_coded32_into_neon16(
    coeff: &[i16],
    bit_depth: u8,
    nx: usize,
    coded_groups: u64,
    out: &mut [i16],
) {
    let coeff = &coeff[..32 * 32];
    let out = &mut out[..32 * 32];
    // SAFETY: the decoder uses the AArch64 NEON backend and both blocks are full-sized.
    unsafe { inv_dct_n_into_neon_16::<32>(coeff, bit_depth, nx, coded_groups, out) }
}

pub(crate) fn inv_transform_dst_into_neon16(coeff: &[i16], bit_depth: u8, out: &mut [i16]) {
    unsafe { inv_transform_dst4_into_neon_16(coeff, bit_depth, out) }
}

#[cfg(test)]
mod tests {
    use super::{
        inv_transform_coded32_into_neon16, inv_transform_dst_into_neon16, inv_transform_into_neon16,
    };

    #[test]
    fn sparse_32_second_pass_matches_dense_neon() {
        const N: usize = 32;
        for nx in [1usize, 4, 5, 8, 9, 12, 13, 16, 17, 32] {
            let coeff: [i16; N * N] =
                std::array::from_fn(|i| if i % N < nx { (i * 7_313) as i16 } else { 0 });
            // Every active strip has coefficients in all group rows, so its
            // first pass is dense in both invocations.
            let row_mask = (1u64 << nx.div_ceil(4)) - 1;
            let groups = row_mask * 0x0101_0101_0101_0101;
            for bit_depth in [8, 10, 12] {
                let mut dense = [0i16; N * N];
                let mut sparse = [0i16; N * N];
                inv_transform_into_neon16(&coeff, N, bit_depth, nx, &mut dense);
                inv_transform_coded32_into_neon16(&coeff, bit_depth, nx, groups, &mut sparse);
                assert_eq!(sparse, dense, "{nx} columns, bit depth {bit_depth}");
            }
        }
    }

    #[test]
    fn sparse_32_first_pass_matches_dense_neon() {
        const N: usize = 32;
        let shapes = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [2, 5, 3, 0, 1, 4, 2, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
            [1, 2, 4, 0, 2, 0, 0, 0],
        ];
        for shape in shapes {
            let mut coeff = [0i16; N * N];
            let mut groups = 0u64;
            let mut nx = 0;
            for (col, &rows) in shape.iter().enumerate() {
                for row in 0..rows {
                    // Make holes in a strip without removing its last group.
                    if row % 3 == 1 && row + 1 < rows {
                        continue;
                    }
                    groups |= 1 << (row * 8 + col);
                    nx = 4 * col + 4;
                    for y in row * 4..row * 4 + 4 {
                        for x in col * 4..col * 4 + 4 {
                            coeff[y * N + x] = (y * 7_313 + x * 3_973) as i16;
                        }
                    }
                }
            }
            for bit_depth in [8, 10, 12] {
                let mut dense = [0i16; N * N];
                let mut sparse = [0i16; N * N];
                inv_transform_into_neon16(&coeff, N, bit_depth, nx, &mut dense);
                inv_transform_coded32_into_neon16(&coeff, bit_depth, nx, groups, &mut sparse);
                assert_eq!(sparse, dense, "{shape:?}, bit depth {bit_depth}");
            }
        }
    }
    use crate::transform::{inv_transform_dst_into_scalar16, inv_transform_into_scalar16};

    #[test]
    fn rounded_i16_neon_transforms_match_scalar() {
        let coeff: [i16; 32 * 32] = std::array::from_fn(|i| ((i * 79) % 127) as i16 - 63);
        let mut expected = [0i16; 32 * 32];
        let mut actual = [0i16; 32 * 32];
        for bit_depth in [8, 10, 12] {
            inv_transform_dst_into_scalar16(&coeff[..16], bit_depth, &mut expected[..16]);
            inv_transform_dst_into_neon16(&coeff[..16], bit_depth, &mut actual[..16]);
            assert_eq!(&actual[..16], &expected[..16], "DST bit depth {bit_depth}");

            for n in [4, 8, 16, 32] {
                let count = n * n;
                inv_transform_into_scalar16(
                    &coeff[..count],
                    n,
                    bit_depth,
                    n,
                    &mut expected[..count],
                );
                inv_transform_into_neon16(&coeff[..count], n, bit_depth, n, &mut actual[..count]);
                assert_eq!(
                    &actual[..count],
                    &expected[..count],
                    "DCT {n}x{n}, bit depth {bit_depth}"
                );
            }
        }
    }
}
