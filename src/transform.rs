/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

/// 4×4 HEVC transform matrix.
static T4: [[i32; 4]; 4] = [
    [64, 64, 64, 64],
    [83, 36, -36, -83],
    [64, -64, -64, 64],
    [36, -83, 83, -36],
];

/// 8×8 HEVC transform matrix.
static T8: [[i32; 8]; 8] = [
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18],
];

static T16: [[i32; 16]; 16] = [
    [
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    ],
    [
        90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90,
    ],
    [
        89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
    ],
    [
        87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87,
    ],
    [
        83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
    ],
    [
        80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80,
    ],
    [
        75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
    ],
    [
        70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70,
    ],
    [
        64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
    ],
    [
        57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57,
    ],
    [
        50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
    ],
    [
        43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43,
    ],
    [
        36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
    ],
    [
        25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25,
    ],
    [
        18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
    ],
    [
        9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9,
    ],
];

static T32: [[i32; 32]; 32] = [
    [
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 64,
    ],
    [
        90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46,
        -54, -61, -67, -73, -78, -82, -85, -88, -90, -90,
    ],
    [
        90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70,
        -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90,
    ],
    [
        90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78,
        88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90,
    ],
    [
        89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18,
        -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
    ],
    [
        88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90,
        -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88,
    ],
    [
        87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80,
        90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87,
    ],
    [
        85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4,
        -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85,
    ],
    [
        83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83,
        -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
    ],
    [
        82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67,
        90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82,
    ],
    [
        80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25,
        -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80,
    ],
    [
        78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54,
        -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78,
    ],
    [
        75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50,
        50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
    ],
    [
        73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88,
        46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73,
    ],
    [
        70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90,
        -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70,
    ],
    [
        67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82,
        31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67,
    ],
    [
        64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
        64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
    ],
    [
        61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38,
        -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61,
    ],
    [
        57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9,
        87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57,
    ],
    [
        54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22,
        78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54,
    ],
    [
        50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75,
        -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
    ],
    [
        46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13,
        85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46,
    ],
    [
        43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25,
        87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43,
    ],
    [
        38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90,
        -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38,
    ],
    [
        36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36,
        -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
    ],
    [
        31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67,
        90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31,
    ],
    [
        25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80,
        -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25,
    ],
    [
        22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13,
        -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22,
    ],
    [
        18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89,
        89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
    ],
    [
        13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46,
        -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13,
    ],
    [
        9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70,
        80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9,
    ],
    [
        4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85,
        82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4,
    ],
];

static DST4: [[i32; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

static DEQUANT_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

#[inline(always)]
fn idct_raw_4(c: [i32; 4]) -> [i32; 4] {
    let mut e = [0i32; 4];
    for j in 0..4 {
        let cj = c[j];
        if cj == 0 {
            continue;
        }
        let trow = T4[j];
        for k in 0..4 {
            e[k] += trow[k] * cj;
        }
    }
    e
}

#[inline(always)]
fn idct_raw_8(c: [i32; 8]) -> [i32; 8] {
    let ee = idct_raw_4([c[0], c[2], c[4], c[6]]);
    let mut oo = [0i32; 4];
    for j in 0..4 {
        let co = c[2 * j + 1];
        if co == 0 {
            continue;
        }
        let trow = T8[2 * j + 1]; // odd row of T8, anti-sym: first 4 cols used
        for k in 0..4 {
            oo[k] += trow[k] * co;
        }
    }
    let mut out = [0i32; 8];
    for k in 0..4 {
        out[k] = ee[k] + oo[k];
        out[7 - k] = ee[k] - oo[k];
    }
    out
}

#[inline(always)]
fn idct_raw_16(c: [i32; 16]) -> [i32; 16] {
    let ee = idct_raw_8(std::array::from_fn(|j| c[2 * j]));
    let mut oo = [0i32; 8];
    for j in 0..8 {
        let co = c[2 * j + 1];
        if co == 0 {
            continue;
        }
        let trow = T16[2 * j + 1]; // odd row of T16, first 8 cols used
        for k in 0..8 {
            oo[k] += trow[k] * co;
        }
    }
    let mut out = [0i32; 16];
    for k in 0..8 {
        out[k] = ee[k] + oo[k];
        out[15 - k] = ee[k] - oo[k];
    }
    out
}

#[inline(always)]
fn idct_raw_32(c: [i32; 32]) -> [i32; 32] {
    let ee = idct_raw_16(std::array::from_fn(|j| c[2 * j]));
    let mut oo = [0i32; 16];
    for j in 0..16 {
        let co = c[2 * j + 1];
        if co == 0 {
            continue;
        }
        let trow = T32[2 * j + 1];
        for k in 0..16 {
            oo[k] += trow[k] * co;
        }
    }
    let mut out = [0i32; 32];
    for k in 0..16 {
        out[k] = ee[k] + oo[k];
        out[31 - k] = ee[k] - oo[k];
    }
    out
}

/// 2-D 32×32 partial butterfly IDCT into `out[..1024]`.
#[inline]
fn inv_butterfly_32_into(coeff: &[i32], bit_depth: u8, out: &mut [i32]) {
    const N: usize = 32;
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    let mut tmp = [0i32; N * N];
    for c in 0..N {
        let col: [i32; N] = std::array::from_fn(|k| coeff[k * N + c]);
        let raw = idct_raw_32(col);
        for m in 0..N {
            tmp[m * N + c] = ((raw[m] + add1) >> shift1).clamp(-32768, 32767);
        }
    }
    for r in 0..N {
        let row: [i32; N] = std::array::from_fn(|k| tmp[r * N + k]);
        let raw = idct_raw_32(row);
        for m in 0..N {
            out[r * N + m] = (raw[m] + add2) >> shift2;
        }
    }
}

/// Allocation-free inverse integer transform.
///
/// Two key optimizations over a naive dense matrix multiply:
///   * **Sparse skip** — residual blocks are typically zero except for a few
///     low-frequency coefficients, so each zero input contributes nothing and
///     is skipped before the inner accumulation runs at all.
///   * **Cache-friendly access** — the basis row `t[k]` is read contiguously
///     while accumulating into all `N` outputs, instead of striding down the
///     `m`-th column of `t` (stride `N`) for every output.
#[inline]
fn inv_transform_n_into<const N: usize>(
    coeff: &[i32],
    t: &[[i32; N]; N],
    bit_depth: u8,
    out: &mut [i32],
) {
    let bd = bit_depth as i32;
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bd;
    let add2 = 1i32 << (shift2 - 1);

    let mut tmp = [0i32; 32 * 32];
    let mut acc = [0i32; N];

    for c in 0..N {
        acc[..N].fill(0);
        for k in 0..N {
            let ck = coeff[k * N + c];
            if ck == 0 {
                continue; // sparse skip — most residual coeffs are zero
            }
            let trow = &t[k];
            for m in 0..N {
                acc[m] += trow[m] * ck;
            }
        }
        for m in 0..N {
            tmp[m * N + c] = ((acc[m] + add1) >> shift1).clamp(-32768, 32767);
        }
    }

    for r in 0..N {
        acc[..N].fill(0);
        let rowv = &tmp[r * N..r * N + N];
        for k in 0..N {
            let rk = rowv[k];
            if rk == 0 {
                continue;
            }
            let trow = &t[k];
            for m in 0..N {
                acc[m] += trow[m] * rk;
            }
        }
        for m in 0..N {
            out[r * N + m] = (acc[m] + add2) >> shift2;
        }
    }
}

/// Dequantize into `out[..n*n]` — avoids a heap allocation per transform block.
pub(crate) fn dequantize_i32_into(
    levels: &[i32],
    n: usize,
    qp: u8,
    bit_depth: u8,
    out: &mut [i32],
) {
    let log2n = (n as u32).trailing_zeros() as i64;
    let bd = bit_depth as i64;
    let bd_shift = (bd + log2n - 5).max(1);
    let add = 1i64 << (bd_shift - 1);
    let qp_bd_offset = 6 * (bd - 8);
    let qp_scaled = (qp as i64) + qp_bd_offset;
    let scale = DEQUANT_SCALE[(qp_scaled % 6) as usize];
    let per = 1i64 << (qp_scaled / 6);
    let factor = scale * per * 16;
    for (o, &l) in out[..n * n].iter_mut().zip(levels) {
        // Intermediate stays i64 (l*factor can exceed i32 for high QP); the
        // clamped result is always within i16 range, so storing as i32 is exact.
        *o = ((l as i64 * factor + add) >> bd_shift).clamp(-32768, 32767) as i32;
    }
}

/// Inverse DCT into `out[..n*n]` — no heap allocation.
pub(crate) fn inv_transform_into(coeff: &[i32], n: usize, bit_depth: u8, out: &mut [i32]) {
    match n {
        4 => inv_transform_n_into::<4>(coeff, &T4, bit_depth, out),
        8 => inv_transform_n_into::<8>(coeff, &T8, bit_depth, out),
        16 => inv_transform_n_into::<16>(coeff, &T16, bit_depth, out),
        32 => inv_butterfly_32_into(coeff, bit_depth, &mut out[..1024]),
        _ => panic!("unsupported transform size {n}"),
    }
}

/// Inverse 4×4 DST into `out[..16]` — no heap allocation.
pub(crate) fn inv_transform_dst_into(coeff: &[i32], bit_depth: u8, out: &mut [i32]) {
    inv_transform_n_into::<4>(coeff, &DST4, bit_depth, out);
}
