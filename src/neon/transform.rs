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

use crate::transform::{DST4, inv_transform_into_scalar};

static ODD16: [[i32; 8]; 8] = [
    [90, 87, 80, 70, 57, 43, 25, 9],
    [87, 57, 9, -43, -80, -90, -70, -25],
    [80, 9, -70, -87, -25, 57, 90, 43],
    [70, -43, -87, 9, 90, 25, -80, -57],
    [57, -80, -25, 90, -9, -87, 43, 70],
    [43, -90, 57, 25, -87, 70, 9, -80],
    [25, -70, 90, -80, 43, 9, -57, 87],
    [9, -25, 43, -57, 70, -80, 87, -90],
];

static ODD32: [[i32; 16]; 16] = [
    [
        90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4,
    ],
    [
        90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13,
    ],
    [
        88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22,
    ],
    [
        85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31,
    ],
    [
        82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38,
    ],
    [
        78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46,
    ],
    [
        73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54,
    ],
    [
        67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61,
    ],
    [
        61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67,
    ],
    [
        54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73,
    ],
    [
        46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78,
    ],
    [
        38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82,
    ],
    [
        31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85,
    ],
    [
        22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88,
    ],
    [
        13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90,
    ],
    [
        4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
    ],
];

#[inline]
#[target_feature(enable = "neon")]
fn load_s32x4(src: &[i32]) -> int32x4_t {
    debug_assert!(src.len() >= 4);
    unsafe { vld1q_s32(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_rows4_s32x4(src: &[i32], stride: usize, x: usize) -> int32x4_t {
    debug_assert!(src.len() > 3 * stride + x);
    let lanes = [
        src[x],
        src[stride + x],
        src[2 * stride + x],
        src[3 * stride + x],
    ];
    unsafe { vld1q_s32(lanes.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_s32x4(dst: &mut [i32], v: int32x4_t) {
    debug_assert!(dst.len() >= 4);
    unsafe { vst1q_s32(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_rows4_s32x4(dst: &mut [i32], stride: usize, x: usize, v: int32x4_t) {
    debug_assert!(dst.len() > 3 * stride + x);
    let mut lanes = [0i32; 4];
    unsafe { vst1q_s32(lanes.as_mut_ptr(), v) };
    dst[x] = lanes[0];
    dst[stride + x] = lanes[1];
    dst[2 * stride + x] = lanes[2];
    dst[3 * stride + x] = lanes[3];
}

#[inline]
#[target_feature(enable = "neon")]
fn zero() -> int32x4_t {
    vdupq_n_s32(0)
}

#[inline]
#[target_feature(enable = "neon")]
fn add(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vaddq_s32(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
fn sub(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vsubq_s32(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
fn mul_const(v: int32x4_t, c: i32) -> int32x4_t {
    vmulq_s32(v, vdupq_n_s32(c))
}

#[inline]
#[target_feature(enable = "neon")]
fn madd_const(acc: int32x4_t, v: int32x4_t, c: i32) -> int32x4_t {
    add(acc, mul_const(v, c))
}

#[inline]
#[target_feature(enable = "neon")]
fn round_shift_s32x4(v: int32x4_t, add: i32, shift: i32) -> int32x4_t {
    vshlq_s32(vaddq_s32(v, vdupq_n_s32(add)), vdupq_n_s32(-shift))
}

#[inline]
#[target_feature(enable = "neon")]
fn round_shift_clip_i16_s32x4(v: int32x4_t, add: i32, shift: i32) -> int32x4_t {
    let v = round_shift_s32x4(v, add, shift);
    vmaxq_s32(vminq_s32(v, vdupq_n_s32(32767)), vdupq_n_s32(-32768))
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
    let mut oo = [zero(); 8];

    for j in 0..8 {
        let co = c[2 * j + 1];
        for (acc, &tk) in oo.iter_mut().zip(ODD16[j].iter()) {
            *acc = madd_const(*acc, co, tk);
        }
    }

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
    let mut oo = [zero(); 16];

    for j in 0..16 {
        let co = c[2 * j + 1];
        for (acc, &tk) in oo.iter_mut().zip(ODD32[j].iter()) {
            *acc = madd_const(*acc, co, tk);
        }
    }

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
fn inv_dct_n_into_neon<const N: usize>(coeff: &[i32], bit_depth: u8, out: &mut [i32]) {
    debug_assert!(N == 4 || N == 8 || N == 16 || N == 32);
    debug_assert!(coeff.len() >= N * N);
    debug_assert!(out.len() >= N * N);

    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    let mut tmp = [0i32; 32 * 32];

    for c in (0..N).step_by(4) {
        let src = std::array::from_fn(|k| load_s32x4(&coeff[k * N + c..]));
        let raw = idct_raw_s32x4::<N>(src);
        for (m, raw) in raw.iter().copied().enumerate() {
            let v = round_shift_clip_i16_s32x4(raw, add1, shift1);
            store_s32x4(&mut tmp[m * N + c..], v);
        }
    }

    for r in (0..N).step_by(4) {
        let src = std::array::from_fn(|k| load_rows4_s32x4(&tmp[r * N..], N, k));
        let raw = idct_raw_s32x4::<N>(src);
        for (x, raw) in raw.iter().copied().enumerate() {
            let v = round_shift_s32x4(raw, add2, shift2);
            store_rows4_s32x4(&mut out[r * N..], N, x, v);
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
    for (m, acc) in acc.iter().copied().enumerate() {
        let v = round_shift_clip_i16_s32x4(acc, add1, shift1);
        store_s32x4(&mut tmp[m * 4..], v);
    }

    let src: [int32x4_t; 4] = std::array::from_fn(|k| load_rows4_s32x4(&tmp, 4, k));
    for x in 0..4 {
        let mut acc = zero();
        for (rk, trow) in src.iter().copied().zip(DST4.iter()) {
            acc = madd_const(acc, rk, trow[x]);
        }
        let v = round_shift_s32x4(acc, add2, shift2);
        store_rows4_s32x4(out, 4, x, v);
    }
}

pub(crate) fn inv_transform_into_neon(coeff: &[i32], n: usize, bit_depth: u8, out: &mut [i32]) {
    unsafe {
        match n {
            4 => inv_dct_n_into_neon::<4>(coeff, bit_depth, out),
            8 => inv_dct_n_into_neon::<8>(coeff, bit_depth, out),
            16 => inv_dct_n_into_neon::<16>(coeff, bit_depth, out),
            32 => inv_dct_n_into_neon::<32>(coeff, bit_depth, out),
            _ => inv_transform_into_scalar(coeff, n, bit_depth, out),
        }
    }
}

pub(crate) fn inv_transform_dst_into_neon(coeff: &[i32], bit_depth: u8, out: &mut [i32]) {
    unsafe { inv_transform_dst4_into_neon(coeff, bit_depth, out) }
}
