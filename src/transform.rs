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

pub(crate) static DST4: [[i32; 4]; 4] = [
    [29, 55, 74, 84],
    [74, 74, 0, -74],
    [84, -29, -74, 55],
    [55, -84, 74, -29],
];

const DEQUANT_SCALE: [i64; 6] = [40, 45, 51, 57, 64, 72];

/// Coefficient/residual storage element: `i16` for 8-bit depth, `i32` for 10/12-bit.
/// Butterfly accumulators are always i32; this only picks the in/out storage width.
pub(crate) trait Coeff: Copy + Default + Send + Sync + 'static {
    fn from_i32(v: i32) -> Self;
    fn to_i32(self) -> i32;
}
impl Coeff for i32 {
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self
    }
}
impl Coeff for i16 {
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as i16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
}

#[inline(always)]
fn idct_odd_8(c: [i32; 8]) -> [i32; 4] {
    [
        89 * c[1] + 75 * c[3] + 50 * c[5] + 18 * c[7],
        75 * c[1] + -18 * c[3] + -89 * c[5] + -50 * c[7],
        50 * c[1] + -89 * c[3] + 18 * c[5] + 75 * c[7],
        18 * c[1] + -50 * c[3] + 75 * c[5] + -89 * c[7],
    ]
}
#[inline(always)]
fn idct_odd_16(c: [i32; 16]) -> [i32; 8] {
    [
        90 * c[1]
            + 87 * c[3]
            + 80 * c[5]
            + 70 * c[7]
            + 57 * c[9]
            + 43 * c[11]
            + 25 * c[13]
            + 9 * c[15],
        87 * c[1]
            + 57 * c[3]
            + 9 * c[5]
            + -43 * c[7]
            + -80 * c[9]
            + -90 * c[11]
            + -70 * c[13]
            + -25 * c[15],
        80 * c[1]
            + 9 * c[3]
            + -70 * c[5]
            + -87 * c[7]
            + -25 * c[9]
            + 57 * c[11]
            + 90 * c[13]
            + 43 * c[15],
        70 * c[1]
            + -43 * c[3]
            + -87 * c[5]
            + 9 * c[7]
            + 90 * c[9]
            + 25 * c[11]
            + -80 * c[13]
            + -57 * c[15],
        57 * c[1]
            + -80 * c[3]
            + -25 * c[5]
            + 90 * c[7]
            + -9 * c[9]
            + -87 * c[11]
            + 43 * c[13]
            + 70 * c[15],
        43 * c[1]
            + -90 * c[3]
            + 57 * c[5]
            + 25 * c[7]
            + -87 * c[9]
            + 70 * c[11]
            + 9 * c[13]
            + -80 * c[15],
        25 * c[1]
            + -70 * c[3]
            + 90 * c[5]
            + -80 * c[7]
            + 43 * c[9]
            + 9 * c[11]
            + -57 * c[13]
            + 87 * c[15],
        9 * c[1]
            + -25 * c[3]
            + 43 * c[5]
            + -57 * c[7]
            + 70 * c[9]
            + -80 * c[11]
            + 87 * c[13]
            + -90 * c[15],
    ]
}
#[inline(always)]
fn idct_odd_32(c: [i32; 32]) -> [i32; 16] {
    [
        90 * c[1]
            + 90 * c[3]
            + 88 * c[5]
            + 85 * c[7]
            + 82 * c[9]
            + 78 * c[11]
            + 73 * c[13]
            + 67 * c[15]
            + 61 * c[17]
            + 54 * c[19]
            + 46 * c[21]
            + 38 * c[23]
            + 31 * c[25]
            + 22 * c[27]
            + 13 * c[29]
            + 4 * c[31],
        90 * c[1]
            + 82 * c[3]
            + 67 * c[5]
            + 46 * c[7]
            + 22 * c[9]
            + -4 * c[11]
            + -31 * c[13]
            + -54 * c[15]
            + -73 * c[17]
            + -85 * c[19]
            + -90 * c[21]
            + -88 * c[23]
            + -78 * c[25]
            + -61 * c[27]
            + -38 * c[29]
            + -13 * c[31],
        88 * c[1]
            + 67 * c[3]
            + 31 * c[5]
            + -13 * c[7]
            + -54 * c[9]
            + -82 * c[11]
            + -90 * c[13]
            + -78 * c[15]
            + -46 * c[17]
            + -4 * c[19]
            + 38 * c[21]
            + 73 * c[23]
            + 90 * c[25]
            + 85 * c[27]
            + 61 * c[29]
            + 22 * c[31],
        85 * c[1]
            + 46 * c[3]
            + -13 * c[5]
            + -67 * c[7]
            + -90 * c[9]
            + -73 * c[11]
            + -22 * c[13]
            + 38 * c[15]
            + 82 * c[17]
            + 88 * c[19]
            + 54 * c[21]
            + -4 * c[23]
            + -61 * c[25]
            + -90 * c[27]
            + -78 * c[29]
            + -31 * c[31],
        82 * c[1]
            + 22 * c[3]
            + -54 * c[5]
            + -90 * c[7]
            + -61 * c[9]
            + 13 * c[11]
            + 78 * c[13]
            + 85 * c[15]
            + 31 * c[17]
            + -46 * c[19]
            + -90 * c[21]
            + -67 * c[23]
            + 4 * c[25]
            + 73 * c[27]
            + 88 * c[29]
            + 38 * c[31],
        78 * c[1]
            + -4 * c[3]
            + -82 * c[5]
            + -73 * c[7]
            + 13 * c[9]
            + 85 * c[11]
            + 67 * c[13]
            + -22 * c[15]
            + -88 * c[17]
            + -61 * c[19]
            + 31 * c[21]
            + 90 * c[23]
            + 54 * c[25]
            + -38 * c[27]
            + -90 * c[29]
            + -46 * c[31],
        73 * c[1]
            + -31 * c[3]
            + -90 * c[5]
            + -22 * c[7]
            + 78 * c[9]
            + 67 * c[11]
            + -38 * c[13]
            + -90 * c[15]
            + -13 * c[17]
            + 82 * c[19]
            + 61 * c[21]
            + -46 * c[23]
            + -88 * c[25]
            + -4 * c[27]
            + 85 * c[29]
            + 54 * c[31],
        67 * c[1]
            + -54 * c[3]
            + -78 * c[5]
            + 38 * c[7]
            + 85 * c[9]
            + -22 * c[11]
            + -90 * c[13]
            + 4 * c[15]
            + 90 * c[17]
            + 13 * c[19]
            + -88 * c[21]
            + -31 * c[23]
            + 82 * c[25]
            + 46 * c[27]
            + -73 * c[29]
            + -61 * c[31],
        61 * c[1]
            + -73 * c[3]
            + -46 * c[5]
            + 82 * c[7]
            + 31 * c[9]
            + -88 * c[11]
            + -13 * c[13]
            + 90 * c[15]
            + -4 * c[17]
            + -90 * c[19]
            + 22 * c[21]
            + 85 * c[23]
            + -38 * c[25]
            + -78 * c[27]
            + 54 * c[29]
            + 67 * c[31],
        54 * c[1]
            + -85 * c[3]
            + -4 * c[5]
            + 88 * c[7]
            + -46 * c[9]
            + -61 * c[11]
            + 82 * c[13]
            + 13 * c[15]
            + -90 * c[17]
            + 38 * c[19]
            + 67 * c[21]
            + -78 * c[23]
            + -22 * c[25]
            + 90 * c[27]
            + -31 * c[29]
            + -73 * c[31],
        46 * c[1]
            + -90 * c[3]
            + 38 * c[5]
            + 54 * c[7]
            + -90 * c[9]
            + 31 * c[11]
            + 61 * c[13]
            + -88 * c[15]
            + 22 * c[17]
            + 67 * c[19]
            + -85 * c[21]
            + 13 * c[23]
            + 73 * c[25]
            + -82 * c[27]
            + 4 * c[29]
            + 78 * c[31],
        38 * c[1]
            + -88 * c[3]
            + 73 * c[5]
            + -4 * c[7]
            + -67 * c[9]
            + 90 * c[11]
            + -46 * c[13]
            + -31 * c[15]
            + 85 * c[17]
            + -78 * c[19]
            + 13 * c[21]
            + 61 * c[23]
            + -90 * c[25]
            + 54 * c[27]
            + 22 * c[29]
            + -82 * c[31],
        31 * c[1]
            + -78 * c[3]
            + 90 * c[5]
            + -61 * c[7]
            + 4 * c[9]
            + 54 * c[11]
            + -88 * c[13]
            + 82 * c[15]
            + -38 * c[17]
            + -22 * c[19]
            + 73 * c[21]
            + -90 * c[23]
            + 67 * c[25]
            + -13 * c[27]
            + -46 * c[29]
            + 85 * c[31],
        22 * c[1]
            + -61 * c[3]
            + 85 * c[5]
            + -90 * c[7]
            + 73 * c[9]
            + -38 * c[11]
            + -4 * c[13]
            + 46 * c[15]
            + -78 * c[17]
            + 90 * c[19]
            + -82 * c[21]
            + 54 * c[23]
            + -13 * c[25]
            + -31 * c[27]
            + 67 * c[29]
            + -88 * c[31],
        13 * c[1]
            + -38 * c[3]
            + 61 * c[5]
            + -78 * c[7]
            + 88 * c[9]
            + -90 * c[11]
            + 85 * c[13]
            + -73 * c[15]
            + 54 * c[17]
            + -31 * c[19]
            + 4 * c[21]
            + 22 * c[23]
            + -46 * c[25]
            + 67 * c[27]
            + -82 * c[29]
            + 90 * c[31],
        4 * c[1]
            + -13 * c[3]
            + 22 * c[5]
            + -31 * c[7]
            + 38 * c[9]
            + -46 * c[11]
            + 54 * c[13]
            + -61 * c[15]
            + 67 * c[17]
            + -73 * c[19]
            + 78 * c[21]
            + -82 * c[23]
            + 85 * c[25]
            + -88 * c[27]
            + 90 * c[29]
            + -90 * c[31],
    ]
}
#[inline(always)]
fn idct_raw_4(c: [i32; 4]) -> [i32; 4] {
    let e0 = 64 * (c[0] + c[2]);
    let e1 = 64 * (c[0] - c[2]);
    let o0 = 83 * c[1] + 36 * c[3];
    let o1 = 36 * c[1] - 83 * c[3];

    [e0 + o0, e1 + o1, e1 - o1, e0 - o0]
}

#[inline(always)]
fn idct_raw_8(c: [i32; 8]) -> [i32; 8] {
    let ee = idct_raw_4([c[0], c[2], c[4], c[6]]);
    let oo = idct_odd_8(c);
    let mut out = [0i32; 8];
    for (k, (&ee, &oo)) in ee.iter().zip(oo.iter()).enumerate() {
        out[k] = ee + oo;
        out[7 - k] = ee - oo;
    }
    out
}

#[inline(always)]
fn idct_raw_16(c: [i32; 16]) -> [i32; 16] {
    let ee = idct_raw_8(std::array::from_fn(|j| c[2 * j]));
    let oo = idct_odd_16(c);
    let mut out = [0i32; 16];
    for (k, (&ee, &oo)) in ee.iter().zip(oo.iter()).enumerate() {
        out[k] = ee + oo;
        out[15 - k] = ee - oo;
    }
    out
}

#[inline(always)]
fn idct_raw_32(c: [i32; 32]) -> [i32; 32] {
    let ee = idct_raw_16(std::array::from_fn(|j| c[2 * j]));
    let oo = idct_odd_32(c);
    let mut out = [0i32; 32];
    for (k, (&ee, &oo)) in ee.iter().zip(oo.iter()).enumerate() {
        out[k] = ee + oo;
        out[31 - k] = ee - oo;
    }
    out
}

#[inline(always)]
fn idct_raw<const N: usize>(c: [i32; N]) -> [i32; N] {
    debug_assert!(N == 4 || N == 8 || N == 16 || N == 32);

    match N {
        4 => {
            let src = [c[0], c[1], c[2], c[3]];
            let r = idct_raw_4(src);
            std::array::from_fn(|i| r[i])
        }
        8 => {
            let src = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
            let r = idct_raw_8(src);
            std::array::from_fn(|i| r[i])
        }
        16 => {
            let src = [
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12],
                c[13], c[14], c[15],
            ];
            let r = idct_raw_16(src);
            std::array::from_fn(|i| r[i])
        }
        32 => {
            let src = [
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12],
                c[13], c[14], c[15], c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23], c[24],
                c[25], c[26], c[27], c[28], c[29], c[30], c[31],
            ];
            let r = idct_raw_32(src);
            std::array::from_fn(|i| r[i])
        }
        _ => unreachable!(),
    }
}

/// 2-D partial-butterfly inverse DCT into `out[..N*N]`.
#[inline]
fn inv_dct_n_into<const N: usize, S: Coeff>(coeff: &[S], bit_depth: u8, nx: usize, out: &mut [S]) {
    debug_assert!(N == 4 || N == 8 || N == 16 || N == 32);
    debug_assert!(coeff.len() >= N * N);
    debug_assert!(out.len() >= N * N);

    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bit_depth as i32;
    let add2 = 1i32 << (shift2 - 1);
    // Stage-1 output is always clamped to i16 range regardless of storage width.
    let mut tmp = [0i16; 32 * 32];

    // Columns >= nx are all-zero on input, so their stage-1 output stays zero in tmp.
    let nx = nx.min(N);
    for c in 0..nx {
        let col: [i32; N] = std::array::from_fn(|k| coeff[k * N + c].to_i32());
        let raw = idct_raw::<N>(col);
        for (m, &raw) in raw.iter().enumerate() {
            tmp[m * N + c] = ((raw + add1) >> shift1).clamp(-32768, 32767) as i16;
        }
    }

    for (tmp_row, out_row) in tmp
        .as_chunks::<N>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<N>().0.iter_mut())
    {
        let row: [i32; N] = std::array::from_fn(|k| tmp_row[k] as i32);
        let raw = idct_raw::<N>(row);
        for (dst, &raw) in out_row.iter_mut().zip(raw.iter()) {
            *dst = S::from_i32((raw + add2) >> shift2);
        }
    }
}

#[inline]
fn inv_transform_n_into<const N: usize, S: Coeff>(
    coeff: &[S],
    t: &[[i32; N]; N],
    bit_depth: u8,
    nx: usize,
    out: &mut [S],
) {
    let bd = bit_depth as i32;
    let shift1 = 7i32;
    let add1 = 1i32 << (shift1 - 1);
    let shift2 = 20 - bd;
    let add2 = 1i32 << (shift2 - 1);

    let mut tmp = [0i16; 32 * 32];
    let mut acc = [0i32; N];

    let nx = nx.min(N);
    for c in 0..nx {
        acc[..N].fill(0);
        for k in 0..N {
            let ck = coeff[k * N + c].to_i32();
            if ck == 0 {
                continue; // sparse skip — most residual coeffs are zero
            }
            let trow = &t[k];
            for (acc, &tm) in acc[..N].iter_mut().zip(trow.iter()) {
                *acc += tm * ck;
            }
        }
        for (m, &acc) in acc[..N].iter().enumerate() {
            tmp[m * N + c] = ((acc + add1) >> shift1).clamp(-32768, 32767) as i16;
        }
    }

    for (rowv, out_row) in tmp
        .as_chunks::<N>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<N>().0.iter_mut())
    {
        acc[..N].fill(0);
        for (&rk, trow) in rowv.iter().zip(t.iter()) {
            if rk == 0 {
                continue;
            }
            let rk = rk as i32;
            for (acc, &tm) in acc[..N].iter_mut().zip(trow.iter()) {
                *acc += tm * rk;
            }
        }
        for (dst, &acc) in out_row.iter_mut().zip(acc[..N].iter()) {
            *dst = S::from_i32((acc + add2) >> shift2);
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DequantParams {
    pub factor: i64,
    pub add: i64,
    pub shift: i32,
}

#[derive(Clone, Copy)]
pub(crate) struct TransformSkipParams {
    pub dequant: DequantParams,
    pub tr_add: i32,
    pub tr_shift: i32,
}

#[derive(Clone, Copy)]
pub(crate) struct ScalingMatrix<'a> {
    coeffs: &'a [u8; 64],
    dc: u8,
    log2_size: u32,
    flat_16: bool,
}

impl<'a> ScalingMatrix<'a> {
    #[inline]
    pub(crate) fn new(coeffs: &'a [u8; 64], dc: u8, n: usize, flat_16: bool) -> Self {
        ScalingMatrix {
            coeffs,
            dc,
            log2_size: (n as u32).trailing_zeros(),
            flat_16,
        }
    }

    #[inline]
    fn is_flat_16(self) -> bool {
        self.flat_16
    }

    #[inline]
    pub(crate) fn coeff(self, idx: usize) -> i64 {
        if self.log2_size == 2 {
            return self.coeffs[idx] as i64;
        }
        if idx == 0 && self.log2_size >= 4 {
            return self.dc as i64;
        }

        let n = 1usize << self.log2_size;
        let y = idx / n;
        let x = idx - y * n;
        let downshift = self.log2_size.saturating_sub(3) as usize;
        let sx = x >> downshift;
        let sy = y >> downshift;
        self.coeffs[sy * 8 + sx] as i64
    }
}

#[inline]
fn dequant_params(n: usize, qp_prime: i32, bit_depth: u8) -> DequantParams {
    let log2n = (n as u32).trailing_zeros() as i64;
    let bd = bit_depth as i64;
    let bd_shift = (bd + log2n - 5).max(1);
    let max_qp_prime = 51 + 6 * (bit_depth as i32 - 8);
    let qp_scaled = qp_prime.clamp(0, max_qp_prime) as i64;
    let scale = DEQUANT_SCALE[(qp_scaled % 6) as usize];
    let per = 1i64 << (qp_scaled / 6);
    DequantParams {
        factor: scale * per * 16,
        add: 1i64 << (bd_shift - 1),
        shift: bd_shift as i32,
    }
}

#[inline]
fn transform_skip_params(n: usize, qp_prime: i32, bit_depth: u8) -> TransformSkipParams {
    let dequant = dequant_params(n, qp_prime, bit_depth);
    let log2n = (n as u32).trailing_zeros() as i32;
    // getTransformShift(bitDepth, log2TrSize, maxLog2TrDynamicRange), with
    // maxLog2TrDynamicRange fixed to 15 for the supported HEVC profiles.
    let tr_shift = 15i32 - bit_depth as i32 - log2n;
    let tr_add = if tr_shift > 0 {
        1i32 << (tr_shift - 1)
    } else {
        0
    };
    TransformSkipParams {
        dequant,
        tr_add,
        tr_shift,
    }
}

#[inline]
fn apply_transform_skip_shift(deq: i32, params: TransformSkipParams) -> i32 {
    if params.tr_shift >= 0 {
        (deq + params.tr_add) >> params.tr_shift
    } else {
        deq << (-params.tr_shift)
    }
}

pub(crate) fn dequantize_into_scalar<S: Coeff>(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [S],
) {
    for (o, &l) in out[..n * n].iter_mut().zip(levels.iter()) {
        // l*factor can exceed i32 at high QP, so the intermediate stays i64.
        let v = ((l as i64 * params.factor + params.add) >> params.shift).clamp(-32768, 32767);
        *o = S::from_i32(v as i32);
    }
}

pub(crate) fn dequantize_transform_skip_into_scalar<S: Coeff>(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [S],
) {
    debug_assert!(
        n == 4,
        "HEVC transform_skip_flag is only signalled for 4x4 TUs"
    );

    for (o, &l) in out[..n * n].iter_mut().zip(levels.iter()) {
        let deq = ((l as i64 * params.dequant.factor + params.dequant.add) >> params.dequant.shift)
            .clamp(-32768, 32767) as i32;
        let residual = apply_transform_skip_shift(deq, params).clamp(-32768, 32767);
        *o = S::from_i32(residual);
    }
}

pub(crate) fn dequantize_scaled_into_scalar<S: Coeff>(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [S],
) {
    let base_factor = params.factor / 16;
    for (idx, (o, &l)) in out[..n * n].iter_mut().zip(levels.iter()).enumerate() {
        let factor = base_factor * scaling.coeff(idx);
        let v = ((l as i64 * factor + params.add) >> params.shift).clamp(-32768, 32767);
        *o = S::from_i32(v as i32);
    }
}

pub(crate) fn dequantize_transform_skip_scaled_into_scalar<S: Coeff>(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [S],
) {
    debug_assert!(
        n == 4,
        "HEVC transform_skip_flag is only signalled for 4x4 TUs"
    );

    let base_factor = params.dequant.factor / 16;
    for (idx, (o, &l)) in out[..n * n].iter_mut().zip(levels.iter()).enumerate() {
        let factor = base_factor * scaling.coeff(idx);
        let deq = ((l as i64 * factor + params.dequant.add) >> params.dequant.shift)
            .clamp(-32768, 32767) as i32;
        let residual = apply_transform_skip_shift(deq, params).clamp(-32768, 32767);
        *o = S::from_i32(residual);
    }
}

pub(crate) fn dequantize_into_scalar_i32(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i32],
) {
    dequantize_into_scalar(levels, n, params, out);
}

pub(crate) fn dequantize_into_scalar_i16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    out: &mut [i16],
) {
    dequantize_into_scalar(levels, n, params, out);
}

pub(crate) fn dequantize_transform_skip_into_scalar_i32(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i32],
) {
    dequantize_transform_skip_into_scalar(levels, n, params, out);
}

pub(crate) fn dequantize_transform_skip_into_scalar_i16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    out: &mut [i16],
) {
    dequantize_transform_skip_into_scalar(levels, n, params, out);
}

pub(crate) fn dequantize_scaled_into_scalar_i32(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    dequantize_scaled_into_scalar(levels, n, params, scaling, out);
}

pub(crate) fn dequantize_scaled_into_scalar_i16(
    levels: &[i32],
    n: usize,
    params: DequantParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    dequantize_scaled_into_scalar(levels, n, params, scaling, out);
}

pub(crate) fn dequantize_transform_skip_scaled_into_scalar_i32(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i32],
) {
    dequantize_transform_skip_scaled_into_scalar(levels, n, params, scaling, out);
}

pub(crate) fn dequantize_transform_skip_scaled_into_scalar_i16(
    levels: &[i32],
    n: usize,
    params: TransformSkipParams,
    scaling: ScalingMatrix<'_>,
    out: &mut [i16],
) {
    dequantize_transform_skip_scaled_into_scalar(levels, n, params, scaling, out);
}

// `n` = transform width/height. Output is clipped to the HEVC residual dynamic
// range used by the rest of this decoder.
type DequantFn = fn(&[i32], usize, DequantParams, &mut [i32]);
type DequantFn16 = fn(&[i32], usize, DequantParams, &mut [i16]);
type DequantSkipFn = fn(&[i32], usize, TransformSkipParams, &mut [i32]);
type DequantSkipFn16 = fn(&[i32], usize, TransformSkipParams, &mut [i16]);
type DequantScaledFn = for<'a> fn(&[i32], usize, DequantParams, ScalingMatrix<'a>, &mut [i32]);
type DequantScaledFn16 = for<'a> fn(&[i32], usize, DequantParams, ScalingMatrix<'a>, &mut [i16]);
type DequantSkipScaledFn =
    for<'a> fn(&[i32], usize, TransformSkipParams, ScalingMatrix<'a>, &mut [i32]);
type DequantSkipScaledFn16 =
    for<'a> fn(&[i32], usize, TransformSkipParams, ScalingMatrix<'a>, &mut [i16]);

static DEQUANT: std::sync::OnceLock<DequantFn> = std::sync::OnceLock::new();
static DEQUANT16: std::sync::OnceLock<DequantFn16> = std::sync::OnceLock::new();
static DEQUANT_SKIP: std::sync::OnceLock<DequantSkipFn> = std::sync::OnceLock::new();
static DEQUANT_SKIP16: std::sync::OnceLock<DequantSkipFn16> = std::sync::OnceLock::new();
static DEQUANT_SCALED: std::sync::OnceLock<DequantScaledFn> = std::sync::OnceLock::new();
static DEQUANT_SCALED16: std::sync::OnceLock<DequantScaledFn16> = std::sync::OnceLock::new();
static DEQUANT_SKIP_SCALED: std::sync::OnceLock<DequantSkipScaledFn> = std::sync::OnceLock::new();
static DEQUANT_SKIP_SCALED16: std::sync::OnceLock<DequantSkipScaledFn16> =
    std::sync::OnceLock::new();

#[inline]
fn resolve_dequant() -> DequantFn {
    *DEQUANT.get_or_init(|| {
        let mut _f: DequantFn = dequantize_into_scalar_i32;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::dequantize_into_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::dequantize_into_sse41;
            }
        }

        _f
    })
}

#[inline]
fn resolve_dequant16() -> DequantFn16 {
    *DEQUANT16.get_or_init(|| {
        let mut _f: DequantFn16 = dequantize_into_scalar_i16;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::dequantize_into_neon16;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::dequantize_into_sse41_16;
            }
        }

        _f
    })
}

#[inline]
fn resolve_dequant_skip() -> DequantSkipFn {
    *DEQUANT_SKIP.get_or_init(|| {
        let mut _f: DequantSkipFn = dequantize_transform_skip_into_scalar_i32;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::dequantize_transform_skip_into_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::dequantize_transform_skip_into_sse41;
            }
        }

        _f
    })
}

#[inline]
fn resolve_dequant_skip16() -> DequantSkipFn16 {
    *DEQUANT_SKIP16.get_or_init(|| {
        let mut _f: DequantSkipFn16 = dequantize_transform_skip_into_scalar_i16;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::dequantize_transform_skip_into_neon16;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::dequantize_transform_skip_into_sse41_16;
            }
        }

        _f
    })
}

#[inline]
fn resolve_dequant_scaled() -> DequantScaledFn {
    *DEQUANT_SCALED.get_or_init(|| {
        let mut _f: DequantScaledFn = dequantize_scaled_into_scalar_i32;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::dequantize_scaled_into_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::dequantize_scaled_into_sse41;
            }
        }

        _f
    })
}

#[inline]
fn resolve_dequant_scaled16() -> DequantScaledFn16 {
    *DEQUANT_SCALED16.get_or_init(|| {
        let mut _f: DequantScaledFn16 = dequantize_scaled_into_scalar_i16;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::dequantize_scaled_into_neon16;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::dequantize_scaled_into_sse41_16;
            }
        }

        _f
    })
}

#[inline]
fn resolve_dequant_skip_scaled() -> DequantSkipScaledFn {
    *DEQUANT_SKIP_SCALED.get_or_init(|| {
        let mut _f: DequantSkipScaledFn = dequantize_transform_skip_scaled_into_scalar_i32;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::dequantize_transform_skip_scaled_into_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::dequantize_transform_skip_scaled_into_sse41;
            }
        }

        _f
    })
}

#[inline]
fn resolve_dequant_skip_scaled16() -> DequantSkipScaledFn16 {
    *DEQUANT_SKIP_SCALED16.get_or_init(|| {
        let mut _f: DequantSkipScaledFn16 = dequantize_transform_skip_scaled_into_scalar_i16;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::dequantize_transform_skip_scaled_into_neon16;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::dequantize_transform_skip_scaled_into_sse41_16;
            }
        }

        _f
    })
}

pub(crate) trait DequantTarget: Coeff {
    fn dequantize(levels: &[i32], n: usize, params: DequantParams, out: &mut [Self]);
    fn dequantize_transform_skip(
        levels: &[i32],
        n: usize,
        params: TransformSkipParams,
        out: &mut [Self],
    );
    fn dequantize_scaled(
        levels: &[i32],
        n: usize,
        params: DequantParams,
        scaling: ScalingMatrix<'_>,
        out: &mut [Self],
    );
    fn dequantize_transform_skip_scaled(
        levels: &[i32],
        n: usize,
        params: TransformSkipParams,
        scaling: ScalingMatrix<'_>,
        out: &mut [Self],
    );
}

impl DequantTarget for i32 {
    #[inline]
    fn dequantize(levels: &[i32], n: usize, params: DequantParams, out: &mut [Self]) {
        resolve_dequant()(levels, n, params, out);
    }

    #[inline]
    fn dequantize_transform_skip(
        levels: &[i32],
        n: usize,
        params: TransformSkipParams,
        out: &mut [Self],
    ) {
        resolve_dequant_skip()(levels, n, params, out);
    }

    #[inline]
    fn dequantize_scaled(
        levels: &[i32],
        n: usize,
        params: DequantParams,
        scaling: ScalingMatrix<'_>,
        out: &mut [Self],
    ) {
        resolve_dequant_scaled()(levels, n, params, scaling, out);
    }

    #[inline]
    fn dequantize_transform_skip_scaled(
        levels: &[i32],
        n: usize,
        params: TransformSkipParams,
        scaling: ScalingMatrix<'_>,
        out: &mut [Self],
    ) {
        resolve_dequant_skip_scaled()(levels, n, params, scaling, out);
    }
}

impl DequantTarget for i16 {
    #[inline]
    fn dequantize(levels: &[i32], n: usize, params: DequantParams, out: &mut [Self]) {
        resolve_dequant16()(levels, n, params, out);
    }

    #[inline]
    fn dequantize_transform_skip(
        levels: &[i32],
        n: usize,
        params: TransformSkipParams,
        out: &mut [Self],
    ) {
        resolve_dequant_skip16()(levels, n, params, out);
    }

    #[inline]
    fn dequantize_scaled(
        levels: &[i32],
        n: usize,
        params: DequantParams,
        scaling: ScalingMatrix<'_>,
        out: &mut [Self],
    ) {
        resolve_dequant_scaled16()(levels, n, params, scaling, out);
    }

    #[inline]
    fn dequantize_transform_skip_scaled(
        levels: &[i32],
        n: usize,
        params: TransformSkipParams,
        scaling: ScalingMatrix<'_>,
        out: &mut [Self],
    ) {
        resolve_dequant_skip_scaled16()(levels, n, params, scaling, out);
    }
}

pub(crate) fn dequantize_into<S: DequantTarget>(
    levels: &[i32],
    n: usize,
    qp_prime: i32,
    bit_depth: u8,
    out: &mut [S],
) {
    let params = dequant_params(n, qp_prime, bit_depth);
    S::dequantize(levels, n, params, out);
}

pub(crate) fn dequantize_transform_skip_into<S: DequantTarget>(
    levels: &[i32],
    n: usize,
    qp_prime: i32,
    bit_depth: u8,
    out: &mut [S],
) {
    debug_assert!(
        n == 4,
        "HEVC transform_skip_flag is only signalled for 4x4 TUs"
    );
    let params = transform_skip_params(n, qp_prime, bit_depth);
    S::dequantize_transform_skip(levels, n, params, out);
}

pub(crate) fn dequantize_scaled_into<S: DequantTarget>(
    levels: &[i32],
    n: usize,
    qp_prime: i32,
    bit_depth: u8,
    scaling: Option<ScalingMatrix<'_>>,
    out: &mut [S],
) {
    let Some(scaling) = scaling else {
        dequantize_into(levels, n, qp_prime, bit_depth, out);
        return;
    };
    if scaling.is_flat_16() {
        dequantize_into(levels, n, qp_prime, bit_depth, out);
        return;
    }
    let params = dequant_params(n, qp_prime, bit_depth);
    S::dequantize_scaled(levels, n, params, scaling, out);
}

pub(crate) fn dequantize_transform_skip_scaled_into<S: DequantTarget>(
    levels: &[i32],
    n: usize,
    qp_prime: i32,
    bit_depth: u8,
    scaling: Option<ScalingMatrix<'_>>,
    out: &mut [S],
) {
    debug_assert!(
        n == 4,
        "HEVC transform_skip_flag is only signalled for 4x4 TUs"
    );
    let Some(scaling) = scaling else {
        dequantize_transform_skip_into(levels, n, qp_prime, bit_depth, out);
        return;
    };
    if scaling.is_flat_16() {
        dequantize_transform_skip_into(levels, n, qp_prime, bit_depth, out);
        return;
    }
    let params = transform_skip_params(n, qp_prime, bit_depth);
    S::dequantize_transform_skip_scaled(levels, n, params, scaling, out);
}

// `nx` = number of nonzero input columns (last_x + 1); stage 1 skips the rest.
type InvTransformFn = fn(&[i32], usize, u8, usize, &mut [i32]);
type InvTransform4Fn = fn(&[i32], u8, &mut [i32]);
type InvTransformFn16 = fn(&[i16], usize, u8, usize, &mut [i16]);
type InvTransform4Fn16 = fn(&[i16], u8, &mut [i16]);

static INV_TRANSFORM: std::sync::OnceLock<InvTransformFn> = std::sync::OnceLock::new();
static INV_TRANSFORM_DST4: std::sync::OnceLock<InvTransform4Fn> = std::sync::OnceLock::new();
static INV_TRANSFORM16: std::sync::OnceLock<InvTransformFn16> = std::sync::OnceLock::new();
static INV_TRANSFORM_DST4_16: std::sync::OnceLock<InvTransform4Fn16> = std::sync::OnceLock::new();

#[inline]
fn resolve_inv_transform() -> InvTransformFn {
    *INV_TRANSFORM.get_or_init(|| {
        let mut _f: InvTransformFn = inv_transform_into_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::inv_transform_into_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::inv_transform_into_sse41;
            }
        }

        _f
    })
}

#[inline]
fn resolve_inv_transform_dst4() -> InvTransform4Fn {
    *INV_TRANSFORM_DST4.get_or_init(|| {
        let mut _f: InvTransform4Fn = inv_transform_dst_into_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::inv_transform_dst_into_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::inv_transform_dst_into_sse41;
            }
        }

        _f
    })
}

#[inline]
fn resolve_inv_transform16() -> InvTransformFn16 {
    *INV_TRANSFORM16.get_or_init(|| {
        let mut _f: InvTransformFn16 = inv_transform_into_scalar16;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::inv_transform_into_neon16;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::inv_transform_into_sse41_16;
            }
        }

        _f
    })
}

#[inline]
fn resolve_inv_transform_dst4_16() -> InvTransform4Fn16 {
    *INV_TRANSFORM_DST4_16.get_or_init(|| {
        let mut _f: InvTransform4Fn16 = inv_transform_dst_into_scalar16;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::inv_transform_dst_into_neon16;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::inv_transform_dst_into_sse41_16;
            }
        }

        _f
    })
}

/// Scalar inverse DCT into `out[..n*n]` — no heap allocation.
pub(crate) fn inv_transform_into_scalar(
    coeff: &[i32],
    n: usize,
    bit_depth: u8,
    nx: usize,
    out: &mut [i32],
) {
    match n {
        4 => inv_dct_n_into::<4, i32>(coeff, bit_depth, nx, out),
        8 => inv_dct_n_into::<8, i32>(coeff, bit_depth, nx, out),
        16 => inv_dct_n_into::<16, i32>(coeff, bit_depth, nx, out),
        32 => inv_dct_n_into::<32, i32>(coeff, bit_depth, nx, out),
        _ => panic!("unsupported transform size {n}"),
    }
}

/// Scalar inverse 4×4 DST/ADST-like intra transform into `out[..16]`.
pub(crate) fn inv_transform_dst_into_scalar(coeff: &[i32], bit_depth: u8, out: &mut [i32]) {
    inv_transform_n_into::<4, i32>(coeff, &DST4, bit_depth, 4, out);
}

/// Scalar i16 inverse DCT (8-bit depth path).
pub(crate) fn inv_transform_into_scalar16(
    coeff: &[i16],
    n: usize,
    bit_depth: u8,
    nx: usize,
    out: &mut [i16],
) {
    match n {
        4 => inv_dct_n_into::<4, i16>(coeff, bit_depth, nx, out),
        8 => inv_dct_n_into::<8, i16>(coeff, bit_depth, nx, out),
        16 => inv_dct_n_into::<16, i16>(coeff, bit_depth, nx, out),
        32 => inv_dct_n_into::<32, i16>(coeff, bit_depth, nx, out),
        _ => panic!("unsupported transform size {n}"),
    }
}

/// Scalar i16 inverse 4×4 DST (8-bit depth path).
pub(crate) fn inv_transform_dst_into_scalar16(coeff: &[i16], bit_depth: u8, out: &mut [i16]) {
    inv_transform_n_into::<4, i16>(coeff, &DST4, bit_depth, 4, out);
}

/// Inverse DCT into `out[..n*n]`. `nx` = nonzero column count (last_x + 1).
pub(crate) fn inv_transform_into(
    coeff: &[i32],
    n: usize,
    bit_depth: u8,
    nx: usize,
    out: &mut [i32],
) {
    resolve_inv_transform()(coeff, n, bit_depth, nx, out);
}

/// Inverse 4×4 DST/ADST-like intra transform into `out[..16]` — no heap allocation.
pub(crate) fn inv_transform_dst_into(coeff: &[i32], bit_depth: u8, out: &mut [i32]) {
    resolve_inv_transform_dst4()(coeff, bit_depth, out);
}

/// i16 inverse DCT (8-bit depth). `nx` = nonzero column count (last_x + 1).
pub(crate) fn inv_transform_into16(
    coeff: &[i16],
    n: usize,
    bit_depth: u8,
    nx: usize,
    out: &mut [i16],
) {
    resolve_inv_transform16()(coeff, n, bit_depth, nx, out);
}

/// i16 inverse 4×4 DST (8-bit depth).
pub(crate) fn inv_transform_dst_into16(coeff: &[i16], bit_depth: u8, out: &mut [i16]) {
    resolve_inv_transform_dst4_16()(coeff, bit_depth, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_skip_4x4_8bit_qp22_known_values() {
        let mut levels = [0i32; 16];
        levels[0] = 1;
        levels[1] = -1;
        levels[2] = 2;
        levels[3] = -2;

        let mut out = [0i16; 16];
        dequantize_transform_skip_into(&levels, 4, 22, 8, &mut out);

        assert_eq!(out[0], 8);
        assert_eq!(out[1], -8);
        assert_eq!(out[2], 16);
        assert_eq!(out[3], -16);
        assert!(out[4..].iter().all(|&v| v == 0));
    }

    #[test]
    fn transform_skip_is_dequant_plus_logical_inverse_skip() {
        let levels = [3, -2, 0, 1, -1, 0, 2, -3, 0, 4, -4, 0, 1, 0, -1, 2];
        let mut deq = [0i32; 16];
        let mut out = [0i32; 16];

        dequantize_into(&levels, 4, 18, 8, &mut deq);
        dequantize_transform_skip_into(&levels, 4, 18, 8, &mut out);

        let shift = 15 - 8 - 2;
        let add = 1 << (shift - 1);
        for (actual, &d) in out.iter().zip(deq.iter()) {
            assert_eq!(*actual, (d + add) >> shift);
        }
    }

    #[test]
    fn scalar_dequant_matches_public_dispatch_i16() {
        let levels = [-12, 0, 7, 19, -31, 42, 3, -4, 5, -6, 100, -100, 1, 2, -3, 4];
        let params = dequant_params(4, 34, 8);
        let mut expected = [0i16; 16];
        let mut actual = [0i16; 16];

        dequantize_into_scalar(&levels, 4, params, &mut expected);
        dequantize_into(&levels, 4, 34, 8, &mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn scalar_transform_skip_matches_public_dispatch_i32() {
        let levels = [-12, 0, 7, 19, -31, 42, 3, -4, 5, -6, 100, -100, 1, 2, -3, 4];
        let params = transform_skip_params(4, 34, 10);
        let mut expected = [0i32; 16];
        let mut actual = [0i32; 16];

        dequantize_transform_skip_into_scalar(&levels, 4, params, &mut expected);
        dequantize_transform_skip_into(&levels, 4, 34, 10, &mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn dequant_10bit_accepts_qp_prime_zero() {
        let mut levels = [0i32; 16];
        levels[0] = 1;

        let mut out = [0i32; 16];
        dequantize_into(&levels, 4, 0, 10, &mut out);

        // Nominal 10-bit QpY=-12 maps to qp_prime=0. The decoder must not add
        // QpBdOffset again inside dequantization; doing so would produce 20 here.
        assert_eq!(out[0], 5);
        assert!(out[1..].iter().all(|&v| v == 0));
    }

    #[test]
    fn transform_skip_10bit_accepts_qp_prime_zero() {
        let mut levels = [0i32; 16];
        levels[0] = 1;

        let mut out = [0i32; 16];
        dequantize_transform_skip_into(&levels, 4, 0, 10, &mut out);

        // dequant=5, transform-skip shift=15-10-log2(4)=3 => (5+4)>>3.
        assert_eq!(out[0], 1);
        assert!(out[1..].iter().all(|&v| v == 0));
    }

    #[test]
    fn scaled_dequant_flat_16_matches_default_dispatch() {
        let levels = [-12, 0, 7, 19, -31, 42, 3, -4, 5, -6, 100, -100, 1, 2, -3, 4];
        let coeffs = [16u8; 64];
        let scaling = ScalingMatrix::new(&coeffs, 16, 4, false);
        let mut expected = [0i16; 16];
        let mut actual = [0i16; 16];

        dequantize_into(&levels, 4, 34, 8, &mut expected);
        dequantize_scaled_into(&levels, 4, 34, 8, Some(scaling), &mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn scaled_dequant_uses_per_coefficient_matrix() {
        let mut levels = [0i32; 16];
        levels[0] = 1;
        levels[5] = 1;
        let mut coeffs = [16u8; 64];
        coeffs[0] = 32;
        coeffs[5] = 8;
        let scaling = ScalingMatrix::new(&coeffs, 16, 4, false);
        let mut out = [0i32; 16];

        dequantize_scaled_into(&levels, 4, 0, 8, Some(scaling), &mut out);

        assert_eq!(out[0], 40);
        assert_eq!(out[5], 10);
        assert!(
            out.iter()
                .enumerate()
                .all(|(i, &v)| i == 0 || i == 5 || v == 0)
        );
    }

    #[test]
    fn scaled_dequant_16x16_replicates_8x8_and_uses_dc() {
        let mut levels = [0i32; 256];
        levels[0] = 1;
        levels[1] = 1;
        levels[2] = 1;
        let mut coeffs = [16u8; 64];
        coeffs[0] = 8;
        coeffs[1] = 32;
        let scaling = ScalingMatrix::new(&coeffs, 24, 16, false);
        let mut out = [0i32; 256];

        dequantize_scaled_into(&levels, 16, 0, 8, Some(scaling), &mut out);

        assert_eq!(out[0], 8);
        assert_eq!(out[1], 3);
        assert_eq!(out[2], 10);
    }
}
