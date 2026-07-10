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

static LUMA_FILTER: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

static CHROMA_FILTER: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// Reference block view: a plane slice with stride, and the top-left integer
/// sample position of the PU. Out-of-frame accesses clamp to the edge (HEVC
/// requires edge extension of the reference picture).
pub(crate) struct RefPlane<'a> {
    pub(crate) data: &'a [u16],
    pub(crate) stride: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl<'a> RefPlane<'a> {
    #[inline]
    fn sample(&self, x: isize, y: isize) -> i32 {
        let xc = x.clamp(0, self.width as isize - 1) as usize;
        let yc = y.clamp(0, self.height as isize - 1) as usize;
        self.data[yc * self.stride + xc] as i32
    }
}

/// Internal intermediate bit shift (shift1 = BitDepth - 8).
#[inline]
fn shift1(bd: u8) -> u32 {
    (bd - 8) as u32
}

/// Luma fractional interpolation of a `w x h` block at integer position
/// `(x0, y0)` with quarter-pel fraction `(fx, fy)` into `dst` (16-bit
/// intermediate, stride = w). Follows the separable H-then-V approach.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_interp(
    r: &RefPlane,
    x0: isize,
    y0: isize,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    bd: u8,
    dst: &mut [i16],
) {
    let s1 = shift1(bd);
    let dst = &mut dst[..w * h];

    if fx == 0 && fy == 0 {
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let sy = y0 + y as isize;
            for (x, out) in dst_row.iter_mut().enumerate() {
                let v = r.sample(x0 + x as isize, sy) << 6 >> s1;
                *out = v as i16;
            }
        }
        return;
    }

    let hf = &LUMA_FILTER[fx];
    let vf = &LUMA_FILTER[fy];

    if fy == 0 {
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let sy = y0 + y as isize;
            for (x, out) in dst_row.iter_mut().enumerate() {
                let sx = x0 + x as isize - 3;
                let acc = hf
                    .iter()
                    .enumerate()
                    .fold(0i32, |acc, (t, &c)| acc + c * r.sample(sx + t as isize, sy));
                *out = (acc >> s1) as i16;
            }
        }
        return;
    }

    if fx == 0 {
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let sy = y0 + y as isize - 3;
            for (x, out) in dst_row.iter_mut().enumerate() {
                let sx = x0 + x as isize;
                let acc = vf
                    .iter()
                    .enumerate()
                    .fold(0i32, |acc, (t, &c)| acc + c * r.sample(sx, sy + t as isize));
                *out = (acc >> s1) as i16;
            }
        }
        return;
    }

    // Separable: horizontal into a temp of height h+7, then vertical.
    let tmp_h = h + 7;
    let mut tmp = vec![0i32; w * tmp_h];
    for (ty, tmp_row) in tmp.chunks_exact_mut(w).enumerate() {
        let sy = y0 + ty as isize - 3;
        for (x, out) in tmp_row.iter_mut().enumerate() {
            let sx = x0 + x as isize - 3;
            let acc = hf
                .iter()
                .enumerate()
                .fold(0i32, |acc, (t, &c)| acc + c * r.sample(sx + t as isize, sy));
            *out = acc >> s1;
        }
    }

    let vshift = 6u32;
    for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
        let tmp_rows = &tmp[y * w..(y + 8) * w];
        for (x, out) in dst_row.iter_mut().enumerate() {
            let acc = tmp_rows
                .chunks_exact(w)
                .zip(vf.iter())
                .fold(0i32, |acc, (tmp_row, &c)| acc + c * tmp_row[x]);
            *out = (acc >> vshift) as i16;
        }
    }
}

/// Chroma fractional interpolation (4-tap, eighth-pel fractions).
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_interp(
    r: &RefPlane,
    x0: isize,
    y0: isize,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    bd: u8,
    dst: &mut [i16],
) {
    let s1 = shift1(bd);
    let dst = &mut dst[..w * h];

    if fx == 0 && fy == 0 {
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let sy = y0 + y as isize;
            for (x, out) in dst_row.iter_mut().enumerate() {
                let v = r.sample(x0 + x as isize, sy) << 6 >> s1;
                *out = v as i16;
            }
        }
        return;
    }

    let hf = &CHROMA_FILTER[fx];
    let vf = &CHROMA_FILTER[fy];

    if fy == 0 {
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let sy = y0 + y as isize;
            for (x, out) in dst_row.iter_mut().enumerate() {
                let sx = x0 + x as isize - 1;
                let acc = hf
                    .iter()
                    .enumerate()
                    .fold(0i32, |acc, (t, &c)| acc + c * r.sample(sx + t as isize, sy));
                *out = (acc >> s1) as i16;
            }
        }
        return;
    }

    if fx == 0 {
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let sy = y0 + y as isize - 1;
            for (x, out) in dst_row.iter_mut().enumerate() {
                let sx = x0 + x as isize;
                let acc = vf
                    .iter()
                    .enumerate()
                    .fold(0i32, |acc, (t, &c)| acc + c * r.sample(sx, sy + t as isize));
                *out = (acc >> s1) as i16;
            }
        }
        return;
    }

    let tmp_h = h + 3;
    let mut tmp = vec![0i32; w * tmp_h];
    for (ty, tmp_row) in tmp.chunks_exact_mut(w).enumerate() {
        let sy = y0 + ty as isize - 1;
        for (x, out) in tmp_row.iter_mut().enumerate() {
            let sx = x0 + x as isize - 1;
            let acc = hf
                .iter()
                .enumerate()
                .fold(0i32, |acc, (t, &c)| acc + c * r.sample(sx + t as isize, sy));
            *out = acc >> s1;
        }
    }

    let vshift = 6u32;
    for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
        let tmp_rows = &tmp[y * w..(y + 4) * w];
        for (x, out) in dst_row.iter_mut().enumerate() {
            let acc = tmp_rows
                .chunks_exact(w)
                .zip(vf.iter())
                .fold(0i32, |acc, (tmp_row, &c)| acc + c * tmp_row[x]);
            *out = (acc >> vshift) as i16;
        }
    }
}

#[inline]
fn clip(v: i32, max: i32) -> u16 {
    v.clamp(0, max) as u16
}

/// Uni-prediction combine: intermediate -> final samples (§8.5.3.3.4.2).
pub(crate) fn uni_pred(
    src: &[i16],
    w: usize,
    h: usize,
    bd: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let shift = 14 - bd as i32;
    let offset = if shift > 0 { 1 << (shift - 1) } else { 0 };
    let max = (1 << bd) - 1;
    assert!(dst_stride >= w);
    assert!(dst.len() >= dst_stride.saturating_mul(h));
    let src = &src[..w * h];

    for (src_row, dst_row) in src.chunks_exact(w).zip(dst.chunks_exact_mut(dst_stride)) {
        for (&s, out) in src_row.iter().zip(dst_row.iter_mut().take(w)) {
            let v = (s as i32 + offset) >> shift;
            *out = clip(v, max);
        }
    }
}

/// Bi-prediction average combine (§8.5.3.3.4.2).
pub(crate) fn bi_pred(
    s0: &[i16],
    s1: &[i16],
    w: usize,
    h: usize,
    bd: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let shift = 15 - bd as i32;
    let offset = 1 << (shift - 1);
    let max = (1 << bd) - 1;
    assert!(dst_stride >= w);
    assert!(dst.len() >= dst_stride.saturating_mul(h));
    let s0 = &s0[..w * h];
    let s1 = &s1[..w * h];

    for ((s0_row, s1_row), dst_row) in s0
        .chunks_exact(w)
        .zip(s1.chunks_exact(w))
        .zip(dst.chunks_exact_mut(dst_stride))
    {
        for ((&a, &b), out) in s0_row
            .iter()
            .zip(s1_row.iter())
            .zip(dst_row.iter_mut().take(w))
        {
            let v = (a as i32 + b as i32 + offset) >> shift;
            *out = clip(v, max);
        }
    }
}

/// Explicit weighted uni-prediction (§8.5.3.3.4.3). `log2_denom` is the weight
/// denominator (WpOffsetBdShift folded into `offset` by the caller's o1 term).
#[allow(clippy::too_many_arguments)]
pub(crate) fn uni_pred_weighted(
    src: &[i16],
    w: usize,
    h: usize,
    bd: u8,
    weight: i32,
    offset: i32,
    log2_denom: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    // shift1 brings the 14-bit interpolated sample down; the weight is applied
    // in that domain and then rounded by log2WD = log2_denom + shift1.
    let shift1 = 14 - bd as i32;
    let log2_wd = log2_denom as i32 + shift1;
    let max = (1 << bd) - 1;
    let round = if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 };
    let off = offset << (bd as i32 - 8);
    let src = &src[..w * h];
    for (src_row, dst_row) in src.chunks_exact(w).zip(dst.chunks_exact_mut(dst_stride)) {
        for (&s, out) in src_row.iter().zip(dst_row.iter_mut().take(w)) {
            let v = if log2_wd >= 1 {
                ((s as i32 * weight + round) >> log2_wd) + off
            } else {
                s as i32 * weight + off
            };
            *out = clip(v, max);
        }
    }
}

/// Explicit weighted bi-prediction (§8.5.3.3.4.3).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bi_pred_weighted(
    s0: &[i16],
    s1: &[i16],
    w: usize,
    h: usize,
    bd: u8,
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
    log2_denom: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let shift1 = 14 - bd as i32;
    let log2_wd = log2_denom as i32 + shift1;
    let max = (1 << bd) - 1;
    let bd_off = bd as i32 - 8;
    // Offsets are signalled in 8-bit units; scale to the sample bit depth.
    let o0 = o0 << bd_off;
    let o1 = o1 << bd_off;
    let rnd = ((o0 + o1 + 1) as i64) << log2_wd;
    let s0 = &s0[..w * h];
    let s1 = &s1[..w * h];
    for ((r0, r1), dst_row) in s0
        .chunks_exact(w)
        .zip(s1.chunks_exact(w))
        .zip(dst.chunks_exact_mut(dst_stride))
    {
        for ((&a, &b), out) in r0.iter().zip(r1.iter()).zip(dst_row.iter_mut().take(w)) {
            let v = ((a as i64 * w0 as i64 + b as i64 * w1 as i64 + rnd) >> (log2_wd + 1)) as i32;
            *out = clip(v, max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(data: Vec<u16>, w: usize, h: usize) -> (Vec<u16>, usize, usize) {
        (data, w, h)
    }

    #[test]
    fn full_pel_copy_roundtrips() {
        // 4x4 ramp, full-pel, 8-bit: uni_pred should reproduce the source.
        let (d, w, h) = plane((0..16).map(|v| v as u16).collect(), 4, 4);
        let r = RefPlane {
            data: &d,
            stride: w,
            width: w,
            height: h,
        };
        let mut mid = vec![0i16; 16];
        luma_interp(&r, 0, 0, 0, 0, 4, 4, 8, &mut mid);
        let mut out = vec![0u16; 16];
        uni_pred(&mid, 4, 4, 8, &mut out, 4);
        assert_eq!(out, d);
    }

    #[test]
    fn half_pel_symmetry() {
        // Constant plane -> any fractional position yields the same constant.
        let d = vec![128u16; 8 * 8];
        let r = RefPlane {
            data: &d,
            stride: 8,
            width: 8,
            height: 8,
        };
        let mut mid = vec![0i16; 16];
        luma_interp(&r, 2, 2, 2, 2, 4, 4, 8, &mut mid);
        let mut out = vec![0u16; 16];
        uni_pred(&mid, 4, 4, 8, &mut out, 4);
        assert!(out.iter().all(|&v| v == 128));
    }

    #[test]
    fn half_pel_exact_value() {
        // 8x1 ramp; half-pel horizontal at x=3 should equal spec 8-tap output.
        let d: Vec<u16> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let r = RefPlane {
            data: &d,
            stride: 8,
            width: 8,
            height: 1,
        };
        let mut mid = vec![0i16; 1];
        // position x0=3 means taps cover indices 0..7 (x0-3 .. x0+4).
        luma_interp(&r, 3, 0, 2, 0, 1, 1, 8, &mut mid);
        assert_eq!(mid[0], 2880, "intermediate half-pel");
        let mut out = vec![0u16; 1];
        uni_pred(&mid, 1, 1, 8, &mut out, 1);
        assert_eq!(out[0], 45);
    }

    #[test]
    fn bi_avg_of_equal_is_same() {
        let s = vec![128i16 << 6; 16];
        let mut out = vec![0u16; 16];
        bi_pred(&s, &s, 4, 4, 8, &mut out, 4);
        assert!(out.iter().all(|&v| v == 128));
    }
}
