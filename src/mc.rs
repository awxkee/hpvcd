/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * // BSD-3-Clause OR Apache-2.0
 */

//! HEVC motion compensation: fractional-sample luma (8-tap) and chroma (4-tap)
//! interpolation into 16-bit intermediates, plus uni/bi weighted-prediction
//! combine to final samples. Coefficients per Rec. ITU-T H.265 Tables 8-11/8-12,
//! cross-checked against de265's `fallback-motion`. Kernels are safe scalar Rust
//! parameterised by bit depth; SIMD backends can slot in behind the same
//! function-pointer dispatch used elsewhere (`resolve_*`).

/// Luma 8-tap filters, indexed by quarter-pel fraction 0..=3.
const LUMA_FILTER: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// Chroma 4-tap filters, indexed by eighth-pel fraction 0..=7.
const CHROMA_FILTER: [[i32; 4]; 8] = [
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
