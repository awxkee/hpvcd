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

pub(crate) static LUMA_FILTER: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

pub(crate) static CHROMA_FILTER: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

pub(crate) type LumaInterpFn = for<'a> fn(
    &RefPlane<'a>,
    isize,
    isize,
    usize,
    usize,
    usize,
    usize,
    u8,
    &mut [i16],
    &mut Vec<i32>,
);
pub(crate) type ChromaInterpFn = for<'a> fn(
    &RefPlane<'a>,
    isize,
    isize,
    usize,
    usize,
    usize,
    usize,
    u8,
    &mut [i16],
    &mut Vec<i32>,
);
pub(crate) type UniMcFn = fn(&[i16], usize, usize, usize, usize, u8, &mut [u16], usize);
pub(crate) type BiMcFn = fn(&[i16], &[i16], usize, usize, usize, usize, u8, &mut [u16], usize);
pub(crate) type UniMcWeightedFn =
    fn(&[i16], usize, usize, usize, usize, u8, i32, i32, u8, &mut [u16], usize);
pub(crate) type BiMcWeightedFn =
    fn(&[i16], &[i16], usize, usize, usize, usize, u8, i32, i32, i32, i32, u8, &mut [u16], usize);

static LUMA_INTERP: std::sync::OnceLock<LumaInterpFn> = std::sync::OnceLock::new();
static CHROMA_INTERP: std::sync::OnceLock<ChromaInterpFn> = std::sync::OnceLock::new();
static UNI_MC: std::sync::OnceLock<UniMcFn> = std::sync::OnceLock::new();
static BI_MC: std::sync::OnceLock<BiMcFn> = std::sync::OnceLock::new();
static UNI_MC_WEIGHTED: std::sync::OnceLock<UniMcWeightedFn> = std::sync::OnceLock::new();
static BI_MC_WEIGHTED: std::sync::OnceLock<BiMcWeightedFn> = std::sync::OnceLock::new();

#[inline]
pub(crate) fn resolve_luma_interp() -> LumaInterpFn {
    *LUMA_INTERP.get_or_init(|| {
        let mut _f: LumaInterpFn = luma_interp_scalar_scratch;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::luma_interp_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::luma_interp_sse41;
            }
        }

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::luma_interp_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_chroma_interp() -> ChromaInterpFn {
    *CHROMA_INTERP.get_or_init(|| {
        let mut _f: ChromaInterpFn = chroma_interp_scalar_scratch;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::chroma_interp_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::chroma_interp_sse41;
            }
        }

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::chroma_interp_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_uni_mc() -> UniMcFn {
    *UNI_MC.get_or_init(|| {
        let mut _f: UniMcFn = uni_mc_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::uni_mc_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::uni_mc_sse41;
            }
        }

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::uni_mc_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_bi_mc() -> BiMcFn {
    *BI_MC.get_or_init(|| {
        let mut _f: BiMcFn = bi_mc_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::bi_mc_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::bi_mc_sse41;
            }
        }

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::bi_mc_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_uni_mc_weighted() -> UniMcWeightedFn {
    *UNI_MC_WEIGHTED.get_or_init(|| {
        let mut _f: UniMcWeightedFn = uni_mc_weighted_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::uni_mc_weighted_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::uni_mc_weighted_sse41;
            }
        }

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::uni_mc_weighted_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_bi_mc_weighted() -> BiMcWeightedFn {
    *BI_MC_WEIGHTED.get_or_init(|| {
        let mut _f: BiMcWeightedFn = bi_mc_weighted_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::bi_mc_weighted_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::bi_mc_weighted_sse41;
            }
        }

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::bi_mc_weighted_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn sample_max(bit_depth: u8) -> i32 {
    if bit_depth == 0 {
        0
    } else if bit_depth <= 15 {
        (1i32 << bit_depth) - 1
    } else {
        u16::MAX as i32
    }
}

#[inline]
pub(crate) fn has_motion_dst(dst: &[u16], stride: usize, w: usize, h: usize) -> bool {
    if w == 0 || h == 0 || stride < w {
        return false;
    }
    let Some(last_row) = h.checked_sub(1).and_then(|y| y.checked_mul(stride)) else {
        return false;
    };
    let Some(end) = last_row.checked_add(w) else {
        return false;
    };
    dst.len() >= end
}

#[inline]
pub(crate) fn can_motion_comp(
    src_len: usize,
    pred_w: usize,
    pred_h: usize,
    valid_w: usize,
    valid_h: usize,
    dst: &[u16],
    dst_stride: usize,
) -> bool {
    if valid_w > pred_w || valid_h > pred_h {
        return false;
    }
    let Some(len) = pred_w.checked_mul(pred_h) else {
        return false;
    };
    src_len >= len && has_motion_dst(dst, dst_stride, valid_w, valid_h)
}

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
    fn row(&self, y: isize) -> &[u16] {
        let yc = y.clamp(0, self.height as isize - 1) as usize;
        let start = yc * self.stride;
        &self.data[start..start + self.width]
    }
}

/// Internal intermediate bit shift (shift1 = BitDepth - 8).
#[inline]
pub(crate) fn shift1(bd: u8) -> u32 {
    (bd - 8) as u32
}

/// Whether a rectangular interpolation footprint is fully inside the reference plane.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn interp_in_bounds(
    r: &RefPlane,
    x0: isize,
    y0: isize,
    left: isize,
    top: isize,
    right: usize,
    bottom: usize,
    w: usize,
    h: usize,
) -> bool {
    if w == 0 || h == 0 || r.width == 0 || r.height == 0 {
        return false;
    }
    let x_min = x0 - left;
    let y_min = y0 - top;
    let x_max = x0 + w as isize + right as isize;
    let y_max = y0 + h as isize + bottom as isize;
    x_min >= 0 && y_min >= 0 && x_max <= r.width as isize && y_max <= r.height as isize
}

/// Luma fractional interpolation at quarter-pel precision.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_interp_scalar_scratch(
    r: &RefPlane,
    x0: isize,
    y0: isize,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    bd: u8,
    dst: &mut [i16],
    tmp: &mut Vec<i32>,
) {
    interp_scalar_scratch(
        r,
        x0,
        y0,
        (fx != 0).then(|| &LUMA_FILTER[fx]),
        (fy != 0).then(|| &LUMA_FILTER[fy]),
        w,
        h,
        bd,
        dst,
        tmp,
    );
}

/// Chroma fractional interpolation at eighth-pel precision.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_interp_scalar_scratch(
    r: &RefPlane,
    x0: isize,
    y0: isize,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    bd: u8,
    dst: &mut [i16],
    tmp: &mut Vec<i32>,
) {
    interp_scalar_scratch(
        r,
        x0,
        y0,
        (fx != 0).then(|| &CHROMA_FILTER[fx]),
        (fy != 0).then(|| &CHROMA_FILTER[fy]),
        w,
        h,
        bd,
        dst,
        tmp,
    );
}

// One CTB-wide chunk and up to seven extra horizontal luma taps.
const INTERP_CHUNK: usize = 64;
const INTERP_ROW: usize = INTERP_CHUNK + 7;

#[inline]
fn extended_row<'a>(src: &'a [u16], x0: isize, pad: &'a mut [u16]) -> &'a [u16] {
    let start = x0.max(0) as usize;
    if x0 >= 0
        && let Some(inside) = src.get(start..start + pad.len())
    {
        return inside;
    }

    let left = x0.saturating_neg().clamp(0, pad.len() as isize) as usize;
    let start = start.min(src.len());
    let count = (src.len() - start).min(pad.len() - left);
    let (left, rest) = pad.split_at_mut(left);
    let (inside, right) = rest.split_at_mut(count);
    left.fill(src[0]);
    inside.copy_from_slice(&src[start..start + count]);
    right.fill(src[src.len() - 1]);
    pad
}

#[inline]
fn filter_horizontal_row<const N: usize, T>(
    src: &[u16],
    x0: isize,
    filter: &[i32; N],
    pad: &mut [u16; INTERP_ROW],
    dst: &mut [T],
    convert: impl Fn(i32) -> T,
) {
    for (chunk, out) in dst.chunks_mut(INTERP_CHUNK).enumerate() {
        let start = x0 + (chunk * INTERP_CHUNK) as isize;
        let src = extended_row(src, start, &mut pad[..out.len() + N - 1]);
        for (sample, taps) in out.iter_mut().zip(src.windows(N)) {
            let acc = taps
                .iter()
                .zip(filter)
                .map(|(&p, &c)| i32::from(p) * c)
                .sum();
            *sample = convert(acc);
        }
    }
}

#[inline]
fn filter_vertical_row<const N: usize, T: Copy + Into<i32>>(
    rows: [&[T]; N],
    col: impl Fn(usize) -> usize,
    filter: &[i32; N],
    shift: u32,
    dst: &mut [i16],
) {
    for (x, out) in dst.iter_mut().enumerate() {
        let sx = col(x);
        let acc: i32 = rows
            .iter()
            .zip(filter)
            .map(|(row, &c)| row[sx].into() * c)
            .sum();
        *out = (acc >> shift) as i16;
    }
}

/// Separable interpolation into a 14-bit internal-precision prediction.
#[allow(clippy::too_many_arguments)]
fn interp_scalar_scratch<const N: usize>(
    r: &RefPlane,
    x0: isize,
    y0: isize,
    horizontal: Option<&[i32; N]>,
    vertical: Option<&[i32; N]>,
    w: usize,
    h: usize,
    bd: u8,
    dst: &mut [i16],
    tmp: &mut Vec<i32>,
) {
    let s1 = shift1(bd);
    let halo = (N / 2 - 1) as isize;
    let dst = &mut dst[..w * h];
    let mut pad = [0; INTERP_ROW];

    match (horizontal, vertical) {
        (None, None) => {
            for (y, out) in dst.chunks_exact_mut(w).enumerate() {
                filter_horizontal_row(r.row(y0 + y as isize), x0, &[64], &mut pad, out, |acc| {
                    (acc >> s1) as i16
                });
            }
        }
        (Some(hf), None) => {
            for (y, out) in dst.chunks_exact_mut(w).enumerate() {
                filter_horizontal_row(
                    r.row(y0 + y as isize),
                    x0 - halo,
                    hf,
                    &mut pad,
                    out,
                    |acc| (acc >> s1) as i16,
                );
            }
        }
        (None, Some(vf)) => {
            for (y, out) in dst.chunks_exact_mut(w).enumerate() {
                let sy = y0 + y as isize - halo;
                let rows = std::array::from_fn(|i| r.row(sy + i as isize));
                let col = |x| (x0 + x as isize).clamp(0, r.width as isize - 1) as usize;
                filter_vertical_row(rows, col, vf, s1, out);
            }
        }
        (Some(hf), Some(vf)) => {
            // Every horizontal intermediate is overwritten before the vertical pass.
            tmp.resize(w * (h + N - 1), 0);
            for (y, row) in tmp.chunks_exact_mut(w).enumerate() {
                filter_horizontal_row(
                    r.row(y0 + y as isize - halo),
                    x0 - halo,
                    hf,
                    &mut pad,
                    row,
                    |acc| acc >> s1,
                );
            }

            for (y, out) in dst.chunks_exact_mut(w).enumerate() {
                let rows = std::array::from_fn(|i| &tmp[(y + i) * w..(y + i + 1) * w]);
                filter_vertical_row(rows, |x| x, vf, 6, out);
            }
        }
    }
}

#[inline]
fn clip(v: i32, max: i32) -> u16 {
    v.clamp(0, max) as u16
}

/// Uni-prediction motion-compensation write: combines an intermediate `pred_w × pred_h`
/// prediction into a possibly clipped `valid_w × valid_h` reconstruction region.
#[allow(clippy::too_many_arguments)]
pub(crate) fn uni_mc_scalar(
    src: &[i16],
    pred_w: usize,
    pred_h: usize,
    valid_w: usize,
    valid_h: usize,
    bd: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    if valid_w == 0 || valid_h == 0 {
        return;
    }
    assert!(valid_w <= pred_w && valid_h <= pred_h);
    assert!(can_motion_comp(
        src.len(),
        pred_w,
        pred_h,
        valid_w,
        valid_h,
        dst,
        dst_stride
    ));
    let shift = 14 - bd as i32;
    let offset = if shift > 0 { 1 << (shift - 1) } else { 0 };
    let max = sample_max(bd);
    let src = &src[..pred_w * pred_h];

    for (src_row, dst_row) in src
        .chunks_exact(pred_w)
        .zip(dst.chunks_mut(dst_stride))
        .take(valid_h)
    {
        for (&s, out) in src_row.iter().zip(dst_row.iter_mut()).take(valid_w) {
            let v = (s as i32 + offset) >> shift;
            *out = clip(v, max);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bi_mc_scalar(
    s0: &[i16],
    s1: &[i16],
    pred_w: usize,
    pred_h: usize,
    valid_w: usize,
    valid_h: usize,
    bd: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    if valid_w == 0 || valid_h == 0 {
        return;
    }
    assert!(valid_w <= pred_w && valid_h <= pred_h);
    assert!(can_motion_comp(
        s0.len(),
        pred_w,
        pred_h,
        valid_w,
        valid_h,
        dst,
        dst_stride
    ));
    assert!(s1.len() >= pred_w * pred_h);
    let shift = 15 - bd as i32;
    let offset = 1 << (shift - 1);
    let max = sample_max(bd);
    let s0 = &s0[..pred_w * pred_h];
    let s1 = &s1[..pred_w * pred_h];

    for ((r0, r1), dst_row) in s0
        .chunks_exact(pred_w)
        .zip(s1.chunks_exact(pred_w))
        .zip(dst.chunks_mut(dst_stride))
        .take(valid_h)
    {
        for ((&a, &b), out) in r0
            .iter()
            .zip(r1.iter())
            .zip(dst_row.iter_mut())
            .take(valid_w)
        {
            let v = (a as i32 + b as i32 + offset) >> shift;
            *out = clip(v, max);
        }
    }
}

/// Explicit weighted uni-prediction motion-compensation write.
/// `offset` is already expressed in reconstructed-sample units.
#[allow(clippy::too_many_arguments)]
pub(crate) fn uni_mc_weighted_scalar(
    src: &[i16],
    pred_w: usize,
    pred_h: usize,
    valid_w: usize,
    valid_h: usize,
    bd: u8,
    weight: i32,
    offset: i32,
    log2_denom: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    if valid_w == 0 || valid_h == 0 {
        return;
    }
    assert!(valid_w <= pred_w && valid_h <= pred_h);
    assert!(can_motion_comp(
        src.len(),
        pred_w,
        pred_h,
        valid_w,
        valid_h,
        dst,
        dst_stride
    ));
    let shift1 = 14 - bd as i32;
    let log2_wd = log2_denom as i32 + shift1;
    let max = sample_max(bd);
    let round = if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 };
    let src = &src[..pred_w * pred_h];

    for (src_row, dst_row) in src
        .chunks_exact(pred_w)
        .zip(dst.chunks_mut(dst_stride))
        .take(valid_h)
    {
        for (&s, out) in src_row.iter().zip(dst_row.iter_mut()).take(valid_w) {
            let v = if log2_wd >= 1 {
                ((s as i32 * weight + round) >> log2_wd) + offset
            } else {
                s as i32 * weight + offset
            };
            *out = clip(v, max);
        }
    }
}

/// Explicit weighted bi-prediction motion-compensation write.
/// `o0` and `o1` are already expressed in reconstructed-sample units.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bi_mc_weighted_scalar(
    s0: &[i16],
    s1: &[i16],
    pred_w: usize,
    pred_h: usize,
    valid_w: usize,
    valid_h: usize,
    bd: u8,
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
    log2_denom: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    if valid_w == 0 || valid_h == 0 {
        return;
    }
    assert!(valid_w <= pred_w && valid_h <= pred_h);
    assert!(can_motion_comp(
        s0.len(),
        pred_w,
        pred_h,
        valid_w,
        valid_h,
        dst,
        dst_stride
    ));
    assert!(s1.len() >= pred_w * pred_h);
    let shift1 = 14 - bd as i32;
    let log2_wd = log2_denom as i32 + shift1;
    let max = sample_max(bd);
    let rnd = (o0 as i64 + o1 as i64 + 1) << log2_wd;
    let s0 = &s0[..pred_w * pred_h];
    let s1 = &s1[..pred_w * pred_h];

    for ((r0, r1), dst_row) in s0
        .chunks_exact(pred_w)
        .zip(s1.chunks_exact(pred_w))
        .zip(dst.chunks_mut(dst_stride))
        .take(valid_h)
    {
        for ((&a, &b), out) in r0
            .iter()
            .zip(r1.iter())
            .zip(dst_row.iter_mut())
            .take(valid_w)
        {
            let v = ((a as i64 * w0 as i64 + b as i64 * w1 as i64 + rnd) >> (log2_wd + 1)) as i32;
            *out = clip(v, max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CHROMA_FILTER, LUMA_FILTER, LumaInterpFn, RefPlane, bi_mc_weighted_scalar,
        chroma_interp_scalar_scratch, luma_interp_scalar_scratch, shift1, uni_mc_weighted_scalar,
    };

    fn compare_clamped_interpolation<const N: usize>(interp: LumaInterpFn, filters: &[[i32; N]]) {
        let halo = (N / 2 - 1) as isize;
        let mut tmp = vec![i32::MIN; 1024];
        for (width, height, stride, w, h) in [
            (19, 14, 23, 7, 5),
            (1, 1, 2, 3, 2),
            (73, 11, 78, 67, 4),
            (N + 7, N + 7, N + 7, 8, 8),
            // The interior position exercises multiple SIMD chunks and a tail.
            (81, 18, 86, 67, 8),
        ] {
            for bd in [8, 10, 12] {
                let s1 = shift1(bd);
                let data: Vec<u16> = (0..stride * height)
                    .map(|i| (((i * 1877) ^ (i * i * 93)) & ((1 << bd) - 1)) as u16)
                    .collect();
                let r = RefPlane {
                    data: &data,
                    stride,
                    width,
                    height,
                };
                let sample = |x: isize, y: isize| {
                    let x = x.clamp(0, width as isize - 1) as usize;
                    let y = y.clamp(0, height as isize - 1) as usize;
                    i32::from(data[y * stride + x])
                };
                let positions = [
                    (-90, -30),
                    (-4, 4),
                    (width as isize - 2, -2),
                    (halo, halo),
                    (4, 4),
                    (3, height as isize + 50),
                    (width as isize + 70, height as isize + 60),
                ];
                let mut dst = vec![i16::MIN; w * h];
                for (x0, y0) in positions {
                    for (fx, hf) in filters.iter().enumerate() {
                        for (fy, vf) in filters.iter().enumerate() {
                            interp(&r, x0, y0, fx, fy, w, h, bd, &mut dst, &mut tmp);
                            for (index, &actual) in dst.iter().enumerate() {
                                let x = x0 + (index % w) as isize;
                                let y = y0 + (index / w) as isize;
                                let horizontal = |y| {
                                    let sum: i32 = hf
                                        .iter()
                                        .enumerate()
                                        .map(|(i, &c)| c * sample(x + i as isize - halo, y))
                                        .sum();
                                    sum >> s1
                                };
                                let vertical = |row: &dyn Fn(isize) -> i32| {
                                    vf.iter()
                                        .enumerate()
                                        .map(|(i, &c)| c * row(y + i as isize - halo))
                                        .sum::<i32>()
                                };
                                let expected = match (fx, fy) {
                                    (0, 0) => (sample(x, y) << 6) >> s1,
                                    (_, 0) => horizontal(y),
                                    (0, _) => vertical(&|y| sample(x, y)) >> s1,
                                    _ => vertical(&horizontal) >> 6,
                                };
                                assert_eq!(
                                    actual, expected as i16,
                                    "{N} taps, {bd} bit, ({fx}, {fy}) at ({x}, {y})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_interpolation_matches_per_tap_edge_clamping() {
        compare_clamped_interpolation(luma_interp_scalar_scratch, &LUMA_FILTER);
        compare_clamped_interpolation(chroma_interp_scalar_scratch, &CHROMA_FILTER);
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_interpolation_matches_per_tap_edge_clamping() {
        compare_clamped_interpolation(crate::neon::luma_interp_neon, &LUMA_FILTER);
        compare_clamped_interpolation(crate::neon::chroma_interp_neon, &CHROMA_FILTER);
    }

    #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn sse41_interpolation_matches_per_tap_edge_clamping() {
        if std::is_x86_feature_detected!("sse4.1") {
            compare_clamped_interpolation(crate::sse::luma_interp_sse41, &LUMA_FILTER);
            compare_clamped_interpolation(crate::sse::chroma_interp_sse41, &CHROMA_FILTER);
        }
    }

    #[cfg(all(feature = "avx", target_arch = "x86_64"))]
    #[test]
    fn avx2_interpolation_matches_per_tap_edge_clamping() {
        if std::is_x86_feature_detected!("avx2") {
            compare_clamped_interpolation(crate::avx::luma_interp_avx2, &LUMA_FILTER);
            compare_clamped_interpolation(crate::avx::chroma_interp_avx2, &CHROMA_FILTER);
        }
    }

    #[test]
    fn weighted_uni_consumes_sample_domain_offset() {
        // Interpolation buffers use 14-bit internal precision. At 10-bit,
        // sample 100 is represented as 100 << 4.
        let src = [100i16 << 4];
        let mut dst = [0u16; 1];
        uni_mc_weighted_scalar(&src, 1, 1, 1, 1, 10, 1, 7, 0, &mut dst, 1);
        assert_eq!(dst[0], 107);
    }

    #[test]
    fn weighted_bi_consumes_sample_domain_offsets() {
        let src0 = [100i16 << 4];
        let src1 = [200i16 << 4];
        let mut dst = [0u16; 1];
        bi_mc_weighted_scalar(&src0, &src1, 1, 1, 1, 1, 10, 1, 7, 1, -3, 0, &mut dst, 1);
        assert_eq!(dst[0], 152);
    }
}
