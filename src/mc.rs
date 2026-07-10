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
    pub(crate) fn sample(&self, x: isize, y: isize) -> i32 {
        let xc = x.clamp(0, self.width as isize - 1) as usize;
        let yc = y.clamp(0, self.height as isize - 1) as usize;
        self.data[yc * self.stride + xc] as i32
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

/// Luma fractional interpolation of a `w x h` block at integer position
/// `(x0, y0)` with quarter-pel fraction `(fx, fy)` into `dst` (16-bit
/// intermediate, stride = w). Follows the separable H-then-V approach.
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

    // Separable: horizontal into scratch of height h+7, then vertical.
    let tmp_h = h + 7;
    tmp.clear();
    tmp.resize(w * tmp_h, 0);
    let tmp = &mut tmp[..w * tmp_h];
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
    tmp.clear();
    tmp.resize(w * tmp_h, 0);
    let tmp = &mut tmp[..w * tmp_h];
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
    let off = offset << (bd as i32 - 8);
    let src = &src[..pred_w * pred_h];

    for (src_row, dst_row) in src
        .chunks_exact(pred_w)
        .zip(dst.chunks_mut(dst_stride))
        .take(valid_h)
    {
        for (&s, out) in src_row.iter().zip(dst_row.iter_mut()).take(valid_w) {
            let v = if log2_wd >= 1 {
                ((s as i32 * weight + round) >> log2_wd) + off
            } else {
                s as i32 * weight + off
            };
            *out = clip(v, max);
        }
    }
}

/// Explicit weighted bi-prediction motion-compensation write.
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
    let bd_off = bd as i32 - 8;
    let o0 = o0 << bd_off;
    let o1 = o1 << bd_off;
    let rnd = ((o0 + o1 + 1) as i64) << log2_wd;
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
