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
use crate::color::{Cicp, MatrixCoefficients};
use crate::fmt::{BitDepth, ChromaFormat, ImageBuffer};

const Q13: u32 = 13;
const Q13_ONE: i64 = 1 << Q13; // 8192 == 1.0
const Q13_ROUND: i64 = 1 << (Q13 - 1); // 4096 == 0.5
/// Limited-range luma scale 255/219 ≈ 1.16438 in Q0.13.
const Q13_KY_LIMITED: i64 = 9539;
/// Limited-range chroma scale 255/224 ≈ 1.13839 in Q0.13.
const Q13_KUV_LIMITED: i64 = 9326;

/// Planar YCbCr image produced by the HEVC decoder.
#[allow(dead_code)]
impl YuvPlanes {
    pub(crate) fn luma_len(&self) -> usize {
        self.y.len()
    }
    pub(crate) fn dims(&self) -> (usize, usize) {
        (self.width, self.height)
    }
    pub(crate) fn y_u8(&self) -> Vec<u8> {
        self.y.iter().map(|&v| v as u8).collect()
    }
    pub(crate) fn cb_u8(&self) -> Vec<u8> {
        self.cb.iter().map(|&v| v as u8).collect()
    }
    pub(crate) fn cr_u8(&self) -> Vec<u8> {
        self.cr.iter().map(|&v| v as u8).collect()
    }
}
pub(crate) struct YuvPlanes {
    pub(crate) y: Vec<u16>,
    pub(crate) cb: Vec<u16>,
    pub(crate) cr: Vec<u16>,
    pub(crate) width: usize, // coded (64-multiple)
    pub(crate) height: usize,
    pub(crate) chroma: ChromaFormat,
    pub(crate) bit_depth: BitDepth,
}

use crate::threadpool::{DisjointMut, ThreadPool, parallel_for};

/// Precomputed conversion constants; row-range methods let bands run in parallel.
struct Cvt<'a> {
    yuv: &'a YuvPlanes,
    dw: usize,
    x0: usize,
    y0: usize,
    max_val: i64,
    y_black: i64,
    neutral: i64,
    k_y: i64,
    cr_to_r: i64,
    cb_to_g: i64,
    cr_to_g: i64,
    cb_to_b: i64,
    sub_w: usize,
    sub_h: usize,
    cw: usize,
    ch: usize,
    identity: bool,
}

impl Cvt<'_> {
    fn new<'a>(yuv: &'a YuvPlanes, dw: usize, x0: usize, y0: usize, color: &Cicp) -> Cvt<'a> {
        let scale = 1i64 << yuv.bit_depth.minus8();
        let k_y: i64 = if color.full_range {
            Q13_ONE
        } else {
            Q13_KY_LIMITED
        };
        let k_uv: i64 = if color.full_range {
            Q13_ONE
        } else {
            Q13_KUV_LIMITED
        };
        let (cr_r0, cb_g0, cr_g0, cb_b0) = match color.matrix {
            MatrixCoefficients::Bt470Bg | MatrixCoefficients::Smpte170m => {
                (11485i64, -2819i64, -5851i64, 14516i64)
            }
            MatrixCoefficients::Bt2020Ncl => (12080i64, -1348i64, -4681i64, 17546i64),
            MatrixCoefficients::Identity => (0, 0, 0, 0),
            _ => (12901i64, -1534i64, -3835i64, 15201i64), // BT.709
        };
        let sub_w = yuv.chroma.sub_w();
        let sub_h = yuv.chroma.sub_h();
        Cvt {
            yuv,
            dw,
            x0,
            y0,
            max_val: yuv.bit_depth.max_val() as i64,
            y_black: if color.full_range { 0 } else { 16 * scale },
            neutral: 128 * scale,
            k_y,
            // Fold the chroma range scale into the matrix once, staying in Q0.13.
            cr_to_r: (cr_r0 * k_uv + Q13_ROUND) >> Q13,
            cb_to_g: (cb_g0 * k_uv + Q13_ROUND) >> Q13,
            cr_to_g: (cr_g0 * k_uv + Q13_ROUND) >> Q13,
            cb_to_b: (cb_b0 * k_uv + Q13_ROUND) >> Q13,
            sub_w,
            sub_h,
            cw: yuv.width.div_ceil(sub_w),
            ch: yuv.height.div_ceil(sub_h),
            identity: color.matrix == MatrixCoefficients::Identity,
        }
    }

    fn pixel(&self, y_pix: usize, x_pix: usize) -> (i64, i64, i64) {
        let yuv = self.yuv;
        let src_y = self.y0.saturating_add(y_pix);
        let src_x = self.x0.saturating_add(x_pix);
        let y_row = src_y.min(yuv.height - 1);
        let c_row = (src_y / self.sub_h).min(self.ch - 1);
        let x_col = src_x.min(yuv.width - 1);
        let c_col = (src_x / self.sub_w).min(self.cw - 1);

        let luma_raw = yuv.y[y_row * yuv.width + x_col] as i64;

        if self.identity {
            let y_scaled = (self.k_y * (luma_raw - self.y_black) + Q13_ROUND) >> Q13;
            (y_scaled, y_scaled, y_scaled)
        } else {
            let cb_raw = yuv.cb[c_row * self.cw + c_col] as i64;
            let cr_raw = yuv.cr[c_row * self.cw + c_col] as i64;
            let y_term = self.k_y * (luma_raw - self.y_black);
            let cb_c = cb_raw - self.neutral;
            let cr_c = cr_raw - self.neutral;
            let r = (y_term + self.cr_to_r * cr_c + Q13_ROUND) >> Q13;
            let g = (y_term + self.cb_to_g * cb_c + self.cr_to_g * cr_c + Q13_ROUND) >> Q13;
            let b = (y_term + self.cb_to_b * cb_c + Q13_ROUND) >> Q13;
            (r, g, b)
        }
    }
}

trait PxCast: Copy + Default + Send {
    fn cast(v: i64) -> Self;
}
impl PxCast for u8 {
    fn cast(v: i64) -> Self {
        v as u8
    }
}
impl PxCast for u16 {
    fn cast(v: i64) -> Self {
        v as u16
    }
}

impl Cvt<'_> {
    fn mono_rows<T: PxCast>(&self, y0: usize, out: &mut [T], cmax: i64) {
        let yuv = self.yuv;
        let src_x = self.x0.min(yuv.width - 1);
        for (dy, dst_row) in out.chunks_exact_mut(self.dw).enumerate() {
            let row = self.y0.saturating_add(y0 + dy).min(yuv.height - 1);
            let src_row = &yuv.y[row * yuv.width..][..yuv.width];
            let copy_w = self.dw.min(yuv.width - src_x);
            let (dst_copy, dst_edge) = dst_row.split_at_mut(copy_w);
            for (dst, &luma) in dst_copy.iter_mut().zip(src_row[src_x..].iter()) {
                let v = (self.k_y * (luma as i64 - self.y_black) + Q13_ROUND) >> Q13;
                *dst = T::cast(v.clamp(0, cmax));
            }
            if !dst_edge.is_empty() {
                let last = src_row[src_x + copy_w.saturating_sub(1)];
                let v = (self.k_y * (last as i64 - self.y_black) + Q13_ROUND) >> Q13;
                dst_edge.fill(T::cast(v.clamp(0, cmax)));
            }
        }
    }

    /// Non-identity fast path: requires the requested visible window to fit inside
    /// the coded planes.
    fn fast_rows<T: PxCast>(&self, y0: usize, out: &mut [T], cmax: i64) {
        let yuv = self.yuv;
        for (dy, row_out) in out.chunks_exact_mut(self.dw * 3).enumerate() {
            let y_pix = self.y0 + y0 + dy;
            let luma_base = y_pix * yuv.width + self.x0;
            let c_base = (y_pix / self.sub_h) * self.cw;
            let luma_row = &yuv.y[luma_base..][..self.dw];
            let cb_row = &yuv.cb[c_base..][..self.cw];
            let cr_row = &yuv.cr[c_base..][..self.cw];
            for (x_pix, (dst, &luma_raw)) in row_out
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(luma_row.iter())
                .enumerate()
            {
                let c_col = (self.x0 + x_pix) / self.sub_w;
                let cb_c = cb_row[c_col] as i64 - self.neutral;
                let cr_c = cr_row[c_col] as i64 - self.neutral;
                let yv = self.k_y * (luma_raw as i64 - self.y_black);
                let r = (yv + self.cr_to_r * cr_c + Q13_ROUND) >> Q13;
                let g = (yv + self.cb_to_g * cb_c + self.cr_to_g * cr_c + Q13_ROUND) >> Q13;
                let b = (yv + self.cb_to_b * cb_c + Q13_ROUND) >> Q13;
                dst[0] = T::cast(r.clamp(0, cmax));
                dst[1] = T::cast(g.clamp(0, cmax));
                dst[2] = T::cast(b.clamp(0, cmax));
            }
        }
    }

    fn slow_rows<T: PxCast>(&self, y0: usize, out: &mut [T], cmax: i64) {
        for (dy, row_out) in out.chunks_exact_mut(self.dw * 3).enumerate() {
            for (x_pix, dst) in row_out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let (r, g, b) = self.pixel(y0 + dy, x_pix);
                dst[0] = T::cast(r.clamp(0, cmax));
                dst[1] = T::cast(g.clamp(0, cmax));
                dst[2] = T::cast(b.clamp(0, cmax));
            }
        }
    }

    fn ycgco_rows<T: PxCast>(&self, y0: usize, out: &mut [T], cmax: i64) {
        let yuv = self.yuv;
        for (dy, row_out) in out.chunks_exact_mut(self.dw * 3).enumerate() {
            let src_y = self.y0.saturating_add(y0 + dy);
            let y_row = src_y.min(yuv.height - 1);
            let c_row = (src_y / self.sub_h).min(self.ch - 1);
            let l_row = &yuv.y[y_row * yuv.width..][..yuv.width];
            let cb_row = &yuv.cb[c_row * self.cw..][..self.cw];
            let cr_row = &yuv.cr[c_row * self.cw..][..self.cw];
            for (x_pix, dst) in row_out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let src_x = self.x0.saturating_add(x_pix);
                let x_col = src_x.min(yuv.width - 1);
                let c_col = (src_x / self.sub_w).min(self.cw - 1);
                let y = l_row[x_col] as i64;
                let cg = cb_row[c_col] as i64 - self.neutral;
                let co = cr_row[c_col] as i64 - self.neutral;
                let t = y - cg;
                dst[0] = T::cast((t + co).clamp(0, cmax)); // R
                dst[1] = T::cast((y + cg).clamp(0, cmax)); // G
                dst[2] = T::cast((t - co).clamp(0, cmax)); // B
            }
        }
    }
}

/// Fill `dh` output rows via `f(y0, band)`, splitting into bands on the pool
/// when profitable; serial otherwise. Output is byte-identical either way.
fn banded<T, F>(pool: Option<&ThreadPool>, dw: usize, dh: usize, chn: usize, f: F) -> Vec<T>
where
    T: Default + Copy + Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let total = dw * dh * chn;
    if let Some(p) = pool
        && p.threads() > 1
        && dh > 1
    {
        let band_rows = dh.div_ceil((p.threads() * 4).min(dh));
        let bands = dh.div_ceil(band_rows);
        let dm = DisjointMut::new(vec![T::default(); total]);
        parallel_for(p, bands, |b| {
            let y0 = b * band_rows;
            let nr = band_rows.min(dh - y0);
            let mut band = dm.slice_mut(y0 * dw * chn..(y0 + nr) * dw * chn);
            f(y0, &mut band);
        });
        return dm.into_inner();
    }
    let mut v = vec![T::default(); total];
    f(0, &mut v);
    v
}

pub(crate) fn yuv_to_rgb_window_with_color(
    yuv: &YuvPlanes,
    dw: usize,
    dh: usize,
    crop_left: usize,
    crop_top: usize,
    color: &Cicp,
) -> ImageBuffer {
    yuv_to_rgb_window_with_color_pool(yuv, dw, dh, crop_left, crop_top, color, None)
}

pub(crate) fn yuv_to_rgb_window_with_color_pool(
    yuv: &YuvPlanes,
    dw: usize,
    dh: usize,
    crop_left: usize,
    crop_top: usize,
    color: &Cicp,
    pool: Option<&ThreadPool>,
) -> ImageBuffer {
    if dw == 0 || dh == 0 || yuv.width == 0 || yuv.height == 0 {
        return if yuv.chroma.is_monochrome() {
            if yuv.bit_depth == BitDepth::Eight {
                ImageBuffer::Luma8(Vec::new())
            } else {
                ImageBuffer::Luma16(Vec::new())
            }
        } else if yuv.bit_depth == BitDepth::Eight {
            ImageBuffer::Rgb8(Vec::new())
        } else {
            ImageBuffer::Rgb16(Vec::new())
        };
    }

    let crop_left = crop_left.min(yuv.width - 1);
    let crop_top = crop_top.min(yuv.height - 1);
    let cvt = Cvt::new(yuv, dw, crop_left, crop_top, color);

    if yuv.chroma.is_monochrome() {
        return if yuv.bit_depth == BitDepth::Eight {
            ImageBuffer::Luma8(banded(pool, dw, dh, 1, |y0, out| {
                cvt.mono_rows::<u8>(y0, out, 255)
            }))
        } else {
            ImageBuffer::Luma16(banded(pool, dw, dh, 1, |y0, out| {
                cvt.mono_rows::<u16>(y0, out, cvt.max_val)
            }))
        };
    }

    if color.matrix == MatrixCoefficients::YCgCo {
        return if yuv.bit_depth == BitDepth::Eight {
            ImageBuffer::Rgb8(banded(pool, dw, dh, 3, |y0, out| {
                cvt.ycgco_rows::<u8>(y0, out, 255)
            }))
        } else {
            ImageBuffer::Rgb16(banded(pool, dw, dh, 3, |y0, out| {
                cvt.ycgco_rows::<u16>(y0, out, cvt.max_val)
            }))
        };
    }

    // Fast path assumes the visible window fits inside the coded planes, making
    // the per-pixel edge clamps in `pixel` provably no-ops; output is bit-exact.
    let fast = dw <= yuv.width - crop_left && dh <= yuv.height - crop_top;
    if fast && !cvt.identity {
        return if yuv.bit_depth == BitDepth::Eight {
            ImageBuffer::Rgb8(banded(pool, dw, dh, 3, |y0, out| {
                cvt.fast_rows::<u8>(y0, out, 255)
            }))
        } else {
            ImageBuffer::Rgb16(banded(pool, dw, dh, 3, |y0, out| {
                cvt.fast_rows::<u16>(y0, out, cvt.max_val)
            }))
        };
    }

    if yuv.bit_depth == BitDepth::Eight {
        ImageBuffer::Rgb8(banded(pool, dw, dh, 3, |y0, out| {
            cvt.slow_rows::<u8>(y0, out, 255)
        }))
    } else {
        ImageBuffer::Rgb16(banded(pool, dw, dh, 3, |y0, out| {
            cvt.slow_rows::<u16>(y0, out, cvt.max_val)
        }))
    }
}
