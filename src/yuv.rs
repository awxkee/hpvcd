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
pub(crate) struct YuvPlanes {
    pub(crate) y: Vec<u16>,
    pub(crate) cb: Vec<u16>,
    pub(crate) cr: Vec<u16>,
    pub(crate) width: usize, // coded (64-multiple)
    pub(crate) height: usize,
    pub(crate) chroma: ChromaFormat,
    pub(crate) bit_depth: BitDepth,
}

///
/// Matrix coefficients supported (ISO/IEC 23091-2):
/// * 1  – BT.709 (default for HD content)
/// * 9  – BT.2020 NCL (HDR, recent iPhones)
/// * 0  – Identity (GBR, no color transform)
/// * other – fall back to BT.709
///
/// Output length = `dw * dh * 3`, each channel at `bit_depth`'s native scale.
pub(crate) fn yuv_to_rgb_with_color(
    yuv: &YuvPlanes,
    dw: usize,
    dh: usize,
    color: &Cicp,
) -> ImageBuffer {
    if yuv.chroma.is_monochrome() {
        let y_black = if color.full_range {
            0i64
        } else {
            16i64 << yuv.bit_depth.minus8()
        };
        let k_y: i64 = if color.full_range {
            Q13_ONE
        } else {
            Q13_KY_LIMITED
        };
        let max_val = yuv.bit_depth.max_val() as i64;
        return if yuv.bit_depth == BitDepth::Eight {
            let mut out = vec![0u8; dw * dh];
            for y in 0..dh {
                for x in 0..dw {
                    let row = y.min(yuv.height - 1);
                    let col = x.min(yuv.width - 1);
                    let luma = yuv.y[row * yuv.width + col] as i64;
                    out[y * dw + x] =
                        ((k_y * (luma - y_black) + Q13_ROUND) >> Q13).clamp(0, 255) as u8;
                }
            }
            ImageBuffer::Luma8(out)
        } else {
            let mut out = vec![0u16; dw * dh];
            for y in 0..dh {
                for x in 0..dw {
                    let row = y.min(yuv.height - 1);
                    let col = x.min(yuv.width - 1);
                    let luma = yuv.y[row * yuv.width + col] as i64;
                    out[y * dw + x] =
                        ((k_y * (luma - y_black) + Q13_ROUND) >> Q13).clamp(0, max_val) as u16;
                }
            }
            ImageBuffer::Luma16(out)
        };
    }

    if color.matrix == MatrixCoefficients::YCgCo {
        return ycgco_to_rgb(yuv, dw, dh);
    }

    let max_val = yuv.bit_depth.max_val() as i64;
    let scale = 1i64 << (yuv.bit_depth.minus8());

    let sub_w = yuv.chroma.sub_w();
    let sub_h = yuv.chroma.sub_h();
    let cw = yuv.width.div_ceil(sub_w);

    let y_black = if color.full_range { 0 } else { 16 * scale };
    let neutral = 128 * scale;
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
    // Fold the chroma range scale into the matrix once (not per pixel), keeping
    // everything in Q0.13: effective = round(coeff * k_uv / 8192).
    let cr_to_r = (cr_r0 * k_uv + Q13_ROUND) >> Q13;
    let cb_to_g = (cb_g0 * k_uv + Q13_ROUND) >> Q13;
    let cr_to_g = (cr_g0 * k_uv + Q13_ROUND) >> Q13;
    let cb_to_b = (cb_b0 * k_uv + Q13_ROUND) >> Q13;

    let pixel = |y_pix: usize, x_pix: usize| -> (i64, i64, i64) {
        let y_row = y_pix.min(yuv.height - 1);
        let c_row = (y_pix / sub_h).min(yuv.height.div_ceil(sub_h) - 1);
        let x_col = x_pix.min(yuv.width - 1);
        let c_col = (x_pix / sub_w).min(cw - 1);

        let luma_raw = yuv.y[y_row * yuv.width + x_col] as i64;

        if color.matrix == MatrixCoefficients::Identity {
            let y_scaled = (k_y * (luma_raw - y_black) + Q13_ROUND) >> Q13;
            (y_scaled, y_scaled, y_scaled)
        } else {
            let cb_raw = yuv.cb[c_row * cw + c_col] as i64;
            let cr_raw = yuv.cr[c_row * cw + c_col] as i64;
            let y_term = k_y * (luma_raw - y_black);
            let cb_c = cb_raw - neutral;
            let cr_c = cr_raw - neutral;
            let r = (y_term + cr_to_r * cr_c + Q13_ROUND) >> Q13;
            let g = (y_term + cb_to_g * cb_c + cr_to_g * cr_c + Q13_ROUND) >> Q13;
            let b = (y_term + cb_to_b * cb_c + Q13_ROUND) >> Q13;
            (r, g, b)
        }
    };

    // Fast paths below assume the visible window fits inside the coded planes
    // (always true: display dims ≤ coded dims), so the per-pixel `.min()` edge
    // clamps in `pixel` are provably no-ops and are omitted. The arithmetic is
    // otherwise identical to `pixel`, so output is bit-exact.
    let fast = dw <= yuv.width && dh <= yuv.height;
    let ch = yuv.height.div_ceil(sub_h);

    if fast && color.matrix != MatrixCoefficients::Identity {
        macro_rules! convert_loop {
            ($T:ty, $clampmax:expr, $variant:path) => {{
                let mut rgb = vec![0 as $T; dw * dh * 3];
                for y_pix in 0..dh {
                    let luma_base = y_pix * yuv.width;
                    let c_base = (y_pix / sub_h) * cw;
                    let row_out = &mut rgb[y_pix * dw * 3..];
                    for x_pix in 0..dw {
                        let luma_raw = yuv.y[luma_base + x_pix] as i64;
                        let c_col = x_pix / sub_w;
                        let cb_raw = yuv.cb[c_base + c_col] as i64;
                        let cr_raw = yuv.cr[c_base + c_col] as i64;
                        let yv = k_y * (luma_raw - y_black);
                        let cb_c = cb_raw - neutral;
                        let cr_c = cr_raw - neutral;
                        let r = (yv + cr_to_r * cr_c + Q13_ROUND) >> Q13;
                        let g = (yv + cb_to_g * cb_c + cr_to_g * cr_c + Q13_ROUND) >> Q13;
                        let b = (yv + cb_to_b * cb_c + Q13_ROUND) >> Q13;
                        let o = &mut row_out[x_pix * 3..];
                        o[0] = r.clamp(0, $clampmax) as $T;
                        o[1] = g.clamp(0, $clampmax) as $T;
                        o[2] = b.clamp(0, $clampmax) as $T;
                    }
                }
                return $variant(rgb);
            }};
        }
        if yuv.bit_depth == BitDepth::Eight {
            convert_loop!(u8, 255, ImageBuffer::Rgb8);
        } else {
            convert_loop!(u16, max_val, ImageBuffer::Rgb16);
        }
    }
    let _ = ch;

    if yuv.bit_depth == BitDepth::Eight {
        let mut rgb = vec![0u8; dw * dh * 3];
        for y_pix in 0..dh {
            for x_pix in 0..dw {
                let (r, g, b) = pixel(y_pix, x_pix);
                let out = &mut rgb[(y_pix * dw + x_pix) * 3..];
                out[0] = r.clamp(0, 255) as u8;
                out[1] = g.clamp(0, 255) as u8;
                out[2] = b.clamp(0, 255) as u8;
            }
        }
        ImageBuffer::Rgb8(rgb)
    } else {
        let mut rgb = vec![0u16; dw * dh * 3];
        for y_pix in 0..dh {
            for x_pix in 0..dw {
                let (r, g, b) = pixel(y_pix, x_pix);
                let out = &mut rgb[(y_pix * dw + x_pix) * 3..];
                out[0] = r.clamp(0, max_val) as u16;
                out[1] = g.clamp(0, max_val) as u16;
                out[2] = b.clamp(0, max_val) as u16;
            }
        }
        ImageBuffer::Rgb16(rgb)
    }
}

pub(crate) fn ycgco_to_rgb(yuv: &YuvPlanes, dw: usize, dh: usize) -> ImageBuffer {
    let scale = 1i64 << yuv.bit_depth.minus8();
    let neutral = 128 * scale; // 1 << (bit_depth - 1)
    let max_val = yuv.bit_depth.max_val() as i64;
    let sub_w = yuv.chroma.sub_w();
    let sub_h = yuv.chroma.sub_h();
    let cw = yuv.width.div_ceil(sub_w);
    let ch = yuv.height.div_ceil(sub_h);

    macro_rules! run {
        ($T: ty, $cmax: expr, $variant: path) => {{
            let mut rgb = vec![0 as $T; dw * dh * 3];
            for y_pix in 0..dh {
                let y_row = y_pix.min(yuv.height - 1);
                let c_row = (y_pix / sub_h).min(ch - 1);
                let l_base = y_row * yuv.width;
                let c_base = c_row * cw;
                let row_out = &mut rgb[y_pix * dw * 3..];
                for (x_pix, dst) in row_out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let x_col = x_pix.min(yuv.width - 1);
                    let c_col = (x_pix / sub_w).min(cw - 1);
                    let y = yuv.y[l_base + x_col] as i64;
                    let cg = yuv.cb[c_base + c_col] as i64 - neutral;
                    let co = yuv.cr[c_base + c_col] as i64 - neutral;
                    let t = y - cg;
                    dst[0] = (t + co).clamp(0, $cmax) as $T; // R
                    dst[1] = (y + cg).clamp(0, $cmax) as $T; // G
                    dst[2] = (t - co).clamp(0, $cmax) as $T; // B
                }
            }
            $variant(rgb)
        }};
    }

    if yuv.bit_depth == BitDepth::Eight {
        run!(u8, 255, ImageBuffer::Rgb8)
    } else {
        run!(u16, max_val, ImageBuffer::Rgb16)
    }
}
