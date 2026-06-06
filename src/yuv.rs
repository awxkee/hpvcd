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
use crate::color::{ColorEncoding, MatrixCoefficients};
use crate::fmt::{BitDepth, ChromaFormat, ImageBuffer};

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
/// * 0  – Identity (GBR, no colour transform)
/// * other – fall back to BT.709
///
/// Output length = `dw * dh * 3`, each channel at `bit_depth`'s native scale.
pub(crate) fn yuv_to_rgb_with_color(
    yuv: &YuvPlanes,
    dw: usize,
    dh: usize,
    color: &ColorEncoding,
) -> ImageBuffer {
    // Monochrome: return luma plane directly — no colour conversion needed.
    if yuv.chroma.is_monochrome() {
        let y_black = if color.full_range {
            0i64
        } else {
            16i64 << yuv.bit_depth.minus8()
        };
        let k_y: i64 = if color.full_range { 10000 } else { 11644 };
        let max_val = yuv.bit_depth.max_val() as i64;
        return if yuv.bit_depth == BitDepth::Eight {
            let mut out = vec![0u8; dw * dh];
            for y in 0..dh {
                for x in 0..dw {
                    let row = y.min(yuv.height - 1);
                    let col = x.min(yuv.width - 1);
                    let luma = yuv.y[row * yuv.width + col] as i64;
                    out[y * dw + x] = ((k_y * (luma - y_black) + 5000) / 10000).clamp(0, 255) as u8;
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
                        ((k_y * (luma - y_black) + 5000) / 10000).clamp(0, max_val) as u16;
                }
            }
            ImageBuffer::Luma16(out)
        };
    }

    let max_val = yuv.bit_depth.max_val() as i64;
    let scale = 1i64 << (yuv.bit_depth.minus8());

    let sub_w = yuv.chroma.sub_w();
    let sub_h = yuv.chroma.sub_h();
    let cw = yuv.width.div_ceil(sub_w);

    // ── Range handling ──────────────────────────────────────────────────────
    let y_black = if color.full_range { 0 } else { 16 * scale };
    let k_y: i64 = if color.full_range { 10000 } else { 11644 };
    let neutral = 128 * scale;
    let k_uv: i64 = if color.full_range { 10000 } else { 11384 };

    // ── Matrix coefficients ─────────────────────────────────────────────────
    let (cr_to_r, cb_to_g, cr_to_g, cb_to_b) = match color.matrix {
        MatrixCoefficients::Bt470Bg | MatrixCoefficients::Smpte170m => {
            (14020i64, -3441i64, -7141i64, 17720i64)
        }
        MatrixCoefficients::Bt2020Ncl => (14746i64, -1645i64, -5714i64, 21418i64),
        MatrixCoefficients::Identity => (0, 0, 0, 0),
        _ => (15748i64, -1873i64, -4681i64, 18556i64),
    };

    let pixel = |y_pix: usize, x_pix: usize| -> (i64, i64, i64) {
        let y_row = y_pix.min(yuv.height - 1);
        let c_row = (y_pix / sub_h).min(yuv.height.div_ceil(sub_h) - 1);
        let x_col = x_pix.min(yuv.width - 1);
        let c_col = (x_pix / sub_w).min(cw - 1);

        let luma_raw = yuv.y[y_row * yuv.width + x_col] as i64;

        if color.matrix == MatrixCoefficients::Identity {
            let y_scaled = (k_y * (luma_raw - y_black) + 5000) / 10000;
            (y_scaled, y_scaled, y_scaled)
        } else {
            let cb_raw = yuv.cb[c_row * cw + c_col] as i64;
            let cr_raw = yuv.cr[c_row * cw + c_col] as i64;
            let y = k_y * (luma_raw - y_black);
            let cb = k_uv * (cb_raw - neutral);
            let cr = k_uv * (cr_raw - neutral);
            let r = (y + cr_to_r * cr / 10000 + 5000) / 10000;
            let g = (y + cb_to_g * cb / 10000 + cr_to_g * cr / 10000 + 5000) / 10000;
            let b = (y + cb_to_b * cb / 10000 + 5000) / 10000;
            (r, g, b)
        }
    };

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
