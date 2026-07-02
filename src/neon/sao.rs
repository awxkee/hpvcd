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

use crate::sao::{apply_sao_plane_banded_scalar, apply_sao_plane_scalar};

#[inline]
#[target_feature(enable = "neon")]
fn shr_u32_neon(v: uint32x4_t, shift: u8) -> uint32x4_t {
    let shift = shift.min(11) as i32;
    vshlq_u32(v, vdupq_n_s32(-shift))
}

#[inline]
#[target_feature(enable = "neon")]
fn band_offset4_neon(
    samples: int32x4_t,
    offsets: &[i32; 4],
    band_pos: int32x4_t,
    shift: u8,
    zero: int32x4_t,
    max: int32x4_t,
) -> (int32x4_t, uint32x4_t) {
    let band = vreinterpretq_s32_u32(shr_u32_neon(vreinterpretq_u32_s32(samples), shift));
    let rel = vsubq_s32(band, band_pos);
    let mut off = zero;

    let m0 = vceqq_s32(rel, vdupq_n_s32(0));
    off = vbslq_s32(m0, vdupq_n_s32(offsets[0]), off);
    let m1 = vceqq_s32(rel, vdupq_n_s32(1));
    off = vbslq_s32(m1, vdupq_n_s32(offsets[1]), off);
    let m2 = vceqq_s32(rel, vdupq_n_s32(2));
    off = vbslq_s32(m2, vdupq_n_s32(offsets[2]), off);
    let m3 = vceqq_s32(rel, vdupq_n_s32(3));
    off = vbslq_s32(m3, vdupq_n_s32(offsets[3]), off);

    let active = vorrq_u32(vorrq_u32(m0, m1), vorrq_u32(m2, m3));
    let v = vaddq_s32(samples, off);
    (vminq_s32(vmaxq_s32(v, zero), max), active)
}

#[inline]
#[target_feature(enable = "neon")]
fn band_offset8_neon(
    dst: &[u16],
    src: &[u16],
    offsets: &[i32; 4],
    band_pos: int32x4_t,
    shift: u8,
    zero: int32x4_t,
    max: int32x4_t,
) -> uint16x8_t {
    debug_assert!(dst.len() >= 8);
    debug_assert!(src.len() >= 8);
    let old = unsafe { vld1q_u16(dst.as_ptr()) };
    let s = unsafe { vld1q_u16(src.as_ptr()) };
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s)));
    let (lo, mlo) = band_offset4_neon(lo, offsets, band_pos, shift, zero, max);
    let (hi, mhi) = band_offset4_neon(hi, offsets, band_pos, shift, zero, max);
    let out = vcombine_u16(
        vqmovn_u32(vreinterpretq_u32_s32(lo)),
        vqmovn_u32(vreinterpretq_u32_s32(hi)),
    );
    let mask = vcombine_u16(vmovn_u32(mlo), vmovn_u32(mhi));
    vbslq_u16(mask, out, old)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn apply_sao_band_offset_neon_impl(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    band_pos: u8,
    bd: u8,
) {
    let max_val = ((1u32 << bd) - 1) as i32;
    let max = vdupq_n_s32(max_val);
    let zero = vdupq_n_s32(0);
    let band_pos_v = vdupq_n_s32(band_pos as i32);
    let shift = bd - 5;

    for y in y0..y_end {
        let row = y * w;
        let mut x = x0;

        while x + 8 <= x_end {
            let out = band_offset8_neon(
                &dst[row + x..],
                &src[row + x..],
                offsets,
                band_pos_v,
                shift,
                zero,
                max,
            );
            unsafe { vst1q_u16(dst[row + x..].as_mut_ptr(), out) };
            x += 8;
        }

        for x in x..x_end {
            let s = src[row + x] as i32;
            let band = (s >> shift) as u8;
            let rel = band.wrapping_sub(band_pos);
            if rel < 4 {
                dst[row + x] = (s + offsets[rel as usize]).clamp(0, max_val) as u16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_plane_neon(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    type_idx: u8,
    offsets: &[i32; 4],
    band_pos: u8,
    eo_class: u8,
    bd: u8,
) {
    if type_idx != 1 || x_end <= x0 || y_end <= y0 {
        apply_sao_plane_scalar(
            dst, src, w, h, x0, y0, x_end, y_end, type_idx, offsets, band_pos, eo_class, bd,
        );
        return;
    }

    unsafe {
        apply_sao_band_offset_neon_impl(dst, src, w, x0, y0, x_end, y_end, offsets, band_pos, bd)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn apply_sao_band_offset_banded_neon_impl(
    dst_band: &mut [u16],
    src_full: &[u16],
    w: usize,
    band_y0: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    band_pos: u8,
    bd: u8,
) {
    if w == 0 || x_end <= x0 || y_end <= y0 || y_end <= band_y0 {
        return;
    }
    let Some(max_val) = (1u32)
        .checked_shl(bd as u32)
        .map(|v| v.saturating_sub(1) as i32)
    else {
        return;
    };
    let max = vdupq_n_s32(max_val);
    let zero = vdupq_n_s32(0);
    let band_pos_v = vdupq_n_s32(band_pos as i32);
    let shift = bd.saturating_sub(5);

    for y in y0..y_end {
        let Some(dst_base) = y.checked_sub(band_y0).and_then(|v| v.checked_mul(w)) else {
            continue;
        };
        let Some(src_base) = y.checked_mul(w) else {
            continue;
        };
        let src_range = src_base + x0..src_base + x_end;
        let dst_range = dst_base + x0..dst_base + x_end;
        let (Some(src_row), Some(dst_row)) = (src_full.get(src_range), dst_band.get_mut(dst_range))
        else {
            continue;
        };

        let mut x = 0usize;
        while x + 8 <= src_row.len() {
            let out = band_offset8_neon(
                &dst_row[x..],
                &src_row[x..],
                offsets,
                band_pos_v,
                shift,
                zero,
                max,
            );
            unsafe { vst1q_u16(dst_row.as_mut_ptr().add(x), out) };
            x += 8;
        }

        for (s, dst) in src_row[x..].iter().copied().zip(dst_row[x..].iter_mut()) {
            let s = s as i32;
            let band = (s >> shift) as u8;
            let rel = band.wrapping_sub(band_pos);
            if rel < 4 {
                *dst = (s + offsets[rel as usize]).clamp(0, max_val) as u16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_plane_banded_neon(
    dst_band: &mut [u16],
    src_full: &[u16],
    w: usize,
    h: usize,
    band_y0: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    type_idx: u8,
    offsets: &[i32; 4],
    band_pos: u8,
    eo_class: u8,
    bd: u8,
) {
    if type_idx != 1 || x_end <= x0 || y_end <= y0 {
        apply_sao_plane_banded_scalar(
            dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, type_idx, offsets, band_pos,
            eo_class, bd,
        );
        return;
    }

    unsafe {
        apply_sao_band_offset_banded_neon_impl(
            dst_band, src_full, w, band_y0, x0, y0, x_end, y_end, offsets, band_pos, bd,
        )
    }
}
