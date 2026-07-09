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

#[inline(always)]
fn sao_max_value_neon(bd: u8) -> Option<i32> {
    1u32.checked_shl(bd as u32)
        .map(|v| v.saturating_sub(1) as i32)
}

#[inline(always)]
fn dst_row_base_neon(y: usize, w: usize, band_y0: usize) -> Option<usize> {
    y.checked_sub(band_y0).and_then(|yy| yy.checked_mul(w))
}

#[inline(always)]
fn src_row_base_neon(y: usize, w: usize) -> Option<usize> {
    y.checked_mul(w)
}

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
    dst: &[u16; 8],
    src: &[u16; 8],
    offsets: &[i32; 4],
    band_pos: int32x4_t,
    shift: u8,
    zero: int32x4_t,
    max: int32x4_t,
) -> uint16x8_t {
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

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x8(dst: &mut [u16; 8], v: uint16x8_t) {
    unsafe { vst1q_u16(dst.as_mut_ptr(), v) };
}

#[inline]
#[target_feature(enable = "neon")]
fn band_offset_tail8_inplace_neon(
    dst: &mut [u16],
    offsets: &[i32; 4],
    band_pos: int32x4_t,
    shift: u8,
    zero: int32x4_t,
    max: int32x4_t,
) {
    debug_assert!(dst.len() <= 8);
    if dst.is_empty() {
        return;
    }

    let len = dst.len();
    let mut tmp = [0u16; 8];
    tmp[..len].copy_from_slice(dst);
    let out = band_offset8_neon(&tmp, &tmp, offsets, band_pos, shift, zero, max);
    store_u16x8(&mut tmp, out);
    dst.copy_from_slice(&tmp[..len]);
}

#[inline]
#[target_feature(enable = "neon")]
fn edge_offset4_neon(
    samples: int32x4_t,
    n1: int32x4_t,
    n2: int32x4_t,
    valid: uint32x4_t,
    offsets: &[i32; 4],
    zero: int32x4_t,
    max: int32x4_t,
) -> (int32x4_t, uint32x4_t) {
    let gt1 = vcgtq_s32(samples, n1);
    let lt1 = vcgtq_s32(n1, samples);
    let eq1 = vceqq_s32(samples, n1);
    let gt2 = vcgtq_s32(samples, n2);
    let lt2 = vcgtq_s32(n2, samples);
    let eq2 = vceqq_s32(samples, n2);

    let m0 = vandq_u32(lt1, lt2);
    let m1 = vorrq_u32(vandq_u32(lt1, eq2), vandq_u32(lt2, eq1));
    let m3 = vorrq_u32(vandq_u32(gt1, eq2), vandq_u32(gt2, eq1));
    let m4 = vandq_u32(gt1, gt2);

    let mut off = zero;
    off = vbslq_s32(m0, vdupq_n_s32(offsets[0]), off);
    off = vbslq_s32(m1, vdupq_n_s32(offsets[1]), off);
    off = vbslq_s32(m3, vdupq_n_s32(offsets[2]), off);
    off = vbslq_s32(m4, vdupq_n_s32(offsets[3]), off);

    let mut active = vdupq_n_u32(0);
    if offsets[0] != 0 {
        active = vorrq_u32(active, m0);
    }
    if offsets[1] != 0 {
        active = vorrq_u32(active, m1);
    }
    if offsets[2] != 0 {
        active = vorrq_u32(active, m3);
    }
    if offsets[3] != 0 {
        active = vorrq_u32(active, m4);
    }
    let v = vaddq_s32(samples, off);
    (vminq_s32(vmaxq_s32(v, zero), max), vandq_u32(active, valid))
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn edge_offset8_masked_neon(
    dst: &[u16; 8],
    samples: &[u16; 8],
    n1: &[u16; 8],
    n2: &[u16; 8],
    valid: &[u16; 8],
    offsets: &[i32; 4],
    zero: int32x4_t,
    max: int32x4_t,
) -> uint16x8_t {
    let old = unsafe { vld1q_u16(dst.as_ptr()) };
    let s = unsafe { vld1q_u16(samples.as_ptr()) };
    let a = unsafe { vld1q_u16(n1.as_ptr()) };
    let b = unsafe { vld1q_u16(n2.as_ptr()) };
    let v = unsafe { vld1q_u16(valid.as_ptr()) };

    let s_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s)));
    let s_hi = vreinterpretq_s32_u32(vmovl_high_u16(s));
    let a_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(a)));
    let a_hi = vreinterpretq_s32_u32(vmovl_high_u16(a));
    let b_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(b)));
    let b_hi = vreinterpretq_s32_u32(vmovl_high_u16(b));
    let v_lo = vmovl_u16(vget_low_u16(v));
    let v_hi = vmovl_high_u16(v);
    let valid_lo = vcgtq_u32(v_lo, vdupq_n_u32(0));
    let valid_hi = vcgtq_u32(v_hi, vdupq_n_u32(0));

    let (lo, mlo) = edge_offset4_neon(s_lo, a_lo, b_lo, valid_lo, offsets, zero, max);
    let (hi, mhi) = edge_offset4_neon(s_hi, a_hi, b_hi, valid_hi, offsets, zero, max);
    let out = vcombine_u16(
        vqmovn_u32(vreinterpretq_u32_s32(lo)),
        vqmovn_u32(vreinterpretq_u32_s32(hi)),
    );
    let mask = vcombine_u16(vmovn_u32(mlo), vmovn_u32(mhi));
    vbslq_u16(mask, out, old)
}

#[inline]
#[target_feature(enable = "neon")]
fn edge_offset8_neon(
    dst: &[u16; 8],
    samples: &[u16; 8],
    n1: &[u16; 8],
    n2: &[u16; 8],
    offsets: &[i32; 4],
    zero: int32x4_t,
    max: int32x4_t,
) -> uint16x8_t {
    let valid = [1u16; 8];
    edge_offset8_masked_neon(dst, samples, n1, n2, &valid, offsets, zero, max)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn edge_offset_tail8_neon(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    y: usize,
    x0: usize,
    dx: i32,
    dy: i32,
    offsets: &[i32; 4],
    zero: int32x4_t,
    max: int32x4_t,
) {
    debug_assert!(dst.len() <= 8);
    if dst.is_empty() {
        return;
    }

    let mut d = [0u16; 8];
    let mut s = [0u16; 8];
    let mut n1 = [0u16; 8];
    let mut n2 = [0u16; 8];
    let mut valid = [0u16; 8];
    let Some(src_base) = src_row_base_neon(y, w) else {
        return;
    };

    for lane in 0..dst.len() {
        let x = x0 + lane;
        let sample = src.get(src_base + x).copied().unwrap_or(0);
        let x1 = x as i32 + dx;
        let y1 = y as i32 + dy;
        let x2 = x as i32 - dx;
        let y2 = y as i32 - dy;
        let ok1 = x1 >= 0 && y1 >= 0 && (x1 as usize) < w && (y1 as usize) < h;
        let ok2 = x2 >= 0 && y2 >= 0 && (x2 as usize) < w && (y2 as usize) < h;
        d[lane] = dst[lane];
        s[lane] = sample;
        if ok1 && ok2 {
            n1[lane] = src[y1 as usize * w + x1 as usize];
            n2[lane] = src[y2 as usize * w + x2 as usize];
            valid[lane] = 1;
        } else {
            n1[lane] = sample;
            n2[lane] = sample;
        }
    }

    let out = edge_offset8_masked_neon(&d, &s, &n1, &n2, &valid, offsets, zero, max);
    let mut tmp = [0u16; 8];
    store_u16x8(&mut tmp, out);
    let len = dst.len();
    dst.copy_from_slice(&tmp[..len]);
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn edge_offset_segment_tail_neon(
    dst_plane: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    band_y0: usize,
    y: usize,
    x0: usize,
    x_end: usize,
    dx: i32,
    dy: i32,
    offsets: &[i32; 4],
    zero: int32x4_t,
    max: int32x4_t,
) {
    if x_end <= x0 {
        return;
    }
    let Some(dst_base) = dst_row_base_neon(y, w, band_y0) else {
        return;
    };
    let mut x = x0;
    while x < x_end {
        let len = (x_end - x).min(8);
        let Some(dst_tail) = dst_plane.get_mut(dst_base + x..dst_base + x + len) else {
            return;
        };
        edge_offset_tail8_neon(dst_tail, src, w, h, y, x, dx, dy, offsets, zero, max);
        x += len;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn apply_sao_edge_offset_horizontal_neon_impl(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    bd: u8,
) {
    apply_sao_edge_offset_horizontal_banded_neon_impl(
        dst, src, w, h, 0, x0, y0, x_end, y_end, offsets, bd,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn apply_sao_edge_offset_vertical_neon_impl(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    bd: u8,
) {
    apply_sao_edge_offset_vertical_banded_neon_impl(
        dst, src, w, h, 0, x0, y0, x_end, y_end, offsets, bd,
    );
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
        let row_range = row + x0..row + x_end;
        let (Some(src_row), Some(dst_row)) = (src.get(row_range.clone()), dst.get_mut(row_range))
        else {
            continue;
        };
        let (src8, src_tail) = src_row.as_chunks::<8>();
        let (dst8, dst_tail) = dst_row.as_chunks_mut::<8>();

        for (src, dst) in src8.iter().zip(dst8.iter_mut()) {
            let out = band_offset8_neon(dst, src, offsets, band_pos_v, shift, zero, max);
            store_u16x8(dst, out);
        }

        for (s, dst) in src_tail.iter().copied().zip(dst_tail.iter_mut()) {
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
#[inline]
#[target_feature(enable = "neon")]
fn apply_sao_band_offset_inplace_neon_impl(
    dst_plane: &mut [u16],
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
        let dst_range = dst_base + x0..dst_base + x_end;
        let Some(dst_row) = dst_plane.get_mut(dst_range) else {
            continue;
        };

        let (dst8, dst_tail) = dst_row.as_chunks_mut::<8>();
        for dst in dst8.iter_mut() {
            let out = band_offset8_neon(&*dst, &*dst, offsets, band_pos_v, shift, zero, max);
            store_u16x8(dst, out);
        }
        band_offset_tail8_inplace_neon(dst_tail, offsets, band_pos_v, shift, zero, max);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_band_offset_inplace_neon(
    dst: &mut [u16],
    w: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    band_pos: u8,
    bd: u8,
) {
    unsafe {
        apply_sao_band_offset_inplace_neon_impl(
            dst, w, 0, x0, y0, x_end, y_end, offsets, band_pos, bd,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_band_offset_banded_inplace_neon(
    dst_band: &mut [u16],
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
    unsafe {
        apply_sao_band_offset_inplace_neon_impl(
            dst_band, w, band_y0, x0, y0, x_end, y_end, offsets, band_pos, bd,
        )
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
    if x_end <= x0 || y_end <= y0 {
        return;
    }

    unsafe {
        match type_idx {
            1 => apply_sao_band_offset_neon_impl(
                dst, src, w, x0, y0, x_end, y_end, offsets, band_pos, bd,
            ),
            2 => match eo_class {
                0 => apply_sao_edge_offset_horizontal_neon_impl(
                    dst, src, w, h, x0, y0, x_end, y_end, offsets, bd,
                ),
                1 => apply_sao_edge_offset_vertical_neon_impl(
                    dst, src, w, h, x0, y0, x_end, y_end, offsets, bd,
                ),
                _ => apply_sao_edge_offset_diagonal_neon_impl(
                    dst, src, w, h, 0, x0, y0, x_end, y_end, offsets, eo_class, bd,
                ),
            },
            _ => apply_sao_plane_scalar(
                dst, src, w, h, x0, y0, x_end, y_end, type_idx, offsets, band_pos, eo_class, bd,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn apply_sao_edge_offset_horizontal_banded_neon_impl(
    dst_band: &mut [u16],
    src_full: &[u16],
    w: usize,
    h: usize,
    band_y0: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    bd: u8,
) {
    if w == 0 || x_end <= x0 || y_end <= y0 || y_end <= band_y0 {
        return;
    }
    let Some(max_val) = sao_max_value_neon(bd) else {
        return;
    };
    let max = vdupq_n_s32(max_val);
    let zero = vdupq_n_s32(0);
    let vec_x0 = x0.max(1);
    let vec_x1 = x_end.min(w.saturating_sub(1));

    if vec_x0 >= vec_x1 {
        for y in y0..y_end {
            edge_offset_segment_tail_neon(
                dst_band, src_full, w, h, band_y0, y, x0, x_end, 1, 0, offsets, zero, max,
            );
        }
        return;
    }

    for y in y0..y_end {
        if x0 < vec_x0 {
            edge_offset_segment_tail_neon(
                dst_band, src_full, w, h, band_y0, y, x0, vec_x0, 1, 0, offsets, zero, max,
            );
        }
        if vec_x0 < vec_x1 {
            let Some(src_base) = src_row_base_neon(y, w) else {
                continue;
            };
            let Some(dst_base) = dst_row_base_neon(y, w, band_y0) else {
                continue;
            };
            let mid_range = src_base + vec_x0..src_base + vec_x1;
            let left_range = src_base + vec_x0 - 1..src_base + vec_x1 - 1;
            let right_range = src_base + vec_x0 + 1..src_base + vec_x1 + 1;
            let dst_range = dst_base + vec_x0..dst_base + vec_x1;
            let (Some(src_mid), Some(src_left), Some(src_right), Some(dst_mid)) = (
                src_full.get(mid_range),
                src_full.get(left_range),
                src_full.get(right_range),
                dst_band.get_mut(dst_range),
            ) else {
                continue;
            };

            let (mid8, mid_tail) = src_mid.as_chunks::<8>();
            let (left8, _) = src_left.as_chunks::<8>();
            let (right8, _) = src_right.as_chunks::<8>();
            let (dst8, dst_tail) = dst_mid.as_chunks_mut::<8>();
            for (((s, l), r), d) in mid8
                .iter()
                .zip(left8.iter())
                .zip(right8.iter())
                .zip(dst8.iter_mut())
            {
                let out = edge_offset8_neon(d, s, r, l, offsets, zero, max);
                store_u16x8(d, out);
            }
            if !mid_tail.is_empty() {
                let tail_x0 = vec_x0 + mid8.len() * 8;
                edge_offset_tail8_neon(
                    dst_tail, src_full, w, h, y, tail_x0, 1, 0, offsets, zero, max,
                );
            }
        }
        if vec_x1 < x_end {
            edge_offset_segment_tail_neon(
                dst_band, src_full, w, h, band_y0, y, vec_x1, x_end, 1, 0, offsets, zero, max,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn apply_sao_edge_offset_vertical_banded_neon_impl(
    dst_band: &mut [u16],
    src_full: &[u16],
    w: usize,
    h: usize,
    band_y0: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    bd: u8,
) {
    if w == 0 || x_end <= x0 || y_end <= y0 || y_end <= band_y0 {
        return;
    }
    let Some(max_val) = sao_max_value_neon(bd) else {
        return;
    };
    let max = vdupq_n_s32(max_val);
    let zero = vdupq_n_s32(0);
    let vec_y0 = y0.max(1);
    let vec_y1 = y_end.min(h.saturating_sub(1));

    if vec_y0 >= vec_y1 {
        for y in y0..y_end {
            edge_offset_segment_tail_neon(
                dst_band, src_full, w, h, band_y0, y, x0, x_end, 0, 1, offsets, zero, max,
            );
        }
        return;
    }

    for y in y0..vec_y0.min(y_end) {
        edge_offset_segment_tail_neon(
            dst_band, src_full, w, h, band_y0, y, x0, x_end, 0, 1, offsets, zero, max,
        );
    }

    for y in vec_y0..vec_y1 {
        let above = (y - 1) * w;
        let row = y * w;
        let below = (y + 1) * w;
        let Some(dst_base) = dst_row_base_neon(y, w, band_y0) else {
            continue;
        };
        let row_range = row + x0..row + x_end;
        let above_range = above + x0..above + x_end;
        let below_range = below + x0..below + x_end;
        let dst_range = dst_base + x0..dst_base + x_end;
        let (Some(src_row), Some(src_above), Some(src_below), Some(dst_row)) = (
            src_full.get(row_range),
            src_full.get(above_range),
            src_full.get(below_range),
            dst_band.get_mut(dst_range),
        ) else {
            continue;
        };

        let (src8, src_tail) = src_row.as_chunks::<8>();
        let (above8, _) = src_above.as_chunks::<8>();
        let (below8, _) = src_below.as_chunks::<8>();
        let (dst8, dst_tail) = dst_row.as_chunks_mut::<8>();
        for (((s, a), b), d) in src8
            .iter()
            .zip(above8.iter())
            .zip(below8.iter())
            .zip(dst8.iter_mut())
        {
            let out = edge_offset8_neon(d, s, b, a, offsets, zero, max);
            store_u16x8(d, out);
        }
        if !src_tail.is_empty() {
            let tail_x0 = x0 + src8.len() * 8;
            edge_offset_tail8_neon(
                dst_tail, src_full, w, h, y, tail_x0, 0, 1, offsets, zero, max,
            );
        }
    }

    for y in vec_y1.max(y0)..y_end {
        edge_offset_segment_tail_neon(
            dst_band, src_full, w, h, band_y0, y, x0, x_end, 0, 1, offsets, zero, max,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn apply_sao_edge_offset_diagonal_neon_impl(
    dst_band: &mut [u16],
    src_full: &[u16],
    w: usize,
    h: usize,
    band_y0: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    eo_class: u8,
    bd: u8,
) {
    if w == 0 || x_end <= x0 || y_end <= y0 || y_end <= band_y0 {
        return;
    }
    let Some(max_val) = sao_max_value_neon(bd) else {
        return;
    };
    let max = vdupq_n_s32(max_val);
    let zero = vdupq_n_s32(0);
    let dy = if eo_class == 2 { 1 } else { -1 };
    let vec_y0 = y0.max(1);
    let vec_y1 = y_end.min(h.saturating_sub(1));
    let vec_x0 = x0.max(1);
    let vec_x1 = x_end.min(w.saturating_sub(1));

    if vec_y0 >= vec_y1 {
        for y in y0..y_end {
            edge_offset_segment_tail_neon(
                dst_band, src_full, w, h, band_y0, y, x0, x_end, 1, dy, offsets, zero, max,
            );
        }
        return;
    }

    for y in y0..vec_y0.min(y_end) {
        edge_offset_segment_tail_neon(
            dst_band, src_full, w, h, band_y0, y, x0, x_end, 1, dy, offsets, zero, max,
        );
    }

    for y in vec_y0..vec_y1 {
        if vec_x0 >= vec_x1 {
            edge_offset_segment_tail_neon(
                dst_band, src_full, w, h, band_y0, y, x0, x_end, 1, dy, offsets, zero, max,
            );
            continue;
        }
        if x0 < vec_x0 {
            edge_offset_segment_tail_neon(
                dst_band, src_full, w, h, band_y0, y, x0, vec_x0, 1, dy, offsets, zero, max,
            );
        }
        if vec_x0 < vec_x1 {
            let n1_y = (y as i32 + dy) as usize;
            let n2_y = (y as i32 - dy) as usize;
            let row = y * w;
            let n1_base = n1_y * w;
            let n2_base = n2_y * w;
            let Some(dst_base) = dst_row_base_neon(y, w, band_y0) else {
                continue;
            };
            let row_range = row + vec_x0..row + vec_x1;
            let n1_range = n1_base + vec_x0 + 1..n1_base + vec_x1 + 1;
            let n2_range = n2_base + vec_x0 - 1..n2_base + vec_x1 - 1;
            let dst_range = dst_base + vec_x0..dst_base + vec_x1;
            let (Some(src_row), Some(src_n1), Some(src_n2), Some(dst_row)) = (
                src_full.get(row_range),
                src_full.get(n1_range),
                src_full.get(n2_range),
                dst_band.get_mut(dst_range),
            ) else {
                continue;
            };

            let (src8, src_tail) = src_row.as_chunks::<8>();
            let (n1_8, _) = src_n1.as_chunks::<8>();
            let (n2_8, _) = src_n2.as_chunks::<8>();
            let (dst8, dst_tail) = dst_row.as_chunks_mut::<8>();
            for (((s, a), b), d) in src8
                .iter()
                .zip(n1_8.iter())
                .zip(n2_8.iter())
                .zip(dst8.iter_mut())
            {
                let out = edge_offset8_neon(d, s, a, b, offsets, zero, max);
                store_u16x8(d, out);
            }
            if !src_tail.is_empty() {
                let tail_x0 = vec_x0 + src8.len() * 8;
                edge_offset_tail8_neon(
                    dst_tail, src_full, w, h, y, tail_x0, 1, dy, offsets, zero, max,
                );
            }
        }
        if vec_x1 < x_end {
            edge_offset_segment_tail_neon(
                dst_band, src_full, w, h, band_y0, y, vec_x1, x_end, 1, dy, offsets, zero, max,
            );
        }
    }

    for y in vec_y1.max(y0)..y_end {
        edge_offset_segment_tail_neon(
            dst_band, src_full, w, h, band_y0, y, x0, x_end, 1, dy, offsets, zero, max,
        );
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

        let (src8, src_tail) = src_row.as_chunks::<8>();
        let (dst8, dst_tail) = dst_row.as_chunks_mut::<8>();

        for (src, dst) in src8.iter().zip(dst8.iter_mut()) {
            let out = band_offset8_neon(dst, src, offsets, band_pos_v, shift, zero, max);
            store_u16x8(dst, out);
        }

        for (s, dst) in src_tail.iter().copied().zip(dst_tail.iter_mut()) {
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
    if x_end <= x0 || y_end <= y0 {
        return;
    }

    unsafe {
        match type_idx {
            1 => apply_sao_band_offset_banded_neon_impl(
                dst_band, src_full, w, band_y0, x0, y0, x_end, y_end, offsets, band_pos, bd,
            ),
            2 => match eo_class {
                0 => apply_sao_edge_offset_horizontal_banded_neon_impl(
                    dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, offsets, bd,
                ),
                1 => apply_sao_edge_offset_vertical_banded_neon_impl(
                    dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, offsets, bd,
                ),
                _ => apply_sao_edge_offset_diagonal_neon_impl(
                    dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, offsets, eo_class, bd,
                ),
            },
            _ => apply_sao_plane_banded_scalar(
                dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, type_idx, offsets,
                band_pos, eo_class, bd,
            ),
        }
    }
}
