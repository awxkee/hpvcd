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

use crate::mc::{
    bi_mc_scalar, bi_mc_weighted_scalar, can_motion_comp, sample_max, uni_mc_scalar,
    uni_mc_weighted_scalar,
};

#[inline]
fn interp_data_ok(r: &crate::mc::RefPlane<'_>) -> bool {
    r.stride >= r.width && r.data.len() >= r.stride.saturating_mul(r.height)
}

#[inline]
fn luma_simd_bounds(
    r: &crate::mc::RefPlane<'_>,
    x0: isize,
    y0: isize,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
) -> bool {
    let left: isize = if fx == 0 { 0 } else { 3 };
    let top: isize = if fy == 0 { 0 } else { 3 };
    let right: usize = if fx == 0 { 0 } else { 4 };
    let bottom: usize = if fy == 0 { 0 } else { 4 };
    interp_data_ok(r) && crate::mc::interp_in_bounds(r, x0, y0, left, top, right, bottom, w, h)
}

#[inline]
fn chroma_simd_bounds(
    r: &crate::mc::RefPlane<'_>,
    x0: isize,
    y0: isize,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
) -> bool {
    let left: isize = if fx == 0 { 0 } else { 1 };
    let top: isize = if fy == 0 { 0 } else { 1 };
    let right: usize = if fx == 0 { 0 } else { 2 };
    let bottom: usize = if fy == 0 { 0 } else { 2 };
    interp_data_ok(r) && crate::mc::interp_in_bounds(r, x0, y0, left, top, right, bottom, w, h)
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x8_i32_pair(src: &[u16]) -> (int32x4_t, int32x4_t) {
    debug_assert!(src.len() >= 8);
    let v = unsafe { vld1q_u16(src.as_ptr()) };
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x4(src: &[i32]) -> int32x4_t {
    debug_assert!(src.len() >= 4);
    unsafe { vld1q_s32(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4(dst: &mut [i32], v: int32x4_t) {
    debug_assert!(dst.len() >= 4);
    unsafe { vst1q_s32(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x8_interp(dst: &mut [i16], lo: int32x4_t, hi: int32x4_t) {
    debug_assert!(dst.len() >= 8);
    unsafe {
        vst1q_s16(
            dst.as_mut_ptr(),
            vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)),
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn conv_u16x8_i32_8tap(src: &[u16], f: &[i32; 8]) -> (int32x4_t, int32x4_t) {
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    for (i, &c) in f.iter().enumerate() {
        let (slo, shi) = load_u16x8_i32_pair(&src[i..]);
        let c = vdupq_n_s32(c);
        lo = vmlaq_s32(lo, slo, c);
        hi = vmlaq_s32(hi, shi, c);
    }
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn conv_u16x8_i32_4tap(src: &[u16], f: &[i32; 4]) -> (int32x4_t, int32x4_t) {
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    for (i, &c) in f.iter().enumerate() {
        let (slo, shi) = load_u16x8_i32_pair(&src[i..]);
        let c = vdupq_n_s32(c);
        lo = vmlaq_s32(lo, slo, c);
        hi = vmlaq_s32(hi, shi, c);
    }
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn copy_scaled_row_neon(src: &[u16], dst: &mut [i16], s1: i32) {
    let (src8, src_tail) = src.as_chunks::<8>();
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    for (src, dst) in src8.iter().zip(dst8.iter_mut()) {
        let (lo, hi) = load_u16x8_i32_pair(src);
        store_i16x8_interp(
            dst,
            shr_s32(vshlq_n_s32::<6>(lo), s1),
            shr_s32(vshlq_n_s32::<6>(hi), s1),
        );
    }
    for (&s, out) in src_tail.iter().zip(dst_tail.iter_mut()) {
        *out = (((s as i32) << 6) >> s1) as i16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_h_8tap_row_neon(src: &[u16], dst: &mut [i16], f: &[i32; 8], s1: i32) {
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    for (chunk_idx, dst) in dst8.iter_mut().enumerate() {
        let x = chunk_idx * 8;
        let (lo, hi) = conv_u16x8_i32_8tap(&src[x..], f);
        store_i16x8_interp(dst, shr_s32(lo, s1), shr_s32(hi, s1));
    }
    let x0 = dst8.len() * 8;
    for (x, out) in dst_tail.iter_mut().enumerate() {
        let x = x0 + x;
        let acc = f
            .iter()
            .enumerate()
            .fold(0i32, |acc, (t, &c)| acc + c * src[x + t] as i32);
        *out = (acc >> s1) as i16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_h_4tap_row_neon(src: &[u16], dst: &mut [i16], f: &[i32; 4], s1: i32) {
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    for (chunk_idx, dst) in dst8.iter_mut().enumerate() {
        let x = chunk_idx * 8;
        let (lo, hi) = conv_u16x8_i32_4tap(&src[x..], f);
        store_i16x8_interp(dst, shr_s32(lo, s1), shr_s32(hi, s1));
    }
    let x0 = dst8.len() * 8;
    for (x, out) in dst_tail.iter_mut().enumerate() {
        let x = x0 + x;
        let acc = f
            .iter()
            .enumerate()
            .fold(0i32, |acc, (t, &c)| acc + c * src[x + t] as i32);
        *out = (acc >> s1) as i16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_h_8tap_tmp_row_neon(src: &[u16], dst: &mut [i32], f: &[i32; 8], s1: i32) {
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    for (chunk_idx, dst) in dst8.iter_mut().enumerate() {
        let x = chunk_idx * 8;
        let (lo, hi) = conv_u16x8_i32_8tap(&src[x..], f);
        store_i32x4(&mut dst[..4], shr_s32(lo, s1));
        store_i32x4(&mut dst[4..], shr_s32(hi, s1));
    }
    let x0 = dst8.len() * 8;
    for (x, out) in dst_tail.iter_mut().enumerate() {
        let x = x0 + x;
        let acc = f
            .iter()
            .enumerate()
            .fold(0i32, |acc, (t, &c)| acc + c * src[x + t] as i32);
        *out = acc >> s1;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_h_4tap_tmp_row_neon(src: &[u16], dst: &mut [i32], f: &[i32; 4], s1: i32) {
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    for (chunk_idx, dst) in dst8.iter_mut().enumerate() {
        let x = chunk_idx * 8;
        let (lo, hi) = conv_u16x8_i32_4tap(&src[x..], f);
        store_i32x4(&mut dst[..4], shr_s32(lo, s1));
        store_i32x4(&mut dst[4..], shr_s32(hi, s1));
    }
    let x0 = dst8.len() * 8;
    for (x, out) in dst_tail.iter_mut().enumerate() {
        let x = x0 + x;
        let acc = f
            .iter()
            .enumerate()
            .fold(0i32, |acc, (t, &c)| acc + c * src[x + t] as i32);
        *out = acc >> s1;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_v_8tap_row_neon(rows: [&[u16]; 8], dst: &mut [i16], f: &[i32; 8], s1: i32) {
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    for (chunk_idx, dst) in dst8.iter_mut().enumerate() {
        let x = chunk_idx * 8;
        let mut lo = vdupq_n_s32(0);
        let mut hi = vdupq_n_s32(0);
        for (row, &c) in rows.iter().zip(f.iter()) {
            let (slo, shi) = load_u16x8_i32_pair(&row[x..]);
            let c = vdupq_n_s32(c);
            lo = vmlaq_s32(lo, slo, c);
            hi = vmlaq_s32(hi, shi, c);
        }
        store_i16x8_interp(dst, shr_s32(lo, s1), shr_s32(hi, s1));
    }
    let x0 = dst8.len() * 8;
    for (x, out) in dst_tail.iter_mut().enumerate() {
        let x = x0 + x;
        let acc = rows
            .iter()
            .zip(f.iter())
            .fold(0i32, |acc, (row, &c)| acc + c * row[x] as i32);
        *out = (acc >> s1) as i16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_v_4tap_row_neon(rows: [&[u16]; 4], dst: &mut [i16], f: &[i32; 4], s1: i32) {
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    for (chunk_idx, dst) in dst8.iter_mut().enumerate() {
        let x = chunk_idx * 8;
        let mut lo = vdupq_n_s32(0);
        let mut hi = vdupq_n_s32(0);
        for (row, &c) in rows.iter().zip(f.iter()) {
            let (slo, shi) = load_u16x8_i32_pair(&row[x..]);
            let c = vdupq_n_s32(c);
            lo = vmlaq_s32(lo, slo, c);
            hi = vmlaq_s32(hi, shi, c);
        }
        store_i16x8_interp(dst, shr_s32(lo, s1), shr_s32(hi, s1));
    }
    let x0 = dst8.len() * 8;
    for (x, out) in dst_tail.iter_mut().enumerate() {
        let x = x0 + x;
        let acc = rows
            .iter()
            .zip(f.iter())
            .fold(0i32, |acc, (row, &c)| acc + c * row[x] as i32);
        *out = (acc >> s1) as i16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_tmp_v_8tap_row_neon(rows: [&[i32]; 8], dst: &mut [i16], f: &[i32; 8]) {
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    for (chunk_idx, dst) in dst8.iter_mut().enumerate() {
        let x = chunk_idx * 8;
        let mut lo = vdupq_n_s32(0);
        let mut hi = vdupq_n_s32(0);
        for (row, &c) in rows.iter().zip(f.iter()) {
            let c = vdupq_n_s32(c);
            lo = vmlaq_s32(lo, load_i32x4(&row[x..]), c);
            hi = vmlaq_s32(hi, load_i32x4(&row[x + 4..]), c);
        }
        store_i16x8_interp(dst, shr_s32(lo, 6), shr_s32(hi, 6));
    }
    let x0 = dst8.len() * 8;
    for (x, out) in dst_tail.iter_mut().enumerate() {
        let x = x0 + x;
        let acc = rows
            .iter()
            .zip(f.iter())
            .fold(0i32, |acc, (row, &c)| acc + c * row[x]);
        *out = (acc >> 6) as i16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_tmp_v_4tap_row_neon(rows: [&[i32]; 4], dst: &mut [i16], f: &[i32; 4]) {
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    for (chunk_idx, dst) in dst8.iter_mut().enumerate() {
        let x = chunk_idx * 8;
        let mut lo = vdupq_n_s32(0);
        let mut hi = vdupq_n_s32(0);
        for (row, &c) in rows.iter().zip(f.iter()) {
            let c = vdupq_n_s32(c);
            lo = vmlaq_s32(lo, load_i32x4(&row[x..]), c);
            hi = vmlaq_s32(hi, load_i32x4(&row[x + 4..]), c);
        }
        store_i16x8_interp(dst, shr_s32(lo, 6), shr_s32(hi, 6));
    }
    let x0 = dst8.len() * 8;
    for (x, out) in dst_tail.iter_mut().enumerate() {
        let x = x0 + x;
        let acc = rows
            .iter()
            .zip(f.iter())
            .fold(0i32, |acc, (row, &c)| acc + c * row[x]);
        *out = (acc >> 6) as i16;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn luma_interp_neon_impl(
    r: &crate::mc::RefPlane<'_>,
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
    let s1 = crate::mc::shift1(bd) as i32;
    let dst = &mut dst[..w * h];
    if fx == 0 && fy == 0 {
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let row_start = (y0 as usize + y) * r.stride + x0 as usize;
            copy_scaled_row_neon(&r.data[row_start..row_start + w], dst_row, s1);
        }
    } else if fy == 0 {
        let hf = &crate::mc::LUMA_FILTER[fx];
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let row_start = (y0 as usize + y) * r.stride + (x0 - 3) as usize;
            filter_h_8tap_row_neon(&r.data[row_start..row_start + w + 7], dst_row, hf, s1);
        }
    } else if fx == 0 {
        let vf = &crate::mc::LUMA_FILTER[fy];
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let sy = (y0 - 3 + y as isize) as usize;
            let rows = [
                &r.data[sy * r.stride + x0 as usize..sy * r.stride + x0 as usize + w],
                &r.data[(sy + 1) * r.stride + x0 as usize..(sy + 1) * r.stride + x0 as usize + w],
                &r.data[(sy + 2) * r.stride + x0 as usize..(sy + 2) * r.stride + x0 as usize + w],
                &r.data[(sy + 3) * r.stride + x0 as usize..(sy + 3) * r.stride + x0 as usize + w],
                &r.data[(sy + 4) * r.stride + x0 as usize..(sy + 4) * r.stride + x0 as usize + w],
                &r.data[(sy + 5) * r.stride + x0 as usize..(sy + 5) * r.stride + x0 as usize + w],
                &r.data[(sy + 6) * r.stride + x0 as usize..(sy + 6) * r.stride + x0 as usize + w],
                &r.data[(sy + 7) * r.stride + x0 as usize..(sy + 7) * r.stride + x0 as usize + w],
            ];
            filter_v_8tap_row_neon(rows, dst_row, vf, s1);
        }
    } else {
        let hf = &crate::mc::LUMA_FILTER[fx];
        let vf = &crate::mc::LUMA_FILTER[fy];
        let tmp_h = h + 7;
        tmp.clear();
        tmp.resize(w * tmp_h, 0);
        for (ty, tmp_row) in tmp[..w * tmp_h].chunks_exact_mut(w).enumerate() {
            let row_start = (y0 - 3 + ty as isize) as usize * r.stride + (x0 - 3) as usize;
            filter_h_8tap_tmp_row_neon(&r.data[row_start..row_start + w + 7], tmp_row, hf, s1);
        }
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let base = y * w;
            let rows = [
                &tmp[base..base + w],
                &tmp[base + w..base + 2 * w],
                &tmp[base + 2 * w..base + 3 * w],
                &tmp[base + 3 * w..base + 4 * w],
                &tmp[base + 4 * w..base + 5 * w],
                &tmp[base + 5 * w..base + 6 * w],
                &tmp[base + 6 * w..base + 7 * w],
                &tmp[base + 7 * w..base + 8 * w],
            ];
            filter_tmp_v_8tap_row_neon(rows, dst_row, vf);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn chroma_interp_neon_impl(
    r: &crate::mc::RefPlane<'_>,
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
    let s1 = crate::mc::shift1(bd) as i32;
    let dst = &mut dst[..w * h];
    if fx == 0 && fy == 0 {
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let row_start = (y0 as usize + y) * r.stride + x0 as usize;
            copy_scaled_row_neon(&r.data[row_start..row_start + w], dst_row, s1);
        }
    } else if fy == 0 {
        let hf = &crate::mc::CHROMA_FILTER[fx];
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let row_start = (y0 as usize + y) * r.stride + (x0 - 1) as usize;
            filter_h_4tap_row_neon(&r.data[row_start..row_start + w + 3], dst_row, hf, s1);
        }
    } else if fx == 0 {
        let vf = &crate::mc::CHROMA_FILTER[fy];
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let sy = (y0 - 1 + y as isize) as usize;
            let rows = [
                &r.data[sy * r.stride + x0 as usize..sy * r.stride + x0 as usize + w],
                &r.data[(sy + 1) * r.stride + x0 as usize..(sy + 1) * r.stride + x0 as usize + w],
                &r.data[(sy + 2) * r.stride + x0 as usize..(sy + 2) * r.stride + x0 as usize + w],
                &r.data[(sy + 3) * r.stride + x0 as usize..(sy + 3) * r.stride + x0 as usize + w],
            ];
            filter_v_4tap_row_neon(rows, dst_row, vf, s1);
        }
    } else {
        let hf = &crate::mc::CHROMA_FILTER[fx];
        let vf = &crate::mc::CHROMA_FILTER[fy];
        let tmp_h = h + 3;
        tmp.clear();
        tmp.resize(w * tmp_h, 0);
        for (ty, tmp_row) in tmp[..w * tmp_h].chunks_exact_mut(w).enumerate() {
            let row_start = (y0 - 1 + ty as isize) as usize * r.stride + (x0 - 1) as usize;
            filter_h_4tap_tmp_row_neon(&r.data[row_start..row_start + w + 3], tmp_row, hf, s1);
        }
        for (y, dst_row) in dst.chunks_exact_mut(w).enumerate() {
            let base = y * w;
            let rows = [
                &tmp[base..base + w],
                &tmp[base + w..base + 2 * w],
                &tmp[base + 2 * w..base + 3 * w],
                &tmp[base + 3 * w..base + 4 * w],
            ];
            filter_tmp_v_4tap_row_neon(rows, dst_row, vf);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_interp_neon(
    r: &crate::mc::RefPlane<'_>,
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
    if fx >= crate::mc::LUMA_FILTER.len()
        || fy >= crate::mc::LUMA_FILTER.len()
        || !luma_simd_bounds(r, x0, y0, fx, fy, w, h)
        || dst.len() < w.saturating_mul(h)
    {
        crate::mc::luma_interp_scalar_scratch(r, x0, y0, fx, fy, w, h, bd, dst, tmp);
        return;
    }
    unsafe { luma_interp_neon_impl(r, x0, y0, fx, fy, w, h, bd, dst, tmp) }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_interp_neon(
    r: &crate::mc::RefPlane<'_>,
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
    if fx >= crate::mc::CHROMA_FILTER.len()
        || fy >= crate::mc::CHROMA_FILTER.len()
        || !chroma_simd_bounds(r, x0, y0, fx, fy, w, h)
        || dst.len() < w.saturating_mul(h)
    {
        crate::mc::chroma_interp_scalar_scratch(r, x0, y0, fx, fy, w, h, bd, dst, tmp);
        return;
    }
    unsafe { chroma_interp_neon_impl(r, x0, y0, fx, fy, w, h, bd, dst, tmp) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x8(src: &[i16]) -> int16x8_t {
    debug_assert!(src.len() >= 8);
    unsafe { vld1q_s16(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x8(dst: &mut [u16], v: uint16x8_t) {
    debug_assert!(dst.len() >= 8);
    unsafe { vst1q_u16(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn shr_s32(v: int32x4_t, shift: i32) -> int32x4_t {
    debug_assert!((0..32).contains(&shift));
    if shift == 0 {
        v
    } else {
        vshlq_s32(v, vdupq_n_s32(-shift))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn clip_s32(v: int32x4_t, max: int32x4_t) -> int32x4_t {
    vminq_s32(vmaxq_s32(v, vdupq_n_s32(0)), max)
}

#[inline]
#[target_feature(enable = "neon")]
fn pack_u16x8(lo: int32x4_t, hi: int32x4_t) -> uint16x8_t {
    vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn uni_pair(src: &[i16], shift: i32, offset: int32x4_t, max: int32x4_t) -> uint16x8_t {
    let s = load_i16x8(src);
    let lo = vaddq_s32(vmovl_s16(vget_low_s16(s)), offset);
    let hi = vaddq_s32(vmovl_s16(vget_high_s16(s)), offset);
    pack_u16x8(
        clip_s32(shr_s32(lo, shift), max),
        clip_s32(shr_s32(hi, shift), max),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn bi_pair(a: &[i16], b: &[i16], shift: i32, offset: int32x4_t, max: int32x4_t) -> uint16x8_t {
    let a = load_i16x8(a);
    let b = load_i16x8(b);
    let lo = vaddq_s32(
        vaddq_s32(vmovl_s16(vget_low_s16(a)), vmovl_s16(vget_low_s16(b))),
        offset,
    );
    let hi = vaddq_s32(
        vaddq_s32(vmovl_s16(vget_high_s16(a)), vmovl_s16(vget_high_s16(b))),
        offset,
    );
    pack_u16x8(
        clip_s32(shr_s32(lo, shift), max),
        clip_s32(shr_s32(hi, shift), max),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn uni_weighted_pair(
    src: &[i16],
    weight: int32x4_t,
    round: int32x4_t,
    off: int32x4_t,
    shift: i32,
    max: int32x4_t,
) -> uint16x8_t {
    let s = load_i16x8(src);
    let lo = vaddq_s32(vmulq_s32(vmovl_s16(vget_low_s16(s)), weight), round);
    let hi = vaddq_s32(vmulq_s32(vmovl_s16(vget_high_s16(s)), weight), round);
    let lo = clip_s32(vaddq_s32(shr_s32(lo, shift), off), max);
    let hi = clip_s32(vaddq_s32(shr_s32(hi, shift), off), max);
    pack_u16x8(lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn bi_weighted_pair(
    s0: &[i16],
    s1: &[i16],
    w0: int32x4_t,
    w1: int32x4_t,
    rnd: int32x4_t,
    shift: i32,
    max: int32x4_t,
) -> uint16x8_t {
    let a = load_i16x8(s0);
    let b = load_i16x8(s1);
    let lo = vaddq_s32(
        vaddq_s32(
            vmulq_s32(vmovl_s16(vget_low_s16(a)), w0),
            vmulq_s32(vmovl_s16(vget_low_s16(b)), w1),
        ),
        rnd,
    );
    let hi = vaddq_s32(
        vaddq_s32(
            vmulq_s32(vmovl_s16(vget_high_s16(a)), w0),
            vmulq_s32(vmovl_s16(vget_high_s16(b)), w1),
        ),
        rnd,
    );
    pack_u16x8(
        clip_s32(shr_s32(lo, shift), max),
        clip_s32(shr_s32(hi, shift), max),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn uni_row_neon(
    src: &[i16],
    dst: &mut [u16],
    w: usize,
    shift: i32,
    offset: int32x4_t,
    max: int32x4_t,
) {
    let src = &src[..w];
    let dst = &mut dst[..w];
    let (src8, src_tail) = src.as_chunks::<8>();
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();

    for (src, dst) in src8.iter().zip(dst8.iter_mut()) {
        let v = uni_pair(src, shift, offset, max);
        store_u16x8(dst, v);
    }

    let offset = vgetq_lane_s32::<0>(offset);
    let max = vgetq_lane_s32::<0>(max);
    for (&s, out) in src_tail.iter().zip(dst_tail.iter_mut()) {
        let v = (s as i32 + offset) >> shift;
        *out = v.clamp(0, max) as u16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn bi_row_neon(
    s0: &[i16],
    s1: &[i16],
    dst: &mut [u16],
    w: usize,
    shift: i32,
    offset: int32x4_t,
    max: int32x4_t,
) {
    let s0 = &s0[..w];
    let s1 = &s1[..w];
    let dst = &mut dst[..w];
    let (s0_8, s0_tail) = s0.as_chunks::<8>();
    let (s1_8, s1_tail) = s1.as_chunks::<8>();
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();

    for ((s0, s1), dst) in s0_8.iter().zip(s1_8.iter()).zip(dst8.iter_mut()) {
        let v = bi_pair(s0, s1, shift, offset, max);
        store_u16x8(dst, v);
    }

    let off = vgetq_lane_s32::<0>(offset);
    let max = vgetq_lane_s32::<0>(max);
    for ((&a, &b), out) in s0_tail.iter().zip(s1_tail.iter()).zip(dst_tail.iter_mut()) {
        let v = (a as i32 + b as i32 + off) >> shift;
        *out = v.clamp(0, max) as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn uni_weighted_row_neon(
    src: &[i16],
    dst: &mut [u16],
    w: usize,
    weight: int32x4_t,
    round: int32x4_t,
    off: int32x4_t,
    shift: i32,
    max: int32x4_t,
) {
    let src = &src[..w];
    let dst = &mut dst[..w];
    let (src8, src_tail) = src.as_chunks::<8>();
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();

    for (src, dst) in src8.iter().zip(dst8.iter_mut()) {
        let v = uni_weighted_pair(src, weight, round, off, shift, max);
        store_u16x8(dst, v);
    }

    let weight = vgetq_lane_s32::<0>(weight);
    let round = vgetq_lane_s32::<0>(round);
    let off = vgetq_lane_s32::<0>(off);
    let max = vgetq_lane_s32::<0>(max);
    for (&s, out) in src_tail.iter().zip(dst_tail.iter_mut()) {
        let v = ((s as i32 * weight + round) >> shift) + off;
        *out = v.clamp(0, max) as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn bi_weighted_row_neon(
    s0: &[i16],
    s1: &[i16],
    dst: &mut [u16],
    w: usize,
    w0: int32x4_t,
    w1: int32x4_t,
    rnd: int32x4_t,
    shift: i32,
    max: int32x4_t,
) {
    let s0 = &s0[..w];
    let s1 = &s1[..w];
    let dst = &mut dst[..w];
    let (s0_8, s0_tail) = s0.as_chunks::<8>();
    let (s1_8, s1_tail) = s1.as_chunks::<8>();
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();

    for ((s0, s1), dst) in s0_8.iter().zip(s1_8.iter()).zip(dst8.iter_mut()) {
        let v = bi_weighted_pair(s0, s1, w0, w1, rnd, shift, max);
        store_u16x8(dst, v);
    }

    let w0 = vgetq_lane_s32::<0>(w0);
    let w1 = vgetq_lane_s32::<0>(w1);
    let rnd = vgetq_lane_s32::<0>(rnd);
    let max = vgetq_lane_s32::<0>(max);
    for ((&a, &b), out) in s0_tail.iter().zip(s1_tail.iter()).zip(dst_tail.iter_mut()) {
        let v = (a as i32 * w0 + b as i32 * w1 + rnd) >> shift;
        *out = v.clamp(0, max) as u16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn uni_mc_neon_impl(
    src: &[i16],
    pred_w: usize,
    pred_h: usize,
    valid_w: usize,
    valid_h: usize,
    bd: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let src = &src[..pred_w * pred_h];
    let shift = 14 - bd as i32;
    let offset = vdupq_n_s32(if shift > 0 { 1 << (shift - 1) } else { 0 });
    let max = vdupq_n_s32(sample_max(bd));
    for (src_row, dst_row) in src
        .chunks_exact(pred_w)
        .zip(dst.chunks_mut(dst_stride))
        .take(valid_h)
    {
        uni_row_neon(src_row, dst_row, valid_w, shift.max(0), offset, max);
    }
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn bi_mc_neon_impl(
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
    let len = pred_w * pred_h;
    let s0 = &s0[..len];
    let s1 = &s1[..len];
    let shift = 15 - bd as i32;
    let offset = vdupq_n_s32(1 << (shift - 1));
    let max = vdupq_n_s32(sample_max(bd));
    for ((s0_row, s1_row), dst_row) in s0
        .chunks_exact(pred_w)
        .zip(s1.chunks_exact(pred_w))
        .zip(dst.chunks_mut(dst_stride))
        .take(valid_h)
    {
        bi_row_neon(s0_row, s1_row, dst_row, valid_w, shift, offset, max);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn uni_mc_weighted_neon_impl(
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
    let src = &src[..pred_w * pred_h];
    let log2_wd = log2_denom as i32 + 14 - bd as i32;
    let round = if log2_wd >= 1 { 1 << (log2_wd - 1) } else { 0 };
    let off = offset;
    let weight = vdupq_n_s32(weight);
    let round = vdupq_n_s32(round);
    let off = vdupq_n_s32(off);
    let max = vdupq_n_s32(sample_max(bd));
    for (src_row, dst_row) in src
        .chunks_exact(pred_w)
        .zip(dst.chunks_mut(dst_stride))
        .take(valid_h)
    {
        uni_weighted_row_neon(src_row, dst_row, valid_w, weight, round, off, log2_wd, max);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn bi_mc_weighted_neon_impl(
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
    let len = pred_w * pred_h;
    let s0 = &s0[..len];
    let s1 = &s1[..len];
    let log2_wd = log2_denom as i32 + 14 - bd as i32;
    let rnd = (o0 as i64 + o1 as i64 + 1) << log2_wd;
    let w0 = vdupq_n_s32(w0);
    let w1 = vdupq_n_s32(w1);
    let rnd = vdupq_n_s32(rnd as i32);
    let max = vdupq_n_s32(sample_max(bd));
    let shift = log2_wd + 1;
    for ((s0_row, s1_row), dst_row) in s0
        .chunks_exact(pred_w)
        .zip(s1.chunks_exact(pred_w))
        .zip(dst.chunks_mut(dst_stride))
        .take(valid_h)
    {
        bi_weighted_row_neon(s0_row, s1_row, dst_row, valid_w, w0, w1, rnd, shift, max);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn uni_mc_neon(
    src: &[i16],
    pred_w: usize,
    pred_h: usize,
    valid_w: usize,
    valid_h: usize,
    bd: u8,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let shift = 14 - bd as i32;
    if !(0..32).contains(&shift)
        || !can_motion_comp(src.len(), pred_w, pred_h, valid_w, valid_h, dst, dst_stride)
    {
        uni_mc_scalar(src, pred_w, pred_h, valid_w, valid_h, bd, dst, dst_stride);
        return;
    }
    unsafe { uni_mc_neon_impl(src, pred_w, pred_h, valid_w, valid_h, bd, dst, dst_stride) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bi_mc_neon(
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
    let shift = 15 - bd as i32;
    let Some(len) = pred_w.checked_mul(pred_h) else {
        bi_mc_scalar(
            s0, s1, pred_w, pred_h, valid_w, valid_h, bd, dst, dst_stride,
        );
        return;
    };
    if !(1..32).contains(&shift)
        || !can_motion_comp(s0.len(), pred_w, pred_h, valid_w, valid_h, dst, dst_stride)
        || s1.len() < len
    {
        bi_mc_scalar(
            s0, s1, pred_w, pred_h, valid_w, valid_h, bd, dst, dst_stride,
        );
        return;
    }
    unsafe {
        bi_mc_neon_impl(
            s0, s1, pred_w, pred_h, valid_w, valid_h, bd, dst, dst_stride,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn uni_mc_weighted_neon(
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
    let log2_wd = log2_denom as i32 + 14 - bd as i32;
    if !(1..32).contains(&log2_wd)
        || bd < 8
        || weight.unsigned_abs() > 8192
        || !can_motion_comp(src.len(), pred_w, pred_h, valid_w, valid_h, dst, dst_stride)
    {
        uni_mc_weighted_scalar(
            src, pred_w, pred_h, valid_w, valid_h, bd, weight, offset, log2_denom, dst, dst_stride,
        );
        return;
    }
    unsafe {
        uni_mc_weighted_neon_impl(
            src, pred_w, pred_h, valid_w, valid_h, bd, weight, offset, log2_denom, dst, dst_stride,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn bi_mc_weighted_neon(
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
    let log2_wd = log2_denom as i32 + 14 - bd as i32;
    let rnd = if bd >= 8 && (0..31).contains(&log2_wd) {
        Some((o0 as i64 + o1 as i64 + 1) << log2_wd)
    } else {
        None
    };
    let Some(len) = pred_w.checked_mul(pred_h) else {
        bi_mc_weighted_scalar(
            s0, s1, pred_w, pred_h, valid_w, valid_h, bd, w0, o0, w1, o1, log2_denom, dst,
            dst_stride,
        );
        return;
    };
    if !(1..32).contains(&(log2_wd + 1))
        || w0.unsigned_abs() > 8192
        || w1.unsigned_abs() > 8192
        || rnd.is_none_or(|r| i32::try_from(r).is_err())
        || !can_motion_comp(s0.len(), pred_w, pred_h, valid_w, valid_h, dst, dst_stride)
        || s1.len() < len
    {
        bi_mc_weighted_scalar(
            s0, s1, pred_w, pred_h, valid_w, valid_h, bd, w0, o0, w1, o1, log2_denom, dst,
            dst_stride,
        );
        return;
    }
    unsafe {
        bi_mc_weighted_neon_impl(
            s0, s1, pred_w, pred_h, valid_w, valid_h, bd, w0, o0, w1, o1, log2_denom, dst,
            dst_stride,
        )
    }
}
