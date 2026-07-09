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

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::sao::{apply_sao_plane_banded_scalar, apply_sao_plane_scalar};

#[inline]
#[target_feature(enable = "sse4.1")]
fn shr_epi32_sse41(v: __m128i, shift: u8) -> __m128i {
    let cnt = _mm_cvtsi32_si128(shift.min(11) as i32);
    _mm_srl_epi32(v, cnt)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn band_offset4_sse41(
    samples: __m128i,
    offsets: &[i32; 4],
    band_pos: __m128i,
    shift: u8,
    zero: __m128i,
    max: __m128i,
) -> (__m128i, __m128i) {
    let band = shr_epi32_sse41(samples, shift);
    let rel = _mm_sub_epi32(band, band_pos);
    let mut off = zero;

    let m0 = _mm_cmpeq_epi32(rel, _mm_setzero_si128());
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[0]), m0);
    let m1 = _mm_cmpeq_epi32(rel, _mm_set1_epi32(1));
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[1]), m1);
    let m2 = _mm_cmpeq_epi32(rel, _mm_set1_epi32(2));
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[2]), m2);
    let m3 = _mm_cmpeq_epi32(rel, _mm_set1_epi32(3));
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[3]), m3);

    let active = _mm_or_si128(_mm_or_si128(m0, m1), _mm_or_si128(m2, m3));
    let v = _mm_add_epi32(samples, off);
    (_mm_min_epi32(_mm_max_epi32(v, zero), max), active)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn band_offset8_sse41(
    dst: &[u16; 8],
    src: &[u16; 8],
    offsets: &[i32; 4],
    band_pos: __m128i,
    shift: u8,
    zero: __m128i,
    max: __m128i,
) -> __m128i {
    let old = unsafe { _mm_loadu_si128(dst.as_ptr().cast::<__m128i>()) };
    let s = unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) };
    let lo = _mm_cvtepu16_epi32(s);
    let hi = _mm_unpackhi_epi16(s, _mm_setzero_si128());
    let (lo, mlo) = band_offset4_sse41(lo, offsets, band_pos, shift, zero, max);
    let (hi, mhi) = band_offset4_sse41(hi, offsets, band_pos, shift, zero, max);
    let out = _mm_packus_epi32(lo, hi);
    let mask = _mm_packs_epi32(mlo, mhi);
    _mm_blendv_epi8(old, out, mask)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x8(dst: &mut [u16; 8], v: __m128i) {
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) };
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn band_offset_tail8_inplace_sse41(
    dst: &mut [u16],
    offsets: &[i32; 4],
    band_pos: __m128i,
    shift: u8,
    zero: __m128i,
    max: __m128i,
) {
    debug_assert!(dst.len() <= 8);
    if dst.is_empty() {
        return;
    }

    let len = dst.len();
    let mut tmp = [0u16; 8];
    tmp[..len].copy_from_slice(dst);
    let out = band_offset8_sse41(&tmp, &tmp, offsets, band_pos, shift, zero, max);
    store_u16x8(&mut tmp, out);
    dst.copy_from_slice(&tmp[..len]);
}

#[inline]
fn apply_eo_sample_inbounds_sse41(
    dst: &mut u16,
    s: u16,
    n1: u16,
    n2: u16,
    offsets: &[i32; 4],
    max_val: i32,
) {
    let s = s as i32;
    let n1 = n1 as i32;
    let n2 = n2 as i32;
    let sign1 = (s > n1) as i32 - (s < n1) as i32;
    let sign2 = (s > n2) as i32 - (s < n2) as i32;
    let offset = match sign1 + sign2 + 2 {
        0 => offsets[0],
        1 => offsets[1],
        3 => offsets[2],
        4 => offsets[3],
        _ => 0,
    };
    if offset != 0 {
        *dst = (s + offset).clamp(0, max_val) as u16;
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn edge_offset4_sse41(
    samples: __m128i,
    n1: __m128i,
    n2: __m128i,
    offsets: &[i32; 4],
    zero: __m128i,
    max: __m128i,
) -> (__m128i, __m128i) {
    let gt1 = _mm_cmpgt_epi32(samples, n1);
    let lt1 = _mm_cmpgt_epi32(n1, samples);
    let eq1 = _mm_cmpeq_epi32(samples, n1);
    let gt2 = _mm_cmpgt_epi32(samples, n2);
    let lt2 = _mm_cmpgt_epi32(n2, samples);
    let eq2 = _mm_cmpeq_epi32(samples, n2);

    let m0 = _mm_and_si128(lt1, lt2);
    let m1 = _mm_or_si128(_mm_and_si128(lt1, eq2), _mm_and_si128(lt2, eq1));
    let m3 = _mm_or_si128(_mm_and_si128(gt1, eq2), _mm_and_si128(gt2, eq1));
    let m4 = _mm_and_si128(gt1, gt2);

    let mut off = zero;
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[0]), m0);
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[1]), m1);
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[2]), m3);
    off = _mm_blendv_epi8(off, _mm_set1_epi32(offsets[3]), m4);

    let mut active = zero;
    if offsets[0] != 0 {
        active = _mm_or_si128(active, m0);
    }
    if offsets[1] != 0 {
        active = _mm_or_si128(active, m1);
    }
    if offsets[2] != 0 {
        active = _mm_or_si128(active, m3);
    }
    if offsets[3] != 0 {
        active = _mm_or_si128(active, m4);
    }
    let v = _mm_add_epi32(samples, off);
    (_mm_min_epi32(_mm_max_epi32(v, zero), max), active)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn edge_offset8_sse41(
    dst: &[u16; 8],
    samples: &[u16; 8],
    n1: &[u16; 8],
    n2: &[u16; 8],
    offsets: &[i32; 4],
    zero: __m128i,
    max: __m128i,
) -> __m128i {
    let old = unsafe { _mm_loadu_si128(dst.as_ptr().cast::<__m128i>()) };
    let s = unsafe { _mm_loadu_si128(samples.as_ptr().cast::<__m128i>()) };
    let a = unsafe { _mm_loadu_si128(n1.as_ptr().cast::<__m128i>()) };
    let b = unsafe { _mm_loadu_si128(n2.as_ptr().cast::<__m128i>()) };

    let s_lo = _mm_cvtepu16_epi32(s);
    let s_hi = _mm_unpackhi_epi16(s, _mm_setzero_si128());
    let a_lo = _mm_cvtepu16_epi32(a);
    let a_hi = _mm_unpackhi_epi16(a, _mm_setzero_si128());
    let b_lo = _mm_cvtepu16_epi32(b);
    let b_hi = _mm_unpackhi_epi16(b, _mm_setzero_si128());

    let (lo, mlo) = edge_offset4_sse41(s_lo, a_lo, b_lo, offsets, zero, max);
    let (hi, mhi) = edge_offset4_sse41(s_hi, a_hi, b_hi, offsets, zero, max);
    let out = _mm_packus_epi32(lo, hi);
    let mask = _mm_packs_epi32(mlo, mhi);
    _mm_blendv_epi8(old, out, mask)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_sao_edge_offset_horizontal_sse41_impl(
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
    if w == 0 || x_end <= x0 || y_end <= y0 {
        return;
    }
    let vec_x0 = x0.max(1);
    let vec_x1 = x_end.min(w.saturating_sub(1));
    if vec_x0 >= vec_x1 {
        apply_sao_plane_scalar(dst, src, w, h, x0, y0, x_end, y_end, 2, offsets, 0, 0, bd);
        return;
    }

    if x0 < vec_x0 {
        apply_sao_plane_scalar(dst, src, w, h, x0, y0, vec_x0, y_end, 2, offsets, 0, 0, bd);
    }
    if vec_x1 < x_end {
        apply_sao_plane_scalar(
            dst, src, w, h, vec_x1, y0, x_end, y_end, 2, offsets, 0, 0, bd,
        );
    }

    let max_val = ((1u32 << bd) - 1) as i32;
    let max = _mm_set1_epi32(max_val);
    let zero = _mm_setzero_si128();

    for y in y0..y_end {
        let row = y * w;
        let mid_range = row + vec_x0..row + vec_x1;
        let left_range = row + vec_x0 - 1..row + vec_x1 - 1;
        let right_range = row + vec_x0 + 1..row + vec_x1 + 1;
        let (Some(src_mid), Some(src_left), Some(src_right), Some(dst_mid)) = (
            src.get(mid_range.clone()),
            src.get(left_range),
            src.get(right_range),
            dst.get_mut(mid_range),
        ) else {
            continue;
        };

        let (mid8, mid_tail) = src_mid.as_chunks::<8>();
        let (left8, left_tail) = src_left.as_chunks::<8>();
        let (right8, right_tail) = src_right.as_chunks::<8>();
        let (dst8, dst_tail) = dst_mid.as_chunks_mut::<8>();

        for (((s, l), r), d) in mid8
            .iter()
            .zip(left8.iter())
            .zip(right8.iter())
            .zip(dst8.iter_mut())
        {
            let out = edge_offset8_sse41(d, s, r, l, offsets, zero, max);
            store_u16x8(d, out);
        }

        for (((&s, &l), &r), d) in mid_tail
            .iter()
            .zip(left_tail.iter())
            .zip(right_tail.iter())
            .zip(dst_tail.iter_mut())
        {
            apply_eo_sample_inbounds_sse41(d, s, r, l, offsets, max_val);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_sao_edge_offset_vertical_sse41_impl(
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
    if w == 0 || x_end <= x0 || y_end <= y0 {
        return;
    }
    let vec_y0 = y0.max(1);
    let vec_y1 = y_end.min(h.saturating_sub(1));
    if vec_y0 >= vec_y1 {
        apply_sao_plane_scalar(dst, src, w, h, x0, y0, x_end, y_end, 2, offsets, 0, 1, bd);
        return;
    }

    if y0 < vec_y0 {
        apply_sao_plane_scalar(dst, src, w, h, x0, y0, x_end, vec_y0, 2, offsets, 0, 1, bd);
    }
    if vec_y1 < y_end {
        apply_sao_plane_scalar(
            dst, src, w, h, x0, vec_y1, x_end, y_end, 2, offsets, 0, 1, bd,
        );
    }

    let max_val = ((1u32 << bd) - 1) as i32;
    let max = _mm_set1_epi32(max_val);
    let zero = _mm_setzero_si128();

    for y in vec_y0..vec_y1 {
        let above = (y - 1) * w;
        let row = y * w;
        let below = (y + 1) * w;
        let row_range = row + x0..row + x_end;
        let above_range = above + x0..above + x_end;
        let below_range = below + x0..below + x_end;
        let (Some(src_row), Some(src_above), Some(src_below), Some(dst_row)) = (
            src.get(row_range.clone()),
            src.get(above_range),
            src.get(below_range),
            dst.get_mut(row_range),
        ) else {
            continue;
        };

        let (src8, src_tail) = src_row.as_chunks::<8>();
        let (above8, above_tail) = src_above.as_chunks::<8>();
        let (below8, below_tail) = src_below.as_chunks::<8>();
        let (dst8, dst_tail) = dst_row.as_chunks_mut::<8>();

        for (((s, a), b), d) in src8
            .iter()
            .zip(above8.iter())
            .zip(below8.iter())
            .zip(dst8.iter_mut())
        {
            let out = edge_offset8_sse41(d, s, b, a, offsets, zero, max);
            store_u16x8(d, out);
        }

        for (((&s, &a), &b), d) in src_tail
            .iter()
            .zip(above_tail.iter())
            .zip(below_tail.iter())
            .zip(dst_tail.iter_mut())
        {
            apply_eo_sample_inbounds_sse41(d, s, b, a, offsets, max_val);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_sao_band_offset_sse41_impl(
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
    let max = _mm_set1_epi32(max_val);
    let zero = _mm_setzero_si128();
    let band_pos_v = _mm_set1_epi32(band_pos as i32);
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
            let out = band_offset8_sse41(dst, src, offsets, band_pos_v, shift, zero, max);
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
#[target_feature(enable = "sse4.1")]
fn apply_sao_band_offset_inplace_sse41_impl(
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
    let max = _mm_set1_epi32(max_val);
    let zero = _mm_setzero_si128();
    let band_pos_v = _mm_set1_epi32(band_pos as i32);
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
            let out = band_offset8_sse41(&*dst, &*dst, offsets, band_pos_v, shift, zero, max);
            store_u16x8(dst, out);
        }
        band_offset_tail8_inplace_sse41(dst_tail, offsets, band_pos_v, shift, zero, max);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_band_offset_inplace_sse41(
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
        apply_sao_band_offset_inplace_sse41_impl(
            dst, w, 0, x0, y0, x_end, y_end, offsets, band_pos, bd,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_band_offset_banded_inplace_sse41(
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
        apply_sao_band_offset_inplace_sse41_impl(
            dst_band, w, band_y0, x0, y0, x_end, y_end, offsets, band_pos, bd,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_plane_sse41(
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
        match (type_idx, eo_class) {
            (1, _) => apply_sao_band_offset_sse41_impl(
                dst, src, w, x0, y0, x_end, y_end, offsets, band_pos, bd,
            ),
            (2, 0) => apply_sao_edge_offset_horizontal_sse41_impl(
                dst, src, w, h, x0, y0, x_end, y_end, offsets, bd,
            ),
            (2, 1) => apply_sao_edge_offset_vertical_sse41_impl(
                dst, src, w, h, x0, y0, x_end, y_end, offsets, bd,
            ),
            _ => apply_sao_plane_scalar(
                dst, src, w, h, x0, y0, x_end, y_end, type_idx, offsets, band_pos, eo_class, bd,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_sao_edge_offset_horizontal_banded_sse41_impl(
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
    let vec_x0 = x0.max(1);
    let vec_x1 = x_end.min(w.saturating_sub(1));
    if vec_x0 >= vec_x1 {
        apply_sao_plane_banded_scalar(
            dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, 2, offsets, 0, 0, bd,
        );
        return;
    }

    if x0 < vec_x0 {
        apply_sao_plane_banded_scalar(
            dst_band, src_full, w, h, band_y0, x0, y0, vec_x0, y_end, 2, offsets, 0, 0, bd,
        );
    }
    if vec_x1 < x_end {
        apply_sao_plane_banded_scalar(
            dst_band, src_full, w, h, band_y0, vec_x1, y0, x_end, y_end, 2, offsets, 0, 0, bd,
        );
    }

    let Some(max_val) = (1u32)
        .checked_shl(bd as u32)
        .map(|v| v.saturating_sub(1) as i32)
    else {
        return;
    };
    let max = _mm_set1_epi32(max_val);
    let zero = _mm_setzero_si128();

    for y in y0..y_end {
        let Some(dst_base) = y.checked_sub(band_y0).and_then(|v| v.checked_mul(w)) else {
            continue;
        };
        let Some(src_base) = y.checked_mul(w) else {
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
        let (left8, left_tail) = src_left.as_chunks::<8>();
        let (right8, right_tail) = src_right.as_chunks::<8>();
        let (dst8, dst_tail) = dst_mid.as_chunks_mut::<8>();

        for (((s, l), r), d) in mid8
            .iter()
            .zip(left8.iter())
            .zip(right8.iter())
            .zip(dst8.iter_mut())
        {
            let out = edge_offset8_sse41(d, s, r, l, offsets, zero, max);
            store_u16x8(d, out);
        }

        for (((&s, &l), &r), d) in mid_tail
            .iter()
            .zip(left_tail.iter())
            .zip(right_tail.iter())
            .zip(dst_tail.iter_mut())
        {
            apply_eo_sample_inbounds_sse41(d, s, r, l, offsets, max_val);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_sao_edge_offset_vertical_banded_sse41_impl(
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
    let vec_y0 = y0.max(1);
    let vec_y1 = y_end.min(h.saturating_sub(1));
    if vec_y0 >= vec_y1 {
        apply_sao_plane_banded_scalar(
            dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, 2, offsets, 0, 1, bd,
        );
        return;
    }

    if y0 < vec_y0 {
        apply_sao_plane_banded_scalar(
            dst_band, src_full, w, h, band_y0, x0, y0, x_end, vec_y0, 2, offsets, 0, 1, bd,
        );
    }
    if vec_y1 < y_end {
        apply_sao_plane_banded_scalar(
            dst_band, src_full, w, h, band_y0, x0, vec_y1, x_end, y_end, 2, offsets, 0, 1, bd,
        );
    }

    let Some(max_val) = (1u32)
        .checked_shl(bd as u32)
        .map(|v| v.saturating_sub(1) as i32)
    else {
        return;
    };
    let max = _mm_set1_epi32(max_val);
    let zero = _mm_setzero_si128();

    for y in vec_y0..vec_y1 {
        let Some(dst_base) = y.checked_sub(band_y0).and_then(|v| v.checked_mul(w)) else {
            continue;
        };
        let above = (y - 1) * w;
        let row = y * w;
        let below = (y + 1) * w;
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
        let (above8, above_tail) = src_above.as_chunks::<8>();
        let (below8, below_tail) = src_below.as_chunks::<8>();
        let (dst8, dst_tail) = dst_row.as_chunks_mut::<8>();

        for (((s, a), b), d) in src8
            .iter()
            .zip(above8.iter())
            .zip(below8.iter())
            .zip(dst8.iter_mut())
        {
            let out = edge_offset8_sse41(d, s, b, a, offsets, zero, max);
            store_u16x8(d, out);
        }

        for (((&s, &a), &b), d) in src_tail
            .iter()
            .zip(above_tail.iter())
            .zip(below_tail.iter())
            .zip(dst_tail.iter_mut())
        {
            apply_eo_sample_inbounds_sse41(d, s, b, a, offsets, max_val);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_sao_band_offset_banded_sse41_impl(
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
    let max = _mm_set1_epi32(max_val);
    let zero = _mm_setzero_si128();
    let band_pos_v = _mm_set1_epi32(band_pos as i32);
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
            let out = band_offset8_sse41(dst, src, offsets, band_pos_v, shift, zero, max);
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
pub(crate) fn apply_sao_plane_banded_sse41(
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
        match (type_idx, eo_class) {
            (1, _) => apply_sao_band_offset_banded_sse41_impl(
                dst_band, src_full, w, band_y0, x0, y0, x_end, y_end, offsets, band_pos, bd,
            ),
            (2, 0) => apply_sao_edge_offset_horizontal_banded_sse41_impl(
                dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, offsets, bd,
            ),
            (2, 1) => apply_sao_edge_offset_vertical_banded_sse41_impl(
                dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, offsets, bd,
            ),
            _ => apply_sao_plane_banded_scalar(
                dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, type_idx, offsets,
                band_pos, eo_class, bd,
            ),
        }
    }
}
