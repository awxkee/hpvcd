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

use core::arch::x86_64::*;

#[inline(always)]
fn sao_max_value(bd: u8) -> Option<i32> {
    1u32.checked_shl(bd as u32)
        .map(|v| v.saturating_sub(1) as i32)
}

#[inline(always)]
fn dst_row_base(y: usize, w: usize, band_y0: usize) -> Option<usize> {
    y.checked_sub(band_y0).and_then(|yy| yy.checked_mul(w))
}

#[inline(always)]
fn src_row_base(y: usize, w: usize) -> Option<usize> {
    y.checked_mul(w)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x16(src: &[u16; 16]) -> __m256i {
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x16(dst: &mut [u16; 16], v: __m256i) {
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast::<__m256i>(), v) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn shr_epi32_avx2(v: __m256i, shift: u8) -> __m256i {
    _mm256_srl_epi32(v, _mm_cvtsi32_si128(shift.min(11) as i32))
}

#[inline]
#[target_feature(enable = "avx2")]
fn u16x16_to_i32x8_pair(v: __m256i) -> (__m256i, __m256i) {
    let lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v));
    let hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(v));
    (lo, hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_u16x16_from_i32x8_pair(lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_mask_u16x16_from_i32x8_pair(lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn clamp_i32x8(v: __m256i, zero: __m256i, max: __m256i) -> __m256i {
    _mm256_min_epi32(_mm256_max_epi32(v, zero), max)
}

#[inline]
#[target_feature(enable = "avx2")]
fn band_offset8_avx2(
    samples: __m256i,
    offsets: &[i32; 4],
    band_pos: __m256i,
    shift: u8,
    zero: __m256i,
    max: __m256i,
) -> (__m256i, __m256i) {
    let band = shr_epi32_avx2(samples, shift);
    let rel = _mm256_sub_epi32(band, band_pos);
    let mut off = zero;

    let m0 = _mm256_cmpeq_epi32(rel, zero);
    off = _mm256_blendv_epi8(off, _mm256_set1_epi32(offsets[0]), m0);
    let m1 = _mm256_cmpeq_epi32(rel, _mm256_set1_epi32(1));
    off = _mm256_blendv_epi8(off, _mm256_set1_epi32(offsets[1]), m1);
    let m2 = _mm256_cmpeq_epi32(rel, _mm256_set1_epi32(2));
    off = _mm256_blendv_epi8(off, _mm256_set1_epi32(offsets[2]), m2);
    let m3 = _mm256_cmpeq_epi32(rel, _mm256_set1_epi32(3));
    off = _mm256_blendv_epi8(off, _mm256_set1_epi32(offsets[3]), m3);

    let active = _mm256_or_si256(_mm256_or_si256(m0, m1), _mm256_or_si256(m2, m3));
    (
        clamp_i32x8(_mm256_add_epi32(samples, off), zero, max),
        active,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn band_offset16_avx2(
    dst: &[u16; 16],
    src: &[u16; 16],
    offsets: &[i32; 4],
    band_pos: __m256i,
    shift: u8,
    zero: __m256i,
    max: __m256i,
) -> __m256i {
    let old = load_u16x16(dst);
    let s = load_u16x16(src);
    let (s_lo, s_hi) = u16x16_to_i32x8_pair(s);
    let (lo, mlo) = band_offset8_avx2(s_lo, offsets, band_pos, shift, zero, max);
    let (hi, mhi) = band_offset8_avx2(s_hi, offsets, band_pos, shift, zero, max);
    let out = pack_u16x16_from_i32x8_pair(lo, hi);
    let mask = pack_mask_u16x16_from_i32x8_pair(mlo, mhi);
    _mm256_blendv_epi8(old, out, mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn edge_offset8_avx2(
    samples: __m256i,
    n1: __m256i,
    n2: __m256i,
    offsets: &[i32; 4],
    zero: __m256i,
    max: __m256i,
) -> (__m256i, __m256i) {
    let gt1 = _mm256_cmpgt_epi32(samples, n1);
    let lt1 = _mm256_cmpgt_epi32(n1, samples);
    let eq1 = _mm256_cmpeq_epi32(samples, n1);
    let gt2 = _mm256_cmpgt_epi32(samples, n2);
    let lt2 = _mm256_cmpgt_epi32(n2, samples);
    let eq2 = _mm256_cmpeq_epi32(samples, n2);

    let m0 = _mm256_and_si256(lt1, lt2);
    let m1 = _mm256_or_si256(_mm256_and_si256(lt1, eq2), _mm256_and_si256(lt2, eq1));
    let m3 = _mm256_or_si256(_mm256_and_si256(gt1, eq2), _mm256_and_si256(gt2, eq1));
    let m4 = _mm256_and_si256(gt1, gt2);

    let mut off = zero;
    off = _mm256_blendv_epi8(off, _mm256_set1_epi32(offsets[0]), m0);
    off = _mm256_blendv_epi8(off, _mm256_set1_epi32(offsets[1]), m1);
    off = _mm256_blendv_epi8(off, _mm256_set1_epi32(offsets[2]), m3);
    off = _mm256_blendv_epi8(off, _mm256_set1_epi32(offsets[3]), m4);

    let mut active = zero;
    if offsets[0] != 0 {
        active = _mm256_or_si256(active, m0);
    }
    if offsets[1] != 0 {
        active = _mm256_or_si256(active, m1);
    }
    if offsets[2] != 0 {
        active = _mm256_or_si256(active, m3);
    }
    if offsets[3] != 0 {
        active = _mm256_or_si256(active, m4);
    }

    (
        clamp_i32x8(_mm256_add_epi32(samples, off), zero, max),
        active,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn edge_offset16_avx2(
    dst: &[u16; 16],
    samples: &[u16; 16],
    n1: &[u16; 16],
    n2: &[u16; 16],
    offsets: &[i32; 4],
    zero: __m256i,
    max: __m256i,
) -> __m256i {
    let old = load_u16x16(dst);
    let s = load_u16x16(samples);
    let a = load_u16x16(n1);
    let b = load_u16x16(n2);

    let (s_lo, s_hi) = u16x16_to_i32x8_pair(s);
    let (a_lo, a_hi) = u16x16_to_i32x8_pair(a);
    let (b_lo, b_hi) = u16x16_to_i32x8_pair(b);

    let (lo, mlo) = edge_offset8_avx2(s_lo, a_lo, b_lo, offsets, zero, max);
    let (hi, mhi) = edge_offset8_avx2(s_hi, a_hi, b_hi, offsets, zero, max);
    let out = pack_u16x16_from_i32x8_pair(lo, hi);
    let mask = pack_mask_u16x16_from_i32x8_pair(mlo, mhi);
    _mm256_blendv_epi8(old, out, mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn band_offset_tail16_avx2(
    dst: &mut [u16],
    src: &[u16],
    offsets: &[i32; 4],
    band_pos: __m256i,
    shift: u8,
    zero: __m256i,
    max: __m256i,
) {
    debug_assert!(dst.len() <= 16 && src.len() >= dst.len());
    if dst.is_empty() {
        return;
    }

    let mut d = [0u16; 16];
    let mut s = [0u16; 16];
    d[..dst.len()].copy_from_slice(&dst[..]);
    s[..dst.len()].copy_from_slice(&src[..dst.len()]);
    let out = band_offset16_avx2(&d, &s, offsets, band_pos, shift, zero, max);
    let mut tmp = [0u16; 16];
    store_u16x16(&mut tmp, out);
    let len = dst.len();
    dst.copy_from_slice(&tmp[..len]);
}

#[inline]
fn neighbor_sample_or_self(src: &[u16], w: usize, h: usize, x: i32, y: i32, s: u16) -> u16 {
    if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
        src.get(y as usize * w + x as usize).copied().unwrap_or(s)
    } else {
        s
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn edge_offset_tail16_avx2(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    y: usize,
    x0: usize,
    dx: i32,
    dy: i32,
    offsets: &[i32; 4],
    zero: __m256i,
    max: __m256i,
) {
    debug_assert!(dst.len() <= 16);
    if dst.is_empty() {
        return;
    }

    let mut d = [0u16; 16];
    let mut s = [0u16; 16];
    let mut n1 = [0u16; 16];
    let mut n2 = [0u16; 16];
    let Some(src_base) = src_row_base(y, w) else {
        return;
    };

    for lane in 0..dst.len() {
        let x = x0 + lane;
        let sample = src.get(src_base + x).copied().unwrap_or(0);
        d[lane] = dst[lane];
        s[lane] = sample;
        n1[lane] = neighbor_sample_or_self(src, w, h, x as i32 + dx, y as i32 + dy, sample);
        n2[lane] = neighbor_sample_or_self(src, w, h, x as i32 - dx, y as i32 - dy, sample);
    }

    let out = edge_offset16_avx2(&d, &s, &n1, &n2, offsets, zero, max);
    let mut tmp = [0u16; 16];
    store_u16x16(&mut tmp, out);
    let len = dst.len();
    dst.copy_from_slice(&tmp[..len]);
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn edge_offset_segment_tail_avx2(
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
    zero: __m256i,
    max: __m256i,
) {
    if x_end <= x0 {
        return;
    }
    let Some(dst_base) = dst_row_base(y, w, band_y0) else {
        return;
    };
    let mut x = x0;
    while x < x_end {
        let len = (x_end - x).min(16);
        let Some(dst_tail) = dst_plane.get_mut(dst_base + x..dst_base + x + len) else {
            return;
        };
        edge_offset_tail16_avx2(dst_tail, src, w, h, y, x, dx, dy, offsets, zero, max);
        x += len;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn apply_sao_band_offset_avx2_impl(
    dst_plane: &mut [u16],
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
    let Some(max_val) = sao_max_value(bd) else {
        return;
    };
    let max = _mm256_set1_epi32(max_val);
    let zero = _mm256_setzero_si256();
    let band_pos_v = _mm256_set1_epi32(band_pos as i32);
    let shift = bd.saturating_sub(5);

    for y in y0..y_end {
        let Some(src_base) = src_row_base(y, w) else {
            continue;
        };
        let Some(dst_base) = dst_row_base(y, w, band_y0) else {
            continue;
        };
        let src_range = src_base + x0..src_base + x_end;
        let dst_range = dst_base + x0..dst_base + x_end;
        let (Some(src_row), Some(dst_row)) =
            (src_full.get(src_range), dst_plane.get_mut(dst_range))
        else {
            continue;
        };

        let (src16, src_tail) = src_row.as_chunks::<16>();
        let (dst16, dst_tail) = dst_row.as_chunks_mut::<16>();
        for (src, dst) in src16.iter().zip(dst16.iter_mut()) {
            let out = band_offset16_avx2(dst, src, offsets, band_pos_v, shift, zero, max);
            store_u16x16(dst, out);
        }
        band_offset_tail16_avx2(dst_tail, src_tail, offsets, band_pos_v, shift, zero, max);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn apply_sao_edge_offset_horizontal_avx2_impl(
    dst_plane: &mut [u16],
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
    let Some(max_val) = sao_max_value(bd) else {
        return;
    };
    let max = _mm256_set1_epi32(max_val);
    let zero = _mm256_setzero_si256();
    let vec_x0 = x0.max(1);
    let vec_x1 = x_end.min(w.saturating_sub(1));

    if vec_x0 >= vec_x1 {
        for y in y0..y_end {
            edge_offset_segment_tail_avx2(
                dst_plane, src_full, w, h, band_y0, y, x0, x_end, 1, 0, offsets, zero, max,
            );
        }
        return;
    }

    for y in y0..y_end {
        if x0 < vec_x0 {
            edge_offset_segment_tail_avx2(
                dst_plane, src_full, w, h, band_y0, y, x0, vec_x0, 1, 0, offsets, zero, max,
            );
        }
        if vec_x0 < vec_x1 {
            let Some(src_base) = src_row_base(y, w) else {
                continue;
            };
            let Some(dst_base) = dst_row_base(y, w, band_y0) else {
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
                dst_plane.get_mut(dst_range),
            ) else {
                continue;
            };

            let (mid16, mid_tail) = src_mid.as_chunks::<16>();
            let (left16, _) = src_left.as_chunks::<16>();
            let (right16, _) = src_right.as_chunks::<16>();
            let (dst16, dst_tail) = dst_mid.as_chunks_mut::<16>();
            for (((s, l), r), d) in mid16
                .iter()
                .zip(left16.iter())
                .zip(right16.iter())
                .zip(dst16.iter_mut())
            {
                let out = edge_offset16_avx2(d, s, r, l, offsets, zero, max);
                store_u16x16(d, out);
            }
            let tail_x0 = vec_x0 + mid16.len() * 16;
            if !mid_tail.is_empty() {
                edge_offset_tail16_avx2(
                    dst_tail, src_full, w, h, y, tail_x0, 1, 0, offsets, zero, max,
                );
            }
        }
        if vec_x1 < x_end {
            edge_offset_segment_tail_avx2(
                dst_plane, src_full, w, h, band_y0, y, vec_x1, x_end, 1, 0, offsets, zero, max,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn apply_sao_edge_offset_vertical_avx2_impl(
    dst_plane: &mut [u16],
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
    let Some(max_val) = sao_max_value(bd) else {
        return;
    };
    let max = _mm256_set1_epi32(max_val);
    let zero = _mm256_setzero_si256();
    let vec_y0 = y0.max(1);
    let vec_y1 = y_end.min(h.saturating_sub(1));

    if vec_y0 >= vec_y1 {
        for y in y0..y_end {
            edge_offset_segment_tail_avx2(
                dst_plane, src_full, w, h, band_y0, y, x0, x_end, 0, 1, offsets, zero, max,
            );
        }
        return;
    }

    for y in y0..vec_y0.min(y_end) {
        edge_offset_segment_tail_avx2(
            dst_plane, src_full, w, h, band_y0, y, x0, x_end, 0, 1, offsets, zero, max,
        );
    }

    for y in vec_y0..vec_y1 {
        let above = (y - 1) * w;
        let row = y * w;
        let below = (y + 1) * w;
        let Some(dst_base) = dst_row_base(y, w, band_y0) else {
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
            dst_plane.get_mut(dst_range),
        ) else {
            continue;
        };

        let (src16, src_tail) = src_row.as_chunks::<16>();
        let (above16, _) = src_above.as_chunks::<16>();
        let (below16, _) = src_below.as_chunks::<16>();
        let (dst16, dst_tail) = dst_row.as_chunks_mut::<16>();
        for (((s, a), b), d) in src16
            .iter()
            .zip(above16.iter())
            .zip(below16.iter())
            .zip(dst16.iter_mut())
        {
            let out = edge_offset16_avx2(d, s, b, a, offsets, zero, max);
            store_u16x16(d, out);
        }
        if !src_tail.is_empty() {
            let tail_x0 = x0 + src16.len() * 16;
            edge_offset_tail16_avx2(
                dst_tail, src_full, w, h, y, tail_x0, 0, 1, offsets, zero, max,
            );
        }
    }

    for y in vec_y1.max(y0)..y_end {
        edge_offset_segment_tail_avx2(
            dst_plane, src_full, w, h, band_y0, y, x0, x_end, 0, 1, offsets, zero, max,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn apply_sao_edge_offset_diagonal_avx2_impl(
    dst_plane: &mut [u16],
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
    let Some(max_val) = sao_max_value(bd) else {
        return;
    };
    let max = _mm256_set1_epi32(max_val);
    let zero = _mm256_setzero_si256();
    let dy = if eo_class == 2 { 1 } else { -1 };
    let vec_y0 = y0.max(1);
    let vec_y1 = y_end.min(h.saturating_sub(1));
    let vec_x0 = x0.max(1);
    let vec_x1 = x_end.min(w.saturating_sub(1));

    if vec_y0 >= vec_y1 {
        for y in y0..y_end {
            edge_offset_segment_tail_avx2(
                dst_plane, src_full, w, h, band_y0, y, x0, x_end, 1, dy, offsets, zero, max,
            );
        }
        return;
    }

    for y in y0..vec_y0.min(y_end) {
        edge_offset_segment_tail_avx2(
            dst_plane, src_full, w, h, band_y0, y, x0, x_end, 1, dy, offsets, zero, max,
        );
    }

    for y in vec_y0..vec_y1 {
        if vec_x0 >= vec_x1 {
            edge_offset_segment_tail_avx2(
                dst_plane, src_full, w, h, band_y0, y, x0, x_end, 1, dy, offsets, zero, max,
            );
            continue;
        }
        if x0 < vec_x0 {
            edge_offset_segment_tail_avx2(
                dst_plane, src_full, w, h, band_y0, y, x0, vec_x0, 1, dy, offsets, zero, max,
            );
        }
        if vec_x0 < vec_x1 {
            let n1_y = (y as i32 + dy) as usize;
            let n2_y = (y as i32 - dy) as usize;
            let row = y * w;
            let n1_base = n1_y * w;
            let n2_base = n2_y * w;
            let Some(dst_base) = dst_row_base(y, w, band_y0) else {
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
                dst_plane.get_mut(dst_range),
            ) else {
                continue;
            };

            let (src16, src_tail) = src_row.as_chunks::<16>();
            let (n1_16, _) = src_n1.as_chunks::<16>();
            let (n2_16, _) = src_n2.as_chunks::<16>();
            let (dst16, dst_tail) = dst_row.as_chunks_mut::<16>();
            for (((s, a), b), d) in src16
                .iter()
                .zip(n1_16.iter())
                .zip(n2_16.iter())
                .zip(dst16.iter_mut())
            {
                let out = edge_offset16_avx2(d, s, a, b, offsets, zero, max);
                store_u16x16(d, out);
            }
            if !src_tail.is_empty() {
                let tail_x0 = vec_x0 + src16.len() * 16;
                edge_offset_tail16_avx2(
                    dst_tail, src_full, w, h, y, tail_x0, 1, dy, offsets, zero, max,
                );
            }
        }
        if vec_x1 < x_end {
            edge_offset_segment_tail_avx2(
                dst_plane, src_full, w, h, band_y0, y, vec_x1, x_end, 1, dy, offsets, zero, max,
            );
        }
    }

    for y in vec_y1.max(y0)..y_end {
        edge_offset_segment_tail_avx2(
            dst_plane, src_full, w, h, band_y0, y, x0, x_end, 1, dy, offsets, zero, max,
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn band_offset_tail16_inplace_avx2(
    dst: &mut [u16],
    offsets: &[i32; 4],
    band_pos: __m256i,
    shift: u8,
    zero: __m256i,
    max: __m256i,
) {
    debug_assert!(dst.len() <= 16);
    if dst.is_empty() {
        return;
    }

    let mut d: [u16; 16] = [0u16; 16];
    d[..dst.len()].copy_from_slice(&dst[..]);
    let out = band_offset16_avx2(&d, &d, offsets, band_pos, shift, zero, max);
    let mut tmp = [0u16; 16];
    store_u16x16(&mut tmp, out);
    let len = dst.len();
    dst.copy_from_slice(&tmp[..len]);
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn apply_sao_band_offset_inplace_avx2_impl(
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
    let Some(max_val) = sao_max_value(bd) else {
        return;
    };
    let max = _mm256_set1_epi32(max_val);
    let zero = _mm256_setzero_si256();
    let band_pos_v = _mm256_set1_epi32(band_pos as i32);
    let shift = bd.saturating_sub(5);

    for y in y0..y_end {
        let Some(dst_base) = dst_row_base(y, w, band_y0) else {
            continue;
        };
        let dst_range = dst_base + x0..dst_base + x_end;
        let Some(dst_row) = dst_plane.get_mut(dst_range) else {
            continue;
        };

        let (dst16, dst_tail) = dst_row.as_chunks_mut::<16>();
        for dst in dst16.iter_mut() {
            let out = band_offset16_avx2(&*dst, &*dst, offsets, band_pos_v, shift, zero, max);
            store_u16x16(dst, out);
        }
        band_offset_tail16_inplace_avx2(dst_tail, offsets, band_pos_v, shift, zero, max);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_band_offset_inplace_avx2(
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
        apply_sao_band_offset_inplace_avx2_impl(
            dst, w, 0, x0, y0, x_end, y_end, offsets, band_pos, bd,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_band_offset_banded_inplace_avx2(
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
        apply_sao_band_offset_inplace_avx2_impl(
            dst_band, w, band_y0, x0, y0, x_end, y_end, offsets, band_pos, bd,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_plane_avx2(
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
            (1, _) => apply_sao_band_offset_avx2_impl(
                dst, src, w, 0, x0, y0, x_end, y_end, offsets, band_pos, bd,
            ),
            (2, 0) => apply_sao_edge_offset_horizontal_avx2_impl(
                dst, src, w, h, 0, x0, y0, x_end, y_end, offsets, bd,
            ),
            (2, 1) => apply_sao_edge_offset_vertical_avx2_impl(
                dst, src, w, h, 0, x0, y0, x_end, y_end, offsets, bd,
            ),
            (2, _) => apply_sao_edge_offset_diagonal_avx2_impl(
                dst, src, w, h, 0, x0, y0, x_end, y_end, offsets, eo_class, bd,
            ),
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_plane_banded_avx2(
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
            (1, _) => apply_sao_band_offset_avx2_impl(
                dst_band, src_full, w, band_y0, x0, y0, x_end, y_end, offsets, band_pos, bd,
            ),
            (2, 0) => apply_sao_edge_offset_horizontal_avx2_impl(
                dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, offsets, bd,
            ),
            (2, 1) => apply_sao_edge_offset_vertical_avx2_impl(
                dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, offsets, bd,
            ),
            (2, _) => apply_sao_edge_offset_diagonal_avx2_impl(
                dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, offsets, eo_class, bd,
            ),
            _ => {}
        }
    }
}
