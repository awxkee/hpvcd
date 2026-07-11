/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice,
 * // this list of conditions and the following disclaimer.
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

use crate::deblock::LumaDecision;
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x4(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 4);
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast::<__m128i>(), v) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x8(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 8);
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn combine_i32x4(lo: __m128i, hi: __m128i) -> __m256i {
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn lo_i32x4(v: __m256i) -> __m128i {
    _mm256_castsi256_si128(v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn hi_i32x4(v: __m256i) -> __m128i {
    _mm256_extracti128_si256::<1>(v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn zero_hi_i32x4(v: __m128i) -> __m256i {
    combine_i32x4(v, _mm_setzero_si128())
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_i32x8_to_u16x8(v: __m256i) -> __m128i {
    _mm_packus_epi32(lo_i32x4(v), hi_i32x4(v))
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_lo_i32x4_to_u16x4(v: __m256i) -> __m128i {
    _mm_packus_epi32(lo_i32x4(v), _mm_setzero_si128())
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4_to_i32x8(src: &[u16]) -> __m256i {
    zero_hi_i32x4(_mm_cvtepu16_epi32(load_u16x4(src)))
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8_to_i32x8(src: &[u16]) -> __m256i {
    _mm256_cvtepu16_epi32(load_u16x8(src))
}

#[inline]
#[target_feature(enable = "avx2")]
fn bias_pair4(a: i32, b: i32) -> __m256i {
    _mm256_setr_epi32(a, a, a, a, b, b, b, b)
}

#[inline]
#[target_feature(enable = "avx2")]
fn abs_i32x8(v: __m256i) -> __m256i {
    _mm256_abs_epi32(v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn cmplt_i32x8(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epi32(b, a)
}

#[inline]
#[target_feature(enable = "avx2")]
fn clamp_i32x8(v: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_min_epi32(_mm256_max_epi32(v, lo), hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn blend_i32x8(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
    _mm256_blendv_epi8(a, b, mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn chroma_filter8_avx2(
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    tc: __m256i,
    zero: __m256i,
    maxv: __m256i,
) -> (__m256i, __m256i) {
    let four = _mm256_set1_epi32(4);
    let delta = _mm256_srai_epi32::<3>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_sub_epi32(q0, p0), four),
            _mm256_sub_epi32(p1, q1),
        ),
        four,
    ));
    let delta = clamp_i32x8(delta, _mm256_sub_epi32(zero, tc), tc);
    let p0n = clamp_i32x8(_mm256_add_epi32(p0, delta), zero, maxv);
    let q0n = clamp_i32x8(_mm256_sub_epi32(q0, delta), zero, maxv);
    (p0n, q0n)
}

#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn decompose(d: LumaDecision) -> (bool, bool, bool) {
    match d {
        LumaDecision::Skip => (false, false, false),
        LumaDecision::Strong => (true, false, false),
        LumaDecision::Weak { do_p1, do_q1 } => (false, do_p1, do_q1),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn luma_filter8_avx2(
    p3: __m256i,
    p2: __m256i,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    q2: __m256i,
    q3: __m256i,
    strong_lo: bool,
    strong_hi: bool,
    do_p1_lo: bool,
    do_p1_hi: bool,
    do_q1_lo: bool,
    do_q1_hi: bool,
    tc: __m256i,
    maxv: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i, __m256i, __m256i) {
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi32(1);
    let two = _mm256_set1_epi32(2);
    let three = _mm256_set1_epi32(3);
    let four = _mm256_set1_epi32(4);
    let eight = _mm256_set1_epi32(8);

    // Build per-half uniform masks (low 4 lanes = segment 0, high 4 = segment 1).
    let half = |lo: bool, hi: bool| {
        _mm256_set_epi32(
            if hi { -1 } else { 0 },
            if hi { -1 } else { 0 },
            if hi { -1 } else { 0 },
            if hi { -1 } else { 0 },
            if lo { -1 } else { 0 },
            if lo { -1 } else { 0 },
            if lo { -1 } else { 0 },
            if lo { -1 } else { 0 },
        )
    };
    let strong = half(strong_lo, strong_hi);
    let do_p1_mask = half(do_p1_lo, do_p1_hi);
    let do_q1_mask = half(do_q1_lo, do_q1_hi);

    let p0s = clamp_i32x8(
        _mm256_srai_epi32::<3>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    p2,
                    _mm256_add_epi32(_mm256_slli_epi32::<1>(p1), _mm256_slli_epi32::<1>(p0)),
                ),
                _mm256_add_epi32(_mm256_slli_epi32::<1>(q0), q1),
            ),
            four,
        )),
        zero,
        maxv,
    );
    let p1s = clamp_i32x8(
        _mm256_srai_epi32::<2>(_mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(p2, p1), _mm256_add_epi32(p0, q0)),
            two,
        )),
        zero,
        maxv,
    );
    let p2s = clamp_i32x8(
        _mm256_srai_epi32::<3>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(_mm256_slli_epi32::<1>(p3), _mm256_mullo_epi32(three, p2)),
                _mm256_add_epi32(p1, p0),
            ),
            _mm256_add_epi32(q0, four),
        )),
        zero,
        maxv,
    );
    let q0s = clamp_i32x8(
        _mm256_srai_epi32::<3>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(p1, _mm256_slli_epi32::<1>(p0)),
                _mm256_add_epi32(_mm256_slli_epi32::<1>(q0), _mm256_slli_epi32::<1>(q1)),
            ),
            _mm256_add_epi32(q2, four),
        )),
        zero,
        maxv,
    );
    let q1s = clamp_i32x8(
        _mm256_srai_epi32::<2>(_mm256_add_epi32(
            _mm256_add_epi32(_mm256_add_epi32(p0, q0), _mm256_add_epi32(q1, q2)),
            two,
        )),
        zero,
        maxv,
    );
    let q2s = clamp_i32x8(
        _mm256_srai_epi32::<3>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(p0, q0),
                _mm256_add_epi32(q1, _mm256_mullo_epi32(three, q2)),
            ),
            _mm256_add_epi32(_mm256_slli_epi32::<1>(q3), four),
        )),
        zero,
        maxv,
    );

    let delta = _mm256_srai_epi32::<4>(_mm256_add_epi32(
        _mm256_sub_epi32(
            _mm256_mullo_epi32(_mm256_set1_epi32(9), _mm256_sub_epi32(q0, p0)),
            _mm256_mullo_epi32(three, _mm256_sub_epi32(q1, p1)),
        ),
        eight,
    ));
    // Per-line weak gate |delta0| < 10*tc.
    let weak_active = cmplt_i32x8(
        abs_i32x8(delta),
        _mm256_mullo_epi32(_mm256_set1_epi32(10), tc),
    );
    let delta = clamp_i32x8(delta, _mm256_sub_epi32(zero, tc), tc);
    let p0w = blend_i32x8(
        p0,
        clamp_i32x8(_mm256_add_epi32(p0, delta), zero, maxv),
        weak_active,
    );
    let q0w = blend_i32x8(
        q0,
        clamp_i32x8(_mm256_sub_epi32(q0, delta), zero, maxv),
        weak_active,
    );

    let half_tc = _mm256_srai_epi32::<1>(tc);
    let neg_half_tc = _mm256_sub_epi32(zero, half_tc);

    let dp1 = clamp_i32x8(
        _mm256_srai_epi32::<1>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_srai_epi32::<1>(_mm256_add_epi32(_mm256_add_epi32(p2, p0), one)),
                p1,
            ),
            delta,
        )),
        neg_half_tc,
        half_tc,
    );
    let p1_apply = _mm256_and_si256(weak_active, do_p1_mask);
    let p1w = blend_i32x8(
        p1,
        clamp_i32x8(_mm256_add_epi32(p1, dp1), zero, maxv),
        p1_apply,
    );

    let dq1 = clamp_i32x8(
        _mm256_srai_epi32::<1>(_mm256_sub_epi32(
            _mm256_sub_epi32(
                _mm256_srai_epi32::<1>(_mm256_add_epi32(_mm256_add_epi32(q2, q0), one)),
                q1,
            ),
            delta,
        )),
        neg_half_tc,
        half_tc,
    );
    let q1_apply = _mm256_and_si256(weak_active, do_q1_mask);
    let q1w = blend_i32x8(
        q1,
        clamp_i32x8(_mm256_add_epi32(q1, dq1), zero, maxv),
        q1_apply,
    );

    (
        blend_i32x8(p0w, p0s, strong),
        blend_i32x8(p1w, p1s, strong),
        blend_i32x8(p2, p2s, strong),
        blend_i32x8(q0w, q0s, strong),
        blend_i32x8(q1w, q1s, strong),
        blend_i32x8(q2, q2s, strong),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_chroma_vertical4x4(
    pix: &[u16],
    cw: usize,
    base: usize,
    edge_start: usize,
) -> (__m128i, __m128i, __m128i, __m128i) {
    debug_assert!(pix.len() >= base + 3 * cw + edge_start + 4);

    let r0 = load_u16x4(&pix[base + edge_start..]);
    let r1 = load_u16x4(&pix[base + cw + edge_start..]);
    let r2 = load_u16x4(&pix[base + 2 * cw + edge_start..]);
    let r3 = load_u16x4(&pix[base + 3 * cw + edge_start..]);

    let r01 = _mm_unpacklo_epi16(r0, r1);
    let r23 = _mm_unpacklo_epi16(r2, r3);
    let p1p0 = _mm_unpacklo_epi32(r01, r23);
    let q0q1 = _mm_unpackhi_epi32(r01, r23);

    (
        _mm_cvtepu16_epi32(p1p0),
        _mm_unpackhi_epi16(p1p0, _mm_setzero_si128()),
        _mm_cvtepu16_epi32(q0q1),
        _mm_unpackhi_epi16(q0q1, _mm_setzero_si128()),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn store_chroma_vertical4x4(
    pix: &mut [u16],
    cw: usize,
    base: usize,
    edge_start: usize,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
) {
    debug_assert!(pix.len() >= base + 3 * cw + edge_start + 4);

    let zero = _mm_setzero_si128();
    let p1 = _mm_packus_epi32(p1, zero);
    let p0 = _mm_packus_epi32(p0, zero);
    let q0 = _mm_packus_epi32(q0, zero);
    let q1 = _mm_packus_epi32(q1, zero);

    let p1p0 = _mm_unpacklo_epi16(p1, p0);
    let q0q1 = _mm_unpacklo_epi16(q0, q1);
    let row01 = _mm_unpacklo_epi32(p1p0, q0q1);
    let row23 = _mm_unpackhi_epi32(p1p0, q0q1);

    store_u16x4(&mut pix[base + edge_start..], row01);
    store_u16x4(
        &mut pix[base + cw + edge_start..],
        _mm_srli_si128::<8>(row01),
    );
    store_u16x4(&mut pix[base + 2 * cw + edge_start..], row23);
    store_u16x4(
        &mut pix[base + 3 * cw + edge_start..],
        _mm_srli_si128::<8>(row23),
    );
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_luma_vertical8x4(
    pix: &[u16],
    w: usize,
    base: usize,
    edge_start: usize,
) -> (
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
) {
    debug_assert!(pix.len() >= base + 3 * w + edge_start + 8);
    let r0 = load_u16x8(&pix[base + edge_start..]);
    let r1 = load_u16x8(&pix[base + w + edge_start..]);
    let r2 = load_u16x8(&pix[base + 2 * w + edge_start..]);
    let r3 = load_u16x8(&pix[base + 3 * w + edge_start..]);

    let r01_lo = _mm_unpacklo_epi16(r0, r1);
    let r23_lo = _mm_unpacklo_epi16(r2, r3);
    let c01 = _mm_unpacklo_epi32(r01_lo, r23_lo);
    let c23 = _mm_unpackhi_epi32(r01_lo, r23_lo);

    let r01_hi = _mm_unpackhi_epi16(r0, r1);
    let r23_hi = _mm_unpackhi_epi16(r2, r3);
    let c45 = _mm_unpacklo_epi32(r01_hi, r23_hi);
    let c67 = _mm_unpackhi_epi32(r01_hi, r23_hi);

    (
        _mm_cvtepu16_epi32(c01),
        _mm_unpackhi_epi16(c01, _mm_setzero_si128()),
        _mm_cvtepu16_epi32(c23),
        _mm_unpackhi_epi16(c23, _mm_setzero_si128()),
        _mm_cvtepu16_epi32(c45),
        _mm_unpackhi_epi16(c45, _mm_setzero_si128()),
        _mm_cvtepu16_epi32(c67),
        _mm_unpackhi_epi16(c67, _mm_setzero_si128()),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn interleave_col_pair(a: __m128i, b: __m128i) -> __m128i {
    let ab = _mm_packus_epi32(a, b);
    _mm_unpacklo_epi16(ab, _mm_srli_si128::<8>(ab))
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn store_luma_vertical8x4(
    pix: &mut [u16],
    w: usize,
    base: usize,
    edge_start: usize,
    c0: __m128i,
    c1: __m128i,
    c2: __m128i,
    c3: __m128i,
    c4: __m128i,
    c5: __m128i,
    c6: __m128i,
    c7: __m128i,
) {
    debug_assert!(pix.len() >= base + 3 * w + edge_start + 8);

    let c01 = interleave_col_pair(c0, c1);
    let c23 = interleave_col_pair(c2, c3);
    let c45 = interleave_col_pair(c4, c5);
    let c67 = interleave_col_pair(c6, c7);

    let r01_l = _mm_unpacklo_epi32(c01, c23);
    let r23_l = _mm_unpackhi_epi32(c01, c23);
    let r01_r = _mm_unpacklo_epi32(c45, c67);
    let r23_r = _mm_unpackhi_epi32(c45, c67);

    let r0 = _mm_unpacklo_epi64(r01_l, r01_r);
    let r1 = _mm_unpackhi_epi64(r01_l, r01_r);
    let r2 = _mm_unpacklo_epi64(r23_l, r23_r);
    let r3 = _mm_unpackhi_epi64(r23_l, r23_r);

    store_u16x8(&mut pix[base + edge_start..], r0);
    store_u16x8(&mut pix[base + w + edge_start..], r1);
    store_u16x8(&mut pix[base + 2 * w + edge_start..], r2);
    store_u16x8(&mut pix[base + 3 * w + edge_start..], r3);
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn chroma_horizontal_plane_avx2_impl(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    scan: usize,
    crow0: usize,
    tc_c: i32,
    maxv_c: i32,
) {
    let p1_base = (edge - 2 - crow0) * cw + scan;
    let p0_base = (edge - 1 - crow0) * cw + scan;
    let q0_base = (edge - crow0) * cw + scan;
    let q1_base = (edge + 1 - crow0) * cw + scan;

    let p1 = load_u16x4_to_i32x8(&pix[p1_base..]);
    let p0 = load_u16x4_to_i32x8(&pix[p0_base..]);
    let q0 = load_u16x4_to_i32x8(&pix[q0_base..]);
    let q1 = load_u16x4_to_i32x8(&pix[q1_base..]);

    let zero = _mm256_setzero_si256();
    let maxv = _mm256_set1_epi32(maxv_c);
    let tc = _mm256_set1_epi32(tc_c);
    let (p0n, q0n) = chroma_filter8_avx2(p1, p0, q0, q1, tc, zero, maxv);
    store_u16x4(&mut pix[p0_base..], pack_lo_i32x4_to_u16x4(p0n));
    store_u16x4(&mut pix[q0_base..], pack_lo_i32x4_to_u16x4(q0n));
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn chroma_horizontal_plane_pair_avx2_impl(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    scan: usize,
    crow0: usize,
    tc0: i32,
    tc1: i32,
    maxv_c: i32,
) {
    let p1_base = (edge - 2 - crow0) * cw + scan;
    let p0_base = (edge - 1 - crow0) * cw + scan;
    let q0_base = (edge - crow0) * cw + scan;
    let q1_base = (edge + 1 - crow0) * cw + scan;

    let p1 = load_u16x8_to_i32x8(&pix[p1_base..]);
    let p0 = load_u16x8_to_i32x8(&pix[p0_base..]);
    let q0 = load_u16x8_to_i32x8(&pix[q0_base..]);
    let q1 = load_u16x8_to_i32x8(&pix[q1_base..]);

    let zero = _mm256_setzero_si256();
    let maxv = _mm256_set1_epi32(maxv_c);
    let tc = bias_pair4(tc0, tc1);
    let (p0n, q0n) = chroma_filter8_avx2(p1, p0, q0, q1, tc, zero, maxv);
    store_u16x8(&mut pix[p0_base..], pack_i32x8_to_u16x8(p0n));
    store_u16x8(&mut pix[q0_base..], pack_i32x8_to_u16x8(q0n));
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn chroma_vertical_plane_avx2_impl(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    s: usize,
    crow0: usize,
    tc_c: i32,
    maxv_c: i32,
) {
    let base = (s - crow0) * cw;
    let edge_start = edge - 2;
    let (p1, p0, q0, q1) = load_chroma_vertical4x4(pix, cw, base, edge_start);

    let zero = _mm256_setzero_si256();
    let maxv = _mm256_set1_epi32(maxv_c);
    let tc = _mm256_set1_epi32(tc_c);
    let (p0n, q0n) = chroma_filter8_avx2(
        zero_hi_i32x4(p1),
        zero_hi_i32x4(p0),
        zero_hi_i32x4(q0),
        zero_hi_i32x4(q1),
        tc,
        zero,
        maxv,
    );
    store_chroma_vertical4x4(
        pix,
        cw,
        base,
        edge_start,
        p1,
        lo_i32x4(p0n),
        lo_i32x4(q0n),
        q1,
    );
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn chroma_vertical_plane_pair_avx2_impl(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    s: usize,
    crow0: usize,
    tc0: i32,
    tc1: i32,
    maxv_c: i32,
) {
    let base0 = (s - crow0) * cw;
    let base1 = base0 + 4 * cw;
    let edge_start = edge - 2;
    let (p1a, p0a, q0a, q1a) = load_chroma_vertical4x4(pix, cw, base0, edge_start);
    let (p1b, p0b, q0b, q1b) = load_chroma_vertical4x4(pix, cw, base1, edge_start);

    let zero = _mm256_setzero_si256();
    let maxv = _mm256_set1_epi32(maxv_c);
    let tc = bias_pair4(tc0, tc1);
    let (p0n, q0n) = chroma_filter8_avx2(
        combine_i32x4(p1a, p1b),
        combine_i32x4(p0a, p0b),
        combine_i32x4(q0a, q0b),
        combine_i32x4(q1a, q1b),
        tc,
        zero,
        maxv,
    );
    store_chroma_vertical4x4(
        pix,
        cw,
        base0,
        edge_start,
        p1a,
        lo_i32x4(p0n),
        lo_i32x4(q0n),
        q1a,
    );
    store_chroma_vertical4x4(
        pix,
        cw,
        base1,
        edge_start,
        p1b,
        hi_i32x4(p0n),
        hi_i32x4(q0n),
        q1b,
    );
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn luma_horizontal_plane_avx2_impl(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    scan: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    let (strong, do_p1, do_q1) = decompose(decision);
    let p3_base = (edge - 4 - row0) * w + scan;
    let p2_base = (edge - 3 - row0) * w + scan;
    let p1_base = (edge - 2 - row0) * w + scan;
    let p0_base = (edge - 1 - row0) * w + scan;
    let q0_base = (edge - row0) * w + scan;
    let q1_base = (edge + 1 - row0) * w + scan;
    let q2_base = (edge + 2 - row0) * w + scan;
    let q3_base = (edge + 3 - row0) * w + scan;

    let p3 = load_u16x4_to_i32x8(&pix[p3_base..]);
    let p2 = load_u16x4_to_i32x8(&pix[p2_base..]);
    let p1 = load_u16x4_to_i32x8(&pix[p1_base..]);
    let p0 = load_u16x4_to_i32x8(&pix[p0_base..]);
    let q0 = load_u16x4_to_i32x8(&pix[q0_base..]);
    let q1 = load_u16x4_to_i32x8(&pix[q1_base..]);
    let q2 = load_u16x4_to_i32x8(&pix[q2_base..]);
    let q3 = load_u16x4_to_i32x8(&pix[q3_base..]);

    let (p0n, p1n, p2n, q0n, q1n, q2n) = luma_filter8_avx2(
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        strong,
        strong,
        do_p1,
        do_p1,
        do_q1,
        do_q1,
        _mm256_set1_epi32(tc),
        _mm256_set1_epi32(maxv),
    );
    store_u16x4(&mut pix[p0_base..], pack_lo_i32x4_to_u16x4(p0n));
    store_u16x4(&mut pix[p1_base..], pack_lo_i32x4_to_u16x4(p1n));
    store_u16x4(&mut pix[p2_base..], pack_lo_i32x4_to_u16x4(p2n));
    store_u16x4(&mut pix[q0_base..], pack_lo_i32x4_to_u16x4(q0n));
    store_u16x4(&mut pix[q1_base..], pack_lo_i32x4_to_u16x4(q1n));
    store_u16x4(&mut pix[q2_base..], pack_lo_i32x4_to_u16x4(q2n));
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn luma_horizontal_plane_pair_avx2_impl(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    scan: usize,
    row0: usize,
    decision0: LumaDecision,
    tc0: i32,
    decision1: LumaDecision,
    tc1: i32,
    maxv: i32,
) {
    let (strong0, do_p1_0, do_q1_0) = decompose(decision0);
    let (strong1, do_p1_1, do_q1_1) = decompose(decision1);
    let p3_base = (edge - 4 - row0) * w + scan;
    let p2_base = (edge - 3 - row0) * w + scan;
    let p1_base = (edge - 2 - row0) * w + scan;
    let p0_base = (edge - 1 - row0) * w + scan;
    let q0_base = (edge - row0) * w + scan;
    let q1_base = (edge + 1 - row0) * w + scan;
    let q2_base = (edge + 2 - row0) * w + scan;
    let q3_base = (edge + 3 - row0) * w + scan;

    let p3 = load_u16x8_to_i32x8(&pix[p3_base..]);
    let p2 = load_u16x8_to_i32x8(&pix[p2_base..]);
    let p1 = load_u16x8_to_i32x8(&pix[p1_base..]);
    let p0 = load_u16x8_to_i32x8(&pix[p0_base..]);
    let q0 = load_u16x8_to_i32x8(&pix[q0_base..]);
    let q1 = load_u16x8_to_i32x8(&pix[q1_base..]);
    let q2 = load_u16x8_to_i32x8(&pix[q2_base..]);
    let q3 = load_u16x8_to_i32x8(&pix[q3_base..]);

    let (p0n, p1n, p2n, q0n, q1n, q2n) = luma_filter8_avx2(
        p3,
        p2,
        p1,
        p0,
        q0,
        q1,
        q2,
        q3,
        strong0,
        strong1,
        do_p1_0,
        do_p1_1,
        do_q1_0,
        do_q1_1,
        bias_pair4(tc0, tc1),
        _mm256_set1_epi32(maxv),
    );
    store_u16x8(&mut pix[p0_base..], pack_i32x8_to_u16x8(p0n));
    store_u16x8(&mut pix[p1_base..], pack_i32x8_to_u16x8(p1n));
    store_u16x8(&mut pix[p2_base..], pack_i32x8_to_u16x8(p2n));
    store_u16x8(&mut pix[q0_base..], pack_i32x8_to_u16x8(q0n));
    store_u16x8(&mut pix[q1_base..], pack_i32x8_to_u16x8(q1n));
    store_u16x8(&mut pix[q2_base..], pack_i32x8_to_u16x8(q2n));
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn luma_vertical_plane_avx2_impl(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    s: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    let (strong, do_p1, do_q1) = decompose(decision);
    let base = (s - row0) * w;
    let edge_start = edge - 4;
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_luma_vertical8x4(pix, w, base, edge_start);
    let (p0n, p1n, p2n, q0n, q1n, q2n) = luma_filter8_avx2(
        zero_hi_i32x4(p3),
        zero_hi_i32x4(p2),
        zero_hi_i32x4(p1),
        zero_hi_i32x4(p0),
        zero_hi_i32x4(q0),
        zero_hi_i32x4(q1),
        zero_hi_i32x4(q2),
        zero_hi_i32x4(q3),
        strong,
        strong,
        do_p1,
        do_p1,
        do_q1,
        do_q1,
        _mm256_set1_epi32(tc),
        _mm256_set1_epi32(maxv),
    );
    store_luma_vertical8x4(
        pix,
        w,
        base,
        edge_start,
        p3,
        lo_i32x4(p2n),
        lo_i32x4(p1n),
        lo_i32x4(p0n),
        lo_i32x4(q0n),
        lo_i32x4(q1n),
        lo_i32x4(q2n),
        q3,
    );
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn luma_vertical_plane_pair_avx2_impl(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    s: usize,
    row0: usize,
    decision0: LumaDecision,
    tc0: i32,
    decision1: LumaDecision,
    tc1: i32,
    maxv: i32,
) {
    let (strong0, do_p1_0, do_q1_0) = decompose(decision0);
    let (strong1, do_p1_1, do_q1_1) = decompose(decision1);
    let base0 = (s - row0) * w;
    let base1 = base0 + 4 * w;
    let edge_start = edge - 4;
    let (p3a, p2a, p1a, p0a, q0a, q1a, q2a, q3a) = load_luma_vertical8x4(pix, w, base0, edge_start);
    let (p3b, p2b, p1b, p0b, q0b, q1b, q2b, q3b) = load_luma_vertical8x4(pix, w, base1, edge_start);

    let (p0n, p1n, p2n, q0n, q1n, q2n) = luma_filter8_avx2(
        combine_i32x4(p3a, p3b),
        combine_i32x4(p2a, p2b),
        combine_i32x4(p1a, p1b),
        combine_i32x4(p0a, p0b),
        combine_i32x4(q0a, q0b),
        combine_i32x4(q1a, q1b),
        combine_i32x4(q2a, q2b),
        combine_i32x4(q3a, q3b),
        strong0,
        strong1,
        do_p1_0,
        do_p1_1,
        do_q1_0,
        do_q1_1,
        bias_pair4(tc0, tc1),
        _mm256_set1_epi32(maxv),
    );
    store_luma_vertical8x4(
        pix,
        w,
        base0,
        edge_start,
        p3a,
        lo_i32x4(p2n),
        lo_i32x4(p1n),
        lo_i32x4(p0n),
        lo_i32x4(q0n),
        lo_i32x4(q1n),
        lo_i32x4(q2n),
        q3a,
    );
    store_luma_vertical8x4(
        pix,
        w,
        base1,
        edge_start,
        p3b,
        hi_i32x4(p2n),
        hi_i32x4(p1n),
        hi_i32x4(p0n),
        hi_i32x4(q0n),
        hi_i32x4(q1n),
        hi_i32x4(q2n),
        q3b,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_horizontal_plane_avx2(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    scan: usize,
    crow0: usize,
    tc_c: i32,
    maxv_c: i32,
) {
    if cw == 0 || scan + 4 > cw || edge < crow0 + 2 || edge + 1 < crow0 {
        return;
    }
    let q1_row = edge + 1 - crow0;
    if pix.len() < (q1_row + 1).saturating_mul(cw) {
        return;
    }
    unsafe { chroma_horizontal_plane_avx2_impl(pix, cw, edge, scan, crow0, tc_c, maxv_c) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_horizontal_plane_pair_avx2(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    scan: usize,
    crow0: usize,
    tc0: i32,
    tc1: i32,
    maxv_c: i32,
) {
    if cw == 0 || scan + 8 > cw || edge < crow0 + 2 || edge + 1 < crow0 {
        return;
    }
    let q1_row = edge + 1 - crow0;
    if pix.len() < (q1_row + 1).saturating_mul(cw) {
        return;
    }
    unsafe { chroma_horizontal_plane_pair_avx2_impl(pix, cw, edge, scan, crow0, tc0, tc1, maxv_c) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_vertical_plane_avx2(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    s: usize,
    crow0: usize,
    tc_c: i32,
    maxv_c: i32,
) {
    if cw == 0 || edge < 2 || edge + 1 >= cw || s < crow0 {
        return;
    }
    let local_row = s - crow0;
    if pix.len() < (local_row + 4).saturating_mul(cw) {
        return;
    }
    unsafe { chroma_vertical_plane_avx2_impl(pix, cw, edge, s, crow0, tc_c, maxv_c) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_vertical_plane_pair_avx2(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    s: usize,
    crow0: usize,
    tc0: i32,
    tc1: i32,
    maxv_c: i32,
) {
    if cw == 0 || edge < 2 || edge + 1 >= cw || s < crow0 {
        return;
    }
    let local_row = s - crow0;
    if pix.len() < (local_row + 8).saturating_mul(cw) {
        return;
    }
    unsafe { chroma_vertical_plane_pair_avx2_impl(pix, cw, edge, s, crow0, tc0, tc1, maxv_c) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_horizontal_plane_avx2(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    scan: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    if w == 0 || scan + 4 > w || edge < row0 + 4 || edge + 3 < row0 {
        return;
    }
    let q3_row = edge + 3 - row0;
    if pix.len() < (q3_row + 1).saturating_mul(w) {
        return;
    }
    unsafe { luma_horizontal_plane_avx2_impl(pix, w, edge, scan, row0, decision, tc, maxv) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_horizontal_plane_pair_avx2(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    scan: usize,
    row0: usize,
    decision0: LumaDecision,
    tc0: i32,
    decision1: LumaDecision,
    tc1: i32,
    maxv: i32,
) {
    if w == 0 || scan + 8 > w || edge < row0 + 4 || edge + 3 < row0 {
        return;
    }
    let q3_row = edge + 3 - row0;
    if pix.len() < (q3_row + 1).saturating_mul(w) {
        return;
    }
    unsafe {
        luma_horizontal_plane_pair_avx2_impl(
            pix, w, edge, scan, row0, decision0, tc0, decision1, tc1, maxv,
        )
    };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_vertical_plane_avx2(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    s: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    if w == 0 || edge < 4 || edge + 3 >= w || s < row0 {
        return;
    }
    let local_row = s - row0;
    if pix.len() < (local_row + 4).saturating_mul(w) {
        return;
    }
    unsafe { luma_vertical_plane_avx2_impl(pix, w, edge, s, row0, decision, tc, maxv) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_vertical_plane_pair_avx2(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    s: usize,
    row0: usize,
    decision0: LumaDecision,
    tc0: i32,
    decision1: LumaDecision,
    tc1: i32,
    maxv: i32,
) {
    if w == 0 || edge < 4 || edge + 3 >= w || s < row0 {
        return;
    }
    let local_row = s - row0;
    if pix.len() < (local_row + 8).saturating_mul(w) {
        return;
    }
    unsafe {
        luma_vertical_plane_pair_avx2_impl(
            pix, w, edge, s, row0, decision0, tc0, decision1, tc1, maxv,
        )
    };
}
