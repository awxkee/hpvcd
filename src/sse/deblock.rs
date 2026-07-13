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

use crate::deblock::{
    LumaDecision, chroma_horizontal_plane_scalar, chroma_vertical_plane_scalar,
    luma_horizontal_plane_scalar, luma_vertical_plane_scalar,
};

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u16x4(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_loadl_epi64(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x4(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 4);
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast::<__m128i>(), v) };
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u16x8(src: &[u16]) -> __m128i {
    debug_assert!(src.len() >= 8);
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x8(dst: &mut [u16], v: __m128i) {
    debug_assert!(dst.len() >= 8);
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) };
}

#[inline]
#[target_feature(enable = "sse4.1")]
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

    // Rows are [p1, p0, q0, q1]. Transpose the 4x4 u16 block so the
    // filter receives one vector per column, one lane per row.
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
#[target_feature(enable = "sse4.1")]
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

    // Inverse transpose. We store four contiguous [p1, p0, q0, q1]
    // rows instead of scalar strided p0/q0 scatter stores. p1/q1 are
    // unchanged but writing them back is cheaper than lane extraction.
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
#[target_feature(enable = "sse4.1")]
fn chroma_filter4_sse41(
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    tc: __m128i,
    zero: __m128i,
    maxv: __m128i,
) -> (__m128i, __m128i) {
    let four = _mm_set1_epi32(4);
    let delta = _mm_srai_epi32::<3>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_mullo_epi32(_mm_sub_epi32(q0, p0), four),
            _mm_sub_epi32(p1, q1),
        ),
        four,
    ));
    let delta = _mm_min_epi32(_mm_max_epi32(delta, _mm_sub_epi32(zero, tc)), tc);
    let p0n = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(p0, delta), zero), maxv);
    let q0n = _mm_min_epi32(_mm_max_epi32(_mm_sub_epi32(q0, delta), zero), maxv);
    (p0n, q0n)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn chroma_horizontal_plane_sse41_impl(
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

    let p1 = _mm_cvtepu16_epi32(load_u16x4(&pix[p1_base..]));
    let p0 = _mm_cvtepu16_epi32(load_u16x4(&pix[p0_base..]));
    let q0 = _mm_cvtepu16_epi32(load_u16x4(&pix[q0_base..]));
    let q1 = _mm_cvtepu16_epi32(load_u16x4(&pix[q1_base..]));

    let zero = _mm_setzero_si128();
    let maxv = _mm_set1_epi32(maxv_c);
    let tc = _mm_set1_epi32(tc_c);
    let (p0n, q0n) = chroma_filter4_sse41(p1, p0, q0, q1, tc, zero, maxv);
    store_u16x4(&mut pix[p0_base..], _mm_packus_epi32(p0n, zero));
    store_u16x4(&mut pix[q0_base..], _mm_packus_epi32(q0n, zero));
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn chroma_vertical_plane_sse41_impl(
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

    let zero = _mm_setzero_si128();
    let maxv = _mm_set1_epi32(maxv_c);
    let tc = _mm_set1_epi32(tc_c);
    let (p0n, q0n) = chroma_filter4_sse41(p1, p0, q0, q1, tc, zero, maxv);
    store_chroma_vertical4x4(pix, cw, base, edge_start, p1, p0n, q0n, q1);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_horizontal_plane_sse41(
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
        chroma_horizontal_plane_scalar(pix, cw, edge, scan, crow0, tc_c, maxv_c);
        return;
    }
    unsafe { chroma_horizontal_plane_sse41_impl(pix, cw, edge, scan, crow0, tc_c, maxv_c) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_vertical_plane_sse41(
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
        chroma_vertical_plane_scalar(pix, cw, edge, s, crow0, tc_c, maxv_c);
        return;
    }
    unsafe { chroma_vertical_plane_sse41_impl(pix, cw, edge, s, crow0, tc_c, maxv_c) };
}

#[inline]
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
fn interleave_col_pair_sse41(a: __m128i, b: __m128i) -> __m128i {
    let ab = _mm_packus_epi32(a, b);
    _mm_unpacklo_epi16(ab, _mm_srli_si128::<8>(ab))
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
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

    let c01 = interleave_col_pair_sse41(c0, c1);
    let c23 = interleave_col_pair_sse41(c2, c3);
    let c45 = interleave_col_pair_sse41(c4, c5);
    let c67 = interleave_col_pair_sse41(c6, c7);

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
#[target_feature(enable = "sse4.1")]
fn abs_i32x4(v: __m128i) -> __m128i {
    let sign = _mm_srai_epi32::<31>(v);
    _mm_sub_epi32(_mm_xor_si128(v, sign), sign)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn cmplt_i32x4(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epi32(b, a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn clamp_i32x4(v: __m128i, lo: __m128i, hi: __m128i) -> __m128i {
    _mm_min_epi32(_mm_max_epi32(v, lo), hi)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn blend_i32x4(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
    _mm_blendv_epi8(a, b, mask)
}

#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn decompose(d: LumaDecision) -> (bool, bool, bool) {
    match d {
        LumaDecision::Skip => (false, false, false),
        LumaDecision::Strong => (true, false, false),
        LumaDecision::Weak { do_p1, do_q1 } => (false, do_p1, do_q1),
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
fn luma_filter4_sse41(
    p3: __m128i,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    q3: __m128i,
    strong_all: bool,
    do_p1: bool,
    do_q1: bool,
    tc: i32,
    maxv: i32,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    let zero = _mm_setzero_si128();
    let maxv_v = _mm_set1_epi32(maxv);
    let tc_v = _mm_set1_epi32(tc);
    let neg_tc_v = _mm_set1_epi32(-tc);
    let one = _mm_set1_epi32(1);
    let two = _mm_set1_epi32(2);
    let three = _mm_set1_epi32(3);
    let four = _mm_set1_epi32(4);
    let eight = _mm_set1_epi32(8);

    // Uniform per-segment strong decision (splat all lanes).
    let strong = if strong_all { _mm_set1_epi32(-1) } else { zero };

    // §8.7.2.5.4: each strongly-filtered sample is clipped to ±2·tc around its
    // original value, then to the sample range. `strong_clip(v, orig)` folds both
    // clamps by tightening [0, maxv] to [orig−2tc, orig+2tc]. Omitting the ±2·tc
    // clip (scalar `deblock_luma_segment` applies it) over-filters high-contrast
    // edges and diverges from the serial decoder.
    let two_tc = _mm_set1_epi32(2 * tc);
    let strong_clip = |v: __m128i, orig: __m128i| {
        clamp_i32x4(
            v,
            _mm_max_epi32(zero, _mm_sub_epi32(orig, two_tc)),
            _mm_min_epi32(maxv_v, _mm_add_epi32(orig, two_tc)),
        )
    };

    let p0s = strong_clip(
        _mm_srai_epi32::<3>(_mm_add_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    p2,
                    _mm_add_epi32(_mm_slli_epi32::<1>(p1), _mm_slli_epi32::<1>(p0)),
                ),
                _mm_add_epi32(_mm_slli_epi32::<1>(q0), q1),
            ),
            four,
        )),
        p0,
    );
    let p1s = strong_clip(
        _mm_srai_epi32::<2>(_mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(p2, p1), _mm_add_epi32(p0, q0)),
            two,
        )),
        p1,
    );
    let p2s = strong_clip(
        _mm_srai_epi32::<3>(_mm_add_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_slli_epi32::<1>(p3), _mm_mullo_epi32(three, p2)),
                _mm_add_epi32(p1, p0),
            ),
            _mm_add_epi32(q0, four),
        )),
        p2,
    );
    let q0s = strong_clip(
        _mm_srai_epi32::<3>(_mm_add_epi32(
            _mm_add_epi32(
                _mm_add_epi32(p1, _mm_slli_epi32::<1>(p0)),
                _mm_add_epi32(_mm_slli_epi32::<1>(q0), _mm_slli_epi32::<1>(q1)),
            ),
            _mm_add_epi32(q2, four),
        )),
        q0,
    );
    let q1s = strong_clip(
        _mm_srai_epi32::<2>(_mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(p0, q0), _mm_add_epi32(q1, q2)),
            two,
        )),
        q1,
    );
    let q2s = strong_clip(
        _mm_srai_epi32::<3>(_mm_add_epi32(
            _mm_add_epi32(
                _mm_add_epi32(p0, q0),
                _mm_add_epi32(q1, _mm_mullo_epi32(three, q2)),
            ),
            _mm_add_epi32(_mm_slli_epi32::<1>(q3), four),
        )),
        q2,
    );

    // Weak filter (§8.7.2.5.7). delta0 unclamped; a line is filtered only when
    // |delta0| < 10*tc (per-line mask). p1/q1 updates gated by uniform do_p1/do_q1.
    let delta0 = _mm_srai_epi32::<4>(_mm_add_epi32(
        _mm_sub_epi32(
            _mm_mullo_epi32(_mm_set1_epi32(9), _mm_sub_epi32(q0, p0)),
            _mm_mullo_epi32(three, _mm_sub_epi32(q1, p1)),
        ),
        eight,
    ));
    let weak_active = cmplt_i32x4(abs_i32x4(delta0), _mm_set1_epi32(tc * 10));
    let delta = clamp_i32x4(delta0, neg_tc_v, tc_v);
    let p0w = blend_i32x4(
        p0,
        clamp_i32x4(_mm_add_epi32(p0, delta), zero, maxv_v),
        weak_active,
    );
    let q0w = blend_i32x4(
        q0,
        clamp_i32x4(_mm_sub_epi32(q0, delta), zero, maxv_v),
        weak_active,
    );

    let half_tc = tc >> 1;
    let neg_half_tc_v = _mm_set1_epi32(-half_tc);
    let half_tc_v = _mm_set1_epi32(half_tc);
    // delta>>1 (arithmetic), matches scalar (dp1 uses +delta then >>1). Here the
    // scalar computes ((p2+p0+1)>>1 - p1 + delta) >> 1, so keep the +delta inside.
    let dp1 = clamp_i32x4(
        _mm_srai_epi32::<1>(_mm_add_epi32(
            _mm_sub_epi32(
                _mm_srai_epi32::<1>(_mm_add_epi32(_mm_add_epi32(p2, p0), one)),
                p1,
            ),
            delta,
        )),
        neg_half_tc_v,
        half_tc_v,
    );
    let p1_mask = if do_p1 { weak_active } else { zero };
    let p1w = blend_i32x4(
        p1,
        clamp_i32x4(_mm_add_epi32(p1, dp1), zero, maxv_v),
        p1_mask,
    );

    let dq1 = clamp_i32x4(
        _mm_srai_epi32::<1>(_mm_sub_epi32(
            _mm_sub_epi32(
                _mm_srai_epi32::<1>(_mm_add_epi32(_mm_add_epi32(q2, q0), one)),
                q1,
            ),
            delta,
        )),
        neg_half_tc_v,
        half_tc_v,
    );
    let q1_mask = if do_q1 { weak_active } else { zero };
    let q1w = blend_i32x4(
        q1,
        clamp_i32x4(_mm_add_epi32(q1, dq1), zero, maxv_v),
        q1_mask,
    );

    (
        blend_i32x4(p0w, p0s, strong),
        blend_i32x4(p1w, p1s, strong),
        blend_i32x4(p2, p2s, strong),
        blend_i32x4(q0w, q0s, strong),
        blend_i32x4(q1w, q1s, strong),
        blend_i32x4(q2, q2s, strong),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
fn luma_horizontal_plane_sse41_impl(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    scan: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    let (strong_all, do_p1, do_q1) = decompose(decision);
    let p3_base = (edge - 4 - row0) * w + scan;
    let p2_base = (edge - 3 - row0) * w + scan;
    let p1_base = (edge - 2 - row0) * w + scan;
    let p0_base = (edge - 1 - row0) * w + scan;
    let q0_base = (edge - row0) * w + scan;
    let q1_base = (edge + 1 - row0) * w + scan;
    let q2_base = (edge + 2 - row0) * w + scan;
    let q3_base = (edge + 3 - row0) * w + scan;

    let p3 = _mm_cvtepu16_epi32(load_u16x4(&pix[p3_base..]));
    let p2 = _mm_cvtepu16_epi32(load_u16x4(&pix[p2_base..]));
    let p1 = _mm_cvtepu16_epi32(load_u16x4(&pix[p1_base..]));
    let p0 = _mm_cvtepu16_epi32(load_u16x4(&pix[p0_base..]));
    let q0 = _mm_cvtepu16_epi32(load_u16x4(&pix[q0_base..]));
    let q1 = _mm_cvtepu16_epi32(load_u16x4(&pix[q1_base..]));
    let q2 = _mm_cvtepu16_epi32(load_u16x4(&pix[q2_base..]));
    let q3 = _mm_cvtepu16_epi32(load_u16x4(&pix[q3_base..]));

    let (p0n, p1n, p2n, q0n, q1n, q2n) = luma_filter4_sse41(
        p3, p2, p1, p0, q0, q1, q2, q3, strong_all, do_p1, do_q1, tc, maxv,
    );
    let zero = _mm_setzero_si128();
    store_u16x4(&mut pix[p0_base..], _mm_packus_epi32(p0n, zero));
    store_u16x4(&mut pix[p1_base..], _mm_packus_epi32(p1n, zero));
    store_u16x4(&mut pix[p2_base..], _mm_packus_epi32(p2n, zero));
    store_u16x4(&mut pix[q0_base..], _mm_packus_epi32(q0n, zero));
    store_u16x4(&mut pix[q1_base..], _mm_packus_epi32(q1n, zero));
    store_u16x4(&mut pix[q2_base..], _mm_packus_epi32(q2n, zero));
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
fn luma_vertical_plane_sse41_impl(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    s: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    let (strong_all, do_p1, do_q1) = decompose(decision);
    let base = (s - row0) * w;
    let edge_start = edge - 4;
    let (p3, p2, p1, p0, q0, q1, q2, q3) = load_luma_vertical8x4(pix, w, base, edge_start);

    let (p0n, p1n, p2n, q0n, q1n, q2n) = luma_filter4_sse41(
        p3, p2, p1, p0, q0, q1, q2, q3, strong_all, do_p1, do_q1, tc, maxv,
    );
    store_luma_vertical8x4(
        pix, w, base, edge_start, p3, p2n, p1n, p0n, q0n, q1n, q2n, q3,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_horizontal_plane_sse41(
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
        luma_horizontal_plane_scalar(pix, w, edge, scan, row0, decision, tc, maxv);
        return;
    }
    unsafe { luma_horizontal_plane_sse41_impl(pix, w, edge, scan, row0, decision, tc, maxv) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_vertical_plane_sse41(
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
        luma_vertical_plane_scalar(pix, w, edge, s, row0, decision, tc, maxv);
        return;
    }
    unsafe { luma_vertical_plane_sse41_impl(pix, w, edge, s, row0, decision, tc, maxv) };
}
