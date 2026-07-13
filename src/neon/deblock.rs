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

use crate::deblock::{
    LumaDecision, chroma_horizontal_plane_scalar, chroma_vertical_plane_scalar,
    luma_horizontal_plane_scalar, luma_vertical_plane_scalar,
};

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x4(src: &[u16]) -> uint16x4_t {
    debug_assert!(src.len() >= 4);
    unsafe { vld1_u16(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x4(dst: &mut [u16], v: uint16x4_t) {
    debug_assert!(dst.len() >= 4);
    unsafe { vst1_u16(dst.as_mut_ptr(), v) };
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x8(src: &[u16]) -> uint16x8_t {
    debug_assert!(src.len() >= 8);
    unsafe { vld1q_u16(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x8(dst: &mut [u16], v: uint16x8_t) {
    debug_assert!(dst.len() >= 8);
    unsafe { vst1q_u16(dst.as_mut_ptr(), v) };
}

#[inline]
#[target_feature(enable = "neon")]
fn load_chroma_vertical4x4(
    pix: &[u16],
    cw: usize,
    base: usize,
    edge_start: usize,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    debug_assert!(pix.len() >= base + 3 * cw + edge_start + 4);

    let r0 = load_u16x4(&pix[base + edge_start..]);
    let r1 = load_u16x4(&pix[base + cw + edge_start..]);
    let r2 = load_u16x4(&pix[base + 2 * cw + edge_start..]);
    let r3 = load_u16x4(&pix[base + 3 * cw + edge_start..]);

    // Rows are [p1, p0, q0, q1]. Transpose the 4x4 u16 block so the
    // filter receives one vector per column, one lane per row.
    let r01 = vtrn_u16(r0, r1);
    let r23 = vtrn_u16(r2, r3);
    let even = vtrn_u32(vreinterpret_u32_u16(r01.0), vreinterpret_u32_u16(r23.0));
    let odd = vtrn_u32(vreinterpret_u32_u16(r01.1), vreinterpret_u32_u16(r23.1));

    (
        vreinterpretq_s32_u32(vmovl_u16(vreinterpret_u16_u32(even.0))),
        vreinterpretq_s32_u32(vmovl_u16(vreinterpret_u16_u32(odd.0))),
        vreinterpretq_s32_u32(vmovl_u16(vreinterpret_u16_u32(even.1))),
        vreinterpretq_s32_u32(vmovl_u16(vreinterpret_u16_u32(odd.1))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn store_chroma_vertical4x4(
    pix: &mut [u16],
    cw: usize,
    base: usize,
    edge_start: usize,
    p1: int32x4_t,
    p0: int32x4_t,
    q0: int32x4_t,
    q1: int32x4_t,
) {
    debug_assert!(pix.len() >= base + 3 * cw + edge_start + 4);

    let p1 = narrow_u16x4(p1);
    let p0 = narrow_u16x4(p0);
    let q0 = narrow_u16x4(q0);
    let q1 = narrow_u16x4(q1);

    // Inverse transpose. Store four contiguous [p1, p0, q0, q1]
    // rows instead of scalar strided p0/q0 scatter stores. p1/q1 are
    // unchanged but writing them back avoids lane extraction.
    let p = vtrn_u16(p1, p0);
    let q = vtrn_u16(q0, q1);
    let rows02 = vtrn_u32(vreinterpret_u32_u16(p.0), vreinterpret_u32_u16(q.0));
    let rows13 = vtrn_u32(vreinterpret_u32_u16(p.1), vreinterpret_u32_u16(q.1));

    store_u16x4(
        &mut pix[base + edge_start..],
        vreinterpret_u16_u32(rows02.0),
    );
    store_u16x4(
        &mut pix[base + cw + edge_start..],
        vreinterpret_u16_u32(rows13.0),
    );
    store_u16x4(
        &mut pix[base + 2 * cw + edge_start..],
        vreinterpret_u16_u32(rows02.1),
    );
    store_u16x4(
        &mut pix[base + 3 * cw + edge_start..],
        vreinterpret_u16_u32(rows13.1),
    );
}

#[inline]
#[target_feature(enable = "neon")]
fn chroma_filter4_neon(
    p1: int32x4_t,
    p0: int32x4_t,
    q0: int32x4_t,
    q1: int32x4_t,
    tc: int32x4_t,
    zero: int32x4_t,
    maxv: int32x4_t,
) -> (int32x4_t, int32x4_t) {
    let delta = vshrq_n_s32::<3>(vaddq_s32(
        vaddq_s32(vmulq_n_s32(vsubq_s32(q0, p0), 4), vsubq_s32(p1, q1)),
        vdupq_n_s32(4),
    ));
    let delta = vminq_s32(vmaxq_s32(delta, vsubq_s32(zero, tc)), tc);
    let p0n = vminq_s32(vmaxq_s32(vaddq_s32(p0, delta), zero), maxv);
    let q0n = vminq_s32(vmaxq_s32(vsubq_s32(q0, delta), zero), maxv);
    (p0n, q0n)
}

#[inline]
#[target_feature(enable = "neon")]
fn narrow_u16x4(v: int32x4_t) -> uint16x4_t {
    vqmovun_s32(v)
}

#[inline]
#[target_feature(enable = "neon")]
fn chroma_horizontal_plane_neon_impl(
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

    let p1 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[p1_base..])));
    let p0 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[p0_base..])));
    let q0 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[q0_base..])));
    let q1 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[q1_base..])));

    let zero = vdupq_n_s32(0);
    let maxv = vdupq_n_s32(maxv_c);
    let tc = vdupq_n_s32(tc_c);
    let (p0n, q0n) = chroma_filter4_neon(p1, p0, q0, q1, tc, zero, maxv);
    store_u16x4(&mut pix[p0_base..], narrow_u16x4(p0n));
    store_u16x4(&mut pix[q0_base..], narrow_u16x4(q0n));
}

#[inline]
#[target_feature(enable = "neon")]
fn chroma_vertical_plane_neon_impl(
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

    let zero = vdupq_n_s32(0);
    let maxv = vdupq_n_s32(maxv_c);
    let tc = vdupq_n_s32(tc_c);
    let (p0n, q0n) = chroma_filter4_neon(p1, p0, q0, q1, tc, zero, maxv);
    store_chroma_vertical4x4(pix, cw, base, edge_start, p1, p0n, q0n, q1);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_horizontal_plane_neon(
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
    unsafe { chroma_horizontal_plane_neon_impl(pix, cw, edge, scan, crow0, tc_c, maxv_c) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_vertical_plane_neon(
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
    unsafe { chroma_vertical_plane_neon_impl(pix, cw, edge, s, crow0, tc_c, maxv_c) };
}

#[inline]
#[target_feature(enable = "neon")]
fn load_luma_vertical8x4(
    pix: &[u16],
    w: usize,
    base: usize,
    edge_start: usize,
) -> (
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
) {
    debug_assert!(pix.len() >= base + 3 * w + edge_start + 8);
    let r0 = load_u16x8(&pix[base + edge_start..]);
    let r1 = load_u16x8(&pix[base + w + edge_start..]);
    let r2 = load_u16x8(&pix[base + 2 * w + edge_start..]);
    let r3 = load_u16x8(&pix[base + 3 * w + edge_start..]);

    let r01 = vtrnq_u16(r0, r1);
    let r23 = vtrnq_u16(r2, r3);
    let even = vtrnq_u32(vreinterpretq_u32_u16(r01.0), vreinterpretq_u32_u16(r23.0));
    let odd = vtrnq_u32(vreinterpretq_u32_u16(r01.1), vreinterpretq_u32_u16(r23.1));

    let c0 = vget_low_u16(vreinterpretq_u16_u32(even.0));
    let c4 = vget_high_u16(vreinterpretq_u16_u32(even.0));
    let c2 = vget_low_u16(vreinterpretq_u16_u32(even.1));
    let c6 = vget_high_u16(vreinterpretq_u16_u32(even.1));
    let c1 = vget_low_u16(vreinterpretq_u16_u32(odd.0));
    let c5 = vget_high_u16(vreinterpretq_u16_u32(odd.0));
    let c3 = vget_low_u16(vreinterpretq_u16_u32(odd.1));
    let c7 = vget_high_u16(vreinterpretq_u16_u32(odd.1));

    (
        vreinterpretq_s32_u32(vmovl_u16(c0)),
        vreinterpretq_s32_u32(vmovl_u16(c1)),
        vreinterpretq_s32_u32(vmovl_u16(c2)),
        vreinterpretq_s32_u32(vmovl_u16(c3)),
        vreinterpretq_s32_u32(vmovl_u16(c4)),
        vreinterpretq_s32_u32(vmovl_u16(c5)),
        vreinterpretq_s32_u32(vmovl_u16(c6)),
        vreinterpretq_s32_u32(vmovl_u16(c7)),
    )
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn store_luma_vertical8x4(
    pix: &mut [u16],
    w: usize,
    base: usize,
    edge_start: usize,
    c0: int32x4_t,
    c1: int32x4_t,
    c2: int32x4_t,
    c3: int32x4_t,
    c4: int32x4_t,
    c5: int32x4_t,
    c6: int32x4_t,
    c7: int32x4_t,
) {
    debug_assert!(pix.len() >= base + 3 * w + edge_start + 8);

    let c0 = narrow_u16x4(c0);
    let c1 = narrow_u16x4(c1);
    let c2 = narrow_u16x4(c2);
    let c3 = narrow_u16x4(c3);
    let c4 = narrow_u16x4(c4);
    let c5 = narrow_u16x4(c5);
    let c6 = narrow_u16x4(c6);
    let c7 = narrow_u16x4(c7);

    let even0 = vcombine_u16(c0, c4);
    let even1 = vcombine_u16(c2, c6);
    let odd0 = vcombine_u16(c1, c5);
    let odd1 = vcombine_u16(c3, c7);

    let even = vtrnq_u32(vreinterpretq_u32_u16(even0), vreinterpretq_u32_u16(even1));
    let odd = vtrnq_u32(vreinterpretq_u32_u16(odd0), vreinterpretq_u32_u16(odd1));

    let rows01 = vtrnq_u16(vreinterpretq_u16_u32(even.0), vreinterpretq_u16_u32(odd.0));
    let rows23 = vtrnq_u16(vreinterpretq_u16_u32(even.1), vreinterpretq_u16_u32(odd.1));

    store_u16x8(&mut pix[base + edge_start..], rows01.0);
    store_u16x8(&mut pix[base + w + edge_start..], rows01.1);
    store_u16x8(&mut pix[base + 2 * w + edge_start..], rows23.0);
    store_u16x8(&mut pix[base + 3 * w + edge_start..], rows23.1);
}

#[inline]
#[target_feature(enable = "neon")]
fn blend_i32x4(a: int32x4_t, b: int32x4_t, mask: uint32x4_t) -> int32x4_t {
    vreinterpretq_s32_u32(vbslq_u32(
        mask,
        vreinterpretq_u32_s32(b),
        vreinterpretq_u32_s32(a),
    ))
}

#[inline]
#[target_feature(enable = "neon")]
fn clamp_i32x4(v: int32x4_t, lo: int32x4_t, hi: int32x4_t) -> int32x4_t {
    vminq_s32(vmaxq_s32(v, lo), hi)
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
#[inline]
fn decompose(d: LumaDecision) -> (bool, bool, bool) {
    match d {
        LumaDecision::Skip => (false, false, false),
        LumaDecision::Strong => (true, false, false),
        LumaDecision::Weak { do_p1, do_q1 } => (false, do_p1, do_q1),
    }
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn luma_filter4_neon(
    p3: int32x4_t,
    p2: int32x4_t,
    p1: int32x4_t,
    p0: int32x4_t,
    q0: int32x4_t,
    q1: int32x4_t,
    q2: int32x4_t,
    q3: int32x4_t,
    strong_all: bool,
    do_p1: bool,
    do_q1: bool,
    tc: i32,
    maxv: i32,
) -> (
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
) {
    let zero = vdupq_n_s32(0);
    let maxv_v = vdupq_n_s32(maxv);
    let three = vdupq_n_s32(3);

    // Uniform per-segment strong decision (all lanes equal).
    let strong = vdupq_n_u32(if strong_all { u32::MAX } else { 0 });

    // §8.7.2.5.4: each strongly-filtered sample is clipped to ±2·tc around its
    // original value, then to the sample range. `strong_clip(v, orig)` folds both
    // clamps by tightening the [0, maxv] range to [orig−2tc, orig+2tc]. Omitting
    // the ±2·tc clip (scalar `deblock_luma_segment` applies it) over-filters
    // high-contrast edges and diverges from the serial decoder.
    let two_tc = vdupq_n_s32(2 * tc);
    let strong_clip = |v: int32x4_t, orig: int32x4_t| {
        clamp_i32x4(
            v,
            vmaxq_s32(zero, vsubq_s32(orig, two_tc)),
            vminq_s32(maxv_v, vaddq_s32(orig, two_tc)),
        )
    };

    let p0s = strong_clip(
        vshrq_n_s32::<3>(vaddq_s32(
            vaddq_s32(
                vaddq_s32(p2, vaddq_s32(vshlq_n_s32::<1>(p1), vshlq_n_s32::<1>(p0))),
                vaddq_s32(vshlq_n_s32::<1>(q0), q1),
            ),
            vdupq_n_s32(4),
        )),
        p0,
    );
    let p1s = strong_clip(
        vshrq_n_s32::<2>(vaddq_s32(
            vaddq_s32(vaddq_s32(p2, p1), vaddq_s32(p0, q0)),
            vdupq_n_s32(2),
        )),
        p1,
    );
    let p2s = strong_clip(
        vshrq_n_s32::<3>(vaddq_s32(
            vaddq_s32(
                vaddq_s32(vshlq_n_s32::<1>(p3), vmulq_s32(three, p2)),
                vaddq_s32(p1, p0),
            ),
            vaddq_s32(q0, vdupq_n_s32(4)),
        )),
        p2,
    );
    let q0s = strong_clip(
        vshrq_n_s32::<3>(vaddq_s32(
            vaddq_s32(
                vaddq_s32(p1, vshlq_n_s32::<1>(p0)),
                vaddq_s32(vshlq_n_s32::<1>(q0), vshlq_n_s32::<1>(q1)),
            ),
            vaddq_s32(q2, vdupq_n_s32(4)),
        )),
        q0,
    );
    let q1s = strong_clip(
        vshrq_n_s32::<2>(vaddq_s32(
            vaddq_s32(vaddq_s32(p0, q0), vaddq_s32(q1, q2)),
            vdupq_n_s32(2),
        )),
        q1,
    );
    let q2s = strong_clip(
        vshrq_n_s32::<3>(vaddq_s32(
            vaddq_s32(vaddq_s32(p0, q0), vaddq_s32(q1, vmulq_s32(three, q2))),
            vaddq_s32(vshlq_n_s32::<1>(q3), vdupq_n_s32(4)),
        )),
        q2,
    );

    let delta0 = vshrq_n_s32::<4>(vaddq_s32(
        vsubq_s32(
            vmulq_n_s32(vsubq_s32(q0, p0), 9),
            vmulq_n_s32(vsubq_s32(q1, p1), 3),
        ),
        vdupq_n_s32(8),
    ));
    let weak_active = vcltq_s32(vabsq_s32(delta0), vdupq_n_s32(tc * 10));
    let delta = clamp_i32x4(delta0, vdupq_n_s32(-tc), vdupq_n_s32(tc));
    let p0w = blend_i32x4(
        p0,
        clamp_i32x4(vaddq_s32(p0, delta), zero, maxv_v),
        weak_active,
    );
    let q0w = blend_i32x4(
        q0,
        clamp_i32x4(vsubq_s32(q0, delta), zero, maxv_v),
        weak_active,
    );

    let half_tc = tc >> 1;
    let dp1 = clamp_i32x4(
        vshrq_n_s32::<1>(vaddq_s32(
            vsubq_s32(
                vshrq_n_s32::<1>(vaddq_s32(vaddq_s32(p2, p0), vdupq_n_s32(1))),
                p1,
            ),
            delta,
        )),
        vdupq_n_s32(-half_tc),
        vdupq_n_s32(half_tc),
    );
    let p1_mask = if do_p1 { weak_active } else { vdupq_n_u32(0) };
    let p1w = blend_i32x4(p1, clamp_i32x4(vaddq_s32(p1, dp1), zero, maxv_v), p1_mask);

    let dq1 = clamp_i32x4(
        vshrq_n_s32::<1>(vsubq_s32(
            vsubq_s32(
                vshrq_n_s32::<1>(vaddq_s32(vaddq_s32(q2, q0), vdupq_n_s32(1))),
                q1,
            ),
            delta,
        )),
        vdupq_n_s32(-half_tc),
        vdupq_n_s32(half_tc),
    );
    let q1_mask = if do_q1 { weak_active } else { vdupq_n_u32(0) };
    let q1w = blend_i32x4(q1, clamp_i32x4(vaddq_s32(q1, dq1), zero, maxv_v), q1_mask);

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
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn luma_horizontal_plane_neon_impl(
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

    let p3 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[p3_base..])));
    let p2 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[p2_base..])));
    let p1 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[p1_base..])));
    let p0 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[p0_base..])));
    let q0 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[q0_base..])));
    let q1 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[q1_base..])));
    let q2 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[q2_base..])));
    let q3 = vreinterpretq_s32_u32(vmovl_u16(load_u16x4(&pix[q3_base..])));

    let (p0n, p1n, p2n, q0n, q1n, q2n) = luma_filter4_neon(
        p3, p2, p1, p0, q0, q1, q2, q3, strong_all, do_p1, do_q1, tc, maxv,
    );
    store_u16x4(&mut pix[p0_base..], narrow_u16x4(p0n));
    store_u16x4(&mut pix[p1_base..], narrow_u16x4(p1n));
    store_u16x4(&mut pix[p2_base..], narrow_u16x4(p2n));
    store_u16x4(&mut pix[q0_base..], narrow_u16x4(q0n));
    store_u16x4(&mut pix[q1_base..], narrow_u16x4(q1n));
    store_u16x4(&mut pix[q2_base..], narrow_u16x4(q2n));
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn luma_vertical_plane_neon_impl(
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

    let (p0n, p1n, p2n, q0n, q1n, q2n) = luma_filter4_neon(
        p3, p2, p1, p0, q0, q1, q2, q3, strong_all, do_p1, do_q1, tc, maxv,
    );
    store_luma_vertical8x4(
        pix, w, base, edge_start, p3, p2n, p1n, p0n, q0n, q1n, q2n, q3,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_horizontal_plane_neon(
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
    unsafe { luma_horizontal_plane_neon_impl(pix, w, edge, scan, row0, decision, tc, maxv) };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_vertical_plane_neon(
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
    unsafe { luma_vertical_plane_neon_impl(pix, w, edge, s, row0, decision, tc, maxv) };
}
