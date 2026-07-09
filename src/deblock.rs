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

//! Parallel in-loop deblocking (HEVC §8.7.2), split from the serial version in
//! `decode.rs`. The picture is filtered in two ordered passes — all vertical
//! edges, then all horizontal — for luma and then chroma. Each pass is a pure
//! function of the buffer state at the pass's start (no edge reads another
//! edge's writes within the same pass, because edges are 8 px apart while a
//! filter touches at most ±4 px), which is what lets a pass run in parallel.
//!
//! Decomposition: each pass is split into CTB-aligned row bands.
//!   * Vertical pass — every pixel access for row `s` is `s*w + …`, so row
//!     bands are perfectly disjoint with no halo. CTB alignment (a multiple of
//!     both 4 and 8) keeps each 4-row filter segment inside one band, so the
//!     joint `d_total` decision is identical to the serial path.
//!   * Horizontal pass — `horiz_bands` places each band boundary in the row gap
//!     between neighboring horizontal filters. Every edge's read/write footprint
//!     is therefore inside exactly one band, so the pass runs in place without a
//!     whole-plane snapshot.

use crate::exec::ExecContext;
use crate::threadpool::{DisjointMut, ThreadPool, parallel_for};

#[rustfmt::skip]
static BETA: [i32; 52] = [
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 6, 7, 8, 9,
    10,11,12,13,14,15,16,17,18,20,
    22,24,26,28,30,32,34,36,38,40,
    42,44,46,48,50,52,54,56,58,60,
    62,64,
];
#[rustfmt::skip]
static TC: [i32; 54] = [
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,
    1,1,1,1,1,1,1,2,2,2,
    2,3,3,3,3,4,4,4,5,5,
    6,6,7,8,9,10,11,13,14,16,
    18,20,22,24,
];

/// Immutable geometry, offsets and lookup tables shared by every band worker.
pub(crate) struct DeblockCtx<'a> {
    pub exec: ExecContext,
    pub w: usize,
    pub h: usize,
    pub cw: usize,
    pub ch: usize,
    pub gw: usize,
    pub gh: usize,
    pub sub_w: usize,
    pub sub_h: usize,
    pub bd: u8,
    pub bd_c: u8,
    pub beta_offset: i32,
    pub tc_offset: i32,
    pub qp_bd_offset_y: i32,
    pub qp_bd_offset_c: i32,
    pub default_qp: i16,
    pub qp_y_map: &'a [i16],
    pub tqb: &'a [bool],
}

impl DeblockCtx<'_> {
    #[inline]
    fn qp_at(&self, px: usize, py: usize) -> i32 {
        if self.qp_y_map.is_empty() || self.gw == 0 || self.gh == 0 {
            return self.default_qp as i32;
        }
        let gx = (px / 4).min(self.gw.saturating_sub(1));
        let gy = (py / 4).min(self.gh.saturating_sub(1));
        gy.checked_mul(self.gw)
            .and_then(|base| base.checked_add(gx))
            .and_then(|idx| self.qp_y_map.get(idx))
            .copied()
            .unwrap_or(self.default_qp) as i32
    }

    #[inline]
    fn tqb_at(&self, px: usize, py: usize) -> bool {
        if px >= self.w || py >= self.h {
            return false;
        }
        let idx = (py / 4) * self.gw + (px / 4);
        self.tqb.get(idx).copied().unwrap_or(false)
    }
}

#[inline]
fn luma_vertical_segment_params(
    ctx: &DeblockCtx<'_>,
    y: &[u16],
    w: usize,
    row0: usize,
    row1: usize,
    edge: usize,
    s: usize,
) -> Option<(i32, i32)> {
    let mid = s + 1;
    let qp_p = ctx.qp_at(edge - 1, mid);
    let qp_q = ctx.qp_at(edge, mid);
    if ctx.tqb_at(edge - 1, mid) || ctx.tqb_at(edge, mid) {
        return None;
    }
    let avg_qp = (qp_p + qp_q + 1) >> 1;
    let beta_prime = (avg_qp + ctx.qp_bd_offset_y + ctx.beta_offset).clamp(0, 51);
    let tc_prime = (avg_qp + ctx.qp_bd_offset_y + 2 + ctx.tc_offset).clamp(0, 53);
    let beta = BETA[beta_prime as usize];
    let tc = TC[tc_prime as usize];
    if tc == 0 {
        return None;
    }

    let seg_end = (s + 4).min(row1);
    let mut d_total = 0i32;
    for r in s..seg_end {
        let lr = r - row0;
        let p = |o: usize| y[lr * w + edge - 1 - o] as i32;
        let q = |o: usize| y[lr * w + edge + o] as i32;
        d_total += (p(2) - 2 * p(1) + p(0)).abs() + (q(0) - 2 * q(1) + q(2)).abs();
    }
    (d_total < beta).then_some((beta, tc))
}

#[inline]
fn luma_horizontal_segment_params(
    ctx: &DeblockCtx<'_>,
    y: &[u16],
    w: usize,
    row0: usize,
    edge: usize,
    scan: usize,
) -> Option<(i32, i32)> {
    let mid = scan + 1;
    let qp_p = ctx.qp_at(mid, edge - 1);
    let qp_q = ctx.qp_at(mid, edge);
    if ctx.tqb_at(mid, edge - 1) || ctx.tqb_at(mid, edge) {
        return None;
    }
    let avg_qp = (qp_p + qp_q + 1) >> 1;
    let beta_prime = (avg_qp + ctx.qp_bd_offset_y + ctx.beta_offset).clamp(0, 51);
    let tc_prime = (avg_qp + ctx.qp_bd_offset_y + 2 + ctx.tc_offset).clamp(0, 53);
    let beta = BETA[beta_prime as usize];
    let tc = TC[tc_prime as usize];
    if tc == 0 {
        return None;
    }

    let at = |buf: &[u16], gy: usize, x: usize| -> i32 { buf[(gy - row0) * w + x] as i32 };
    let mut d_total = 0i32;
    for c in scan..scan + 4 {
        let p = |o: usize| at(y, edge - 1 - o, c);
        let q = |o: usize| at(y, edge + o, c);
        d_total += (p(2) - 2 * p(1) + p(0)).abs() + (q(0) - 2 * q(1) + q(2)).abs();
    }
    (d_total < beta).then_some((beta, tc))
}

/// Luma vertical edges for the global rows `[row0, row1)`. Writes and reads only
/// touch those rows, so `y` may be a band slice whose local row 0 is `row0`.
fn luma_vertical(ctx: &DeblockCtx<'_>, y: &mut [u16], row0: usize, row1: usize) {
    let w = ctx.w;
    let maxv = (1i32 << ctx.bd) - 1;
    let filter = ctx.exec.luma_deblock_vertical;
    let pair_filter = ctx.exec.luma_deblock_vertical_pair;
    let last_full_edge = w.saturating_sub(4);
    let mut edge = 8;
    while edge <= last_full_edge {
        let mut s = row0;
        while s + 4 <= row1 {
            if let Some(pair_filter) = pair_filter
                && s + 8 <= row1
            {
                let first = luma_vertical_segment_params(ctx, y, w, row0, row1, edge, s);
                let second = luma_vertical_segment_params(ctx, y, w, row0, row1, edge, s + 4);
                match (first, second) {
                    (Some((beta0, tc0)), Some((beta1, tc1))) => {
                        pair_filter(y, w, edge, s, row0, beta0, tc0, beta1, tc1, maxv);
                        s += 8;
                        continue;
                    }
                    (None, _) => {
                        s += 4;
                        continue;
                    }
                    (Some((beta, tc)), None) => {
                        filter(y, w, edge, s, row0, beta, tc, maxv);
                        s += 4;
                        continue;
                    }
                }
            }

            if let Some((beta, tc)) = luma_vertical_segment_params(ctx, y, w, row0, row1, edge, s) {
                filter(y, w, edge, s, row0, beta, tc, maxv);
            }
            s += 4;
        }
        edge += 8;
    }
}

/// Luma horizontal edges writing into global rows `[row0, row1)`. The band is
/// laid out with local row 0 == `row0`. Horizontal edges are 8 samples apart and
/// their write spans do not overlap, so the pass is safe in-place; every row the
/// filter reads for an edge inside this band is also inside this band because
/// `horiz_bands` places boundaries at the row gap between neighboring edges.
fn luma_horizontal(ctx: &DeblockCtx<'_>, y: &mut [u16], row0: usize, row1: usize) {
    let w = ctx.w;
    let h = ctx.h;
    let maxv = (1i32 << ctx.bd) - 1;
    let filter = ctx.exec.luma_deblock_horizontal;
    let pair_filter = ctx.exec.luma_deblock_horizontal_pair;
    let last_full_edge = h.saturating_sub(4);
    let mut edge = (row0.div_ceil(8) * 8).max(8);
    while edge <= last_full_edge {
        if edge >= row1 {
            break;
        }
        let mut scan = 0;
        while scan + 4 <= w {
            if let Some(pair_filter) = pair_filter
                && scan + 8 <= w
            {
                let first = luma_horizontal_segment_params(ctx, y, w, row0, edge, scan);
                let second = luma_horizontal_segment_params(ctx, y, w, row0, edge, scan + 4);
                match (first, second) {
                    (Some((beta0, tc0)), Some((beta1, tc1))) => {
                        pair_filter(y, w, edge, scan, row0, beta0, tc0, beta1, tc1, maxv);
                        scan += 8;
                        continue;
                    }
                    (None, _) => {
                        scan += 4;
                        continue;
                    }
                    (Some((beta, tc)), None) => {
                        filter(y, w, edge, scan, row0, beta, tc, maxv);
                        scan += 4;
                        continue;
                    }
                }
            }

            if let Some((beta, tc)) = luma_horizontal_segment_params(ctx, y, w, row0, edge, scan) {
                filter(y, w, edge, scan, row0, beta, tc, maxv);
            }
            scan += 4;
        }
        edge += 8;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) type LumaDeblockPlaneFn = fn(&mut [u16], usize, usize, usize, usize, i32, i32, i32);
#[allow(clippy::too_many_arguments)]
pub(crate) type LumaDeblockPairFn =
    fn(&mut [u16], usize, usize, usize, usize, i32, i32, i32, i32, i32);

static LUMA_VERTICAL_PLANE: std::sync::OnceLock<LumaDeblockPlaneFn> = std::sync::OnceLock::new();
static LUMA_HORIZONTAL_PLANE: std::sync::OnceLock<LumaDeblockPlaneFn> = std::sync::OnceLock::new();
static LUMA_VERTICAL_PAIR: std::sync::OnceLock<Option<LumaDeblockPairFn>> =
    std::sync::OnceLock::new();
static LUMA_HORIZONTAL_PAIR: std::sync::OnceLock<Option<LumaDeblockPairFn>> =
    std::sync::OnceLock::new();

#[inline]
pub(crate) fn resolve_luma_vertical_plane() -> LumaDeblockPlaneFn {
    *LUMA_VERTICAL_PLANE.get_or_init(|| {
        let mut _f: LumaDeblockPlaneFn = luma_vertical_plane_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::luma_vertical_plane_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::luma_vertical_plane_sse41;
            }
        }

        #[cfg(all(feature = "sse", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::luma_vertical_plane_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_luma_vertical_pair() -> Option<LumaDeblockPairFn> {
    *LUMA_VERTICAL_PAIR.get_or_init(|| {
        let mut _f: Option<LumaDeblockPairFn> = None;

        #[cfg(all(feature = "sse", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::luma_vertical_plane_pair_avx2);
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_luma_horizontal_plane() -> LumaDeblockPlaneFn {
    *LUMA_HORIZONTAL_PLANE.get_or_init(|| {
        let mut _f: LumaDeblockPlaneFn = luma_horizontal_plane_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::luma_horizontal_plane_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::luma_horizontal_plane_sse41;
            }
        }

        #[cfg(all(feature = "sse", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::luma_horizontal_plane_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_luma_horizontal_pair() -> Option<LumaDeblockPairFn> {
    *LUMA_HORIZONTAL_PAIR.get_or_init(|| {
        let mut _f: Option<LumaDeblockPairFn> = None;

        #[cfg(all(feature = "sse", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::luma_horizontal_plane_pair_avx2);
            }
        }

        _f
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_vertical_plane_scalar(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    s: usize,
    row0: usize,
    beta: i32,
    tc: i32,
    maxv: i32,
) {
    for r in s..s + 4 {
        let lr = r - row0;
        let base_p = lr * w + edge - 1;
        let base_q = lr * w + edge;
        let (p0, p1, p2, p3) = (
            pix[base_p] as i32,
            pix[base_p - 1] as i32,
            pix[base_p - 2] as i32,
            pix[base_p - 3] as i32,
        );
        let (q0, q1, q2, q3) = (
            pix[base_q] as i32,
            pix[base_q + 1] as i32,
            pix[base_q + 2] as i32,
            pix[base_q + 3] as i32,
        );
        luma_filter_lane_vertical(
            pix, w, lr, edge, p0, p1, p2, p3, q0, q1, q2, q3, beta, tc, maxv,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_horizontal_plane_scalar(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    scan: usize,
    row0: usize,
    beta: i32,
    tc: i32,
    maxv: i32,
) {
    for c in scan..scan + 4 {
        let at = |buf: &[u16], gy: usize, x: usize| -> i32 { buf[(gy - row0) * w + x] as i32 };
        let (p0, p1, p2, p3) = (
            at(pix, edge - 1, c),
            at(pix, edge - 2, c),
            at(pix, edge - 3, c),
            at(pix, edge - 4, c),
        );
        let (q0, q1, q2, q3) = (
            at(pix, edge, c),
            at(pix, edge + 1, c),
            at(pix, edge + 2, c),
            at(pix, edge + 3, c),
        );
        luma_filter_lane_horizontal(
            pix, w, edge, c, row0, p0, p1, p2, p3, q0, q1, q2, q3, beta, tc, maxv,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn luma_filter_lane_vertical(
    pix: &mut [u16],
    w: usize,
    lr: usize,
    edge: usize,
    p0: i32,
    p1: i32,
    p2: i32,
    p3: i32,
    q0: i32,
    q1: i32,
    q2: i32,
    q3: i32,
    beta: i32,
    tc: i32,
    maxv: i32,
) {
    let base_p = lr * w + edge - 1;
    let base_q = lr * w + edge;
    let put = |dst: &mut [u16], idx: usize, val: i32| {
        dst[idx] = val.clamp(0, maxv) as u16;
    };
    let dp = (p2 - 2 * p1 + p0).abs();
    let dq = (q2 - 2 * q1 + q0).abs();
    let d = dp + dq;
    let strong = d < (beta >> 2)
        && (p0 - q0).abs() < (5 * tc + 1) >> 1
        && (p3 - p0).abs() + (q0 - q3).abs() < (beta * 3) >> 3;
    if strong {
        put(pix, base_p, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
        put(pix, base_p - 1, (p2 + p1 + p0 + q0 + 2) >> 2);
        put(pix, base_p - 2, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
        put(pix, base_q, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
        put(pix, base_q + 1, (p0 + q0 + q1 + q2 + 2) >> 2);
        put(pix, base_q + 2, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3);
    } else {
        let delta = ((9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4).clamp(-tc, tc);
        put(pix, base_p, p0 + delta);
        put(pix, base_q, q0 - delta);
        let thres = (tc * 10 + 1) >> 1;
        if (2 * (p0 - p1) - delta).abs() < thres {
            let dp1 = (((p2 + p0 + 1) >> 1) - p1 + (delta >> 1)).clamp(-(tc >> 1), tc >> 1);
            put(pix, base_p - 1, p1 + dp1);
        }
        if (2 * (q0 - q1) + delta).abs() < thres {
            let dq1 = (((q2 + q0 + 1) >> 1) - q1 - (delta >> 1)).clamp(-(tc >> 1), tc >> 1);
            put(pix, base_q + 1, q1 + dq1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn luma_filter_lane_horizontal(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    c: usize,
    row0: usize,
    p0: i32,
    p1: i32,
    p2: i32,
    p3: i32,
    q0: i32,
    q1: i32,
    q2: i32,
    q3: i32,
    beta: i32,
    tc: i32,
    maxv: i32,
) {
    let put = |dst: &mut [u16], gy: usize, val: i32| {
        dst[(gy - row0) * w + c] = val.clamp(0, maxv) as u16;
    };
    let dp = (p2 - 2 * p1 + p0).abs();
    let dq = (q2 - 2 * q1 + q0).abs();
    let d = dp + dq;
    let strong = d < (beta >> 2)
        && (p0 - q0).abs() < (5 * tc + 1) >> 1
        && (p3 - p0).abs() + (q0 - q3).abs() < (beta * 3) >> 3;
    if strong {
        put(pix, edge - 1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
        put(pix, edge - 2, (p2 + p1 + p0 + q0 + 2) >> 2);
        put(pix, edge - 3, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
        put(pix, edge, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
        put(pix, edge + 1, (p0 + q0 + q1 + q2 + 2) >> 2);
        put(pix, edge + 2, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3);
    } else {
        let delta = ((9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4).clamp(-tc, tc);
        put(pix, edge - 1, p0 + delta);
        put(pix, edge, q0 - delta);
        let thres = (tc * 10 + 1) >> 1;
        if (2 * (p0 - p1) - delta).abs() < thres {
            let dp1 = (((p2 + p0 + 1) >> 1) - p1 + (delta >> 1)).clamp(-(tc >> 1), tc >> 1);
            put(pix, edge - 2, p1 + dp1);
        }
        if (2 * (q0 - q1) + delta).abs() < thres {
            let dq1 = (((q2 + q0 + 1) >> 1) - q1 - (delta >> 1)).clamp(-(tc >> 1), tc >> 1);
            put(pix, edge + 1, q1 + dq1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) type ChromaDeblockPlaneFn = fn(&mut [u16], usize, usize, usize, usize, i32, i32);
#[allow(clippy::too_many_arguments)]
pub(crate) type ChromaDeblockPairFn = fn(&mut [u16], usize, usize, usize, usize, i32, i32, i32);

static CHROMA_VERTICAL_PLANE: std::sync::OnceLock<ChromaDeblockPlaneFn> =
    std::sync::OnceLock::new();
static CHROMA_HORIZONTAL_PLANE: std::sync::OnceLock<ChromaDeblockPlaneFn> =
    std::sync::OnceLock::new();
static CHROMA_VERTICAL_PAIR: std::sync::OnceLock<Option<ChromaDeblockPairFn>> =
    std::sync::OnceLock::new();
static CHROMA_HORIZONTAL_PAIR: std::sync::OnceLock<Option<ChromaDeblockPairFn>> =
    std::sync::OnceLock::new();

#[inline]
pub(crate) fn resolve_chroma_vertical_plane() -> ChromaDeblockPlaneFn {
    *CHROMA_VERTICAL_PLANE.get_or_init(|| {
        let mut _f: ChromaDeblockPlaneFn = chroma_vertical_plane_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::chroma_vertical_plane_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::chroma_vertical_plane_sse41;
            }
        }

        #[cfg(all(feature = "sse", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::chroma_vertical_plane_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_chroma_vertical_pair() -> Option<ChromaDeblockPairFn> {
    *CHROMA_VERTICAL_PAIR.get_or_init(|| {
        let mut _f: Option<ChromaDeblockPairFn> = None;

        #[cfg(all(feature = "sse", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::chroma_vertical_plane_pair_avx2);
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_chroma_horizontal_plane() -> ChromaDeblockPlaneFn {
    *CHROMA_HORIZONTAL_PLANE.get_or_init(|| {
        let mut _f: ChromaDeblockPlaneFn = chroma_horizontal_plane_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::chroma_horizontal_plane_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::chroma_horizontal_plane_sse41;
            }
        }

        #[cfg(all(feature = "sse", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::chroma_horizontal_plane_avx2;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_chroma_horizontal_pair() -> Option<ChromaDeblockPairFn> {
    *CHROMA_HORIZONTAL_PAIR.get_or_init(|| {
        let mut _f: Option<ChromaDeblockPairFn> = None;

        #[cfg(all(feature = "sse", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::chroma_horizontal_plane_pair_avx2);
            }
        }

        _f
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_vertical_plane_scalar(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    s: usize,
    crow0: usize,
    tc_c: i32,
    maxv_c: i32,
) {
    for r in s..s + 4 {
        let lr = r - crow0;
        let p0 = pix[lr * cw + edge - 1] as i32;
        let p1 = pix[lr * cw + edge - 2] as i32;
        let q0 = pix[lr * cw + edge] as i32;
        let q1 = pix[lr * cw + edge + 1] as i32;
        let delta = (((q0 - p0) * 4 + p1 - q1 + 4) >> 3).clamp(-tc_c, tc_c);
        if delta != 0 {
            pix[lr * cw + edge - 1] = (p0 + delta).clamp(0, maxv_c) as u16;
            pix[lr * cw + edge] = (q0 - delta).clamp(0, maxv_c) as u16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_horizontal_plane_scalar(
    pix: &mut [u16],
    cw: usize,
    edge: usize,
    scan: usize,
    crow0: usize,
    tc_c: i32,
    maxv_c: i32,
) {
    for c in scan..scan + 4 {
        let base = |gy: usize| (gy - crow0) * cw + c;
        let p0 = pix[base(edge - 1)] as i32;
        let p1 = pix[base(edge - 2)] as i32;
        let q0 = pix[base(edge)] as i32;
        let q1 = pix[base(edge + 1)] as i32;
        let delta = (((q0 - p0) * 4 + p1 - q1 + 4) >> 3).clamp(-tc_c, tc_c);
        if delta != 0 {
            pix[base(edge - 1)] = (p0 + delta).clamp(0, maxv_c) as u16;
            pix[base(edge)] = (q0 - delta).clamp(0, maxv_c) as u16;
        }
    }
}

#[inline]
fn chroma_vertical_segment_tc(ctx: &DeblockCtx<'_>, edge: usize, s: usize) -> Option<i32> {
    let mid = s + 1;
    let qlx = edge * ctx.sub_w;
    let qly = mid * ctx.sub_h;
    let avg_qp_l = ctx.qp_at(qlx.min(ctx.w - 1), qly.min(ctx.h - 1));
    let tc_prime_c = (avg_qp_l + ctx.qp_bd_offset_c + 2 + ctx.tc_offset).clamp(0, 53);
    let tc_c = TC[tc_prime_c as usize];
    if tc_c == 0 {
        return None;
    }
    let px_p = (edge - 1) * ctx.sub_w;
    let py_p = mid * ctx.sub_h;
    let px_q = edge * ctx.sub_w;
    let py_q = mid * ctx.sub_h;
    if ctx.tqb_at(px_p, py_p) || ctx.tqb_at(px_q, py_q) {
        return None;
    }
    Some(tc_c)
}

#[inline]
fn chroma_horizontal_segment_tc(ctx: &DeblockCtx<'_>, edge: usize, scan: usize) -> Option<i32> {
    let mid = scan + 1;
    let qlx = mid * ctx.sub_w;
    let qly = edge * ctx.sub_h;
    let avg_qp_l = ctx.qp_at(qlx.min(ctx.w - 1), qly.min(ctx.h - 1));
    let tc_prime_c = (avg_qp_l + ctx.qp_bd_offset_c + 2 + ctx.tc_offset).clamp(0, 53);
    let tc_c = TC[tc_prime_c as usize];
    if tc_c == 0 {
        return None;
    }
    let px_p = mid * ctx.sub_w;
    let py_p = (edge - 1) * ctx.sub_h;
    let px_q = mid * ctx.sub_w;
    let py_q = edge * ctx.sub_h;
    if ctx.tqb_at(px_p, py_p) || ctx.tqb_at(px_q, py_q) {
        return None;
    }
    Some(tc_c)
}

/// Chroma vertical edges for chroma rows `[crow0, crow1)`. Like luma vertical,
/// every access for chroma row `s` is `s*cw + …`, so chroma-row bands are
/// disjoint with no halo. Filters both Cb and Cr.
fn chroma_vertical(
    ctx: &DeblockCtx<'_>,
    cb: &mut [u16],
    cr: &mut [u16],
    crow0: usize,
    crow1: usize,
) {
    let cw = ctx.cw;
    let maxv_c = (1i32 << ctx.bd_c) - 1;
    let filter = ctx.exec.chroma_deblock_vertical;
    let pair_filter = ctx.exec.chroma_deblock_vertical_pair;
    let last_full_chroma_edge = cw.saturating_sub(2);
    let mut edge = 8;
    while edge <= last_full_chroma_edge {
        let mut s = crow0;
        while s + 4 <= crow1 {
            if let Some(pair_filter) = pair_filter
                && s + 8 <= crow1
            {
                let first = chroma_vertical_segment_tc(ctx, edge, s);
                let second = chroma_vertical_segment_tc(ctx, edge, s + 4);
                match (first, second) {
                    (Some(tc0), Some(tc1)) => {
                        pair_filter(cb, cw, edge, s, crow0, tc0, tc1, maxv_c);
                        pair_filter(cr, cw, edge, s, crow0, tc0, tc1, maxv_c);
                        s += 8;
                        continue;
                    }
                    (None, _) => {
                        s += 4;
                        continue;
                    }
                    (Some(tc_c), None) => {
                        filter(cb, cw, edge, s, crow0, tc_c, maxv_c);
                        filter(cr, cw, edge, s, crow0, tc_c, maxv_c);
                        s += 4;
                        continue;
                    }
                }
            }

            if let Some(tc_c) = chroma_vertical_segment_tc(ctx, edge, s) {
                filter(cb, cw, edge, s, crow0, tc_c, maxv_c);
                filter(cr, cw, edge, s, crow0, tc_c, maxv_c);
            }
            s += 4;
        }
        edge += 8;
    }
}

/// Chroma horizontal edges writing chroma rows `[crow0, crow1)`. Like the
/// luma horizontal pass, this is safe in-place because neighboring horizontal
/// chroma edges are 8 samples apart and write only `[e-1, e]`.
fn chroma_horizontal(
    ctx: &DeblockCtx<'_>,
    cb: &mut [u16],
    cr: &mut [u16],
    crow0: usize,
    crow1: usize,
) {
    let cw = ctx.cw;
    let ch = ctx.ch;
    let maxv_c = (1i32 << ctx.bd_c) - 1;
    let filter = ctx.exec.chroma_deblock_horizontal;
    let pair_filter = ctx.exec.chroma_deblock_horizontal_pair;
    let last_full_chroma_edge = ch.saturating_sub(2);
    let mut edge = (crow0.div_ceil(8) * 8).max(8);
    while edge <= last_full_chroma_edge {
        if edge >= crow1 {
            break;
        }
        let mut scan = 0;
        while scan + 4 <= cw {
            if let Some(pair_filter) = pair_filter
                && scan + 8 <= cw
            {
                let first = chroma_horizontal_segment_tc(ctx, edge, scan);
                let second = chroma_horizontal_segment_tc(ctx, edge, scan + 4);
                match (first, second) {
                    (Some(tc0), Some(tc1)) => {
                        pair_filter(cb, cw, edge, scan, crow0, tc0, tc1, maxv_c);
                        pair_filter(cr, cw, edge, scan, crow0, tc0, tc1, maxv_c);
                        scan += 8;
                        continue;
                    }
                    (None, _) => {
                        scan += 4;
                        continue;
                    }
                    (Some(tc_c), None) => {
                        filter(cb, cw, edge, scan, crow0, tc_c, maxv_c);
                        filter(cr, cw, edge, scan, crow0, tc_c, maxv_c);
                        scan += 4;
                        continue;
                    }
                }
            }

            if let Some(tc_c) = chroma_horizontal_segment_tc(ctx, edge, scan) {
                filter(cb, cw, edge, scan, crow0, tc_c, maxv_c);
                filter(cr, cw, edge, scan, crow0, tc_c, maxv_c);
            }
            scan += 4;
        }
        edge += 8;
    }
}

/// CTB-aligned row bands covering `[0, total)`. Each band is a whole number of
/// CTBs tall (except possibly the last), so no filter segment or edge straddles
/// a boundary. Returns `(start, end)` pairs with `end` of the last == `total`.
fn ctb_bands(total: usize, ctb: usize) -> Vec<(usize, usize)> {
    let mut bands = Vec::new();
    let mut r = 0;
    while r < total {
        let end = (r + ctb).min(total);
        bands.push((r, end));
        r = end;
    }
    bands
}

/// Row bands for the *horizontal* passes, whose internal boundaries are placed
/// at rows `≡ 4 (mod 8)`. A horizontal edge sits at a multiple of 8 and writes
/// the rows `[e-3, e+2]` (luma) / `[e-1, e]` (chroma); the gap between adjacent
/// edges' write spans is exactly the rows `≡ 3,4 (mod 8)`, so a boundary at
/// `+4` never splits any edge's writes across two bands. The first band starts
/// at 0 and the last ends at `total`.
fn horiz_bands(total: usize, ctb: usize) -> Vec<(usize, usize)> {
    let mut bands = Vec::new();
    let mut r = 0;
    while r < total {
        // Next boundary: first row > r that is ≡ 4 (mod 8) and ≥ r + ctb-ish
        // spacing. We step in CTB-sized chunks then snap up to the next ≡4 row.
        let target = r + ctb;
        if target >= total {
            bands.push((r, total));
            break;
        }
        // Snap `target` up to the nearest row ≡ 4 (mod 8).
        let rem = target % 8;
        let end = if rem <= 4 {
            target + (4 - rem)
        } else {
            target + (12 - rem)
        };
        let end = end.min(total);
        bands.push((r, end));
        r = end;
    }
    bands
}

/// Result planes after deblocking.
pub(crate) struct DeblockPlanes {
    pub y: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
}

/// Parallel deblocking driver. Runs the four ordered passes; within each pass,
/// CTB-aligned row bands are filtered concurrently on `pool`. Bit-identical to
/// the serial `apply_deblocking`. `log2_ctb` sets band height; larger CTBs give
/// coarser (fewer, bigger) bands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_deblocking_parallel(
    pool: &ThreadPool,
    ctx: &DeblockCtx<'_>,
    log2_ctb: u32,
    mut y: Vec<u16>,
    mut cb: Vec<u16>,
    mut cr: Vec<u16>,
) -> DeblockPlanes {
    let ctb = 1usize << log2_ctb;
    let w = ctx.w;
    let cw = ctx.cw;

    // ---- Luma vertical: row bands, no halo, in place ----
    {
        let bands = ctb_bands(ctx.h, ctb);
        let y_dm = DisjointMut::new(std::mem::take(&mut y));
        parallel_for(pool, bands.len(), |bi| {
            let (r0, r1) = bands[bi];
            let mut band = y_dm.slice_mut(r0 * w..r1 * w);
            luma_vertical(ctx, &mut band, r0, r1);
        });
        y = y_dm.into_inner();
    }

    // ---- Luma horizontal: row bands, in place ----
    {
        let bands = horiz_bands(ctx.h, ctb);
        let y_dm = DisjointMut::new(std::mem::take(&mut y));
        parallel_for(pool, bands.len(), |bi| {
            let (r0, r1) = bands[bi];
            let mut band = y_dm.slice_mut(r0 * w..r1 * w);
            luma_horizontal(ctx, &mut band, r0, r1);
        });
        y = y_dm.into_inner();
    }

    // ---- Chroma vertical: chroma-row bands, no halo, in place ----
    if ctx.cw > 0 && ctx.ch > 0 {
        // Keep chroma bands luma-CTB-aligned by dividing by sub_h.
        let cband = (ctb / ctx.sub_h).max(1);
        let bands = ctb_bands(ctx.ch, cband);
        let cb_dm = DisjointMut::new(std::mem::take(&mut cb));
        let cr_dm = DisjointMut::new(std::mem::take(&mut cr));
        parallel_for(pool, bands.len(), |bi| {
            let (r0, r1) = bands[bi];
            let mut cbb = cb_dm.slice_mut(r0 * cw..r1 * cw);
            let mut crb = cr_dm.slice_mut(r0 * cw..r1 * cw);
            chroma_vertical(ctx, &mut cbb, &mut crb, r0, r1);
        });
        cb = cb_dm.into_inner();
        cr = cr_dm.into_inner();
    }

    // ---- Chroma horizontal: chroma-row bands, in place ----
    if ctx.cw > 0 && ctx.ch > 0 {
        let cband = (ctb / ctx.sub_h).max(1);
        let bands = horiz_bands(ctx.ch, cband);
        let cb_dm = DisjointMut::new(std::mem::take(&mut cb));
        let cr_dm = DisjointMut::new(std::mem::take(&mut cr));
        parallel_for(pool, bands.len(), |bi| {
            let (r0, r1) = bands[bi];
            let mut cbb = cb_dm.slice_mut(r0 * cw..r1 * cw);
            let mut crb = cr_dm.slice_mut(r0 * cw..r1 * cw);
            chroma_horizontal(ctx, &mut cbb, &mut crb, r0, r1);
        });
        cb = cb_dm.into_inner();
        cr = cr_dm.into_inner();
    }

    DeblockPlanes { y, cb, cr }
}
