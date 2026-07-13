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

use crate::decode::{SliceDeblock, qpc};
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
    pub chroma_idc: u8,
    pub cb_qp_offset: i32,
    pub cr_qp_offset: i32,
    pub beta_offset: i32,
    pub tc_offset: i32,
    pub deblocking_disabled: bool,
    pub default_qp: i16,
    pub log2_ctb: u32,
    pub qp_y_map: &'a [i16],
    pub tqb: &'a [bool],
    /// Per-4×4 deblock edge flags and boundary strengths (see `FullDecoder`).
    pub edge_v: &'a [bool],
    pub edge_h: &'a [bool],
    pub bs_v: &'a [u8],
    pub bs_h: &'a [u8],
    pub pcm: &'a [bool],
    pub slice_idx: &'a [u16],
    pub slice_deblock: &'a [SliceDeblock],
    pub slice_lf_across: &'a [bool],
    pub pcm_loop_filter_disabled: bool,
    pub loop_filter_across_slices: bool,
    /// Resolved tile geometry when tiles are enabled and cross-tile filtering is
    /// disabled; `None` otherwise (no tile gating needed).
    pub tile_grid: Option<crate::tiles::TileGrid>,
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
    fn grid(&self, px: usize, py: usize) -> Option<usize> {
        if px >= self.w || py >= self.h || self.gw == 0 {
            return None;
        }
        Some((py / 4) * self.gw + (px / 4))
    }

    /// Bs for the vertical edge to the left of `(px, py)`; 0 if not filtered.
    #[inline]
    fn bs_v_at(&self, px: usize, py: usize) -> u8 {
        if px == 0 {
            return 0;
        }
        let g = match self.grid(px, py) {
            Some(g) => g,
            None => return 0,
        };
        if !self.edge_v.get(g).copied().unwrap_or(false) {
            return 0;
        }
        if !self.filter_across(px - 1, py, px, py) {
            return 0;
        }
        self.bs_v.get(g).copied().unwrap_or(0)
    }

    /// Bs for the horizontal edge above `(px, py)`; 0 if not filtered.
    #[inline]
    fn bs_h_at(&self, px: usize, py: usize) -> u8 {
        if py == 0 {
            return 0;
        }
        let g = match self.grid(px, py) {
            Some(g) => g,
            None => return 0,
        };
        if !self.edge_h.get(g).copied().unwrap_or(false) {
            return 0;
        }
        if !self.filter_across(px, py - 1, px, py) {
            return 0;
        }
        self.bs_h.get(g).copied().unwrap_or(0)
    }

    #[inline]
    fn filter_across(&self, pxp: usize, pyp: usize, pxq: usize, pyq: usize) -> bool {
        // TQB/PCM samples do not suppress the complete edge. HEVC filters using
        // the reconstructed samples and then restores only the exempt side. The
        // parallel driver performs that substitution after every pass, exactly
        // like the serial path.
        let sq = self.slice_at(pxq, pyq);
        let q_across = self
            .slice_lf_across
            .get(sq as usize)
            .copied()
            .unwrap_or(self.loop_filter_across_slices);
        if !q_across && self.slice_at(pxp, pyp) != sq {
            return false;
        }
        if let Some(g) = &self.tile_grid {
            let c = self.log2_ctb;
            if g.tile_id_at(pxp >> c, pyp >> c) != g.tile_id_at(pxq >> c, pyq >> c) {
                return false;
            }
        }
        true
    }

    #[inline]
    fn slice_at(&self, px: usize, py: usize) -> u16 {
        self.grid(px, py)
            .and_then(|g| self.slice_idx.get(g))
            .copied()
            .unwrap_or(0)
    }

    #[inline]
    fn slice_deblock_at(&self, px: usize, py: usize) -> SliceDeblock {
        self.slice_deblock
            .get(self.slice_at(px, py) as usize)
            .copied()
            .unwrap_or(SliceDeblock {
                disabled: self.deblocking_disabled,
                beta_offset_div2: self.beta_offset / 2,
                tc_offset_div2: self.tc_offset / 2,
            })
    }

    #[inline]
    fn sample_suppressed(&self, grid: usize) -> bool {
        self.tqb.get(grid).copied().unwrap_or(false)
            || (self.pcm_loop_filter_disabled && self.pcm.get(grid).copied().unwrap_or(false))
    }
}

/// Per-4-line luma deblock decision (§8.7.2.5.3), computed once from lines 0 and
/// 3 and applied uniformly to all 4 lines. `Weak` carries the dEp/dEq gates.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LumaDecision {
    Skip,
    Strong,
    Weak { do_p1: bool, do_q1: bool },
}

/// Compute the segment decision from lines 0 and 3. `g(line, tap)` reads a sample
/// (tap: -1=p0,-2=p1,-3=p2,-4=p3, 0=q0,1=q1,2=q2,3=q3).
#[inline]
pub(crate) fn luma_decision(beta: i32, tc: i32, g: impl Fn(usize, i32) -> i32) -> LumaDecision {
    let dp0 = (g(0, -3) - 2 * g(0, -2) + g(0, -1)).abs();
    let dp3 = (g(3, -3) - 2 * g(3, -2) + g(3, -1)).abs();
    let dq0 = (g(0, 2) - 2 * g(0, 1) + g(0, 0)).abs();
    let dq3 = (g(3, 2) - 2 * g(3, 1) + g(3, 0)).abs();
    if dp0 + dp3 + dq0 + dq3 >= beta {
        return LumaDecision::Skip;
    }
    let dsam = |line: usize, dpq: i32| -> bool {
        2 * dpq < (beta >> 2)
            && (g(line, -4) - g(line, -1)).abs() + (g(line, 0) - g(line, 3)).abs() < (beta >> 3)
            && (g(line, -1) - g(line, 0)).abs() < (5 * tc + 1) >> 1
    };
    if dsam(0, dp0 + dq0) && dsam(3, dp3 + dq3) {
        LumaDecision::Strong
    } else {
        let side_thr = (beta + (beta >> 1)) >> 3;
        LumaDecision::Weak {
            do_p1: dp0 + dp3 < side_thr,
            do_q1: dq0 + dq3 < side_thr,
        }
    }
}

#[inline]
pub(crate) fn deblock_luma_segment(
    plane: &mut [u16],
    beta: i32,
    tc: i32,
    max_val: i32,
    at: impl Fn(usize, i32) -> usize,
) {
    let g = |plane: &[u16], line: usize, tap: i32| plane[at(line, tap)] as i32;

    let decision = luma_decision(beta, tc, |line, tap| g(plane, line, tap));
    let (strong, do_p1, do_q1) = match decision {
        LumaDecision::Skip => return,
        LumaDecision::Strong => (true, false, false),
        LumaDecision::Weak { do_p1, do_q1 } => (false, do_p1, do_q1),
    };

    let tc2 = tc >> 1;
    for line in 0..4 {
        let p0 = g(plane, line, -1);
        let p1 = g(plane, line, -2);
        let p2 = g(plane, line, -3);
        let p3 = g(plane, line, -4);
        let q0 = g(plane, line, 0);
        let q1 = g(plane, line, 1);
        let q2 = g(plane, line, 2);
        let q3 = g(plane, line, 3);
        if strong {
            // §8.7.2.5.4: each strongly-filtered sample is additionally clipped
            // to ±2·tc around its original value.
            let c = |orig: i32, v: i32| -> u16 {
                v.clamp(orig - 2 * tc, orig + 2 * tc).clamp(0, max_val) as u16
            };
            plane[at(line, -1)] = c(p0, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            plane[at(line, -2)] = c(p1, (p2 + p1 + p0 + q0 + 2) >> 2);
            plane[at(line, -3)] = c(p2, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
            plane[at(line, 0)] = c(q0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            plane[at(line, 1)] = c(q1, (p0 + q0 + q1 + q2 + 2) >> 2);
            plane[at(line, 2)] = c(q2, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3);
        } else {
            let delta0 = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta0.abs() < tc * 10 {
                let delta = delta0.clamp(-tc, tc);
                plane[at(line, -1)] = (p0 + delta).clamp(0, max_val) as u16;
                plane[at(line, 0)] = (q0 - delta).clamp(0, max_val) as u16;
                if do_p1 {
                    let dp1 = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-tc2, tc2);
                    plane[at(line, -2)] = (p1 + dp1).clamp(0, max_val) as u16;
                }
                if do_q1 {
                    let dq1 = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-tc2, tc2);
                    plane[at(line, 1)] = (q1 + dq1).clamp(0, max_val) as u16;
                }
            }
        }
    }
}

fn luma_vertical_segment_params(
    ctx: &DeblockCtx<'_>,
    y: &[u16],
    w: usize,
    row0: usize,
    row1: usize,
    edge: usize,
    s: usize,
) -> Option<(LumaDecision, i32)> {
    let mid = s + 1;
    // Real TU/PU/CU edge with Bs>0 and not a disabled slice/tile/PCM/TQB
    // boundary (§8.7.2). Mirrors the serial `deblock_bs_v` gate.
    let bs = ctx.bs_v_at(edge, mid);
    if bs == 0 {
        return None;
    }
    let bs = bs as i32;
    let sd = ctx.slice_deblock_at(edge, mid);
    if sd.disabled {
        return None;
    }
    let qp_p = ctx.qp_at(edge - 1, mid);
    let qp_q = ctx.qp_at(edge, mid);
    let avg_qp = (qp_p + qp_q + 1) >> 1;
    // §8.7.2.5.3: Q indexes the tables at qPL (+offsets) without QpBdOffset; the
    // looked-up β′/tC′ are then scaled to the sample bit depth. Mirrors the
    // serial `apply_deblocking`.
    let beta_prime = (avg_qp + sd.beta_offset_div2 * 2).clamp(0, 51);
    // tc' = Q(qp + 2*(Bs-1) + tc_offset) (§8.7.2.5.3): Bs=2 adds 2, Bs=1 adds 0.
    let tc_prime = (avg_qp + 2 * (bs - 1) + sd.tc_offset_div2 * 2).clamp(0, 53);
    let bd_shift = (ctx.bd - 8) as u32;
    let beta = BETA[beta_prime as usize] << bd_shift;
    let tc = TC[tc_prime as usize] << bd_shift;
    if tc == 0 {
        return None;
    }
    // Decision from lines 0 and 3 (§8.7.2.5.3). Band-local row = global - row0.
    let dec = luma_decision(beta, tc, |line, tap| {
        let col = (edge as i32 + tap) as usize;
        y[(s + line - row0) * w + col] as i32
    });
    let _ = row1;
    (dec != LumaDecision::Skip).then_some((dec, tc))
}

#[inline]
fn luma_horizontal_segment_params(
    ctx: &DeblockCtx<'_>,
    y: &[u16],
    w: usize,
    row0: usize,
    edge: usize,
    scan: usize,
) -> Option<(LumaDecision, i32)> {
    let mid = scan + 1;
    let bs = ctx.bs_h_at(mid, edge);
    if bs == 0 {
        return None;
    }
    let bs = bs as i32;
    let sd = ctx.slice_deblock_at(mid, edge);
    if sd.disabled {
        return None;
    }
    let qp_p = ctx.qp_at(mid, edge - 1);
    let qp_q = ctx.qp_at(mid, edge);
    let avg_qp = (qp_p + qp_q + 1) >> 1;
    let beta_prime = (avg_qp + sd.beta_offset_div2 * 2).clamp(0, 51);
    let tc_prime = (avg_qp + 2 * (bs - 1) + sd.tc_offset_div2 * 2).clamp(0, 53);
    let bd_shift = (ctx.bd - 8) as u32;
    let beta = BETA[beta_prime as usize] << bd_shift;
    let tc = TC[tc_prime as usize] << bd_shift;
    if tc == 0 {
        return None;
    }
    let dec = luma_decision(beta, tc, |line, tap| {
        let row = (edge as i32 + tap) as usize - row0;
        y[row * w + (scan + line)] as i32
    });
    (dec != LumaDecision::Skip).then_some((dec, tc))
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
        // The per-4-line decision (§8.7.2.5.3) is computed scalar from lines 0
        // and 3; the chosen filter is then applied — vectorized when SIMD is
        // available — uniformly to all 4 lines.
        while s + 4 <= row1 {
            if let Some(pair_filter) = pair_filter
                && s + 8 <= row1
            {
                let first = luma_vertical_segment_params(ctx, y, w, row0, row1, edge, s);
                let second = luma_vertical_segment_params(ctx, y, w, row0, row1, edge, s + 4);
                match (first, second) {
                    (Some((d0, tc0)), Some((d1, tc1))) => {
                        pair_filter(y, w, edge, s, row0, d0, tc0, d1, tc1, maxv);
                        s += 8;
                        continue;
                    }
                    (None, _) => {
                        s += 4;
                        continue;
                    }
                    (Some((d0, tc0)), None) => {
                        filter(y, w, edge, s, row0, d0, tc0, maxv);
                        s += 4;
                        continue;
                    }
                }
            }
            if let Some((dec, tc)) = luma_vertical_segment_params(ctx, y, w, row0, row1, edge, s) {
                filter(y, w, edge, s, row0, dec, tc, maxv);
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
                    (Some((d0, tc0)), Some((d1, tc1))) => {
                        pair_filter(y, w, edge, scan, row0, d0, tc0, d1, tc1, maxv);
                        scan += 8;
                        continue;
                    }
                    (None, _) => {
                        scan += 4;
                        continue;
                    }
                    (Some((d0, tc0)), None) => {
                        filter(y, w, edge, scan, row0, d0, tc0, maxv);
                        scan += 4;
                        continue;
                    }
                }
            }
            if let Some((dec, tc)) = luma_horizontal_segment_params(ctx, y, w, row0, edge, scan) {
                filter(y, w, edge, scan, row0, dec, tc, maxv);
            }
            scan += 4;
        }
        edge += 8;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) type LumaDeblockPlaneFn =
    fn(&mut [u16], usize, usize, usize, usize, LumaDecision, i32, i32);
#[allow(clippy::too_many_arguments)]
pub(crate) type LumaDeblockPairFn =
    fn(&mut [u16], usize, usize, usize, usize, LumaDecision, i32, LumaDecision, i32, i32);

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

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
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

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
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

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
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

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::luma_horizontal_plane_pair_avx2);
            }
        }

        _f
    })
}

#[allow(clippy::too_many_arguments)]
/// Apply a precomputed per-segment `decision` uniformly to the 4 lines of a luma
/// edge segment. `at(line, tap)` gives the flat index (see `luma_decision`).
#[inline]
pub(crate) fn apply_luma_decision(
    plane: &mut [u16],
    decision: LumaDecision,
    tc: i32,
    max_val: i32,
    at: impl Fn(usize, i32) -> usize,
) {
    let (strong, do_p1, do_q1) = match decision {
        LumaDecision::Skip => return,
        LumaDecision::Strong => (true, false, false),
        LumaDecision::Weak { do_p1, do_q1 } => (false, do_p1, do_q1),
    };
    let tc2 = tc >> 1;
    let g = |plane: &[u16], line: usize, tap: i32| plane[at(line, tap)] as i32;
    for line in 0..4 {
        let p0 = g(plane, line, -1);
        let p1 = g(plane, line, -2);
        let p2 = g(plane, line, -3);
        let p3 = g(plane, line, -4);
        let q0 = g(plane, line, 0);
        let q1 = g(plane, line, 1);
        let q2 = g(plane, line, 2);
        let q3 = g(plane, line, 3);
        if strong {
            // §8.7.2.5.4: each strongly-filtered sample is additionally clipped
            // to ±2·tc around its original value.
            let c = |orig: i32, v: i32| -> u16 {
                v.clamp(orig - 2 * tc, orig + 2 * tc).clamp(0, max_val) as u16
            };
            plane[at(line, -1)] = c(p0, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            plane[at(line, -2)] = c(p1, (p2 + p1 + p0 + q0 + 2) >> 2);
            plane[at(line, -3)] = c(p2, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
            plane[at(line, 0)] = c(q0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            plane[at(line, 1)] = c(q1, (p0 + q0 + q1 + q2 + 2) >> 2);
            plane[at(line, 2)] = c(q2, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3);
        } else {
            let delta0 = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            if delta0.abs() < tc * 10 {
                let delta = delta0.clamp(-tc, tc);
                plane[at(line, -1)] = (p0 + delta).clamp(0, max_val) as u16;
                plane[at(line, 0)] = (q0 - delta).clamp(0, max_val) as u16;
                if do_p1 {
                    let dp1 = ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-tc2, tc2);
                    plane[at(line, -2)] = (p1 + dp1).clamp(0, max_val) as u16;
                }
                if do_q1 {
                    let dq1 = ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-tc2, tc2);
                    plane[at(line, 1)] = (q1 + dq1).clamp(0, max_val) as u16;
                }
            }
        }
    }
}

/// Scalar reference: apply `decision` to a vertical-edge segment (4 rows).
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_vertical_plane_scalar(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    s: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    apply_luma_decision(pix, decision, tc, maxv, |line, tap| {
        let col = (edge as i32 + tap) as usize;
        (s + line - row0) * w + col
    });
}

/// Scalar reference: apply `decision` to a horizontal-edge segment (4 cols).
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_horizontal_plane_scalar(
    pix: &mut [u16],
    w: usize,
    edge: usize,
    scan: usize,
    row0: usize,
    decision: LumaDecision,
    tc: i32,
    maxv: i32,
) {
    apply_luma_decision(pix, decision, tc, maxv, |line, tap| {
        let row = (edge as i32 + tap) as usize - row0;
        row * w + (scan + line)
    });
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

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
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

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
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

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
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

        #[cfg(all(feature = "avx", target_arch = "x86_64"))]
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
fn chroma_tcs(ctx: &DeblockCtx<'_>, avg_qp_l: i32, tc_offset: i32) -> (i32, i32) {
    let shift = (ctx.bd_c - 8) as u32;
    let tc_for = |offset: i32| {
        let qp_c = qpc(avg_qp_l + offset, ctx.chroma_idc, ctx.bd_c);
        let tc_prime = (qp_c + 2 + tc_offset).clamp(0, 53);
        TC[tc_prime as usize] << shift
    };
    (tc_for(ctx.cb_qp_offset), tc_for(ctx.cr_qp_offset))
}

#[inline]
fn chroma_vertical_segment_tc(ctx: &DeblockCtx<'_>, edge: usize, s: usize) -> Option<(i32, i32)> {
    let mid = s + 1;
    let qlx = edge * ctx.sub_w;
    let qly = mid * ctx.sub_h;
    // Chroma filters only where the co-located luma edge has Bs == 2.
    if ctx.bs_v_at(qlx, qly) < 2 {
        return None;
    }
    let lqx = qlx.min(ctx.w.saturating_sub(1));
    let lqy = qly.min(ctx.h.saturating_sub(1));
    let sd = ctx.slice_deblock_at(lqx, lqy);
    if sd.disabled {
        return None;
    }
    let lpx = qlx.saturating_sub(1).min(ctx.w.saturating_sub(1));
    let qp_p = ctx.qp_at(lpx, lqy);
    let qp_q = ctx.qp_at(lqx, lqy);
    let avg_qp_l = (qp_p + qp_q + 1) >> 1;
    Some(chroma_tcs(ctx, avg_qp_l, sd.tc_offset_div2 * 2))
}

#[inline]
fn chroma_horizontal_segment_tc(
    ctx: &DeblockCtx<'_>,
    edge: usize,
    scan: usize,
) -> Option<(i32, i32)> {
    let mid = scan + 1;
    let qlx = mid * ctx.sub_w;
    let qly = edge * ctx.sub_h;
    if ctx.bs_h_at(qlx, qly) < 2 {
        return None;
    }
    let lqx = qlx.min(ctx.w.saturating_sub(1));
    let lqy = qly.min(ctx.h.saturating_sub(1));
    let sd = ctx.slice_deblock_at(lqx, lqy);
    if sd.disabled {
        return None;
    }
    let lpy = qly.saturating_sub(1).min(ctx.h.saturating_sub(1));
    let qp_p = ctx.qp_at(lqx, lpy);
    let qp_q = ctx.qp_at(lqx, lqy);
    let avg_qp_l = (qp_p + qp_q + 1) >> 1;
    Some(chroma_tcs(ctx, avg_qp_l, sd.tc_offset_div2 * 2))
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
                    (Some((cb0, cr0)), Some((cb1, cr1))) => {
                        if cb0 != 0 || cb1 != 0 {
                            pair_filter(cb, cw, edge, s, crow0, cb0, cb1, maxv_c);
                        }
                        if cr0 != 0 || cr1 != 0 {
                            pair_filter(cr, cw, edge, s, crow0, cr0, cr1, maxv_c);
                        }
                        s += 8;
                        continue;
                    }
                    (None, _) => {
                        s += 4;
                        continue;
                    }
                    (Some((tc_cb, tc_cr)), None) => {
                        if tc_cb != 0 {
                            filter(cb, cw, edge, s, crow0, tc_cb, maxv_c);
                        }
                        if tc_cr != 0 {
                            filter(cr, cw, edge, s, crow0, tc_cr, maxv_c);
                        }
                        s += 4;
                        continue;
                    }
                }
            }

            if let Some((tc_cb, tc_cr)) = chroma_vertical_segment_tc(ctx, edge, s) {
                if tc_cb != 0 {
                    filter(cb, cw, edge, s, crow0, tc_cb, maxv_c);
                }
                if tc_cr != 0 {
                    filter(cr, cw, edge, s, crow0, tc_cr, maxv_c);
                }
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
                    (Some((cb0, cr0)), Some((cb1, cr1))) => {
                        if cb0 != 0 || cb1 != 0 {
                            pair_filter(cb, cw, edge, scan, crow0, cb0, cb1, maxv_c);
                        }
                        if cr0 != 0 || cr1 != 0 {
                            pair_filter(cr, cw, edge, scan, crow0, cr0, cr1, maxv_c);
                        }
                        scan += 8;
                        continue;
                    }
                    (None, _) => {
                        scan += 4;
                        continue;
                    }
                    (Some((tc_cb, tc_cr)), None) => {
                        if tc_cb != 0 {
                            filter(cb, cw, edge, scan, crow0, tc_cb, maxv_c);
                        }
                        if tc_cr != 0 {
                            filter(cr, cw, edge, scan, crow0, tc_cr, maxv_c);
                        }
                        scan += 4;
                        continue;
                    }
                }
            }

            if let Some((tc_cb, tc_cr)) = chroma_horizontal_segment_tc(ctx, edge, scan) {
                if tc_cb != 0 {
                    filter(cb, cw, edge, scan, crow0, tc_cb, maxv_c);
                }
                if tc_cr != 0 {
                    filter(cr, cw, edge, scan, crow0, tc_cr, maxv_c);
                }
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

#[inline]
fn suppression_active(ctx: &DeblockCtx<'_>) -> bool {
    ctx.tqb.iter().any(|&v| v) || (ctx.pcm_loop_filter_disabled && ctx.pcm.iter().any(|&v| v))
}

/// Restore exempt luma samples from the pre-deblock picture after one ordered
/// pass. The filter must still run across an exempt/non-exempt boundary so the
/// non-exempt side is updated; only the exempt output samples are substituted.
fn restore_luma(ctx: &DeblockCtx<'_>, plane: &mut [u16], snapshot: &[u16]) {
    for gy in 0..ctx.gh {
        let grid_row = gy * ctx.gw;
        for gx in 0..ctx.gw {
            if !ctx.sample_suppressed(grid_row + gx) {
                continue;
            }
            let x0 = gx * 4;
            let y0 = gy * 4;
            let width = ctx.w.saturating_sub(x0).min(4);
            let height = ctx.h.saturating_sub(y0).min(4);
            if width == 0 || height == 0 {
                continue;
            }
            for row in 0..height {
                let off = (y0 + row) * ctx.w + x0;
                plane[off..off + width].copy_from_slice(&snapshot[off..off + width]);
            }
        }
    }
}

fn restore_chroma(ctx: &DeblockCtx<'_>, plane: &mut [u16], snapshot: &[u16]) {
    for cy in 0..ctx.ch {
        let ly = cy * ctx.sub_h;
        let grid_row = (ly / 4) * ctx.gw;
        for cx in 0..ctx.cw {
            let lx = cx * ctx.sub_w;
            let grid = grid_row + lx / 4;
            if ctx.sample_suppressed(grid) {
                let off = cy * ctx.cw + cx;
                plane[off] = snapshot[off];
            }
        }
    }
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
    let suppress = suppression_active(ctx);
    let snap_y = suppress.then(|| y.clone());
    let snap_cb = suppress.then(|| cb.clone());
    let snap_cr = suppress.then(|| cr.clone());

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
        if let Some(snapshot) = &snap_y {
            restore_luma(ctx, &mut y, snapshot);
        }
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
        if let Some(snapshot) = &snap_y {
            restore_luma(ctx, &mut y, snapshot);
        }
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
        if let (Some(scb), Some(scr)) = (&snap_cb, &snap_cr) {
            restore_chroma(ctx, &mut cb, scb);
            restore_chroma(ctx, &mut cr, scr);
        }
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
        if let (Some(scb), Some(scr)) = (&snap_cb, &snap_cr) {
            restore_chroma(ctx, &mut cb, scb);
            restore_chroma(ctx, &mut cr, scr);
        }
    }

    DeblockPlanes { y, cb, cr }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::ExecContext;

    /// Build a 2×2-CTB-worth 8×8-pixel grid (gw=gh=2) DeblockCtx with the given
    /// per-4×4 edge/Bs/slice arrays and no tiles.
    fn ctx<'a>(
        edge_v: &'a [bool],
        edge_h: &'a [bool],
        bs_v: &'a [u8],
        bs_h: &'a [u8],
        tqb: &'a [bool],
        pcm: &'a [bool],
        slice_idx: &'a [u16],
        across_slices: bool,
        pcm_lf_disabled: bool,
    ) -> DeblockCtx<'a> {
        DeblockCtx {
            exec: ExecContext::new(),
            w: 8,
            h: 8,
            cw: 4,
            ch: 4,
            gw: 2,
            gh: 2,
            sub_w: 2,
            sub_h: 2,
            bd: 8,
            bd_c: 8,
            chroma_idc: 1,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            beta_offset: 0,
            tc_offset: 0,
            deblocking_disabled: false,
            default_qp: 26,
            log2_ctb: 6,
            qp_y_map: &[],
            tqb,
            edge_v,
            edge_h,
            bs_v,
            bs_h,
            pcm,
            slice_idx,
            slice_deblock: &[],
            slice_lf_across: &[],
            pcm_loop_filter_disabled: pcm_lf_disabled,
            loop_filter_across_slices: across_slices,
            tile_grid: None,
        }
    }

    #[test]
    fn bs_zero_when_not_a_real_edge() {
        // No edge flags set → nothing filtered even with a Bs value present.
        let edge = [false; 4];
        let bs = [2u8; 4];
        let no = [false; 4];
        let s = [0u16; 4];
        let c = ctx(&edge, &edge, &bs, &bs, &no, &no, &s, true, false);
        assert_eq!(c.bs_v_at(4, 0), 0);
        assert_eq!(c.bs_h_at(0, 4), 0);
    }

    #[test]
    fn bs_reported_at_real_edge() {
        // Mark the 4×4 cell at grid (1,0) (pixel col 4) as a vertical edge, Bs 2.
        let mut edge_v = [false; 4];
        let mut bs_v = [0u8; 4];
        edge_v[1] = true; // grid idx (gy=0,gx=1)
        bs_v[1] = 2;
        let z = [false; 4];
        let zb = [0u8; 4];
        let s = [0u16; 4];
        let c = ctx(&edge_v, &z, &bs_v, &zb, &z, &z, &s, true, false);
        assert_eq!(c.bs_v_at(4, 0), 2);
    }

    #[test]
    fn tqb_and_pcm_suppress_only_the_exempt_samples() {
        let mut edge_v = [false; 4];
        let mut bs_v = [0u8; 4];
        edge_v[1] = true;
        bs_v[1] = 2;
        let z = [false; 4];
        let zb = [0u8; 4];
        let s = [0u16; 4];
        // P side (pixel col 3 → grid 0) is transquant-bypass. The edge must
        // still be evaluated so the non-exempt Q side can be filtered; the
        // driver restores the exempt samples after the pass.
        let mut tqb = [false; 4];
        tqb[0] = true;
        let c = ctx(&edge_v, &z, &bs_v, &zb, &tqb, &z, &s, true, false);
        assert_eq!(c.bs_v_at(4, 0), 2);
        assert!(c.sample_suppressed(0));
        // PCM with loop filtering disabled follows the same substitution rule.
        let mut pcm = [false; 4];
        pcm[1] = true;
        let c2 = ctx(&edge_v, &z, &bs_v, &zb, &z, &pcm, &s, true, true);
        assert_eq!(c2.bs_v_at(4, 0), 2);
        assert!(c2.sample_suppressed(1));
        // PCM present but pcm_loop_filter_disabled=false is not exempt.
        let c3 = ctx(&edge_v, &z, &bs_v, &zb, &z, &pcm, &s, true, false);
        assert_eq!(c3.bs_v_at(4, 0), 2);
        assert!(!c3.sample_suppressed(1));
    }

    #[test]
    fn cross_slice_gate() {
        let mut edge_v = [false; 4];
        let mut bs_v = [0u8; 4];
        edge_v[1] = true;
        bs_v[1] = 2;
        let z = [false; 4];
        let zb = [0u8; 4];
        // P (grid 0) in slice 0, Q (grid 1) in slice 1.
        let s = [0u16, 1, 0, 1];
        // across_slices = false → different slices suppress.
        let c = ctx(&edge_v, &z, &bs_v, &zb, &z, &z, &s, false, false);
        assert_eq!(c.bs_v_at(4, 0), 0);
        // across_slices = true → filtered.
        let c2 = ctx(&edge_v, &z, &bs_v, &zb, &z, &z, &s, true, false);
        assert_eq!(c2.bs_v_at(4, 0), 2);
    }

    #[test]
    fn q_side_slice_controls_cross_slice_and_offsets() {
        let mut edge_v = [false; 4];
        let mut bs_v = [0u8; 4];
        edge_v[1] = true;
        bs_v[1] = 2;
        let z = [false; 4];
        let zb = [0u8; 4];
        let owners = [0u16, 1, 0, 1];
        let slices = [
            SliceDeblock {
                disabled: false,
                beta_offset_div2: 0,
                tc_offset_div2: 0,
            },
            SliceDeblock {
                disabled: false,
                beta_offset_div2: 2,
                tc_offset_div2: -1,
            },
        ];
        let across = [true, false];
        let mut c = ctx(&edge_v, &z, &bs_v, &zb, &z, &z, &owners, true, false);
        c.slice_deblock = &slices;
        c.slice_lf_across = &across;

        // Q is in slice 1, whose per-slice flag forbids crossing even though
        // the PPS/default value supplied by the helper is true.
        assert_eq!(c.bs_v_at(4, 0), 0);
        let sd = c.slice_deblock_at(4, 0);
        assert_eq!(sd.beta_offset_div2, 2);
        assert_eq!(sd.tc_offset_div2, -1);
    }
}
