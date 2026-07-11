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

use crate::dpb::RefEntry;
use crate::inter::{MotionInfo, Mv, PredFlags};

/// Clip3 helper.
#[inline]
fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.clamp(lo, hi)
}

/// Scale a motion vector by the POC-distance ratio (§8.5.3.2.9). `col_dist` is
/// the distance for the source MV's reference; `cur_dist` for the target.
pub(crate) fn scale_mv(mv: Mv, col_dist: i32, cur_dist: i32) -> Mv {
    let td = clip3(-128, 127, col_dist);
    let tb = clip3(-128, 127, cur_dist);
    if td == 0 {
        return mv;
    }
    let tx = (16384 + (td.abs() >> 1)) / td;
    let dsf = clip3(-4096, 4095, (tb * tx + 32) >> 6);
    let sc = |c: i16| -> i16 {
        let p = dsf * c as i32;
        let v = p.signum() * ((p.abs() + 127) >> 8);
        clip3(-32768, 32767, v) as i16
    };
    Mv::new(sc(mv.x), sc(mv.y))
}

/// Motion-neighbor access used by merge and AMVP derivation.
pub(crate) trait Neighbors {
    /// Raw motion at luma position `(x, y)`. Prediction-block availability is
    /// derived separately because a previous PU in the same CU is available
    /// even though its samples are not represented by the picture-level
    /// `decoded` map yet.
    fn motion_at(&self, x: isize, y: isize) -> Option<MotionInfo>;
    /// Z-scan/slice/tile availability for a position outside the current CU.
    fn available(&self, x: isize, y: isize) -> bool;
    /// Collocated temporal motion for target position, scaled to `ref_poc`.
    fn temporal(&self, x: usize, y: usize, list: usize, ref_poc: i32, cur_poc: i32) -> Option<Mv>;
    fn cur_poc(&self) -> i32;
    /// log2 of the CTB size, for the bottom-right temporal candidate's CTB-row
    /// availability check (§8.5.3.2.8).
    fn ctb_log2(&self) -> u32;
}

/// A merge candidate (full bidirectional motion).
#[derive(Clone, Copy, Default)]
pub(crate) struct MergeCand {
    pub(crate) pred: PredFlags,
    pub(crate) mv: [Mv; 2],
    pub(crate) ref_idx: [i8; 2],
}

impl MergeCand {
    fn from_motion(m: &MotionInfo) -> Self {
        MergeCand {
            pred: m.pred,
            mv: m.mv,
            ref_idx: m.ref_idx,
        }
    }
    fn same_motion(&self, o: &MergeCand) -> bool {
        self.pred == o.pred && self.mv == o.mv && self.ref_idx == o.ref_idx
    }
}

/// Position of a PU for neighbor derivation.
pub(crate) struct PuGeom {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) w: usize,
    pub(crate) h: usize,
    /// Top-left and dimensions of the containing coding block. These are
    /// required by the prediction-block availability process (§6.4.2), which
    /// treats same-CU neighbors differently from picture-level z-scan
    /// neighbors.
    pub(crate) cu_x: usize,
    pub(crate) cu_y: usize,
    /// Whether this is a B slice (enables L1 and combined candidates).
    pub(crate) is_b: bool,
    /// Part index and mode for the second-PU redundancy checks (§8.5.3.2.1).
    pub(crate) part_idx: usize,
    /// CU dimensions, used to classify the 2nd-PU split direction (vertical vs
    /// horizontal) for the A1/B1 redundancy exclusion.
    pub(crate) cu_w: usize,
    pub(crate) cu_h: usize,
    /// Current-picture referencing active (SCC): the zero merge candidates
    /// exclude the current picture from the L0 ref-index cycling, matching
    /// HM-SCM / x265 (getInterMergeCandidates: numRefIdx0-- when SCC). Without
    /// this, zero candidates cycle onto the current-picture entry with a (0,0)
    /// BV, which is an invalid block vector.
    pub(crate) curr_pic_ref: bool,
    /// log2 of the parallel merge estimation region size (§7.4.3.3.1). A
    /// spatial candidate in the same region as the PU is unavailable.
    pub(crate) par_mrg_level: u32,
}

/// Derive prediction-block availability (§6.4.2). A neighbor in the same
/// coding block does not use the picture-level reconstructed-sample map. The
/// sole same-CU exclusion is the lower-left region of the second NxN PU;
/// neighbors outside the CU use the regular z-scan/slice/tile process.
#[inline]
fn prediction_block_available<N: Neighbors>(nb: &N, pu: &PuGeom, px: isize, py: isize) -> bool {
    if px < 0 || py < 0 {
        return false;
    }
    let (x, y) = (px as usize, py as usize);
    let same_cb = x >= pu.cu_x
        && y >= pu.cu_y
        && x < pu.cu_x.saturating_add(pu.cu_w)
        && y < pu.cu_y.saturating_add(pu.cu_h);
    if !same_cb {
        return nb.available(px, py);
    }

    !(pu.w.saturating_mul(2) == pu.cu_w
        && pu.h.saturating_mul(2) == pu.cu_h
        && pu.part_idx == 1
        && pu.cu_y.saturating_add(pu.h) <= y
        && pu.cu_x.saturating_add(pu.w) > x)
}

/// Derive the merge candidate list (§8.5.3.2.1) and return the candidate at
/// `merge_idx`. `max_cand` = MaxNumMergeCand. `col_ref` supplies the collocated
/// reference POC for temporal candidates; `list_poc` maps ref_idx→POC per list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_merge<N: Neighbors>(
    nb: &N,
    pu: &PuGeom,
    merge_idx: usize,
    max_cand: usize,
    temporal_enabled: bool,
    list0: &[RefEntry],
    list1: &[RefEntry],
) -> MergeCand {
    let mut cands: Vec<MergeCand> = Vec::with_capacity(max_cand + 2);
    let (x, y, w, h) = (pu.x as isize, pu.y as isize, pu.w as isize, pu.h as isize);

    // Spatial candidates in order A1, B1, B0, A0, B2 (§8.5.3.2.2).
    // A1: left-bottom (x-1, y+h-1).
    let a1 = spatial(nb, x - 1, y + h - 1, pu, SpatialPos::A1);
    if let Some(c) = a1 {
        push_unique(&mut cands, c, max_cand);
    }
    // B1: top-right (x+w-1, y-1).
    let b1 = spatial(nb, x + w - 1, y - 1, pu, SpatialPos::B1);
    if let Some(c) = b1
        && a1.is_none_or(|a| !a.same_motion(&c))
    {
        push_unique(&mut cands, c, max_cand);
    }
    // B0: top-right corner (x+w, y-1).
    if let Some(c) = spatial(nb, x + w, y - 1, pu, SpatialPos::B0)
        && b1.is_none_or(|b| !b.same_motion(&c))
    {
        push_unique(&mut cands, c, max_cand);
    }
    // A0: left-bottom corner (x-1, y+h).
    if let Some(c) = spatial(nb, x - 1, y + h, pu, SpatialPos::A0)
        && a1.is_none_or(|a| !a.same_motion(&c))
    {
        push_unique(&mut cands, c, max_cand);
    }
    // B2: top-left corner (x-1, y-1) — only if fewer than 4 spatial so far.
    if cands.len() < 4
        && let Some(c) = spatial(nb, x - 1, y - 1, pu, SpatialPos::B2)
    {
        let dup = a1.is_some_and(|a| a.same_motion(&c)) || b1.is_some_and(|b| b.same_motion(&c));
        if !dup {
            push_unique(&mut cands, c, max_cand);
        }
    }

    // Temporal candidate (§8.5.3.2.7): collocated bottom-right then center.
    if temporal_enabled
        && cands.len() < max_cand
        && let Some(tc) = temporal_merge(nb, pu, list0, list1)
    {
        cands.push(tc);
    }

    // Combined bi-predictive candidates (B slices).
    if pu.is_b {
        derive_combined(&mut cands, max_cand, list0, list1);
    }

    // SCC: the current picture (last L0 entry) is excluded from the zero-
    // candidate ref-index count (HM-SCM / x265 getInterMergeCandidates does
    // numRefIdx0-- when SCC; only the L0 count is reduced). A zero candidate on
    // the current-picture entry would be a (0,0) block vector, which is never
    // valid.
    let n0 = list0.len().saturating_sub(usize::from(pu.curr_pic_ref));
    derive_zero(&mut cands, max_cand, pu.is_b, n0, list1.len());

    cands
        .get(merge_idx)
        .copied()
        .unwrap_or_else(|| zero_cand(pu.is_b))
}

#[derive(Clone, Copy)]
enum SpatialPos {
    A0,
    A1,
    B0,
    B1,
    B2,
}

fn spatial<N: Neighbors>(
    nb: &N,
    px: isize,
    py: isize,
    pu: &PuGeom,
    pos: SpatialPos,
) -> Option<MergeCand> {
    // Second-PU redundancy (§8.5.3.2.2, Note 1). Only the A1 and B1 candidates
    // are dropped, and only for the second PU of a directional split:
    //   A1 unavailable when partIdx==1 and the CU is split vertically
    //       (Nx2N / nLx2N / nRx2N: PU narrower than the CU),
    //   B1 unavailable when partIdx==1 and the CU is split horizontally
    //       (2NxN / 2NxnU / 2NxnD: PU shorter than the CU).
    // A0/B0/B2 are never excluded on this ground.
    if pu.part_idx == 1 {
        let vertical_split = pu.w < pu.cu_w && pu.h == pu.cu_h;
        let horizontal_split = pu.h < pu.cu_h && pu.w == pu.cu_w;
        match pos {
            SpatialPos::A1 if vertical_split => return None,
            SpatialPos::B1 if horizontal_split => return None,
            _ => {}
        }
    }
    // Parallel merge estimation region (§8.5.3.2.2): a spatial neighbor that
    // falls in the same merge-estimation region as the PU is unavailable, so
    // that all PBs in the region can derive their merge lists in parallel.
    let lvl = pu.par_mrg_level;
    if px >= 0
        && py >= 0
        && (pu.x as isize >> lvl) == (px >> lvl)
        && (pu.y as isize >> lvl) == (py >> lvl)
    {
        return None;
    }
    if !prediction_block_available(nb, pu, px, py) {
        return None;
    }
    let m = nb.motion_at(px, py)?;
    if m.is_intra {
        return None;
    }
    Some(MergeCand::from_motion(&m))
}

fn temporal_merge<N: Neighbors>(
    nb: &N,
    pu: &PuGeom,
    list0: &[RefEntry],
    list1: &[RefEntry],
) -> Option<MergeCand> {
    let cur_poc = nb.cur_poc();
    // Bottom-right collocated, fallback to center. The BR candidate is only
    // available when it stays in the current PU's CTB row (§8.5.3.2.8).
    let ctb_log2 = nb.ctb_log2();
    let br_y = pu.y + pu.h;
    let br: Option<(usize, usize)> = if (br_y >> ctb_log2) == (pu.y >> ctb_log2) {
        Some((pu.x + pu.w, br_y))
    } else {
        None
    };
    let ctr = (pu.x + pu.w / 2, pu.y + pu.h / 2);
    let ref0_poc = list0.first().map(|r| r.poc).unwrap_or(cur_poc);
    let mv0 = br
        .and_then(|b| nb.temporal(b.0, b.1, 0, ref0_poc, cur_poc))
        .or_else(|| nb.temporal(ctr.0, ctr.1, 0, ref0_poc, cur_poc));
    let mv1 = if pu.is_b {
        let ref1_poc = list1.first().map(|r| r.poc).unwrap_or(cur_poc);
        br.and_then(|b| nb.temporal(b.0, b.1, 1, ref1_poc, cur_poc))
            .or_else(|| nb.temporal(ctr.0, ctr.1, 1, ref1_poc, cur_poc))
    } else {
        None
    };
    if mv0.is_none() && mv1.is_none() {
        return None;
    }

    let mut cand = MergeCand {
        pred: PredFlags::default(),
        mv: [Mv::default(); 2],
        ref_idx: [-1; 2],
    };
    if let Some(mv0) = mv0 {
        cand.pred.l0 = true;
        cand.mv[0] = mv0;
        cand.ref_idx[0] = 0;
    }
    if let Some(mv1) = mv1 {
        cand.pred.l1 = true;
        cand.mv[1] = mv1;
        cand.ref_idx[1] = 0;
    }
    Some(cand)
}

/// Combined bi-predictive merge candidates (§8.5.3.2.4).
fn derive_combined(
    cands: &mut Vec<MergeCand>,
    max_cand: usize,
    list0: &[RefEntry],
    list1: &[RefEntry],
) {
    if cands.len() < 2 || cands.len() >= max_cand {
        return;
    }
    static L0: [usize; 12] = [0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3];
    static L1: [usize; 12] = [1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2];
    let n = cands.len();
    let mut k = 0;
    while cands.len() < max_cand && k < n * (n - 1) {
        let i0 = L0[k % 12];
        let i1 = L1[k % 12];
        k += 1;
        if i0 >= n || i1 >= n {
            continue;
        }
        let a = cands[i0];
        let b = cands[i1];
        if a.pred.l0 && b.pred.l1 {
            // A combined candidate is added when the two sides reference
            // different pictures OR have different motion (§8.5.3.2.4). The
            // comparison is on the referenced *picture* (POC), not the ref
            // index — the same index in L0 and L1 can point to different
            // pictures.
            let poc0 = list0.get(a.ref_idx[0].max(0) as usize).map(|r| r.poc);
            let poc1 = list1.get(b.ref_idx[1].max(0) as usize).map(|r| r.poc);
            if poc0 != poc1 || a.mv[0] != b.mv[1] {
                cands.push(MergeCand {
                    pred: PredFlags { l0: true, l1: true },
                    mv: [a.mv[0], b.mv[1]],
                    ref_idx: [a.ref_idx[0], b.ref_idx[1]],
                });
            }
        }
    }
}

/// Zero-motion candidates (§8.5.3.2.5).
fn derive_zero(cands: &mut Vec<MergeCand>, max_cand: usize, is_b: bool, n0: usize, n1: usize) {
    let num_ref = if is_b { n0.min(n1) } else { n0 };
    if num_ref == 0 {
        // No eligible references (e.g. IBC-only list): no zero candidates are
        // appended and the merge list stays short.
        return;
    }
    let mut zero_idx = 0i8;
    while cands.len() < max_cand {
        let r = if (zero_idx as usize) < num_ref {
            zero_idx
        } else {
            0
        };
        cands.push(MergeCand {
            pred: PredFlags { l0: true, l1: is_b },
            mv: [Mv::default(), Mv::default()],
            ref_idx: [r, if is_b { r } else { -1 }],
        });
        zero_idx += 1;
        if zero_idx > 32 {
            break;
        }
    }
}

fn zero_cand(is_b: bool) -> MergeCand {
    MergeCand {
        pred: PredFlags { l0: true, l1: is_b },
        mv: [Mv::default(), Mv::default()],
        ref_idx: [0, if is_b { 0 } else { -1 }],
    }
}

fn push_unique(cands: &mut Vec<MergeCand>, c: MergeCand, max_cand: usize) {
    if cands.len() < max_cand {
        cands.push(c);
    }
}

/// AMVP predictor derivation (§8.5.3.2.6). Returns the two predictor MVs for
/// `mvp_flag` selection. `ref_idx`/`list` identify the target reference.
pub(crate) fn derive_amvp<N: Neighbors>(
    nb: &N,
    pu: &PuGeom,
    list: usize,
    ref_poc: i32,
    temporal_enabled: bool,
    col_list0: &[RefEntry],
    col_list1: &[RefEntry],
) -> [Mv; 2] {
    let (x, y, w, h) = (pu.x as isize, pu.y as isize, pu.w as isize, pu.h as isize);
    let cur_poc = nb.cur_poc();
    let mut preds: Vec<Mv> = Vec::with_capacity(2);

    let a_pos = [(x - 1, y + h), (x - 1, y + h - 1)];
    let b_pos = [(x + w, y - 1), (x + w - 1, y - 1), (x - 1, y - 1)];

    // Current-picture references use the same normative AMVP candidate
    // derivation as every other reference (§8.5.3.2.6–8). Their integer-unit
    // MVD combination is applied by the caller after predictor selection.

    // isScaledFlagLX (§8.5.3.2.7): set when either A block is available for
    // motion — an available, non-intra neighbor. An intra A neighbor does not
    // set the flag (de265 available_pred_blk returns false for intra), which is
    // what lets the B group produce a *scaled* candidate when A is intra/absent.
    let a_inter = |px: isize, py: isize| -> bool {
        prediction_block_available(nb, pu, px, py)
            && nb.motion_at(px, py).is_some_and(|m| !m.is_intra)
    };
    let is_scaled_flag = a_inter(a_pos[0].0, a_pos[0].1) || a_inter(a_pos[1].0, a_pos[1].1);

    // Long-term status of the target reference (§8.5.3.2.7): a spatial candidate
    // that references a picture of a different long-term/short-term type is not
    // usable, and a long-term target is never scaled.
    let target_lt = col_list0
        .iter()
        .chain(col_list1.iter())
        .find(|r| r.poc == ref_poc)
        .map(|r| r.long_term)
        .unwrap_or(false);

    // A candidate: same-POC pass over A0/A1, then same-type (scaled) pass.
    let mut a = amvp_spatial(
        nb, pu, &a_pos, list, ref_poc, cur_poc, true, true, target_lt,
    );
    // B candidate: same-POC pass only (scaling gated on isScaledFlag below).
    let mut b = amvp_spatial(
        nb, pu, &b_pos, list, ref_poc, cur_poc, false, true, target_lt,
    );

    if !is_scaled_flag {
        // No motion-available A block: the unscaled same-POC B result stands in
        // as the A candidate, and B is re-derived taking the first available
        // neighbor of matching reference type and scaling it (§8.5.3.2.7 step 5).
        // This re-derivation does NOT repeat the same-POC pass, so a later
        // same-POC neighbor does not pre-empt an earlier scalable one.
        if a.is_none() {
            a = b;
        }
        b = amvp_spatial(
            nb, pu, &b_pos, list, ref_poc, cur_poc, true, false, target_lt,
        );
    }

    if let Some(mv) = a {
        preds.push(mv);
    }
    if let Some(mv) = b
        && a != Some(mv)
    {
        preds.push(mv);
    }

    // Temporal predictor.
    if preds.len() < 2 && temporal_enabled {
        let ctb_log2 = nb.ctb_log2();
        let br_y = pu.y + pu.h;
        let br: Option<(usize, usize)> = if (br_y >> ctb_log2) == (pu.y >> ctb_log2) {
            Some((pu.x + pu.w, br_y))
        } else {
            None
        };
        let ctr = (pu.x + pu.w / 2, pu.y + pu.h / 2);
        let _ = (col_list0, col_list1);
        if let Some(mv) = br
            .and_then(|b| nb.temporal(b.0, b.1, list, ref_poc, cur_poc))
            .or_else(|| nb.temporal(ctr.0, ctr.1, list, ref_poc, cur_poc))
        {
            preds.push(mv);
        }
    }

    while preds.len() < 2 {
        preds.push(Mv::default());
    }
    [preds[0], preds[1]]
}

#[allow(clippy::too_many_arguments)]
fn amvp_spatial<N: Neighbors>(
    nb: &N,
    pu: &PuGeom,
    positions: &[(isize, isize)],
    list: usize,
    ref_poc: i32,
    cur_poc: i32,
    allow_scaled: bool,
    do_same_ref: bool,
    target_lt: bool,
) -> Option<Mv> {
    // First pass: exact ref (same-POC) match, no scaling.
    if do_same_ref {
        for &(px, py) in positions {
            if !prediction_block_available(nb, pu, px, py) {
                continue;
            }
            if let Some(m) = nb.motion_at(px, py) {
                if m.is_intra {
                    continue;
                }
                for l in [list, 1 - list] {
                    if m.pred_uses(l) && m.ref_poc[l] == ref_poc {
                        return Some(m.mv[l]);
                    }
                }
            }
        }
    }
    // Second pass: scale from the first available neighbor whose reference is
    // the same long-term/short-term type as the target (§8.5.3.2.7 step 5). A
    // long-term target uses the MV unscaled; short-term targets scale by POC
    // distance. Only the A group may always scale; for B this runs solely when
    // isScaledFlagLX==0.
    if !allow_scaled {
        return None;
    }
    for &(px, py) in positions {
        if !prediction_block_available(nb, pu, px, py) {
            continue;
        }
        if let Some(m) = nb.motion_at(px, py) {
            if m.is_intra {
                continue;
            }
            for l in [list, 1 - list] {
                if m.pred_uses(l) && m.ref_lt[l] == target_lt {
                    if target_lt {
                        return Some(m.mv[l]);
                    }
                    let col_dist = cur_poc - m.ref_poc[l];
                    let cur_dist = cur_poc - ref_poc;
                    return Some(scale_mv(m.mv[l], col_dist, cur_dist));
                }
            }
        }
    }
    None
}

impl MotionInfo {
    #[inline]
    fn pred_uses(&self, l: usize) -> bool {
        (l == 0 && self.pred.l0) || (l == 1 && self.pred.l1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct L1OnlyTemporal;

    impl Neighbors for L1OnlyTemporal {
        fn motion_at(&self, _x: isize, _y: isize) -> Option<MotionInfo> {
            None
        }

        fn available(&self, _x: isize, _y: isize) -> bool {
            false
        }

        fn temporal(
            &self,
            _x: usize,
            _y: usize,
            list: usize,
            _ref_poc: i32,
            _cur_poc: i32,
        ) -> Option<Mv> {
            (list == 1).then_some(Mv::new(12, -4))
        }

        fn cur_poc(&self) -> i32 {
            8
        }

        fn ctb_log2(&self) -> u32 {
            6
        }
    }

    #[test]
    fn scale_identity_when_equal_dist() {
        let mv = Mv::new(10, -6);
        let s = scale_mv(mv, 4, 4);
        assert_eq!(s, mv);
    }

    #[test]
    fn scale_halves_on_double_dist() {
        let mv = Mv::new(16, 0);
        // col_dist 8, cur_dist 4 -> factor ~0.5
        let s = scale_mv(mv, 8, 4);
        assert!((s.x - 8).abs() <= 1);
    }

    #[test]
    fn current_picture_amvp_uses_normative_zero_padding() {
        let pu = PuGeom {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
            cu_x: 0,
            cu_y: 0,
            is_b: false,
            part_idx: 0,
            cu_w: 16,
            cu_h: 16,
            par_mrg_level: 2,
            curr_pic_ref: false,
        };
        let current = [RefEntry {
            _dpb_index: usize::MAX,
            poc: 8,
            long_term: true,
        }];

        // With no available spatial or temporal candidate, §8.5.3.2.6 pads
        // the AMVP list with zero MVs. Older SCM-only default block vectors
        // must not be injected here: they alter the explicitly coded MVD.
        assert_eq!(
            derive_amvp(&L1OnlyTemporal, &pu, 0, 8, false, &current, &[]),
            [Mv::default(), Mv::default()]
        );
    }

    #[test]
    fn temporal_merge_keeps_l1_only_candidate() {
        let pu = PuGeom {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
            cu_x: 0,
            cu_y: 0,
            is_b: true,
            part_idx: 0,
            cu_w: 16,
            cu_h: 16,
            par_mrg_level: 2,
            curr_pic_ref: false,
        };
        let refs0 = [RefEntry {
            _dpb_index: 0,
            poc: 4,
            long_term: false,
        }];
        let refs1 = [RefEntry {
            _dpb_index: 1,
            poc: 12,
            long_term: false,
        }];

        let cand = temporal_merge(&L1OnlyTemporal, &pu, &refs0, &refs1).unwrap();
        assert_eq!(
            cand.pred,
            PredFlags {
                l0: false,
                l1: true
            }
        );
        assert_eq!(cand.mv, [Mv::default(), Mv::new(12, -4)]);
        assert_eq!(cand.ref_idx, [-1, 0]);
    }

    #[test]
    fn default_parallel_merge_level_filters_same_region() {
        struct OneNeighbor;
        impl Neighbors for OneNeighbor {
            fn motion_at(&self, _x: isize, _y: isize) -> Option<MotionInfo> {
                Some(MotionInfo {
                    pred: PredFlags {
                        l0: true,
                        l1: false,
                    },
                    ref_idx: [0, -1],
                    ..Default::default()
                })
            }

            fn available(&self, _x: isize, _y: isize) -> bool {
                true
            }

            fn temporal(
                &self,
                _x: usize,
                _y: usize,
                _list: usize,
                _ref_poc: i32,
                _cur_poc: i32,
            ) -> Option<Mv> {
                None
            }

            fn cur_poc(&self) -> i32 {
                0
            }

            fn ctb_log2(&self) -> u32 {
                6
            }
        }

        let pu = PuGeom {
            x: 10,
            y: 10,
            w: 4,
            h: 4,
            cu_x: 8,
            cu_y: 8,
            is_b: false,
            part_idx: 0,
            cu_w: 4,
            cu_h: 4,
            par_mrg_level: 2,
            curr_pic_ref: false,
        };
        assert!(spatial(&OneNeighbor, 9, 10, &pu, SpatialPos::A1).is_none());
    }

    #[test]
    fn previous_pu_in_same_cu_does_not_require_decoded_map() {
        struct SameCuNeighbor;
        impl Neighbors for SameCuNeighbor {
            fn motion_at(&self, _x: isize, _y: isize) -> Option<MotionInfo> {
                Some(MotionInfo {
                    pred: PredFlags {
                        l0: true,
                        l1: false,
                    },
                    ref_idx: [0, -1],
                    ..Default::default()
                })
            }

            fn available(&self, _x: isize, _y: isize) -> bool {
                false
            }

            fn temporal(
                &self,
                _x: usize,
                _y: usize,
                _list: usize,
                _ref_poc: i32,
                _cur_poc: i32,
            ) -> Option<Mv> {
                None
            }

            fn cur_poc(&self) -> i32 {
                0
            }

            fn ctb_log2(&self) -> u32 {
                6
            }
        }

        let pu = PuGeom {
            x: 0,
            y: 8,
            w: 8,
            h: 8,
            cu_x: 0,
            cu_y: 0,
            is_b: false,
            part_idx: 2,
            cu_w: 16,
            cu_h: 16,
            par_mrg_level: 0,
            curr_pic_ref: false,
        };
        assert!(prediction_block_available(&SameCuNeighbor, &pu, 7, 7));
        assert!(spatial(&SameCuNeighbor, 7, 7, &pu, SpatialPos::B1).is_some());
    }

    #[test]
    fn second_nxn_pu_rejects_unavailable_same_cu_region() {
        struct AlwaysAvailable;
        impl Neighbors for AlwaysAvailable {
            fn motion_at(&self, _x: isize, _y: isize) -> Option<MotionInfo> {
                Some(MotionInfo::default())
            }

            fn available(&self, _x: isize, _y: isize) -> bool {
                true
            }

            fn temporal(
                &self,
                _x: usize,
                _y: usize,
                _list: usize,
                _ref_poc: i32,
                _cur_poc: i32,
            ) -> Option<Mv> {
                None
            }

            fn cur_poc(&self) -> i32 {
                0
            }

            fn ctb_log2(&self) -> u32 {
                6
            }
        }

        let pu = PuGeom {
            x: 8,
            y: 0,
            w: 8,
            h: 8,
            cu_x: 0,
            cu_y: 0,
            is_b: false,
            part_idx: 1,
            cu_w: 16,
            cu_h: 16,
            par_mrg_level: 0,
            curr_pic_ref: false,
        };
        assert!(!prediction_block_available(&AlwaysAvailable, &pu, 7, 8));
        assert!(prediction_block_available(&AlwaysAvailable, &pu, 8, 8));
    }

    #[test]
    fn zero_merge_candidates_include_current_picture_reference_slot() {
        let pu = PuGeom {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
            cu_x: 0,
            cu_y: 0,
            is_b: false,
            part_idx: 0,
            cu_w: 16,
            cu_h: 16,
            par_mrg_level: 2,
            curr_pic_ref: false,
        };
        let refs = [
            RefEntry {
                _dpb_index: 0,
                poc: 4,
                long_term: false,
            },
            RefEntry {
                _dpb_index: usize::MAX,
                poc: 8,
                long_term: true,
            },
        ];
        let cand = derive_merge(&L1OnlyTemporal, &pu, 1, 2, false, &refs, &[]);
        assert_eq!(cand.ref_idx, [1, -1]);
        assert_eq!(cand.mv, [Mv::default(), Mv::default()]);
    }
}
