/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * // BSD-3-Clause OR Apache-2.0
 */

//! Motion vector derivation: spatial/temporal merge candidate lists, combined
//! bi-predictive and zero candidates, and AMVP predictor derivation. Ported
//! from de265's `motion.cc` (§8.5.3.2), rewritten in safe Rust. The decoder
//! supplies neighbour motion via the `Neighbours` accessor so this module stays
//! free of picture-buffer details.

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

/// Motion of a spatial neighbour, or `None` if unavailable/intra.
pub(crate) trait Neighbours {
    /// Motion at luma position (x, y) if available and inter-coded and in the
    /// same slice/tile; else None.
    fn motion_at(&self, x: isize, y: isize) -> Option<MotionInfo>;
    /// Whether (x, y) is available for prediction from (cur_x, cur_y).
    fn available(&self, x: isize, y: isize) -> bool;
    /// Collocated temporal motion for target position, scaled to `ref_poc`.
    fn temporal(&self, x: usize, y: usize, list: usize, ref_poc: i32, cur_poc: i32) -> Option<Mv>;
    fn cur_poc(&self) -> i32;
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

/// Position of a PU for neighbour derivation.
pub(crate) struct PuGeom {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) w: usize,
    pub(crate) h: usize,
    /// Whether this is a B slice (enables L1 and combined candidates).
    pub(crate) is_b: bool,
    /// Part index and mode for the second-PU redundancy checks (§8.5.3.2.1).
    pub(crate) part_idx: usize,
    /// Top-left of the CU (for redundancy of 2nd PU).
    pub(crate) cu_x: usize,
    pub(crate) cu_y: usize,
    pub(crate) cu_w: usize,
    pub(crate) cu_h: usize,
}

/// Derive the merge candidate list (§8.5.3.2.1) and return the candidate at
/// `merge_idx`. `max_cand` = MaxNumMergeCand. `col_ref` supplies the collocated
/// reference POC for temporal candidates; `list_poc` maps ref_idx→POC per list.
#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_merge<N: Neighbours>(
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

    // Temporal candidate (§8.5.3.2.7): collocated bottom-right then centre.
    if temporal_enabled
        && cands.len() < max_cand
        && let Some(tc) = temporal_merge(nb, pu, list0, list1)
    {
        cands.push(tc);
    }

    // Combined bi-predictive candidates (B slices).
    if pu.is_b {
        derive_combined(&mut cands, max_cand);
    }

    // Zero motion candidates.
    derive_zero(&mut cands, max_cand, pu.is_b, list0.len(), list1.len());

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

fn spatial<N: Neighbours>(
    nb: &N,
    px: isize,
    py: isize,
    pu: &PuGeom,
    pos: SpatialPos,
) -> Option<MergeCand> {
    // Second-PU redundancy: a neighbour inside the same CU's first PU is
    // unavailable (§8.5.3.2.2). Simplified for common partition shapes.
    if pu.part_idx == 1 && inside_first_pu(px, py, pu, pos) {
        return None;
    }
    if !nb.available(px, py) {
        return None;
    }
    let m = nb.motion_at(px, py)?;
    if m.is_intra {
        return None;
    }
    Some(MergeCand::from_motion(&m))
}

fn inside_first_pu(px: isize, py: isize, pu: &PuGeom, _pos: SpatialPos) -> bool {
    // If the neighbour position lies within the CU but before this PU's start,
    // it belongs to the first PU. Covers Nx2N/2NxN second-PU exclusion.
    let within_cu = px >= pu.cu_x as isize
        && px < (pu.cu_x + pu.cu_w) as isize
        && py >= pu.cu_y as isize
        && py < (pu.cu_y + pu.cu_h) as isize;
    within_cu && !(px >= pu.x as isize && py >= pu.y as isize)
}

fn temporal_merge<N: Neighbours>(
    nb: &N,
    pu: &PuGeom,
    list0: &[RefEntry],
    list1: &[RefEntry],
) -> Option<MergeCand> {
    let cur_poc = nb.cur_poc();
    // Bottom-right collocated, fallback to centre.
    let br = (pu.x + pu.w, pu.y + pu.h);
    let ctr = (pu.x + pu.w / 2, pu.y + pu.h / 2);
    let ref0_poc = list0.first().map(|r| r.poc).unwrap_or(cur_poc);
    let mv0 = nb
        .temporal(br.0, br.1, 0, ref0_poc, cur_poc)
        .or_else(|| nb.temporal(ctr.0, ctr.1, 0, ref0_poc, cur_poc))?;
    let mut cand = MergeCand {
        pred: PredFlags {
            l0: true,
            l1: false,
        },
        mv: [mv0, Mv::default()],
        ref_idx: [0, -1],
    };
    if pu.is_b {
        let ref1_poc = list1.first().map(|r| r.poc).unwrap_or(cur_poc);
        if let Some(mv1) = nb
            .temporal(br.0, br.1, 1, ref1_poc, cur_poc)
            .or_else(|| nb.temporal(ctr.0, ctr.1, 1, ref1_poc, cur_poc))
        {
            cand.pred.l1 = true;
            cand.mv[1] = mv1;
            cand.ref_idx[1] = 0;
        }
    }
    Some(cand)
}

/// Combined bi-predictive merge candidates (§8.5.3.2.4).
fn derive_combined(cands: &mut Vec<MergeCand>, max_cand: usize) {
    if cands.len() < 2 || cands.len() >= max_cand {
        return;
    }
    const L0: [usize; 12] = [0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3];
    const L1: [usize; 12] = [1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2];
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
            // Distinct references required.
            if a.ref_idx[0] != b.ref_idx[1] || a.mv[0] != b.mv[1] {
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
pub(crate) fn derive_amvp<N: Neighbours>(
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

    // isScaledFlagLX (§8.5.3.2.7 step 5): set when either A block is available
    // as a *block* — including intra neighbours. It gates whether the B group
    // may produce a scaled candidate.
    let is_scaled_flag =
        nb.available(a_pos[0].0, a_pos[0].1) || nb.available(a_pos[1].0, a_pos[1].1);

    // A candidate: same-ref pass over A0/A1, then scaled pass (always allowed).
    let mut a = amvp_spatial(nb, &a_pos, list, ref_poc, cur_poc, true);
    // B candidate: same-ref pass only.
    let mut b = amvp_spatial(nb, &b_pos, list, ref_poc, cur_poc, false);

    if !is_scaled_flag {
        // No A blocks: the unscaled B result stands in as the A candidate, and
        // B is re-derived with scaling permitted.
        if a.is_none() {
            a = b;
        }
        b = amvp_spatial(nb, &b_pos, list, ref_poc, cur_poc, true);
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
        let br = (pu.x + pu.w, pu.y + pu.h);
        let ctr = (pu.x + pu.w / 2, pu.y + pu.h / 2);
        let _ = (col_list0, col_list1);
        if let Some(mv) = nb
            .temporal(br.0, br.1, list, ref_poc, cur_poc)
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

fn amvp_spatial<N: Neighbours>(
    nb: &N,
    positions: &[(isize, isize)],
    list: usize,
    ref_poc: i32,
    cur_poc: i32,
    allow_scaled: bool,
) -> Option<Mv> {
    // First pass: exact ref match (no scaling).
    for &(px, py) in positions {
        if !nb.available(px, py) {
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
    // Second pass: scale from any available inter MV. Only the A group may
    // always scale; for B this runs solely when isScaledFlagLX==0 (§8.5.3.2.7).
    if !allow_scaled {
        return None;
    }
    for &(px, py) in positions {
        if !nb.available(px, py) {
            continue;
        }
        if let Some(m) = nb.motion_at(px, py) {
            if m.is_intra {
                continue;
            }
            for l in [list, 1 - list] {
                if m.pred_uses(l) {
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
}
