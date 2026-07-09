/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * // BSD-3-Clause OR Apache-2.0
 */

//! Decoded Picture Buffer (DPB): holds reconstructed reference frames with their
//! POC and per-4x4 motion field, performs RPS-based reference marking, builds
//! RefPicList0/RefPicList1 for the current slice, and emits frames in output
//! (POC) order. Rewritten in safe Rust following de265's `decctx`/`dpb` flow.

use crate::inter::MotionInfo;
use crate::rps::ShortTermRps;
use crate::yuv::YuvPlanes;

/// One stored picture and its decode-time metadata.
pub(crate) struct Frame {
    pub(crate) planes: YuvPlanes,
    pub(crate) poc: i32,
    /// Per-4x4-block motion field in raster order over the coded picture,
    /// `width4 * height4` entries.
    pub(crate) motion: Vec<MotionInfo>,
    pub(crate) width4: usize,
    pub(crate) height4: usize,
    /// Reference-marking state.
    pub(crate) short_term: bool,
    pub(crate) long_term: bool,
    /// Whether this frame still needs to be output.
    pub(crate) needed_for_output: bool,
}

impl Frame {
    #[inline]
    pub(crate) fn is_reference(&self) -> bool {
        self.short_term || self.long_term
    }

    /// Motion info at luma sample (x, y), clamped to the frame.
    #[inline]
    pub(crate) fn motion_at(&self, x: usize, y: usize) -> MotionInfo {
        let bx = (x >> 2).min(self.width4.saturating_sub(1));
        let by = (y >> 2).min(self.height4.saturating_sub(1));
        self.motion
            .get(by * self.width4 + bx)
            .copied()
            .unwrap_or_else(MotionInfo::intra)
    }
}

/// A reference picture list entry: an index into the DPB plus whether it is a
/// long-term reference (affects MV scaling) and its POC.
#[derive(Clone, Copy)]
pub(crate) struct RefEntry {
    pub(crate) dpb_index: usize,
    pub(crate) poc: i32,
    pub(crate) long_term: bool,
}

pub(crate) struct Dpb {
    pub(crate) frames: Vec<Frame>,
    /// Max frames retained before forcing output (from SPS max_dec_pic_buffering).
    max_frames: usize,
}

impl Dpb {
    pub(crate) fn new(max_frames: usize) -> Self {
        Dpb {
            frames: Vec::new(),
            max_frames: max_frames.max(1),
        }
    }

    /// Clear all references (IDR): frames stay only if still needed for output.
    pub(crate) fn clear_refs(&mut self) {
        for f in &mut self.frames {
            f.short_term = false;
            f.long_term = false;
        }
    }

    /// Insert a freshly decoded frame; returns its index.
    pub(crate) fn push(&mut self, frame: Frame) -> usize {
        self.frames.push(frame);
        self.frames.len() - 1
    }

    /// Find a short-term reference frame by POC.
    fn find_by_poc(&self, poc: i32) -> Option<usize> {
        self.frames
            .iter()
            .position(|f| f.poc == poc && f.is_reference())
    }

    /// Apply RPS marking for the current picture (§8.3.2): mark frames whose POC
    /// is in the current RPS as short-term references, all others as unused.
    /// Returns the POC-sorted before/after sets for list construction.
    pub(crate) fn apply_rps(&mut self, cur_poc: i32, rps: &ShortTermRps) -> RpsPocs {
        let mut before = Vec::new(); // negative delta (used_by_curr)
        let mut after = Vec::new(); // positive delta (used_by_curr)
        let mut foll = Vec::new(); // referenced but not used by current

        for (d, &used) in rps.delta_poc_s0.iter().zip(&rps.used_s0) {
            let poc = cur_poc + d;
            if used {
                before.push(poc);
            } else {
                foll.push(poc);
            }
        }
        for (d, &used) in rps.delta_poc_s1.iter().zip(&rps.used_s1) {
            let poc = cur_poc + d;
            if used {
                after.push(poc);
            } else {
                foll.push(poc);
            }
        }

        // Mark: any frame not in {before,after,foll,long-term} loses ref status.
        let keep: Vec<i32> = before
            .iter()
            .chain(after.iter())
            .chain(foll.iter())
            .copied()
            .collect();
        for f in &mut self.frames {
            if f.short_term && !keep.contains(&f.poc) {
                f.short_term = false;
            }
        }
        RpsPocs { before, after }
    }

    /// Build RefPicList0/1 (§8.3.4) from the marked references. `list_mod_*` are
    /// the explicit reorder indices (empty = default order).
    pub(crate) fn build_ref_lists(
        &self,
        pocs: &RpsPocs,
        num_l0: usize,
        num_l1: usize,
        is_b: bool,
        list_mod_l0: &[u32],
        list_mod_l1: &[u32],
    ) -> (Vec<RefEntry>, Vec<RefEntry>) {
        // Temp list0: before (near→far), then after, then long-term.
        let mut temp0: Vec<RefEntry> = Vec::new();
        for &poc in &pocs.before {
            if let Some(i) = self.find_by_poc(poc) {
                temp0.push(RefEntry {
                    dpb_index: i,
                    poc,
                    long_term: false,
                });
            }
        }
        for &poc in &pocs.after {
            if let Some(i) = self.find_by_poc(poc) {
                temp0.push(RefEntry {
                    dpb_index: i,
                    poc,
                    long_term: false,
                });
            }
        }
        // List1 temp: after first, then before.
        let mut temp1: Vec<RefEntry> = Vec::new();
        for &poc in &pocs.after {
            if let Some(i) = self.find_by_poc(poc) {
                temp1.push(RefEntry {
                    dpb_index: i,
                    poc,
                    long_term: false,
                });
            }
        }
        for &poc in &pocs.before {
            if let Some(i) = self.find_by_poc(poc) {
                temp1.push(RefEntry {
                    dpb_index: i,
                    poc,
                    long_term: false,
                });
            }
        }

        let list0 = finalize_list(&temp0, num_l0, list_mod_l0);
        let list1 = if is_b {
            finalize_list(&temp1, num_l1, list_mod_l1)
        } else {
            Vec::new()
        };
        (list0, list1)
    }

    /// Emit frames that are no longer needed for reference and are due for output
    /// (POC order). Called after each picture; returns POC-ordered outputs and
    /// removes them from the DPB. `flush` forces all remaining outputs.
    pub(crate) fn bump(&mut self, flush: bool) -> Vec<Frame> {
        let mut out = Vec::new();
        loop {
            let pending = self.frames.iter().filter(|f| f.needed_for_output).count();
            let over = self.frames.len() > self.max_frames;
            if !(flush && pending > 0) && !(over && pending > 0) {
                break;
            }
            // Output the lowest-POC frame still needing output.
            let mut best: Option<usize> = None;
            for (i, f) in self.frames.iter().enumerate() {
                if f.needed_for_output && best.is_none_or(|b| f.poc < self.frames[b].poc) {
                    best = Some(i);
                }
            }
            let Some(bi) = best else { break };
            self.frames[bi].needed_for_output = false;
            // Remove if also not a reference.
            if !self.frames[bi].is_reference() {
                out.push(self.frames.remove(bi));
            } else {
                // Reference but output: clone-free move is impossible while it
                // stays a reference, so we output a shallow copy of planes.
                out.push(self.frames[bi].shallow_output());
            }
            if !flush {
                break;
            }
        }
        // Drop non-reference, already-output frames to reclaim space.
        self.frames
            .retain(|f| f.is_reference() || f.needed_for_output);
        out
    }
}

impl Frame {
    /// Produce an output-only copy of the frame (planes cloned, motion dropped).
    fn shallow_output(&self) -> Frame {
        Frame {
            planes: self.planes.clone(),
            poc: self.poc,
            motion: Vec::new(),
            width4: 0,
            height4: 0,
            short_term: false,
            long_term: false,
            needed_for_output: false,
        }
    }
}

/// POC-sorted RPS partitions for the current picture.
pub(crate) struct RpsPocs {
    /// used_by_curr negative-delta POCs, nearest first.
    pub(crate) before: Vec<i32>,
    /// used_by_curr positive-delta POCs, nearest first.
    pub(crate) after: Vec<i32>,
}

/// Repeat the temp list to `num` entries, then apply explicit reordering.
fn finalize_list(temp: &[RefEntry], num: usize, list_mod: &[u32]) -> Vec<RefEntry> {
    if temp.is_empty() || num == 0 {
        return Vec::new();
    }
    // RefPicListTemp is cyclically extended to at least `num` entries.
    let mut extended: Vec<RefEntry> = Vec::with_capacity(num.max(temp.len()));
    while extended.len() < num {
        for &e in temp {
            if extended.len() >= num {
                break;
            }
            extended.push(e);
        }
    }
    if list_mod.is_empty() {
        extended.truncate(num);
        extended
    } else {
        list_mod
            .iter()
            .take(num)
            .map(|&idx| extended[(idx as usize).min(extended.len() - 1)])
            .collect()
    }
}

// YuvPlanes needs Clone for shallow_output.
impl Clone for YuvPlanes {
    fn clone(&self) -> Self {
        YuvPlanes {
            y: self.y.clone(),
            cb: self.cb.clone(),
            cr: self.cr.clone(),
            width: self.width,
            height: self.height,
            chroma: self.chroma,
            bit_depth: self.bit_depth,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fmt::{BitDepth, ChromaFormat};

    fn mk_frame(poc: i32) -> Frame {
        Frame {
            planes: YuvPlanes {
                y: vec![0; 4],
                cb: vec![0; 1],
                cr: vec![0; 1],
                width: 2,
                height: 2,
                chroma: ChromaFormat::Yuv420,
                bit_depth: BitDepth::Eight,
            },
            poc,
            motion: vec![MotionInfo::intra(); 1],
            width4: 1,
            height4: 1,
            short_term: true,
            long_term: false,
            needed_for_output: true,
        }
    }

    #[test]
    fn ref_list_orders_before_then_after() {
        let mut dpb = Dpb::new(8);
        dpb.push(mk_frame(0));
        dpb.push(mk_frame(4));
        dpb.push(mk_frame(8));
        // current poc 6: before = {4,0}, after = {8}
        let rps = ShortTermRps {
            delta_poc_s0: vec![-2, -6],
            used_s0: vec![true, true],
            delta_poc_s1: vec![2],
            used_s1: vec![true],
        };
        let pocs = dpb.apply_rps(6, &rps);
        assert_eq!(pocs.before, vec![4, 0]);
        assert_eq!(pocs.after, vec![8]);
        let (l0, l1) = dpb.build_ref_lists(&pocs, 2, 1, true, &[], &[]);
        assert_eq!(l0[0].poc, 4);
        assert_eq!(l0[1].poc, 0);
        assert_eq!(l1[0].poc, 8);
    }
}
