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
//! Decoded Picture Buffer (DPB): holds reconstructed reference frames with their
//! POC and per-4x4 motion field, performs RPS-based reference marking, builds
//! RefPicList0/RefPicList1 for the current slice, and emits frames in output
//! (POC) order. Rewritten in safe Rust following de265's `decctx`/`dpb` flow.

use crate::error::DecodeError;
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
    /// PicLatencyCount (§C.5.2.2), advanced once for each subsequently decoded
    /// picture while this picture remains pending for output.
    pub(crate) latency: usize,
    /// Display metadata (conformance-window crop, dimensions, colour) captured
    /// from the SPS active when this picture was decoded. Carried per-frame so a
    /// picture emitted after a later SPS activation (e.g. a stream of IDRs each
    /// changing resolution) is still cropped with its own SPS, not the newest.
    pub(crate) meta: Option<crate::video::FrameMeta>,
}

impl Frame {
    #[inline]
    pub(crate) fn is_reference(&self) -> bool {
        self.short_term || self.long_term
    }
}

/// A reference picture list entry: an index into the DPB plus whether it is a
/// long-term reference (affects MV scaling) and its POC.
#[derive(Clone, Copy)]
pub(crate) struct RefEntry {
    pub(crate) _dpb_index: usize,
    pub(crate) poc: i32,
    pub(crate) long_term: bool,
}

pub(crate) struct Dpb {
    pub(crate) frames: Vec<Frame>,
    /// Max frames retained before forcing output (from SPS max_dec_pic_buffering).
    max_frames: usize,
    /// sps_max_num_reorder_pics: at most this many output-pending pictures may be
    /// held before the lowest-POC one is bumped for output (§C.5.2.2).
    max_reorder: usize,
    max_latency: Option<usize>,
    /// NoRaslOutputFlag of the current random-access period: when true, RASL
    /// pictures attached to the period's IRAP are discarded.
    no_rasl_output: bool,
}

impl Dpb {
    pub(crate) fn new(max_frames: usize) -> Self {
        Dpb {
            frames: Vec::new(),
            max_frames: max_frames.max(1),
            max_reorder: max_frames.max(1),
            max_latency: None,
            no_rasl_output: false,
        }
    }

    /// Configure the output DPB from the active SPS (§C.5.2.2). `buffering` is
    /// sps_max_dec_pic_buffering (DPB size) and `reorder` is
    /// sps_max_num_reorder_pics (reorder-latency bound).
    pub(crate) fn configure(
        &mut self,
        buffering: usize,
        reorder: usize,
        max_latency: Option<usize>,
    ) {
        self.max_frames = buffering.max(1);
        self.max_reorder = reorder.min(self.max_frames);
        self.max_latency = max_latency;
    }

    /// Whether RASL pictures in the current random-access period are suppressed.
    pub(crate) fn pending_no_rasl_output(&self) -> bool {
        self.no_rasl_output
    }

    /// Record the NoRaslOutputFlag for the random-access period just started.
    pub(crate) fn set_no_rasl_output(&mut self, v: bool) {
        self.no_rasl_output = v;
    }

    /// Discard pictures still pending output without emitting them, for an IRAP
    /// with NoOutputOfPriorPicsFlag == 1 (§C.5.2.2). Clears the output flag and
    /// drops frames no longer needed as references.
    pub(crate) fn discard_pending_output(&mut self) {
        for f in &mut self.frames {
            f.needed_for_output = false;
        }
        self.frames
            .retain(|f| f.is_reference() || f.needed_for_output);
    }

    /// Clear all references (IDR): frames stay only if still needed for output.
    pub(crate) fn clear_refs(&mut self) {
        for f in &mut self.frames {
            f.short_term = false;
            f.long_term = false;
        }
    }

    /// Insert a freshly decoded frame; returns its index.
    ///
    /// §C.5.2.3: when a picture is added to the DPB, PicLatencyCount is set to 0
    /// for it and incremented by one for every other picture still needed for
    /// output. Doing this at store time (rather than at the next before-decode
    /// bump) keeps the latency count in step with decode order, so a picture is
    /// never made latency-due before a lower-POC picture that follows it in
    /// decode order has actually been stored.
    pub(crate) fn push(&mut self, frame: Frame) -> usize {
        for f in &mut self.frames {
            if f.needed_for_output {
                f.latency = f.latency.saturating_add(1);
            }
        }
        self.frames.push(frame);
        self.frames.len() - 1
    }

    /// Run the Annex C output process before decoding the next picture. RPS
    /// marking has already released references not needed by that picture.
    pub(crate) fn bump_before_decode(&mut self) -> Vec<Frame> {
        self.bump_before_picture(true)
    }

    /// At a new CVS, reference-capacity pressure is resolved by the impending
    /// reference reset. Only reorder/latency constraints can make a prior
    /// picture due before no_output_of_prior_pics_flag discards the remainder.
    pub(crate) fn bump_before_irap(&mut self) -> Vec<Frame> {
        self.bump_before_picture(false)
    }

    fn bump_before_picture(&mut self, check_fullness: bool) -> Vec<Frame> {
        // RPS marking runs immediately before this process. Purge pictures that
        // it made unused and that were already output before evaluating DPB
        // fullness; counting those dead slots causes premature leading-picture
        // output at the next IRAP.
        self.frames
            .retain(|f| f.is_reference() || f.needed_for_output);

        let mut out = Vec::new();
        loop {
            let pending = self.frames.iter().filter(|f| f.needed_for_output).count();
            let full = check_fullness && self.frames.len() >= self.max_frames && pending != 0;
            let over_reorder = pending > self.max_reorder;
            let over_latency = self.max_latency.is_some_and(|limit| {
                self.frames
                    .iter()
                    .any(|f| f.needed_for_output && f.latency >= limit)
            });
            if !full && !over_reorder && !over_latency {
                break;
            }
            let Some(frame) = self.bump_one() else { break };
            out.push(frame);
        }
        self.frames
            .retain(|f| f.is_reference() || f.needed_for_output);
        out
    }

    /// Find a short-term reference frame by POC.
    fn find_by_poc(&self, poc: i32) -> Option<usize> {
        self.frames
            .iter()
            .position(|f| f.poc == poc && f.is_reference())
    }

    /// As `apply_rps`, but also marks long-term references. `lt_refs` is the
    /// slice's list of (poc_or_lsb, used_by_curr, has_msb); `max_poc_lsb` sizes
    /// the LSB-only match. Returns the used-by-curr short-term and long-term
    /// POCs for list construction.
    pub(crate) fn apply_rps_lt(
        &mut self,
        cur_poc: i32,
        rps: &ShortTermRps,
        lt_refs: &[(i32, bool, bool, i32)],
        max_poc_lsb: i32,
    ) -> RpsPocs {
        let mut before = Vec::new(); // negative delta (used_by_curr)
        let mut after = Vec::new(); // positive delta (used_by_curr)
        let mut foll = Vec::new(); // referenced but not used by current
        let mut lt = Vec::new(); // used_by_curr long-term
        let mut lt_keep = Vec::new(); // all long-term (used or foll)

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

        // Long-term references: resolve each to a DPB picture's POC. When the
        // MSB cycle was signalled we can reconstruct the full POC (§8.3.2):
        //   pocLt = cur_poc - deltaMsbCycle*MaxPocLsb - (cur_poc_lsb - pocLsbLt)
        // and match by full POC; otherwise only the LSB is known and we match by
        // LSB. `val` holds poc_lsb_lt and `delta_cycle` the MSB cycle.
        for &(val, used, has_msb, delta_cycle) in lt_refs {
            let matched = if has_msb && max_poc_lsb > 0 {
                let cur_lsb = cur_poc.rem_euclid(max_poc_lsb);
                let poc_lt = cur_poc - delta_cycle * max_poc_lsb - (cur_lsb - val);
                self.frames.iter().find(|f| f.poc == poc_lt).map(|f| f.poc)
            } else if max_poc_lsb > 0 {
                self.frames
                    .iter()
                    .find(|f| f.poc.rem_euclid(max_poc_lsb) == val.rem_euclid(max_poc_lsb))
                    .map(|f| f.poc)
            } else {
                None
            };
            if let Some(poc) = matched {
                lt_keep.push(poc);
                if used {
                    lt.push(poc);
                }
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
            if lt_keep.contains(&f.poc) {
                // Reclassify as a long-term reference.
                f.long_term = true;
                f.short_term = false;
            } else if f.long_term && !lt_keep.contains(&f.poc) {
                f.long_term = false;
            }
            if f.short_term && !keep.contains(&f.poc) {
                f.short_term = false;
            }
        }
        RpsPocs { before, after, lt }
    }

    /// Build RefPicList0/1 (§8.3.4) from the marked references. `list_mod_*` are
    /// the explicit reorder indices (empty = default order).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_ref_lists(
        &self,
        pocs: &RpsPocs,
        num_l0: usize,
        num_l1: usize,
        is_b: bool,
        current: Option<RefEntry>,
        list_mod_l0: &[u32],
        list_mod_l1: &[u32],
    ) -> Result<(Vec<RefEntry>, Vec<RefEntry>), DecodeError> {
        // Temp list0: before (near→far), then after, then long-term.
        let mut temp0: Vec<RefEntry> = Vec::new();
        for &poc in &pocs.before {
            if let Some(i) = self.find_by_poc(poc) {
                temp0.push(RefEntry {
                    _dpb_index: i,
                    poc,
                    long_term: false,
                });
            }
        }
        for &poc in &pocs.after {
            if let Some(i) = self.find_by_poc(poc) {
                temp0.push(RefEntry {
                    _dpb_index: i,
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
                    _dpb_index: i,
                    poc,
                    long_term: false,
                });
            }
        }
        for &poc in &pocs.before {
            if let Some(i) = self.find_by_poc(poc) {
                temp1.push(RefEntry {
                    _dpb_index: i,
                    poc,
                    long_term: false,
                });
            }
        }
        // Long-term references follow the short-term ones in both lists (§8.3.4).
        for &poc in &pocs.lt {
            if let Some(i) = self.find_by_poc(poc) {
                temp0.push(RefEntry {
                    _dpb_index: i,
                    poc,
                    long_term: true,
                });
                temp1.push(RefEntry {
                    _dpb_index: i,
                    poc,
                    long_term: true,
                });
            }
        }

        // SCC current-picture referencing is part of RefPicListTemp0/1, not an
        // extra entry appended after list construction (§8.3.4, equations 8-8
        // and 8-10). During picture decoding the current picture is treated as
        // a long-term reference.
        if let Some(curr) = current {
            temp0.push(curr);
            temp1.push(curr);
        }

        let mut list0 = finalize_list(&temp0, num_l0, list_mod_l0)?;
        // With implicit L0 ordering and more temp entries than active entries,
        // SCC forces the current picture into the final active L0 slot (8-9).
        if let Some(curr) = current
            && list_mod_l0.is_empty()
            && num_l0 != 0
            && temp0.len() > num_l0
        {
            list0[num_l0 - 1] = curr;
        }
        let list1 = if is_b {
            finalize_list(&temp1, num_l1, list_mod_l1)?
        } else {
            Vec::new()
        };
        Ok((list0, list1))
    }

    /// Emit frames that are no longer needed for reference and are due for output
    /// (POC order). Called after each picture; returns POC-ordered outputs and
    /// removes them from the DPB. `flush` forces all remaining outputs.
    pub(crate) fn bump(&mut self, flush: bool) -> Vec<Frame> {
        let mut out = Vec::new();
        loop {
            let pending = self.frames.iter().filter(|f| f.needed_for_output).count();
            // §C.5.2.2 bumping: output the lowest-POC pending picture when the
            // number of output-pending pictures exceeds the reorder bound, or the
            // DPB is full, or a flush was requested.
            let over_reorder = pending > self.max_reorder;
            let over_full = self.frames.len() > self.max_frames && pending > 0;
            if !(flush && pending > 0) && !over_reorder && !over_full {
                break;
            }
            let Some(frame) = self.bump_one() else { break };
            out.push(frame);
            if !flush {
                break;
            }
        }
        // Drop non-reference, already-output frames to reclaim space.
        self.frames
            .retain(|f| f.is_reference() || f.needed_for_output);
        out
    }

    fn bump_one(&mut self) -> Option<Frame> {
        let best = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.needed_for_output)
            .min_by_key(|(_, f)| f.poc)
            .map(|(i, _)| i)?;
        self.frames[best].needed_for_output = false;
        if self.frames[best].is_reference() {
            Some(self.frames[best].shallow_output())
        } else {
            Some(self.frames.remove(best))
        }
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
            latency: 0,
            meta: self.meta.clone(),
        }
    }
}

/// POC-sorted RPS partitions for the current picture.
#[derive(Clone)]
pub(crate) struct RpsPocs {
    /// used_by_curr negative-delta POCs, nearest first.
    pub(crate) before: Vec<i32>,
    /// used_by_curr positive-delta POCs, nearest first.
    pub(crate) after: Vec<i32>,
    /// used_by_curr long-term reference POCs (matched to DPB entries).
    pub(crate) lt: Vec<i32>,
}

/// Repeat the temp list to `num` entries, then apply explicit reordering.
fn finalize_list(
    temp: &[RefEntry],
    num: usize,
    list_mod: &[u32],
) -> Result<Vec<RefEntry>, DecodeError> {
    if temp.is_empty() || num == 0 {
        return Ok(Vec::new());
    }
    // RefPicListTempX is cyclically extended to NumRpsCurrTempListX =
    // Max(num_ref_idx_active, NumPicTotalCurr) entries (§8.3.4). `temp` already
    // holds exactly NumPicTotalCurr entries, so extend to Max(num, temp.len());
    // reference-list-modification indices may address any entry in this full
    // list, not just the first `num`.
    let temp_len = num.max(temp.len());
    let mut extended: Vec<RefEntry> = Vec::with_capacity(temp_len);
    while extended.len() < temp_len {
        for &e in temp {
            if extended.len() >= temp_len {
                break;
            }
            extended.push(e);
        }
    }
    if list_mod.is_empty() {
        extended.truncate(num);
        Ok(extended)
    } else {
        // Each list_entry_lX indexes into the full temp list; an out-of-range
        // index is a bitstream/conformance error rather than something to clamp.
        list_mod
            .iter()
            .take(num)
            .map(|&idx| {
                extended
                    .get(idx as usize)
                    .copied()
                    .ok_or_else(|| DecodeError::Bitstream("ref list_entry out of range".into()))
            })
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

    fn frame(poc: i32, short_term: bool, needed_for_output: bool) -> Frame {
        Frame {
            planes: YuvPlanes {
                y: vec![poc as u16],
                cb: Vec::new(),
                cr: Vec::new(),
                width: 1,
                height: 1,
                chroma: ChromaFormat::Monochrome,
                bit_depth: BitDepth::Eight,
            },
            poc,
            motion: Vec::new(),
            width4: 0,
            height4: 0,
            short_term,
            long_term: false,
            needed_for_output,
            latency: 0,
            meta: None,
        }
    }

    #[test]
    fn current_picture_is_an_active_long_term_reference() {
        let dpb = Dpb::new(1);
        let pocs = RpsPocs {
            before: Vec::new(),
            after: Vec::new(),
            lt: Vec::new(),
        };
        let current = RefEntry {
            _dpb_index: usize::MAX,
            poc: 17,
            long_term: true,
        };

        let (l0, l1) = dpb
            .build_ref_lists(&pocs, 1, 1, true, Some(current), &[], &[])
            .unwrap();
        assert_eq!(l0.len(), 1);
        assert_eq!(l1.len(), 1);
        assert_eq!(l0[0].poc, 17);
        assert_eq!(l1[0].poc, 17);
        assert!(l0[0].long_term && l1[0].long_term);
    }

    #[test]
    fn dead_output_reference_does_not_create_false_capacity_pressure() {
        let mut dpb = Dpb::new(2);
        dpb.configure(2, 2, None);
        dpb.push(frame(0, false, false));
        dpb.push(frame(2, true, true));

        assert!(dpb.bump_before_decode().is_empty());
        assert_eq!(dpb.frames.len(), 1);
        assert_eq!(dpb.frames[0].poc, 2);
    }

    #[test]
    fn irap_bumps_only_reorder_due_picture_before_discard() {
        let mut dpb = Dpb::new(6);
        dpb.configure(6, 4, None);
        for poc in 3..=7 {
            dpb.push(frame(poc, true, true));
        }

        let out = dpb.bump_before_irap();
        assert_eq!(out.iter().map(|f| f.poc).collect::<Vec<_>>(), [3]);
        assert_eq!(dpb.frames.iter().filter(|f| f.needed_for_output).count(), 4);
    }

    #[test]
    fn max_latency_forces_output_before_capacity_limit() {
        // SpsMaxLatencyPictures = 2: a pending picture must be output once two
        // later pictures have been decoded, even though the DPB (capacity 8) is
        // nowhere near full and the reorder bound (8) is not exceeded.
        //
        // §C.5.2.3: PicLatencyCount advances when a *subsequent* picture is
        // stored, not on each output-process invocation. So POC 4's latency only
        // reaches the limit after two higher-POC pictures have been pushed — and
        // by then those pictures are in the DPB, so POC 4 (the lowest POC) is the
        // one bumped, never a later picture ahead of it.
        let mut dpb = Dpb::new(8);
        dpb.configure(8, 8, Some(2));
        dpb.push(frame(4, true, true));

        dpb.push(frame(5, true, true)); // POC 4 latency -> 1
        assert!(dpb.bump_before_decode().is_empty());

        dpb.push(frame(6, true, true)); // POC 4 latency -> 2 (>= limit)
        let out = dpb.bump_before_decode();
        assert_eq!(out.iter().map(|f| f.poc).collect::<Vec<_>>(), [4]);
    }
}
