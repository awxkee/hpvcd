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
            no_rasl_output: false,
        }
    }

    /// Configure the output DPB from the active SPS (§C.5.2.2). `buffering` is
    /// sps_max_dec_pic_buffering (DPB size) and `reorder` is
    /// sps_max_num_reorder_pics (reorder-latency bound).
    pub(crate) fn configure(&mut self, buffering: usize, reorder: usize) {
        self.max_frames = buffering.max(1);
        self.max_reorder = reorder.min(self.max_frames);
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
}
