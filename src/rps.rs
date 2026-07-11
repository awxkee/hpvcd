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

//! HEVC reference picture set (RPS) parsing and current-picture RPS derivation.
//!
//! Models `short_term_ref_pic_set` (§7.3.7) as explicit delta-POC lists so the
//! slice layer can build reference picture lists. Ported in spirit from de265's
//! `slice.cc` RPS handling, rewritten in safe Rust.

use crate::bitreader::BitReader;
use crate::error::DecodeError;

fn e(s: &'static str) -> DecodeError {
    DecodeError::Bitstream(s.into())
}

/// One short-term reference picture set: signed delta-POC values (relative to
/// the current picture) split into negative (before) and positive (after)
/// lists, each with a `used_by_curr` flag.
#[derive(Clone, Default, Debug)]
pub(crate) struct ShortTermRps {
    /// delta POC of S0 (negative) pictures, ascending in |delta| (most-negative last).
    pub(crate) delta_poc_s0: Vec<i32>,
    pub(crate) used_s0: Vec<bool>,
    /// delta POC of S1 (positive) pictures, ascending.
    pub(crate) delta_poc_s1: Vec<i32>,
    pub(crate) used_s1: Vec<bool>,
}

impl ShortTermRps {
    #[inline]
    pub(crate) fn num_negative(&self) -> usize {
        self.delta_poc_s0.len()
    }
    #[inline]
    pub(crate) fn num_positive(&self) -> usize {
        self.delta_poc_s1.len()
    }
    #[inline]
    pub(crate) fn num_delta_pocs(&self) -> usize {
        self.num_negative() + self.num_positive()
    }
}

/// Parse one `short_term_ref_pic_set(idx)`. `sets` holds the previously parsed
/// sets (needed for inter-RPS prediction). `num_sets` is the total count being
/// parsed in the SPS (for slice-header RPS this equals `sets.len()`).
pub(crate) fn parse_short_term_rps(
    r: &mut BitReader,
    idx: usize,
    num_sets: usize,
    sets: &[ShortTermRps],
) -> Result<ShortTermRps, DecodeError> {
    let inter_pred = if idx != 0 {
        r.read_flag().map_err(|_| e("inter_ref_pic_set_pred"))?
    } else {
        false
    };

    if inter_pred {
        let delta_idx = if idx == num_sets {
            r.read_ue().map_err(|_| e("delta_idx_minus1"))? as usize + 1
        } else {
            1
        };
        let ref_rps_idx = idx
            .checked_sub(delta_idx)
            .ok_or_else(|| e("delta_idx out of range"))?;
        let src = sets.get(ref_rps_idx).ok_or_else(|| e("ref rps missing"))?;
        let delta_rps_sign = r.read_bit().map_err(|_| e("delta_rps_sign"))?;
        let abs_delta_rps = r.read_ue().map_err(|_| e("abs_delta_rps_minus1"))? as i32 + 1;
        let delta_rps = if delta_rps_sign != 0 {
            -abs_delta_rps
        } else {
            abs_delta_rps
        };

        let n = src.num_delta_pocs();
        let mut used = vec![false; n + 1];
        let mut use_delta = vec![true; n + 1];
        for j in 0..=n {
            let u = r.read_flag().map_err(|_| e("used_by_curr_pic_flag"))?;
            used[j] = u;
            if !u {
                use_delta[j] = r.read_flag().map_err(|_| e("use_delta_flag"))?;
            }
        }
        Ok(derive_inter_rps(src, delta_rps, &used, &use_delta))
    } else {
        let num_neg = r.read_ue().map_err(|_| e("num_negative_pics"))? as usize;
        let num_pos = r.read_ue().map_err(|_| e("num_positive_pics"))? as usize;
        let mut out = ShortTermRps::default();
        let mut prev = 0i32;
        for _ in 0..num_neg {
            let d = r.read_ue().map_err(|_| e("delta_poc_s0_minus1"))? as i32 + 1;
            prev -= d;
            out.delta_poc_s0.push(prev);
            out.used_s0
                .push(r.read_flag().map_err(|_| e("used_by_curr_s0"))?);
        }
        prev = 0;
        for _ in 0..num_pos {
            let d = r.read_ue().map_err(|_| e("delta_poc_s1_minus1"))? as i32 + 1;
            prev += d;
            out.delta_poc_s1.push(prev);
            out.used_s1
                .push(r.read_flag().map_err(|_| e("used_by_curr_s1"))?);
        }
        Ok(out)
    }
}

/// Derive an RPS predicted from `src` (§7.4.8, inter-RPS). Produces S0/S1 lists
/// ordered as required (S0 descending magnitude of negative deltas, S1 ascending).
///
/// The `used_by_curr_pic_flag[j]` / `use_delta_flag[j]` arrays are indexed over
/// the source set's `NumDeltaPocs + 1` candidates in the order: all source S0
/// (negative) entries `[0..nNeg)`, all source S1 (positive) entries
/// `[nNeg..nNeg+nPos)`, then the "frame 0" anchor at the terminal index
/// `[NumDeltaPocs]`. An entry is included when `use_delta_flag` is set and marked
/// used-by-curr from `used_by_curr_pic_flag`.
fn derive_inter_rps(
    src: &ShortTermRps,
    delta_rps: i32,
    used: &[bool],
    use_delta: &[bool],
) -> ShortTermRps {
    let mut out = ShortTermRps::default();
    let n_neg = src.num_negative();
    let n_pos = src.num_positive();
    let total = src.num_delta_pocs(); // terminal index = anchor ("frame 0")

    // --- S0 (negative) of the new set, in decreasing POC order:
    // source S1 reversed, then anchor, then source S0.
    for j in (0..n_pos).rev() {
        let dpoc = src.delta_poc_s1[j] + delta_rps;
        let idx = n_neg + j; // S1 entries live at [nNeg + j]
        if dpoc < 0 && use_delta.get(idx).copied().unwrap_or(false) {
            out.delta_poc_s0.push(dpoc);
            out.used_s0.push(used.get(idx).copied().unwrap_or(false));
        }
    }
    if delta_rps < 0 && use_delta.get(total).copied().unwrap_or(false) {
        out.delta_poc_s0.push(delta_rps);
        out.used_s0.push(used.get(total).copied().unwrap_or(false));
    }
    for j in 0..n_neg {
        let dpoc = src.delta_poc_s0[j] + delta_rps;
        if dpoc < 0 && use_delta.get(j).copied().unwrap_or(false) {
            out.delta_poc_s0.push(dpoc);
            out.used_s0.push(used.get(j).copied().unwrap_or(false));
        }
    }

    // --- S1 (positive) of the new set, in increasing POC order:
    // source S0 reversed, then anchor, then source S1.
    for j in (0..n_neg).rev() {
        let dpoc = src.delta_poc_s0[j] + delta_rps;
        if dpoc > 0 && use_delta.get(j).copied().unwrap_or(false) {
            out.delta_poc_s1.push(dpoc);
            out.used_s1.push(used.get(j).copied().unwrap_or(false));
        }
    }
    if delta_rps > 0 && use_delta.get(total).copied().unwrap_or(false) {
        out.delta_poc_s1.push(delta_rps);
        out.used_s1.push(used.get(total).copied().unwrap_or(false));
    }
    for j in 0..n_pos {
        let dpoc = src.delta_poc_s1[j] + delta_rps;
        let idx = n_neg + j;
        if dpoc > 0 && use_delta.get(idx).copied().unwrap_or(false) {
            out.delta_poc_s1.push(dpoc);
            out.used_s1.push(used.get(idx).copied().unwrap_or(false));
        }
    }

    out
}

/// Derive the picture-order-count of a slice from its `poc_lsb` and the previous
/// picture's POC (§8.3.1). `max_poc_lsb` = 1 << log2_max_poc_lsb.
pub(crate) fn derive_poc(
    poc_lsb: i32,
    prev_poc: i32,
    max_poc_lsb: i32,
    is_irap_no_rasl: bool,
) -> i32 {
    if is_irap_no_rasl {
        return poc_lsb;
    }
    let prev_lsb = prev_poc.rem_euclid(max_poc_lsb);
    let prev_msb = prev_poc - prev_lsb;
    let msb = if poc_lsb < prev_lsb && (prev_lsb - poc_lsb) >= (max_poc_lsb / 2) {
        prev_msb + max_poc_lsb
    } else if poc_lsb > prev_lsb && (poc_lsb - prev_lsb) > (max_poc_lsb / 2) {
        prev_msb - max_poc_lsb
    } else {
        prev_msb
    };
    msb + poc_lsb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    // Build a minimal explicit RPS bitstream: idx 0, num_neg=1 (delta -1, used),
    // num_pos=0. ue(0)->'1', ue for num_neg=1 -> '010', delta_poc_s0_minus1=0 ->
    // '1', used flag=1 -> '1'.
    #[test]
    fn parse_explicit_single_negative() {
        // num_negative_pics = 1 : ue(1) = 010
        // num_positive_pics = 0 : ue(0) = 1
        // delta_poc_s0_minus1 = 0 : ue(0) = 1
        // used_by_curr_pic_s0 = 1
        // bits: 010 1 1 1
        let bytes = [0b0101_1100];
        let mut r = BitReader::new(&bytes);
        let rps = parse_short_term_rps(&mut r, 0, 1, &[]).unwrap();
        assert_eq!(rps.delta_poc_s0, vec![-1]);
        assert_eq!(rps.used_s0, vec![true]);
        assert!(rps.delta_poc_s1.is_empty());
    }

    #[test]
    fn poc_wraps() {
        let max = 256;
        // prev poc 255, lsb small -> msb increments
        let p = derive_poc(2, 255, max, false);
        assert_eq!(p, 258);
        // irap resets
        assert_eq!(derive_poc(0, 999, max, true), 0);
    }

    fn src_rps(s0: &[i32], s1: &[i32]) -> ShortTermRps {
        ShortTermRps {
            delta_poc_s0: s0.to_vec(),
            used_s0: vec![true; s0.len()],
            delta_poc_s1: s1.to_vec(),
            used_s1: vec![true; s1.len()],
        }
    }

    // Inter-RPS prediction (§7.4.8). Flags are indexed over the source set's
    // NumDeltaPocs+1 candidates: [source S0.. , source S1.. , anchor]. These
    // expectations follow the HM/de265 reference construction.
    #[test]
    fn inter_rps_negative_delta() {
        // src: S0=[-1,-2], S1=[4]; indices 0->-1,1->-2,2->4,3->anchor.
        let src = src_rps(&[-1, -2], &[4]);
        let used = [true, true, true, true];
        let use_delta = [true, true, true, true];
        let out = derive_inter_rps(&src, -1, &used, &use_delta);
        // S0: anchor(-1), then -1-1=-2, -2-1=-3  (S1 reversed 4-1=3 is >0 so not S0)
        assert_eq!(out.delta_poc_s0, vec![-1, -2, -3]);
        // S1: only 4-1=3
        assert_eq!(out.delta_poc_s1, vec![3]);
    }

    #[test]
    fn inter_rps_positive_delta() {
        // src: S0=[-2], S1=[3]; indices 0->-2, 1->3, 2->anchor.
        let src = src_rps(&[-2], &[3]);
        let used = [true, true, true];
        let use_delta = [true, true, true];
        let out = derive_inter_rps(&src, 2, &used, &use_delta);
        // S0 (dPoc<0): S1 reversed 3+2=5 (>0, no); anchor 2 (>0, no);
        //             S0 fwd -2+2=0 (not <0). => empty
        assert!(out.delta_poc_s0.is_empty());
        // S1 (dPoc>0): S0 reversed -2+2=0 (no); anchor 2 (>0, yes);
        //              S1 fwd 3+2=5 (yes). => [2,5]
        assert_eq!(out.delta_poc_s1, vec![2, 5]);
    }

    #[test]
    fn inter_rps_use_delta_excludes_entry() {
        // With use_delta_flag false for the source S1 entry (index n_neg+0 = 1),
        // that candidate is dropped from the predicted set.
        let src = src_rps(&[-1], &[3]);
        // indices: 0->S0[-1], 1->S1[3], 2->anchor
        let used = [true, false, true];
        let use_delta = [true, false, true]; // exclude the S1 entry (idx 1)
        let out = derive_inter_rps(&src, 1, &used, &use_delta);
        // S1: S0 reversed -1+1=0 (no); anchor 1 (>0 yes); S1 fwd 3+1=4 but idx1
        //     use_delta=false -> excluded. => [1]
        assert_eq!(out.delta_poc_s1, vec![1]);
        assert!(out.delta_poc_s0.is_empty());
    }
}
