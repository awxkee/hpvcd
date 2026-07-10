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
fn derive_inter_rps(
    src: &ShortTermRps,
    delta_rps: i32,
    used: &[bool],
    use_delta: &[bool],
) -> ShortTermRps {
    let mut out = ShortTermRps::default();
    let n_neg = src.num_negative();
    let n_pos = src.num_positive();
    let total = src.num_delta_pocs();

    // S0 (negative) of the new set: iterate src S1 (reversed) then src's "0" then src S0.
    // Follows the reference construction: for each candidate combined delta, keep
    // those that are < 0.
    // Build combined src delta list in the reference order used by the spec:
    //   index j in 0..=total maps to: j<n_pos -> src.S1[n_pos-1-j]; j==?; else src.S0
    // We implement the standard two-pass approach.

    // Pass for S0 (negative results).
    // j over src positive deltas (reverse), then the implicit "current" (delta 0),
    // then src negative deltas.
    // Simpler: enumerate all src deltas plus the zero anchor, compute dPoc.
    let src_deltas = combined_src_deltas(src);
    // Negative list.
    // First, contributions from src S1 reversed (these can become negative when delta_rps<0).
    for i in (0..n_pos).rev() {
        let dpoc = src.delta_poc_s1[i] + delta_rps;
        let uidx = n_neg + 1 + i; // mapping of S1 entries in used[]/use_delta[]
        if dpoc < 0 && used_or_use(used, use_delta, uidx) {
            out.delta_poc_s0.push(dpoc);
            out.used_s0.push(used.get(uidx).copied().unwrap_or(false));
        }
    }
    // Anchor (delta 0) contributes delta_rps if negative.
    if delta_rps < 0 && used_or_use(used, use_delta, n_neg) {
        out.delta_poc_s0.push(delta_rps);
        out.used_s0.push(used.get(n_neg).copied().unwrap_or(false));
    }
    // src S0 entries.
    for i in 0..n_neg {
        let dpoc = src.delta_poc_s0[i] + delta_rps;
        let uidx = i;
        if dpoc < 0 && used_or_use(used, use_delta, uidx) {
            out.delta_poc_s0.push(dpoc);
            out.used_s0.push(used.get(uidx).copied().unwrap_or(false));
        }
    }

    // Positive list (S1).
    for i in (0..n_neg).rev() {
        let dpoc = src.delta_poc_s0[i] + delta_rps;
        let uidx = i;
        if dpoc > 0 && used_or_use(used, use_delta, uidx) {
            out.delta_poc_s1.push(dpoc);
            out.used_s1.push(used.get(uidx).copied().unwrap_or(false));
        }
    }
    if delta_rps > 0 && used_or_use(used, use_delta, n_neg) {
        out.delta_poc_s1.push(delta_rps);
        out.used_s1.push(used.get(n_neg).copied().unwrap_or(false));
    }
    for i in 0..n_pos {
        let dpoc = src.delta_poc_s1[i] + delta_rps;
        let uidx = n_neg + 1 + i;
        if dpoc > 0 && used_or_use(used, use_delta, uidx) {
            out.delta_poc_s1.push(dpoc);
            out.used_s1.push(used.get(uidx).copied().unwrap_or(false));
        }
    }
    let _ = (total, &src_deltas);
    out
}

#[inline]
fn used_or_use(used: &[bool], use_delta: &[bool], idx: usize) -> bool {
    used.get(idx).copied().unwrap_or(false) || use_delta.get(idx).copied().unwrap_or(false)
}

fn combined_src_deltas(src: &ShortTermRps) -> Vec<i32> {
    let mut v = Vec::with_capacity(src.num_delta_pocs());
    v.extend_from_slice(&src.delta_poc_s0);
    v.extend_from_slice(&src.delta_poc_s1);
    v
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
}
