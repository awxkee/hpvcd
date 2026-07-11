/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

use super::contexts::{ContextSet, CtxModel};
use super::engine::CabacDecoder;

pub(crate) const SCAN_DIAG: u8 = 0;
pub(crate) const SCAN_HORIZ: u8 = 1;
pub(crate) const SCAN_VERT: u8 = 2;

/// Up-right diagonal / horizontal / vertical scan over a W×W grid.
/// Returns (x, y) = (col, row) positions in scan order.
fn build_scan_order(w: usize, scan_idx: u8) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(w * w);
    match scan_idx {
        SCAN_HORIZ => {
            for y in 0..w {
                for x in 0..w {
                    out.push((x, y));
                }
            }
        }
        SCAN_VERT => {
            for x in 0..w {
                for y in 0..w {
                    out.push((x, y));
                }
            }
        }
        _ => {
            // up-right diagonal
            let (mut x, mut y) = (0i32, 0i32);
            loop {
                while y >= 0 {
                    if (x as usize) < w && (y as usize) < w {
                        out.push((x as usize, y as usize));
                    }
                    y -= 1;
                    x += 1;
                }
                y = x;
                x = 0;
                if out.len() >= w * w {
                    break;
                }
            }
        }
    }
    out
}

pub(crate) struct ResidualScanTable {
    order: Vec<(usize, usize)>,
    index: Vec<u8>,
    width: usize,
}

impl ResidualScanTable {
    #[inline(always)]
    fn order(&self) -> &[(usize, usize)] {
        &self.order
    }

    /// Index of position `(px, py)` within this scan order.
    #[inline(always)]
    fn index(&self, px: usize, py: usize) -> usize {
        if px < self.width && py < self.width {
            self.index[py * self.width + px] as usize
        } else {
            0
        }
    }
}

struct ResidualSigCtxTable {
    ctx: Vec<[u8; 4]>,
    width: usize,
}

impl ResidualSigCtxTable {
    #[inline(always)]
    fn ctx(&self, x: usize, y: usize, prev_csbf: u8) -> usize {
        self.ctx[y * self.width + x][prev_csbf as usize] as usize
    }
}

pub(crate) struct ResidualScanTables {
    by_log2: Vec<Vec<ResidualScanTable>>,
    sig_ctx_by_log2: Vec<Vec<[ResidualSigCtxTable; 2]>>,
}

impl ResidualScanTables {
    #[inline(always)]
    fn table(&self, w: usize, scan_idx: u8) -> &ResidualScanTable {
        let lg = w.trailing_zeros() as usize;
        &self.by_log2[lg][scan_idx as usize]
    }

    #[inline(always)]
    fn sig_ctx_table(&self, log2_ts: u32, scan_idx: u8, is_luma: bool) -> &ResidualSigCtxTable {
        &self.sig_ctx_by_log2[log2_ts as usize][scan_idx as usize][is_luma as usize]
    }
}

fn build_scan_table(w: usize, scan_idx: u8) -> ResidualScanTable {
    let order = build_scan_order(w, scan_idx);
    let mut index = vec![0u8; w * w];
    for (i, &(x, y)) in order.iter().enumerate() {
        index[y * w + x] = i as u8;
    }
    ResidualScanTable {
        order,
        index,
        width: w,
    }
}

fn build_sig_ctx_table(log2_ts: u32, scan_idx: u8, is_luma: bool) -> ResidualSigCtxTable {
    let width = 1usize << log2_ts;
    let mut ctx = vec![[0u8; 4]; width * width];
    for y in 0..width {
        for x in 0..width {
            let dst = &mut ctx[y * width + x];
            for prev_csbf in 0..4u8 {
                dst[prev_csbf as usize] =
                    calc_sig_ctx(x, y, prev_csbf, log2_ts, scan_idx, is_luma).min(43) as u8;
            }
        }
    }
    ResidualSigCtxTable { ctx, width }
}

fn dummy_sig_ctx_table() -> ResidualSigCtxTable {
    ResidualSigCtxTable {
        ctx: vec![[0u8; 4]; 1],
        width: 1,
    }
}

/// Cached scan tables. Sub-block grids are at most 8×8 and the in-sub-block
/// scan is always 4×4, so every (width, scan_idx) pair is precomputed once.
///
/// Resolve this once through `ExecContext` and pass the returned reference into
/// residual coding. That keeps `OnceLock`'s atomic load out of the per-TU path.
pub(crate) fn resolve_residual_scan_tables() -> &'static ResidualScanTables {
    use std::sync::OnceLock;

    static TABLES: OnceLock<ResidualScanTables> = OnceLock::new();
    TABLES.get_or_init(|| ResidualScanTables {
        by_log2: (0..4)
            .map(|lg| {
                let w = 1usize << lg;
                (0..3).map(|s| build_scan_table(w, s as u8)).collect()
            })
            .collect(),
        sig_ctx_by_log2: (0..=5)
            .map(|lg| {
                (0..3)
                    .map(|s| {
                        std::array::from_fn(|luma| {
                            if lg >= 2 {
                                build_sig_ctx_table(lg as u32, s as u8, luma != 0)
                            } else {
                                dummy_sig_ctx_table()
                            }
                        })
                    })
                    .collect()
            })
            .collect(),
    })
}

/// sig_coeff_flag context (§9.3.4.2.5), all sizes.
fn calc_sig_ctx(
    xc: usize,
    yc: usize,
    prev_csbf: u8,
    log2_ts: u32,
    scan_idx: u8,
    is_luma: bool,
) -> usize {
    static MAP4: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 99];
    let sb_width = 1usize << (log2_ts - 2);
    let mut s: i32;
    if sb_width == 1 {
        s = MAP4[(yc << 2) + xc] as i32;
    } else if xc + yc == 0 {
        s = 0;
    } else {
        let (xp, yp) = (xc & 3, yc & 3);
        let (xs, ys) = (xc >> 2, yc >> 2);
        s = match prev_csbf {
            0 => {
                if xp + yp >= 3 {
                    0
                } else if xp + yp > 0 {
                    1
                } else {
                    2
                }
            }
            1 => {
                if yp == 0 {
                    2
                } else if yp == 1 {
                    1
                } else {
                    0
                }
            }
            2 => {
                if xp == 0 {
                    2
                } else if xp == 1 {
                    1
                } else {
                    0
                }
            }
            _ => 2,
        };
        if is_luma {
            if xs + ys > 0 {
                s += 3;
            }
            s += if sb_width == 2 {
                if scan_idx == SCAN_DIAG { 9 } else { 15 }
            } else {
                21
            };
        } else {
            s += if sb_width == 2 { 9 } else { 12 };
        }
    }
    if is_luma { s as usize } else { 27 + s as usize }
}

#[inline(always)]
fn decode_last_prefix(
    dec: &mut CabacDecoder<'_>,
    ctx: &mut [CtxModel; 18],
    log2_ts: u32,
    is_luma: bool,
) -> u32 {
    let (ctx_offset, ctx_shift) = if is_luma {
        (3 * (log2_ts - 2) + ((log2_ts - 1) >> 2), (log2_ts + 1) >> 2)
    } else {
        (15, log2_ts - 2)
    };
    static GROUP_IDX: [u32; 32] = [
        0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
        9, 9,
    ];
    let size = 1u32 << log2_ts;
    let max_group = GROUP_IDX[(size - 1) as usize];
    let mut group = 0u32;
    while group < max_group {
        let ci = (ctx_offset + (group >> ctx_shift)) as usize;
        let b = dec.decode_bin(&mut ctx[ci]);
        if b == 0 {
            break;
        }
        group += 1;
    }
    group
}

#[inline(always)]
fn decode_egk_bypass(dec: &mut CabacDecoder<'_>, k: u32) -> u32 {
    let mut prefix = 0u32;
    while prefix < 31 && dec.decode_bypass() != 0 {
        prefix += 1;
    }
    if prefix == 31 || prefix + k >= 32 {
        return 0;
    }
    let suffix = dec.decode_bypass_bits(prefix + k);
    (((1u32 << prefix) - 1) << k).wrapping_add(suffix)
}

#[inline(always)]
fn decode_limited_egk_bypass(dec: &mut CabacDecoder<'_>, rice_param: u32, bit_depth: u8) -> u32 {
    let log2_transform_range = 15u32.max(u32::from(bit_depth) + 6);
    let max_prefix_extension = 28u32.saturating_sub(log2_transform_range);
    let mut extension = 0u32;
    while extension < max_prefix_extension && dec.decode_bypass() != 0 {
        extension += 1;
    }

    let escape_length = if extension == max_prefix_extension {
        // At the limit there is no terminating zero; a fixed transform-range
        // suffix follows the all-ones extension (§9.3.3.4).
        log2_transform_range
    } else {
        extension + rice_param
    };
    if extension >= 32 || rice_param >= 32 || escape_length > 32 {
        return 0;
    }
    let base = ((1u32 << extension) - 1)
        .checked_shl(rice_param)
        .unwrap_or(0);
    base.wrapping_add(dec.decode_bypass_bits(escape_length))
}

#[inline(always)]
fn decode_remaining(
    dec: &mut CabacDecoder<'_>,
    rice: u32,
    extended_precision: bool,
    bit_depth: u8,
) -> u32 {
    // The TR prefix has at most four one-bins. Quotients 0..3 carry a fixed
    // Rice suffix; the all-ones prefix enters EG(k+1), or limited EG(k+1) when
    // extended_precision_processing_flag is enabled (§9.3.3.11).
    let mut quotient = 0u32;
    while quotient < 4 && dec.decode_bypass() != 0 {
        quotient += 1;
    }
    if quotient < 4 {
        return (quotient << rice).wrapping_add(dec.decode_bypass_bits(rice));
    }

    let suffix = if extended_precision {
        decode_limited_egk_bypass(dec, rice + 1, bit_depth)
    } else {
        decode_egk_bypass(dec, rice + 1)
    };
    (4u32 << rice).wrapping_add(suffix)
}

static MIN_GROUP: [u32; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];

/// Decode residual coefficients for one TU. Returns row-major levels (n×n i32).
#[allow(clippy::too_many_arguments)]
/// RExt residual-coding controls (§7.4.3.2.2 flags exercised by SCC streams).
pub(crate) struct RextResidual {
    /// persistent_rice_adaptation_enabled_flag: init each sub-block's Rice
    /// parameter from ContextSet::stat_coeff (§9.3.3.13).
    pub(crate) persistent_rice: bool,
    /// transform_skip_context_enabled_flag: single dedicated sig-coeff context
    /// for transform-skip / transquant-bypass blocks (§9.3.4.2.5).
    pub(crate) transform_skip_context: bool,
    /// explicit_rdpcm_enabled_flag and whether this TU is in an inter
    /// (non-intra-predicted) CU, gating the explicit_rdpcm syntax (§7.3.8.11).
    pub(crate) explicit_rdpcm: bool,
    /// The implicit-RDPCM SPS flag is enabled and this intra TU uses horizontal
    /// or vertical prediction. It becomes active only when transform skip is set.
    pub(crate) implicit_rdpcm: bool,
    /// cabac_bypass_alignment_enabled_flag: align the arithmetic range before
    /// coefficient signs/remaining levels whenever escapeDataPresent is true.
    pub(crate) bypass_alignment: bool,
    /// extended_precision_processing_flag selects limited EGk for coefficient
    /// remaining escape suffixes. `bit_depth` is the active component depth.
    pub(crate) extended_precision: bool,
    pub(crate) bit_depth: u8,
    pub(crate) is_inter: bool,
}

/// Explicit RDPCM outcome for the TU: None, or Some(dir) with 0=horizontal,
/// 1=vertical residual DPCM.
pub(crate) type RdpcmDir = Option<u8>;

#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_coding(
    dec: &mut CabacDecoder<'_>,
    ctx: &mut ContextSet,
    scan_tables: &ResidualScanTables,
    log2_ts: u32,
    is_luma: bool,
    scan_idx: u8,
    sign_data_hiding: bool,
    transform_skip_ctx: Option<usize>, // Some(ctx_idx) if transform_skip allowed (TU≤4)
    transquant_bypass: bool,
    rext: &RextResidual,
    coeffs: &mut [i32],
) -> (bool, usize, usize, i32, RdpcmDir) {
    let n = 1usize << log2_ts;
    coeffs[..n * n].fill(0);
    let mut max_x = 0usize;
    let mut max_abs_level = 0i32;

    // transform_skip_flag
    let mut transform_skip = false;
    if let Some(_ci) = transform_skip_ctx
        && !transquant_bypass
    {
        let idx = if is_luma { 0 } else { 1 };
        transform_skip = dec.decode_bin(&mut ctx.transform_skip_flag[idx]) != 0;
    }

    // explicit_rdpcm_flag / dir (§7.3.8.11): inter TUs with transform skip or
    // transquant bypass, when the SPS enables explicit RDPCM.
    let mut rdpcm: RdpcmDir = None;
    if rext.explicit_rdpcm && rext.is_inter && (transform_skip || transquant_bypass) {
        let ci = if is_luma { 0 } else { 1 };
        if dec.decode_bin(&mut ctx.explicit_rdpcm_flag[ci]) != 0 {
            let dir = dec.decode_bin(&mut ctx.explicit_rdpcm_dir[ci]);
            rdpcm = Some(dir);
        }
    }

    // last_sig_coeff position
    let xp = decode_last_prefix(dec, &mut ctx.last_sig_coeff_x_prefix, log2_ts, is_luma);
    let yp = decode_last_prefix(dec, &mut ctx.last_sig_coeff_y_prefix, log2_ts, is_luma);
    let read_suffix = |dec: &mut CabacDecoder<'_>, g: u32| -> u32 {
        let bits = g.saturating_sub(2) / 2;
        dec.decode_bypass_bits(bits)
    };
    let xs = read_suffix(dec, xp);
    let ys = read_suffix(dec, yp);
    let mut last_x = (MIN_GROUP[xp as usize] + xs) as usize;
    let mut last_y = (MIN_GROUP[yp as usize] + ys) as usize;
    if scan_idx == SCAN_VERT {
        std::mem::swap(&mut last_x, &mut last_y);
    }
    // Scan tables
    let sb_w = (n / 4).max(1);
    let sb_table = scan_tables.table(sb_w, scan_idx); // sub-block grid scan
    let pos_table = scan_tables.table(4, scan_idx); // within 4×4 sub-block
    let sb_scan = sb_table.order();
    let pos_scan = pos_table.order();
    let sig_ctx_table = scan_tables.sig_ctx_table(log2_ts, scan_idx, is_luma);

    // Find last scan position within its sub-block + which sub-block.
    let last_sbx = last_x >> 2;
    let last_sby = last_y >> 2;
    let last_sb = sb_table.index(last_sbx, last_sby);
    let last_in_sb = pos_table.index(last_x & 3, last_y & 3);

    // Coded-sub-block flags fit in one u64 (the grid is at most 8×8).
    let mut csbf = 0u64;
    let mut c1_carry = 1i32; // greater1 ctx carried across sub-blocks
    let mut first_subblock = true;

    for (i, &(sbx, sby)) in sb_scan[..=last_sb].iter().enumerate().rev() {
        let sb_grid = sbx + sby * sb_w;
        let right = if sbx + 1 < sb_w {
            ((csbf >> (sb_grid + 1)) & 1) as u8
        } else {
            0
        };
        let below = if sby + 1 < sb_w {
            ((csbf >> (sb_grid + sb_w)) & 1) as u8
        } else {
            0
        };
        let prev_csbf = (right & 1) | ((below & 1) << 1);

        // coded_sub_block_flag
        let infer_dc = i < last_sb && i > 0;
        let coded = if infer_dc {
            let ctx_inc = ((right | below) & 1) as usize;
            let cg = if is_luma { ctx_inc } else { 2 + ctx_inc };
            dec.decode_bin(&mut ctx.coded_sub_block_flag[cg]) != 0
        } else {
            i == 0 || i == last_sb
        };
        csbf |= (coded as u64) << sb_grid;
        if !coded {
            continue;
        }

        let scan_top = if i == last_sb { last_in_sb } else { 15 };

        // Store both the coefficient destination and the 4×4 scan position in
        // one compact entry.  Entries remain in high→low scan order.
        let mut sig_entries = [0u16; 16];
        let mut sig_len = 0usize;
        if i == last_sb {
            let coeff_index = last_y * n + last_x;
            sig_entries[0] = ((coeff_index as u16) << 4) | last_in_sb as u16;
            sig_len = 1;
        }
        let start = if i == last_sb {
            last_in_sb.checked_sub(1)
        } else {
            Some(scan_top)
        };
        if let Some(start) = start {
            for (k, &(px, py)) in pos_scan[..=start].iter().enumerate().rev() {
                let xc = (sbx << 2) + px;
                let yc = (sby << 2) + py;
                let significant = if k == 0 && infer_dc && sig_len == 0 {
                    true
                } else {
                    // §9.3.4.2.5: with transform_skip_context enabled, TS/TQB
                    // blocks use one dedicated context (luma 42, chroma 16).
                    let ci = if rext.transform_skip_context && (transform_skip || transquant_bypass)
                    {
                        if is_luma { 42 } else { 27 + 16 }
                    } else {
                        sig_ctx_table.ctx(xc, yc, prev_csbf)
                    };
                    dec.decode_bin(&mut ctx.sig_coeff_flag[ci]) != 0
                };

                // This write is always in-bounds: before each of the at most 16
                // candidates, sig_len is at most 15.  Advancing conditionally
                // avoids the unpredictable append branch from the old loop.
                let coeff_index = yc * n + xc;
                sig_entries[sig_len] = ((coeff_index as u16) << 4) | k as u16;
                sig_len += significant as usize;
            }
        }
        if sig_len == 0 {
            continue;
        }
        let sig_entries = &sig_entries[..sig_len];
        let last_sig_pos = usize::from(sig_entries[0] & 15); // highest k
        let first_sig_pos = usize::from(sig_entries[sig_len - 1] & 15); // lowest k
        // ── greater1 ──
        let mut ctx_set: i32 = if i == 0 || !is_luma { 0 } else { 2 };
        if !first_subblock && c1_carry == 0 {
            ctx_set += 1;
        }
        first_subblock = false;
        let mut c1 = 1i32;
        let chroma_off = if is_luma { 0 } else { 16 };
        let mut gr1_mask = 0u16;
        let mut first_gr1_idx = 16usize;
        let n_gr1 = sig_len.min(8);
        for j in 0..n_gr1 {
            let ci = (ctx_set * 4 + c1.min(3)) as usize + chroma_off;
            let greater1 = dec.decode_bin(&mut ctx.coeff_abs_level_greater1[ci]) != 0;
            gr1_mask |= (greater1 as u16) << j;
            if greater1 {
                c1 = 0;
                if first_gr1_idx == 16 {
                    first_gr1_idx = j;
                }
            } else if c1 > 0 && c1 < 3 {
                c1 += 1;
            }
        }
        c1_carry = c1;
        // ── greater2 (only on first greater1 coeff) ──
        let gr2 = if first_gr1_idx < 16 {
            let ci = ctx_set as usize + if is_luma { 0 } else { 4 };
            dec.decode_bin(&mut ctx.coeff_abs_level_greater2[ci]) != 0
        } else {
            false
        };

        // §7.3.8.11 escapeDataPresent becomes true when there is a second
        // greater1 coefficient, when more than eight significant coefficients
        // are present, or when greater2 is set.  With CABAC bypass alignment,
        // both coeff_sign_flag and coeff_abs_level_remaining must start from an
        // arithmetic range of exactly 256 (§9.3.1 / §9.3.4.3.6).
        let escape_data_present = sig_len > 8 || gr1_mask.count_ones() > 1 || gr2;
        if rext.bypass_alignment && escape_data_present {
            dec.align_bypass();
        }

        // ── sign hiding decision ──
        // Sign-data hiding is disabled for every residual-DPCM block. With
        // explicit RDPCM this depends on the decoded flag; with implicit RDPCM
        // it depends on transform skip plus the horizontal/vertical intra mode.
        // Leaving it enabled consumes one fewer bypass sign and silently shifts
        // the arithmetic offset before coeff_abs_level_remaining.
        let rdpcm_active = rdpcm.is_some() || (transform_skip && rext.implicit_rdpcm);
        let sign_hidden = sign_data_hiding
            && (last_sig_pos as i32 - first_sig_pos as i32) > 3
            && !transquant_bypass
            && !rdpcm_active;

        // The hidden sign is always the final (lowest-scan-position) entry, so
        // all coded signs form one contiguous bypass run.  Reverse once to make
        // bit j correspond directly to significant entry j.
        let sign_count = sig_len - sign_hidden as usize;
        let coded_signs = dec.decode_bypass_bits(sign_count as u32) as u16;
        let sign_mask = if sign_count == 0 {
            0
        } else {
            coded_signs.reverse_bits() >> (u16::BITS as usize - sign_count)
        };
        // ── coeff_abs_level_remaining + assemble ──
        // §9.3.3.13: with persistent Rice adaptation the sub-block's initial
        // Rice parameter comes from StatCoeff[sbType], and the first coded
        // remaining value in the sub-block updates that statistic.
        let sb_type = (2 * usize::from(is_luma)) + usize::from(transform_skip || transquant_bypass);
        let mut rice = if rext.persistent_rice {
            u32::from(ctx.stat_coeff[sb_type] >> 2)
        } else {
            0
        };
        let mut first_rem_in_sb = true;
        let mut abs_parity = 0i32;
        for (j, &entry) in sig_entries.iter().enumerate() {
            let g1 = ((gr1_mask >> j) & 1) as i32;
            let has_gr2 = j == first_gr1_idx;
            let base = 1 + g1 + (has_gr2 && gr2) as i32;
            // baseLevel == threshold → read remaining
            let threshold = if j < 8 {
                if has_gr2 { 3 } else { 2 }
            } else {
                1
            };
            let rem = if base == threshold {
                let rem = decode_remaining(dec, rice, rext.extended_precision, rext.bit_depth);
                if rext.persistent_rice && first_rem_in_sb {
                    first_rem_in_sb = false;
                    let stat = &mut ctx.stat_coeff[sb_type];
                    if rem >= (3u32 << (*stat >> 2)) {
                        *stat += 1;
                    } else if 2 * rem < (1u32 << (*stat >> 2)) && *stat > 0 {
                        *stat -= 1;
                    }
                }
                // Rice parameter adaptation happens only when a remaining value
                // is actually decoded (§9.3.3.2). Persistent-rice removes the
                // cap of 4.
                let level = base + rem as i32;
                if level > (3 << rice) {
                    rice = if rext.persistent_rice {
                        rice + 1
                    } else {
                        (rice + 1).min(4)
                    };
                }
                rem
            } else {
                0
            };
            let level = base + rem as i32;
            max_abs_level = max_abs_level.max(level);

            let coeff_index = usize::from(entry >> 4);
            let xc = coeff_index & (n - 1);
            let negative = ((sign_mask >> j) & 1) != 0;
            coeffs[coeff_index] = if negative { -level } else { level };
            max_x = max_x.max(xc);
            abs_parity ^= level & 1;
        }

        // Resolve hidden sign by parity.
        if sign_hidden {
            let coeff_index = usize::from(sig_entries[sig_len - 1] >> 4);
            let coeff = &mut coeffs[coeff_index];
            *coeff = if abs_parity != 0 {
                -coeff.abs()
            } else {
                coeff.abs()
            };
        }
    }

    (transform_skip, max_x, last_y, max_abs_level, rdpcm)
}
