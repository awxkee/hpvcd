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

use super::contexts::ContextSet;
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

fn decode_last_prefix(
    dec: &mut CabacDecoder<'_>,
    ctx: &mut [super::contexts::CtxModel],
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
    let n = ctx.len();
    let mut group = 0u32;
    while group < max_group {
        let ci = ((ctx_offset + (group >> ctx_shift)) as usize).min(n - 1);
        let b = dec.decode_bin(&mut ctx[ci]);
        if b == 0 {
            break;
        }
        group += 1;
    }
    group
}

fn decode_remaining(dec: &mut CabacDecoder<'_>, rice: u32) -> u32 {
    let mut prefix = 0u32;
    while dec.decode_bypass() != 0 {
        prefix += 1;
        if prefix > 32 {
            break;
        }
    }
    if prefix <= 3 {
        let mut suf = 0u32;
        for _ in 0..rice {
            suf = (suf << 1) | dec.decode_bypass() as u32;
        }
        (prefix << rice) + suf
    } else {
        // prefix is ≤ 32, rice ≤ 4 — cap suffix_bits so we never read > 32 bypass bins
        let suffix_bits = (prefix - 3 + rice).min(32);
        let mut cw = 0u32;
        for _ in 0..suffix_bits {
            cw = (cw << 1) | dec.decode_bypass() as u32;
        }
        // Saturate the base calculation: (1<<(p-3)+2)<<rice can overflow u32 when p≈32
        let base = 1u32
            .checked_shl(prefix - 3)
            .unwrap_or(u32::MAX)
            .saturating_add(2)
            .checked_shl(rice)
            .unwrap_or(u32::MAX);
        base.saturating_add(cw)
    }
}

static MIN_GROUP: [u32; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];

/// Decode residual coefficients for one TU. Returns row-major levels (n×n i32).
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
    coeffs: &mut [i32],
) -> (bool, usize, usize) {
    let n = 1usize << log2_ts;
    coeffs[..n * n].fill(0);
    let mut max_x = 0usize;

    // transform_skip_flag
    let mut transform_skip = false;
    if let Some(_ci) = transform_skip_ctx {
        let idx = if is_luma { 0 } else { 1 };
        transform_skip = dec.decode_bin(&mut ctx.transform_skip_flag[idx]) != 0;
    }

    // last_sig_coeff position
    let xp = decode_last_prefix(dec, &mut ctx.last_sig_coeff_x_prefix, log2_ts, is_luma);
    let yp = decode_last_prefix(dec, &mut ctx.last_sig_coeff_y_prefix, log2_ts, is_luma);
    let read_suffix = |dec: &mut CabacDecoder<'_>, g: u32| -> u32 {
        if g <= 3 {
            return 0;
        }
        let bits = (g - 2) / 2;
        let mut v = 0;
        for _ in 0..bits {
            v = (v << 1) | dec.decode_bypass() as u32;
        }
        v
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

    // csbf neighbour tracking
    let mut csbf = [0u8; 64];
    let csbf = &mut csbf[..sb_w * sb_w];
    let mut c1_carry = 1i32; // greater1 ctx carried across sub-blocks
    let mut first_subblock = true;

    for i in (0..=last_sb).rev() {
        let (sbx, sby) = sb_scan[i];
        let sb_grid = sbx + sby * sb_w;

        // coded_sub_block_flag
        let mut infer_dc = false;
        let coded;
        if i < last_sb && i > 0 {
            // neighbour-based context
            let right = if sbx + 1 < sb_w {
                csbf[(sbx + 1) + sby * sb_w]
            } else {
                0
            };
            let below = if sby + 1 < sb_w {
                csbf[sbx + (sby + 1) * sb_w]
            } else {
                0
            };
            let ctx_inc = ((right | below) & 1) as usize;
            let cg = if is_luma { ctx_inc } else { 2 + ctx_inc };
            coded = dec.decode_bin(&mut ctx.coded_sub_block_flag[cg]) != 0;
            infer_dc = true;
        } else {
            coded = i == 0 || i == last_sb;
        }
        csbf[sb_grid] = coded as u8;
        if !coded {
            continue;
        }

        let prev_csbf = {
            let right = if sbx + 1 < sb_w {
                csbf[(sbx + 1) + sby * sb_w]
            } else {
                0
            };
            let below = if sby + 1 < sb_w {
                csbf[sbx + (sby + 1) * sb_w]
            } else {
                0
            };
            (right & 1) | ((below & 1) << 1)
        };

        let scan_top = if i == last_sb { last_in_sb } else { 15 };

        // sig_coeff_flag for positions scan_top-1 .. 0 (last is implicit at i==last_sb).
        // Collect significant scan positions directly in high→low order; this avoids
        // materialising a 16-entry bool map and scanning it again.
        let mut sig_scan = [0usize; 16];
        let mut sig_len = 0usize;
        let mut any_sig = false;
        if i == last_sb {
            sig_scan[0] = last_in_sb;
            sig_len = 1;
            any_sig = true;
        }
        let start = if i == last_sb {
            last_in_sb.checked_sub(1)
        } else {
            Some(scan_top)
        };
        if let Some(start) = start {
            for k in (0..=start).rev() {
                let (px, py) = pos_scan[k];
                let xc = (sbx << 2) + px;
                let yc = (sby << 2) + py;
                let s = if k == 0 && infer_dc && !any_sig {
                    true
                } else {
                    let ci = sig_ctx_table.ctx(xc, yc, prev_csbf);
                    dec.decode_bin(&mut ctx.sig_coeff_flag[ci]) != 0
                };
                if s {
                    sig_scan[sig_len] = k;
                    sig_len += 1;
                    any_sig = true;
                }
            }
        }
        if sig_len == 0 {
            continue;
        }
        let sig_scan = &sig_scan[..sig_len];
        let last_sig_pos = sig_scan[0]; // highest k
        let first_sig_pos = sig_scan[sig_len - 1]; // lowest k

        // ── greater1 ──
        let mut ctx_set: i32 = if i == 0 || !is_luma { 0 } else { 2 };
        if !first_subblock && c1_carry == 0 {
            ctx_set += 1;
        }
        first_subblock = false;
        let mut c1 = 1i32;
        let chroma_off = if is_luma { 0 } else { 16 };
        let mut gr1 = [false; 16];
        let mut last_gr1_idx: Option<usize> = None;
        let n_gr1 = sig_len.min(8);
        for (j, dst) in gr1[..n_gr1].iter_mut().enumerate() {
            let g1ctx = c1.min(3);
            let ci = (ctx_set * 4 + g1ctx) as usize + chroma_off;
            let f = dec.decode_bin(&mut ctx.coeff_abs_level_greater1[ci]) != 0;
            *dst = f;
            if f {
                c1 = 0;
                if last_gr1_idx.is_none() {
                    last_gr1_idx = Some(j);
                }
            } else if c1 > 0 && c1 < 3 {
                c1 += 1;
            }
        }
        c1_carry = c1;

        // ── greater2 (only on first greater1 coeff) ──
        let mut gr2 = false;
        if let Some(j) = last_gr1_idx {
            let ci = ctx_set as usize + if is_luma { 0 } else { 4 };
            gr2 = dec.decode_bin(&mut ctx.coeff_abs_level_greater2[ci]) != 0;
            let _ = j;
        }

        // ── sign hiding decision ──
        let sign_hidden = sign_data_hiding
            && (last_sig_pos as i32 - first_sig_pos as i32) > 3
            && !transquant_bypass;

        // ── signs (bypass) ──
        let mut signs = [0i32; 16];
        for (j, &k) in sig_scan.iter().enumerate() {
            if sign_hidden && k == first_sig_pos {
                signs[j] = 0; // inferred later
            } else {
                signs[j] = if dec.decode_bypass() != 0 { 1 } else { 0 }; // 1 = negative
            }
        }

        // ── coeff_abs_level_remaining + assemble ──
        let mut rice = 0u32;
        let mut sum_abs = 0i64;
        let mut first_sig_j = 0usize;
        for (j, &k) in sig_scan.iter().enumerate() {
            let g1 = if j < n_gr1 { gr1[j] as i32 } else { 0 };
            let g2 = if Some(j) == last_gr1_idx {
                gr2 as i32
            } else {
                0
            };
            let base = 1 + g1 + g2;
            // baseLevel == threshold → read remaining
            let threshold = if j < 8 {
                if Some(j) == last_gr1_idx { 3 } else { 2 }
            } else {
                1
            };
            let rem = if base == threshold {
                decode_remaining(dec, rice)
            } else {
                0
            };
            let level = base + rem as i32;
            if level > (3 << rice) {
                rice = (rice + 1).min(4);
            }

            let (px, py) = pos_scan[k];
            let xc = (sbx << 2) + px;
            let yc = (sby << 2) + py;
            let mut val = level;
            if signs[j] == 1 {
                val = -val;
            }
            coeffs[yc * n + xc] = val;
            max_x = max_x.max(xc);
            sum_abs += level as i64;
            if k == first_sig_pos {
                first_sig_j = j;
            }
        }
        // Resolve hidden sign by parity.
        if sign_hidden {
            let k = first_sig_pos;
            let (px, py) = pos_scan[k];
            let xc = (sbx << 2) + px;
            let yc = (sby << 2) + py;
            if sum_abs % 2 == 1 {
                coeffs[yc * n + xc] = -coeffs[yc * n + xc].abs();
            } else {
                coeffs[yc * n + xc] = coeffs[yc * n + xc].abs();
            }
            let _ = first_sig_j;
        }
    }

    // max_x = true max nonzero column; caller uses (max_x + 1) to bound stage-1 columns.
    (transform_skip, max_x, last_y)
}
