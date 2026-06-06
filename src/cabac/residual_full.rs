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
pub(crate) static TRACE_NEXT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
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

/// Cached scan tables. Sub-block grids are at most 8×8 and the in-sub-block
/// scan is always 4×4, so every (width, scan_idx) pair is precomputed once and
/// returned as a slice — avoiding a per-transform-block heap allocation.
fn scan_order(w: usize, scan_idx: u8) -> &'static [(usize, usize)] {
    use std::sync::OnceLock;
    #[allow(clippy::type_complexity)]
    static TABLE: OnceLock<Vec<Vec<Vec<(usize, usize)>>>> = OnceLock::new();
    let t = TABLE.get_or_init(|| {
        (0..4)
            .map(|lg| {
                let w = 1usize << lg;
                (0..3).map(|s| build_scan_order(w, s as u8)).collect()
            })
            .collect()
    });
    let lg = w.trailing_zeros() as usize;
    &t[lg][scan_idx as usize]
}

/// Index of position `(px, py)` within a cached scan order.
#[inline]
fn scan_index(scan: &[(usize, usize)], px: usize, py: usize) -> usize {
    scan.iter()
        .position(|&(x, y)| x == px && y == py)
        .unwrap_or(0)
}

/// sig_coeff_flag context (§9.3.4.2.5), all sizes.
fn sig_ctx(
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
    dec: &mut CabacDecoder,
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
    let trace = TRACE_NEXT.load(std::sync::atomic::Ordering::Relaxed);
    while group < max_group {
        let ci = ((ctx_offset + (group >> ctx_shift)) as usize).min(n - 1);
        let b = dec.decode_bin(&mut ctx[ci]);
        if trace {
            eprintln!(
                "    last_sig group={} ci={} bin={} range={} off={}",
                group, ci, b, dec.range, dec.offset
            );
        }
        if b == 0 {
            break;
        }
        group += 1;
    }
    group
}

fn decode_remaining(dec: &mut CabacDecoder, rice: u32) -> u32 {
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
        let base = (1u32
            .checked_shl(prefix - 3)
            .unwrap_or(u32::MAX)
            .saturating_add(2))
        .checked_shl(rice)
        .unwrap_or(u32::MAX);
        base.saturating_add(cw)
    }
}

const MIN_GROUP: [u32; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];

/// Decode residual coefficients for one TU. Returns row-major levels (n×n i32).
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_coding(
    dec: &mut CabacDecoder,
    ctx: &mut ContextSet,
    log2_ts: u32,
    is_luma: bool,
    scan_idx: u8,
    sign_data_hiding: bool,
    transform_skip_ctx: Option<usize>, // Some(ctx_idx) if transform_skip allowed (TU≤4)
    transquant_bypass: bool,
) -> (Vec<i32>, bool) {
    let n = 1usize << log2_ts;
    let mut coeffs = vec![0i32; n * n];

    // transform_skip_flag
    let mut transform_skip = false;
    if let Some(_ci) = transform_skip_ctx {
        let idx = if is_luma { 0 } else { 1 };
        transform_skip = dec.decode_bin(&mut ctx.transform_skip_flag[idx]) != 0;
    }

    // last_sig_coeff position
    let xp = decode_last_prefix(dec, &mut ctx.last_sig_coeff_x_prefix, log2_ts, is_luma);
    let yp = decode_last_prefix(dec, &mut ctx.last_sig_coeff_y_prefix, log2_ts, is_luma);
    let read_suffix = |dec: &mut CabacDecoder, g: u32| -> u32 {
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
    if std::env::var("DBG2").is_ok() {
        eprintln!(
            "  resid log2={} luma={} scan={} last=({},{}) xp={} yp={}",
            log2_ts, is_luma, scan_idx, last_x, last_y, xp, yp
        );
    }

    // Scan tables
    let sb_w = (n / 4).max(1);
    let sb_scan = scan_order(sb_w, scan_idx); // sub-block grid scan
    let pos_scan = scan_order(4, scan_idx); // within 4×4 sub-block

    // Find last scan position within its sub-block + which sub-block.
    let last_sbx = last_x >> 2;
    let last_sby = last_y >> 2;
    let last_sb = scan_index(sb_scan, last_sbx, last_sby);
    let last_in_sb = scan_index(pos_scan, last_x & 3, last_y & 3);

    // csbf neighbour tracking
    let mut csbf = vec![0u8; sb_w * sb_w];
    let mut c1_carry = 1i32; // greater1 ctx carried across sub-blocks
    let mut first_subblock = true;

    for i in (0..=last_sb).rev() {
        let (sbx, sby) = sb_scan[i];
        let sb_grid = sbx + sby * sb_w;
        let dbg3 = std::env::var("DBG3").is_ok() && log2_ts == 5 && is_luma;

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

        // sig_coeff_flag for positions scan_top-1 .. 0 (last is implicit at i==last_sb)
        let mut sig = [false; 16];
        let mut any_sig = false;
        if i == last_sb {
            sig[last_in_sb] = true;
            any_sig = true;
        }
        let start = if i == last_sb {
            scan_top as i32 - 1
        } else {
            scan_top as i32
        };
        for k in (0..=start.max(-1)).rev() {
            if k < 0 {
                break;
            }
            let k = k as usize;
            let (px, py) = pos_scan[k];
            let xc = (sbx << 2) + px;
            let yc = (sby << 2) + py;
            if k == 0 && infer_dc && !any_sig {
                sig[0] = true;
                any_sig = true;
                continue;
            }
            let ci = sig_ctx(xc, yc, prev_csbf, log2_ts, scan_idx, is_luma)
                .min(ctx.sig_coeff_flag.len() - 1);
            let s = dec.decode_bin(&mut ctx.sig_coeff_flag[ci]) != 0;
            if TRACE_NEXT.load(std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "    sig k={} (xc={},yc={}) ci={} bin={} range={} off={} state={}/{}",
                    k,
                    xc,
                    yc,
                    ci,
                    s as u8,
                    dec.range,
                    dec.offset,
                    ctx.sig_coeff_flag[ci].p_state_idx,
                    ctx.sig_coeff_flag[ci].val_mps
                );
            }
            sig[k] = s;
            if s {
                any_sig = true;
            }
        }

        // Collect significant scan positions (high→low).
        let mut sig_scan: Vec<usize> = Vec::new();
        for k in (0..16).rev() {
            if sig[k] {
                sig_scan.push(k);
            }
        }
        if sig_scan.is_empty() {
            continue;
        }
        let last_sig_pos = *sig_scan.first().unwrap(); // highest k
        let first_sig_pos = *sig_scan.last().unwrap(); // lowest k

        // ── greater1 ──
        let mut ctx_set: i32 = if i == 0 || !is_luma { 0 } else { 2 };
        if !first_subblock && c1_carry == 0 {
            ctx_set += 1;
        }
        first_subblock = false;
        let mut c1 = 1i32;
        let chroma_off = if is_luma { 0 } else { 16 };
        let mut gr1 = vec![false; sig_scan.len()];
        let mut last_gr1_idx: Option<usize> = None;
        let n_gr1 = sig_scan.len().min(8);
        for (j, dst) in gr1[..n_gr1].iter_mut().enumerate() {
            let g1ctx = c1.min(3);
            let ci = ((ctx_set * 4 + g1ctx) as usize + chroma_off)
                .min(ctx.coeff_abs_level_greater1.len() - 1);
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
        if dbg3 {
            eprintln!(
                "   sb[{}]=({},{}) sig={:?} gr1={:?} lastg1={:?}",
                i, sbx, sby, sig_scan, gr1, last_gr1_idx
            );
        }

        // ── greater2 (only on first greater1 coeff) ──
        let mut gr2 = false;
        if let Some(j) = last_gr1_idx {
            let ci = ((ctx_set) as usize + if is_luma { 0 } else { 4 })
                .min(ctx.coeff_abs_level_greater2.len() - 1);
            gr2 = dec.decode_bin(&mut ctx.coeff_abs_level_greater2[ci]) != 0;
            let _ = j;
        }

        // ── sign hiding decision ──
        let sign_hidden = sign_data_hiding
            && (last_sig_pos as i32 - first_sig_pos as i32) > 3
            && !transquant_bypass;

        // ── signs (bypass) ──
        let mut signs = vec![0i32; sig_scan.len()];
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

    (coeffs, transform_skip)
}
