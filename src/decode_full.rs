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

use crate::cabac::CabacDecoder;
use crate::cabac::{ContextSet, IntraModeContexts};
use crate::cabac::{SCAN_DIAG, SCAN_HORIZ, SCAN_VERT, residual_coding};
use crate::config::{Pps, Sps};
use crate::error::DecodeError;
use crate::fmt::BitDepth;
use crate::intra_full;
use crate::transform;
use crate::yuv::YuvPlanes;

const MODE_PLANAR: u8 = 0;
const MODE_DC: u8 = 1;

#[derive(Clone, Default)]
struct SaoCtb {
    type_idx: [u8; 3], // 0=off,1=band,2=edge
    offsets: [[i32; 4]; 3],
    band_pos: [u8; 3],
    eo_class: [u8; 3],
}

pub(crate) struct FullDecoder<'a> {
    cab: CabacDecoder<'a>,
    ctx: ContextSet,
    ictx: IntraModeContexts,
    sps: Sps,
    pps: Pps,

    // Reconstruction planes (coded dimensions, CTB-aligned).
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
    w: usize,  // coded luma width
    h: usize,  // coded luma height
    cw: usize, // chroma width
    ch: usize, // chroma height
    sub_w: usize,
    sub_h: usize,
    bd: u8,
    bd_c: u8,

    log2_ctb: u32,
    log2_min_cb: u32,
    log2_min_tb: u32,
    log2_max_tb: u32,
    max_trafo_depth_intra: u32,

    // Per-4×4-luma intra mode and "decoded" availability.
    mode_y: Vec<u8>,
    decoded: Vec<bool>, // per 4×4 luma block
    grid_w: usize,      // w/4
    #[allow(dead_code)]
    grid_h: usize, // h/4
    ct_depth: Vec<u8>,  // per 4×4, coding-tree depth (for split_cu_flag ctx)

    // QP tracking
    slice_qp: i32,
    qp_y_prev: i32,
    qp_y_map: Vec<i16>, // per 4×4 luma (QpY ∈ −24..51, fits i16; halves a multi-MB buffer)
    cu_qp_delta_val: i32,
    is_cu_qp_delta_coded: bool,
    log2_qg: u32,
    cur_qp: i32,

    sao: Vec<SaoCtb>,
    ctb_cols: usize,
    ctb_rows: usize,
    sao_luma: bool,
    sao_chroma: bool,

    sign_hiding: bool,

    // WPP context snapshots
    wpp_ctx_snap: Vec<Option<ContextSet>>,
    wpp_ictx_snap: Vec<Option<IntraModeContexts>>,

    /// Pre-allocated scratch memory reused every TU to avoid per-block
    /// heap allocations on the hot path (~4–6 allocs per TU eliminated).
    scratch: intra_full::IntraScratch,
    /// Dequantised coefficient scratch (max 32×32 = 1024 values, clamped to ±32768 → i32)
    deq_scratch: Vec<i32>,
    /// Inverse-transform output scratch (max 32×32 = 1024 i32 values)
    res_scratch: Vec<i32>,
    /// Cached strong_intra_smoothing (avoids env-var lookup per TU)
    strong_smoothing: bool,
}

impl<'a> FullDecoder<'a> {
    /// Maximum allowed dimension per axis and pixel count.
    pub(crate) const MAX_DIM: usize = 16_384;
    pub(crate) const MAX_PIXELS: usize = 64 * 1024 * 1024; // 64 MP

    pub(crate) fn new(
        cabac: &'a [u8],
        sps: Sps,
        pps: Pps,
        slice_qp: i32,
        sao_luma: bool,
        sao_chroma: bool,
    ) -> Result<Self, DecodeError> {
        // Reject dimensions that would cause enormous allocations.
        let w = sps.width as usize;
        let h = sps.height as usize;
        if w == 0
            || h == 0
            || w > Self::MAX_DIM
            || h > Self::MAX_DIM
            || w.saturating_mul(h) > Self::MAX_PIXELS
        {
            return Err(DecodeError::Bitstream(format!(
                "image dimensions {}×{} exceed maximum",
                w, h
            )));
        }
        let cab =
            CabacDecoder::new(cabac).map_err(|_| DecodeError::Bitstream("cabac init".into()))?;
        let log2_ctb = sps.log2_ctb;
        let ctb = 1usize << log2_ctb;
        let sub_w = sps.chroma.sub_w();
        let sub_h = sps.chroma.sub_h();
        let cw = if sps.chroma.is_monochrome() {
            0
        } else {
            w / sub_w
        };
        let ch = if sps.chroma.is_monochrome() {
            0
        } else {
            h / sub_h
        };
        let grid_w = w / 4;
        let grid_h = h / 4;
        let qp = ContextSet::init_islice(slice_qp.clamp(0, 51) as u8);
        let ictx = IntraModeContexts::init_islice(slice_qp.clamp(0, 51) as u8);
        let ctb_cols = w.div_ceil(ctb);
        let ctb_rows = h.div_ceil(ctb);
        let log2_qg = sps.log2_ctb - pps.diff_cu_qp_delta_depth;
        Ok(FullDecoder {
            cab,
            ctx: qp,
            ictx,
            bd: sps.bit_depth_luma,
            bd_c: sps.bit_depth_chroma,
            log2_ctb,
            log2_min_cb: sps.log2_min_cb,
            log2_min_tb: sps.log2_min_tb,
            log2_max_tb: sps.log2_max_tb,
            max_trafo_depth_intra: sps.max_transform_hierarchy_intra,
            y: vec![0; w * h],
            cb: vec![0; cw * ch],
            cr: vec![0; cw * ch],
            w,
            h,
            cw,
            ch,
            sub_w,
            sub_h,
            mode_y: vec![MODE_DC; grid_w * grid_h],
            decoded: vec![false; grid_w * grid_h],
            ct_depth: vec![0; grid_w * grid_h],
            grid_w,
            grid_h,
            slice_qp,
            qp_y_prev: slice_qp,
            qp_y_map: vec![slice_qp as i16; grid_w * grid_h],
            cu_qp_delta_val: 0,
            is_cu_qp_delta_coded: false,
            log2_qg,
            cur_qp: slice_qp,
            sao: vec![SaoCtb::default(); ctb_cols * ctb_rows],
            ctb_cols,
            ctb_rows,
            sao_luma,
            sao_chroma,
            sign_hiding: pps.sign_data_hiding_enabled,
            wpp_ctx_snap: vec![None; ctb_rows],
            wpp_ictx_snap: vec![None; ctb_rows],
            scratch: intra_full::IntraScratch::new(),
            deq_scratch: vec![0i32; 1024],
            res_scratch: vec![0i32; 1024],
            strong_smoothing: std::env::var("NOSTRONG").is_err(),
            sps,
            pps,
        })
    }

    pub(crate) fn decode(&mut self) -> Result<YuvPlanes, DecodeError> {
        let ctb = 1usize << self.log2_ctb;
        let wpp = self.pps.entropy_coding_sync_enabled;

        for ry in 0..self.ctb_rows {
            // WPP: at start of every non-first row, restore saved contexts and
            // reinitialize the CABAC engine from the current stream position
            // (which the previous row's sub-stream end already byte-aligned to).
            if wpp && ry > 0 {
                if let (Some(ctx), Some(ictx)) = (
                    self.wpp_ctx_snap[ry - 1].take(),
                    self.wpp_ictx_snap[ry - 1].take(),
                ) {
                    self.ctx = ctx;
                    self.ictx = ictx;
                }
                self.cab.reinit_engine();
            }

            for rx in 0..self.ctb_cols {
                if self.sps.sao_enabled {
                    self.parse_sao(rx, ry);
                }
                // New CTB → QG reset handled inside coding_unit via QG tracking.
                self.coding_quadtree(rx * ctb, ry * ctb, self.log2_ctb, 0);

                // WPP: save context snapshot after the 2nd CTB of each row.
                if wpp && rx == 1 {
                    self.wpp_ctx_snap[ry] = Some(self.ctx.clone());
                    self.wpp_ictx_snap[ry] = Some(self.ictx);
                }

                let end = self.cab.decode_terminate();
                if end != 0 {
                    // end_of_slice_segment_flag (or end_of_sub_stream if WPP
                    // miscounted) — just finish gracefully.
                    break;
                }
            }

            // WPP: after the last CTB of each non-final row, the stream contains
            // an end_of_sub_stream_one_bit (= 1), then byte-alignment padding,
            // then the next row's sub-stream starts.
            if wpp && ry < self.ctb_rows - 1 {
                let eoss = self.cab.decode_terminate();
                debug_assert_eq!(eoss, 1, "WPP: end_of_sub_stream_one_bit must be 1");
                self.cab.byte_align();
                // Engine reinit happens at the top of the next loop iteration.
            }
        }
        if self.sps.sao_enabled {
            self.apply_deblocking();
            self.apply_sao();
        }
        Ok(YuvPlanes {
            y: std::mem::take(&mut self.y),
            cb: std::mem::take(&mut self.cb),
            cr: std::mem::take(&mut self.cr),
            width: self.w,
            height: self.h,
            chroma: self.sps.chroma,
            bit_depth: self.sps.bit_depth().unwrap_or(BitDepth::Eight),
        })
    }

    fn parse_sao(&mut self, rx: usize, ry: usize) {
        let idx = ry * self.ctb_cols + rx;
        if !self.sao_luma && !self.sao_chroma {
            return;
        }
        let mut merge_left = false;
        let mut merge_up = false;
        if rx > 0 {
            merge_left = self.cab.decode_bin(&mut self.ctx.sao_merge_flag) != 0;
        }
        if !merge_left && ry > 0 {
            merge_up = self.cab.decode_bin(&mut self.ctx.sao_merge_flag) != 0;
        }
        if merge_left {
            self.sao[idx] = self.sao[idx - 1].clone();
            return;
        }
        if merge_up {
            self.sao[idx] = self.sao[idx - self.ctb_cols].clone();
            return;
        }

        let mut s = SaoCtb::default();
        let ncomp = if self.sps.chroma.is_monochrome() {
            1
        } else {
            3
        };
        let cmax = (1i32 << (self.bd.min(10) - 5)) - 1;
        for c in 0..ncomp {
            let enabled = if c == 0 {
                self.sao_luma
            } else {
                self.sao_chroma
            };
            if !enabled {
                continue;
            }
            // sao_type_idx
            let type_idx = if c == 2 {
                s.type_idx[1] // Cr reuses Cb's type
            } else {
                let bin0 = self.cab.decode_bin(&mut self.ctx.sao_type_idx);

                if bin0 == 0 {
                    0
                } else if self.cab.decode_bypass() == 0 {
                    1
                } else {
                    2
                }
            };
            s.type_idx[c] = type_idx;
            if type_idx == 0 {
                continue;
            }
            // 4 offset magnitudes (TR, bypass, cMax)
            let mut absv = [0i32; 4];
            for v in absv.iter_mut() {
                let mut m = 0;
                while m < cmax && self.cab.decode_bypass() != 0 {
                    m += 1;
                }
                *v = m;
            }
            if type_idx == 1 {
                // band: signs for nonzero, then band position
                for dst in absv.iter_mut() {
                    if *dst != 0 && self.cab.decode_bypass() != 0 {
                        *dst = -*dst;
                    }
                }
                let mut bp = 0u8;
                for _ in 0..5 {
                    bp = (bp << 1) | self.cab.decode_bypass();
                }
                s.band_pos[c] = bp;
            } else {
                // edge: offsets are +,+,-,- by convention; eo_class
                absv[2] = -absv[2];
                absv[3] = -absv[3];
                if c != 2 {
                    let mut eo = 0u8;
                    for _ in 0..2 {
                        eo = (eo << 1) | self.cab.decode_bypass();
                    }
                    s.eo_class[c] = eo;
                } else {
                    s.eo_class[2] = s.eo_class[1];
                }
            }
            s.offsets[c] = absv;
        }
        self.sao[idx] = s;
    }

    fn apply_deblocking(&mut self) {
        // HEVC §8.7.2.4 Tables 8-10 / 8-11 (beta, tC)
        #[rustfmt::skip]
        static BETA: [i32; 52] = [
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 6, 7, 8, 9,
            10,11,12,13,14,15,16,17,18,20,
            22,24,26,28,30,32,34,36,38,40,
            42,44,46,48,50,52,54,56,58,60,
            62,64,
        ];
        #[rustfmt::skip]
        static TC: [i32; 54] = [
            0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,1,1,
            1,1,1,1,1,1,1,2,2,2,
            2,3,3,3,3,4,4,4,5,5,
            6,6,7,8,9,10,11,13,14,16,
            18,20,22,24,
        ];

        // Slice-level offsets (default 0 for libheif)
        let beta_offset = self.pps.beta_offset_div2 * 2;
        let tc_offset = self.pps.tc_offset_div2 * 2;
        // HEVC §8.7.2.3: table indices use QP′ = QP + QpBdOffset where
        // QpBdOffset = 6*(BitDepth−8).  For 8-bit this is 0; for 10-bit it's 12.
        let qp_bd_offset_y = 6 * (self.bd as i32 - 8);
        let qp_bd_offset_c = 6 * (self.bd_c as i32 - 8);

        // QP for dequantization (per 4×4 grid). For intra-only all Bs=2.
        // We use the CTB average QP — approximation good enough.
        let w = self.w;
        let h = self.h;
        let cw = self.cw;
        let ch = self.ch;
        let gw = self.grid_w;

        // Helper: look up qp_y_map for a luma pixel (rounded to 4×4 grid)
        let qp_at = |qp_map: &[i16], px: usize, py: usize| -> i32 {
            qp_map[(py / 4).min(h / 4 - 1) * gw + (px / 4).min(w / 4 - 1)] as i32
        };

        // Vertical edges first (filter across columns), then horizontal.
        for pass in 0..2usize {
            // pass 0 = vertical edges (x mod 8 == 0, filter across x boundary)
            // pass 1 = horizontal edges (y mod 8 == 0, filter across y boundary)
            let (edge_step, scan_step, edge_max, scan_max) = if pass == 0 {
                (8, 1, w, h)
            } else {
                (8, 1, h, w)
            };
            let _ = scan_step;

            let mut edge = 8; // skip image boundary
            while edge < edge_max {
                // For each 4-pixel segment along the edge
                let mut scan = 0;
                while scan + 4 <= scan_max {
                    // p-side: pixels going into the block (edge-1, edge-2, edge-3, edge-4)
                    // q-side: pixels going out (edge, edge+1, edge+2, edge+3)
                    if pass == 0 {
                        // vertical edge at x=edge, rows scan..scan+3
                        let mid = scan + 1; // representative row
                        let qp_p = qp_at(&self.qp_y_map, edge - 1, mid);
                        let qp_q = qp_at(&self.qp_y_map, edge, mid);
                        let avg_qp = (qp_p + qp_q + 1) >> 1;
                        let beta_prime = (avg_qp + qp_bd_offset_y + beta_offset).clamp(0, 51);
                        let tc_prime = (avg_qp + qp_bd_offset_y + 2 + tc_offset).clamp(0, 53);
                        let beta = BETA[beta_prime as usize];
                        let tc = TC[tc_prime as usize];
                        if tc == 0 {
                            scan += 4;
                            continue;
                        }

                        // Compute d across all 4 rows of the segment
                        let mut d_total = 0i32;
                        for s in scan..scan + 4 {
                            if s >= h {
                                break;
                            }
                            let p = |o: usize| self.y[s * w + edge - 1 - o] as i32;
                            let q = |o: usize| self.y[s * w + edge + o] as i32;
                            d_total +=
                                (p(2) - 2 * p(1) + p(0)).abs() + (q(0) - 2 * q(1) + q(2)).abs();
                        }
                        if d_total >= beta {
                            scan += 4;
                            continue;
                        }

                        // Apply filter to each of the 4 rows
                        for s in scan..scan + 4 {
                            if s >= h {
                                continue;
                            }
                            let base_p = s * w + edge - 1;
                            let base_q = s * w + edge;
                            let (p0, p1, p2, p3) = (
                                self.y[base_p] as i32,
                                self.y[base_p - 1] as i32,
                                self.y[base_p - 2] as i32,
                                self.y[base_p - 3] as i32,
                            );
                            let (q0, q1, q2, q3) = (
                                self.y[base_q] as i32,
                                self.y[base_q + 1] as i32,
                                self.y[base_q + 2] as i32,
                                self.y[base_q + 3] as i32,
                            );
                            let dp = (p2 - 2 * p1 + p0).abs();
                            let dq = (q2 - 2 * q1 + q0).abs();
                            let d = dp + dq;
                            let strong = d < (beta >> 2)
                                && (p0 - q0).abs() < (5 * tc + 1) >> 1
                                && (p3 - p0).abs() + (q0 - q3).abs() < (beta * 3) >> 3;
                            let maxv = (1i32 << self.bd) - 1;
                            if strong {
                                self.y[base_p] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3)
                                    .clamp(0, maxv)
                                    as u16;
                                self.y[base_p - 1] =
                                    ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(0, maxv) as u16;
                                self.y[base_p - 2] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3)
                                    .clamp(0, maxv)
                                    as u16;
                                self.y[base_q] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3)
                                    .clamp(0, maxv)
                                    as u16;
                                self.y[base_q + 1] =
                                    ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(0, maxv) as u16;
                                self.y[base_q + 2] = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3)
                                    .clamp(0, maxv)
                                    as u16;
                            } else {
                                let delta =
                                    ((9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4).clamp(-tc, tc);
                                self.y[base_p] = (p0 + delta).clamp(0, maxv) as u16;
                                self.y[base_q] = (q0 - delta).clamp(0, maxv) as u16;
                                let thres = (tc * 10 + 1) >> 1;
                                if (2 * (p0 - p1) - delta).abs() < thres {
                                    let dp1 = (((p2 + p0 + 1) >> 1) - p1 + (delta >> 1))
                                        .clamp(-(tc >> 1), tc >> 1);
                                    self.y[base_p - 1] = (p1 + dp1).clamp(0, maxv) as u16;
                                }
                                if (2 * (q0 - q1) + delta).abs() < thres {
                                    let dq1 = (((q2 + q0 + 1) >> 1) - q1 - (delta >> 1))
                                        .clamp(-(tc >> 1), tc >> 1);
                                    self.y[base_q + 1] = (q1 + dq1).clamp(0, maxv) as u16;
                                }
                            }
                        }
                        scan += 4;
                        continue;
                    } else {
                        // horizontal edge at y=edge, cols scan..scan+3
                        let mid = scan + 1;
                        let qp_p = qp_at(&self.qp_y_map, mid, edge - 1);
                        let qp_q = qp_at(&self.qp_y_map, mid, edge);
                        let avg_qp = (qp_p + qp_q + 1) >> 1;
                        let beta_prime = (avg_qp + qp_bd_offset_y + beta_offset).clamp(0, 51);
                        let tc_prime = (avg_qp + qp_bd_offset_y + 2 + tc_offset).clamp(0, 53);
                        let beta = BETA[beta_prime as usize];
                        let tc = TC[tc_prime as usize];
                        if tc == 0 {
                            scan += 4;
                            continue;
                        }

                        let mut d_total = 0i32;
                        for s in scan..scan + 4 {
                            if s >= w {
                                break;
                            }
                            let p = |o: usize| self.y[(edge - 1 - o) * w + s] as i32;
                            let q = |o: usize| self.y[(edge + o) * w + s] as i32;
                            d_total +=
                                (p(2) - 2 * p(1) + p(0)).abs() + (q(0) - 2 * q(1) + q(2)).abs();
                        }
                        if d_total >= beta {
                            scan += 4;
                            continue;
                        }

                        for s in scan..scan + 4 {
                            if s >= w {
                                continue;
                            }
                            let (p0, p1, p2, p3) = (
                                self.y[(edge - 1) * w + s] as i32,
                                self.y[(edge - 2) * w + s] as i32,
                                self.y[(edge - 3) * w + s] as i32,
                                if edge >= 4 {
                                    self.y[(edge - 4) * w + s] as i32
                                } else {
                                    0
                                },
                            );
                            let (q0, q1, q2, q3) = (
                                self.y[(edge) * w + s] as i32,
                                self.y[(edge + 1) * w + s] as i32,
                                self.y[(edge + 2) * w + s] as i32,
                                if edge + 3 < h {
                                    self.y[(edge + 3) * w + s] as i32
                                } else {
                                    0
                                },
                            );
                            let dp = (p2 - 2 * p1 + p0).abs();
                            let dq = (q2 - 2 * q1 + q0).abs();
                            let d = dp + dq;
                            let strong = d < (beta >> 2)
                                && (p0 - q0).abs() < (5 * tc + 1) >> 1
                                && (p3 - p0).abs() + (q0 - q3).abs() < (beta * 3) >> 3;
                            let maxv = (1i32 << self.bd) - 1;
                            if strong {
                                self.y[(edge - 1) * w + s] =
                                    ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(0, maxv)
                                        as u16;
                                self.y[(edge - 2) * w + s] =
                                    ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(0, maxv) as u16;
                                self.y[(edge - 3) * w + s] =
                                    ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(0, maxv)
                                        as u16;
                                self.y[(edge) * w + s] =
                                    ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(0, maxv)
                                        as u16;
                                self.y[(edge + 1) * w + s] =
                                    ((p0 + q0 + q1 + q2 + 2) >> 2).clamp(0, maxv) as u16;
                                self.y[(edge + 2) * w + s] =
                                    ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(0, maxv)
                                        as u16;
                            } else {
                                let delta =
                                    ((9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4).clamp(-tc, tc);
                                self.y[(edge - 1) * w + s] = (p0 + delta).clamp(0, maxv) as u16;
                                self.y[(edge) * w + s] = (q0 - delta).clamp(0, maxv) as u16;
                                let thres = (tc * 10 + 1) >> 1;
                                if (2 * (p0 - p1) - delta).abs() < thres {
                                    let dp1 = (((p2 + p0 + 1) >> 1) - p1 + (delta >> 1))
                                        .clamp(-(tc >> 1), tc >> 1);
                                    self.y[(edge - 2) * w + s] = (p1 + dp1).clamp(0, maxv) as u16;
                                }
                                if (2 * (q0 - q1) + delta).abs() < thres {
                                    let dq1 = (((q2 + q0 + 1) >> 1) - q1 - (delta >> 1))
                                        .clamp(-(tc >> 1), tc >> 1);
                                    self.y[(edge + 1) * w + s] = (q1 + dq1).clamp(0, maxv) as u16;
                                }
                            }
                        }
                        scan += 4;
                        continue;
                    }
                }
                edge += edge_step;
            }
        }

        for pass in 0..2usize {
            let (edge_step, scan_max) = if pass == 0 { (8, ch) } else { (8, cw) };
            let maxv_c = (1i32 << self.bd_c) - 1;

            let mut edge = 8;
            while edge < if pass == 0 { cw } else { ch } {
                let mut scan = 0;
                while scan + 4 <= scan_max {
                    let mid = scan + 1;
                    // QP for chroma — use luma QP at corresponding position
                    let (qlx, qly) = if pass == 0 {
                        (edge * self.sub_w, mid * self.sub_h)
                    } else {
                        (mid * self.sub_w, edge * self.sub_h)
                    };
                    let avg_qp_l = qp_at(&self.qp_y_map, qlx.min(w - 1), qly.min(h - 1));
                    let tc_prime_c = (avg_qp_l + qp_bd_offset_c + 2 + tc_offset).clamp(0, 53);
                    let tc_c = TC[tc_prime_c as usize];
                    if tc_c == 0 {
                        scan += 4;
                        continue;
                    }

                    for plane in 0..2usize {
                        let pix = if plane == 0 {
                            &mut self.cb
                        } else {
                            &mut self.cr
                        };
                        for s in scan..scan + 4 {
                            if s >= scan_max {
                                continue;
                            }
                            let (p0, p1, q0, q1) = if pass == 0 {
                                (
                                    pix[s * cw + edge - 1] as i32,
                                    pix[s * cw + edge - 2] as i32,
                                    pix[s * cw + edge] as i32,
                                    pix[s * cw + edge + 1] as i32,
                                )
                            } else {
                                (
                                    pix[(edge - 1) * cw + s] as i32,
                                    pix[(edge - 2) * cw + s] as i32,
                                    pix[(edge) * cw + s] as i32,
                                    pix[(edge + 1) * cw + s] as i32,
                                )
                            };
                            let delta = ((q0 - p0) * 4 + p1 - q1 + 4) >> 3;
                            let delta = delta.clamp(-tc_c, tc_c);
                            if delta != 0 {
                                let (ip, iq) = if pass == 0 {
                                    (s * cw + edge - 1, s * cw + edge)
                                } else {
                                    ((edge - 1) * cw + s, edge * cw + s)
                                };
                                pix[ip] = (p0 + delta).clamp(0, maxv_c) as u16;
                                pix[iq] = (q0 - delta).clamp(0, maxv_c) as u16;
                            }
                        }
                    }
                    scan += 4;
                }
                edge += edge_step;
            }
        }
    }

    fn apply_sao(&mut self) {
        let ctb = 1usize << self.log2_ctb;
        // Work on clones so EO neighbor lookups always use original values.
        let orig_y = self.y.clone();
        let orig_cb = self.cb.clone();
        let orig_cr = self.cr.clone();

        for ry in 0..self.ctb_rows {
            for rx in 0..self.ctb_cols {
                let idx = ry * self.ctb_cols + rx;
                let sao = &self.sao[idx];
                let x0 = rx * ctb;
                let y0 = ry * ctb;

                // Luma
                if self.sao_luma && sao.type_idx[0] != 0 {
                    let x_end = (x0 + ctb).min(self.w);
                    let y_end = (y0 + ctb).min(self.h);
                    Self::apply_sao_plane(
                        &mut self.y,
                        &orig_y,
                        self.w,
                        self.h,
                        x0,
                        y0,
                        x_end,
                        y_end,
                        sao.type_idx[0],
                        &sao.offsets[0],
                        sao.band_pos[0],
                        sao.eo_class[0],
                        self.bd,
                    );
                }
                // Chroma (Cb, Cr share eo_class)
                if self.sao_chroma {
                    let cw = self.cw;
                    let ch = self.ch;
                    let cx0 = x0 / self.sub_w;
                    let cy0 = y0 / self.sub_h;
                    let cx_end = ((x0 + ctb) / self.sub_w).min(cw);
                    let cy_end = ((y0 + ctb) / self.sub_h).min(ch);

                    if sao.type_idx[1] != 0 {
                        Self::apply_sao_plane(
                            &mut self.cb,
                            &orig_cb,
                            cw,
                            ch,
                            cx0,
                            cy0,
                            cx_end,
                            cy_end,
                            sao.type_idx[1],
                            &sao.offsets[1],
                            sao.band_pos[1],
                            sao.eo_class[1],
                            self.bd_c,
                        );
                    }
                    if sao.type_idx[2] != 0 {
                        Self::apply_sao_plane(
                            &mut self.cr,
                            &orig_cr,
                            cw,
                            ch,
                            cx0,
                            cy0,
                            cx_end,
                            cy_end,
                            sao.type_idx[2],
                            &sao.offsets[2],
                            sao.band_pos[2],
                            sao.eo_class[2],
                            self.bd_c,
                        );
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_sao_plane(
        dst: &mut [u16],
        src: &[u16],
        w: usize,
        h: usize,
        x0: usize,
        y0: usize,
        x_end: usize,
        y_end: usize,
        type_idx: u8,
        offsets: &[i32; 4],
        band_pos: u8,
        eo_class: u8,
        bd: u8,
    ) {
        let max_val = ((1u32 << bd) - 1) as i32;

        match type_idx {
            1 => {
                // Band offset
                let shift = bd - 5;
                for y in y0..y_end {
                    for x in x0..x_end {
                        let s = src[y * w + x] as i32;
                        let band = (s >> shift) as u8;
                        let rel = band.wrapping_sub(band_pos);
                        if rel < 4 {
                            let v = (s + offsets[rel as usize]).clamp(0, max_val);
                            dst[y * w + x] = v as u16;
                        }
                    }
                }
            }
            2 => {
                // Edge offset (§8.7.3.2.4)
                // Direction vectors for the two neighbors
                let (dx, dy): (i32, i32) = match eo_class {
                    0 => (1, 0),  // horizontal
                    1 => (0, 1),  // vertical
                    2 => (1, 1),  // 135°
                    _ => (1, -1), // 45°
                };

                for y in y0..y_end {
                    for x in x0..x_end {
                        let s = src[y * w + x] as i32;

                        // Neighbour 1 (forward direction)
                        let x1 = x as i32 + dx;
                        let y1 = y as i32 + dy;
                        // Neighbour 2 (backward direction)
                        let x2 = x as i32 - dx;
                        let y2 = y as i32 - dy;

                        // Out-of-bounds neighbors count as "equal" (no offset)
                        let inb = |xx: i32, yy: i32| -> bool {
                            xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h
                        };
                        let n1 = if inb(x1, y1) {
                            src[y1 as usize * w + x1 as usize] as i32
                        } else {
                            s
                        };
                        let n2 = if inb(x2, y2) {
                            src[y2 as usize * w + x2 as usize] as i32
                        } else {
                            s
                        };

                        let sign1 = (s > n1) as i32 - (s < n1) as i32;
                        let sign2 = (s > n2) as i32 - (s < n2) as i32;
                        let edge_idx = sign1 + sign2 + 2; // 0..4

                        // category 2 (edge_idx==2) always has offset 0
                        let offset = match edge_idx {
                            0 => offsets[0], // local min → positive offset
                            1 => offsets[1],
                            3 => offsets[2],
                            4 => offsets[3], // local max → negative offset
                            _ => 0,
                        };
                        if offset != 0 {
                            dst[y * w + x] = (s + offset).clamp(0, max_val) as u16;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn coding_quadtree(&mut self, x0: usize, y0: usize, log2_cb: u32, depth: u8) {
        let cb_size = 1usize << log2_cb;
        let in_pic = x0 + cb_size <= self.w && y0 + cb_size <= self.h;
        let can_split = log2_cb > self.log2_min_cb;
        let split = if x0 + cb_size <= self.w && y0 + cb_size <= self.h && can_split {
            // read split_cu_flag with neighbour-depth context
            let ctx_inc = self.split_cu_ctx(x0, y0, depth);
            self.cab.decode_bin(&mut self.ctx.split_cu_flag[ctx_inc]) != 0
        } else {
            can_split && !in_pic
        };
        if split {
            let half = cb_size / 2;
            let d = depth + 1;
            self.coding_quadtree(x0, y0, log2_cb - 1, d);
            if x0 + half < self.w {
                self.coding_quadtree(x0 + half, y0, log2_cb - 1, d);
            }
            if y0 + half < self.h {
                self.coding_quadtree(x0, y0 + half, log2_cb - 1, d);
            }
            if x0 + half < self.w && y0 + half < self.h {
                self.coding_quadtree(x0 + half, y0 + half, log2_cb - 1, d);
            }
        } else {
            self.set_ct_depth(x0, y0, cb_size, depth);
            self.coding_unit(x0, y0, log2_cb);
        }
    }

    fn split_cu_ctx(&self, x0: usize, y0: usize, depth: u8) -> usize {
        let mut inc = 0;
        if x0 >= 4 {
            let g = (y0 / 4) * self.grid_w + (x0 - 1) / 4;
            if self.decoded[g] && self.ct_depth[g] as usize > depth as usize {
                inc += 1;
            }
        }
        if y0 >= 4 {
            let g = ((y0 - 1) / 4) * self.grid_w + x0 / 4;
            if self.decoded[g] && self.ct_depth[g] as usize > depth as usize {
                inc += 1;
            }
        }
        inc
    }

    fn set_ct_depth(&mut self, x0: usize, y0: usize, size: usize, depth: u8) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w && yy < self.h {
                    self.ct_depth[(yy / 4) * self.grid_w + xx / 4] = depth;
                }
            }
        }
    }

    fn coding_unit(&mut self, x0: usize, y0: usize, log2_cb: u32) {
        let dbg = std::env::var("DBG").is_ok() && y0 < 64;
        if dbg {
            eprintln!("CU ({},{}) log2={}", x0, y0, log2_cb);
        }
        // QG handling
        let qg_mask = !((1usize << self.log2_qg) - 1);
        let xqg = x0 & qg_mask;
        let yqg = y0 & qg_mask;
        if x0 == xqg && y0 == yqg {
            self.is_cu_qp_delta_coded = false;
            self.cu_qp_delta_val = 0;
            self.cur_qp = self.predict_qp(xqg, yqg);
        }

        if self.pps.transquant_bypass_enabled {
            // cu_transquant_bypass_flag — libheif disables; not expected.
        }

        let cb_size = 1usize << log2_cb;
        // part_mode: NxN only at min CB
        let nxn = if log2_cb == self.log2_min_cb {
            self.cab.decode_bin(&mut self.ictx.part_mode) == 0
        } else {
            false
        };

        let npu = if nxn { 2 } else { 1 };
        let pu_size = cb_size / npu;
        let mut luma_modes = [MODE_DC; 4];

        // prev_intra_luma_pred_flag for each PU
        let mut prev_flags = [false; 4];
        for dst in prev_flags[..npu * npu].iter_mut() {
            *dst = self
                .cab
                .decode_bin(&mut self.ictx.prev_intra_luma_pred_flag)
                != 0;
        }
        let mut mpm_or_rem = [0u8; 4];
        for (&src, dst) in prev_flags[..npu * npu].iter().zip(mpm_or_rem.iter_mut()) {
            if src {
                // mpm_idx: TR cMax=2, bypass
                let mut v = 0u8;
                if self.cab.decode_bypass() != 0 {
                    v = 1 + self.cab.decode_bypass();
                }
                *dst = v;
            } else {
                let mut v = 0u8;
                for _ in 0..5 {
                    v = (v << 1) | self.cab.decode_bypass();
                }
                *dst = v;
            }
        }
        for i in 0..npu * npu {
            let pux = x0 + (i % npu) * pu_size;
            let puy = y0 + (i / npu) * pu_size;
            let mode = self.derive_luma_mode(pux, puy, prev_flags[i], mpm_or_rem[i]);
            luma_modes[i] = mode;
            self.set_mode(pux, puy, pu_size, mode);
        }
        if dbg {
            eprintln!(
                "  nxn x={} y={} prev={:?} mpm_rem={:?} modes={:?}",
                x0,
                y0,
                &prev_flags[..npu * npu],
                &mpm_or_rem[..npu * npu],
                &luma_modes[..npu * npu]
            );
        }

        // intra_chroma_pred_mode (1 per CU for 4:2:0/4:2:2)
        let chroma_mode = if self.sps.chroma.is_monochrome() {
            MODE_DC
        } else {
            self.decode_chroma_mode(luma_modes[0])
        };

        // transform_tree
        let intra_split = nxn;
        let max_depth = self.max_trafo_depth_intra + intra_split as u32;
        self.transform_tree(
            x0,
            y0,
            x0,
            y0,
            log2_cb,
            0,
            0,
            &luma_modes,
            chroma_mode,
            intra_split,
            max_depth,
            false,
            false,
        );

        // mark decoded
        self.mark_decoded(x0, y0, cb_size);
        self.qp_y_prev = self.cur_qp;
        self.set_qp(x0, y0, cb_size, self.cur_qp);
    }

    fn predict_qp(&self, xqg: usize, yqg: usize) -> i32 {
        let ctb = 1usize << self.log2_ctb;
        let ctb_x = (xqg / ctb) * ctb;
        let ctb_y = (yqg / ctb) * ctb;

        // WPP (HEVC §8.6.1): for the first QG in a CTB row, qPY_PRED = SliceQpY.
        let first_in_ctb_row = xqg == 0 && (yqg & (ctb - 1)) == 0;
        if self.pps.entropy_coding_sync_enabled && first_in_ctb_row {
            return self.slice_qp;
        }

        // qPY_A: left neighbour, must be in same CTB
        let qp_a = if xqg >= 1 && (xqg - 1) >= ctb_x {
            self.qp_y_map[(yqg / 4) * self.grid_w + (xqg - 1) / 4] as i32
        } else {
            self.qp_y_prev
        };
        // qPY_B: above neighbour, must be in same CTB
        let qp_b = if yqg >= 1 && (yqg - 1) >= ctb_y {
            self.qp_y_map[((yqg - 1) / 4) * self.grid_w + xqg / 4] as i32
        } else {
            self.qp_y_prev
        };
        (qp_a + qp_b + 1) >> 1
    }

    fn set_qp(&mut self, x0: usize, y0: usize, size: usize, qp: i32) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w && yy < self.h {
                    self.qp_y_map[(yy / 4) * self.grid_w + xx / 4] = qp as i16;
                }
            }
        }
    }

    fn set_mode(&mut self, x0: usize, y0: usize, size: usize, mode: u8) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w && yy < self.h {
                    self.mode_y[(yy / 4) * self.grid_w + xx / 4] = mode;
                }
            }
        }
    }

    fn mark_decoded(&mut self, x0: usize, y0: usize, size: usize) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w && yy < self.h {
                    self.decoded[(yy / 4) * self.grid_w + xx / 4] = true;
                }
            }
        }
    }

    fn derive_luma_mode(&self, x0: usize, y0: usize, prev: bool, val: u8) -> u8 {
        let cand_a = self.neighbor_mode(x0 as i32 - 1, y0 as i32, true);
        let cand_b = self.neighbor_mode(x0 as i32, y0 as i32 - 1, false);
        let mpm = mpm_list(cand_a, cand_b, y0, self.log2_ctb);
        if prev {
            mpm[val as usize]
        } else {
            let mut sorted = mpm;
            sorted.sort_unstable();
            let mut mode = val;
            for &m in sorted.iter() {
                if mode >= m {
                    mode += 1;
                }
            }
            mode
        }
    }

    fn neighbor_mode(&self, x: i32, y: i32, _left: bool) -> u8 {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return MODE_DC;
        }
        let g = (y as usize / 4) * self.grid_w + x as usize / 4;
        self.mode_y[g]
    }

    fn decode_chroma_mode(&mut self, luma_mode: u8) -> u8 {
        let bin0 = self.cab.decode_bin(&mut self.ictx.intra_chroma_pred_mode);
        if bin0 == 0 {
            return luma_mode; // DM
        }
        let mut idx = 0u8;
        for _ in 0..2 {
            idx = (idx << 1) | self.cab.decode_bypass();
        }
        let cand = [0u8, 26, 10, 1][idx as usize];
        if cand == luma_mode { 34 } else { cand }
    }
}

/// MPM candidate list (§8.4.2).
fn mpm_list(mut cand_a: u8, cand_b: u8, y0: usize, log2_ctb: u32) -> [u8; 3] {
    // candB from a different CTB row → DC
    let cand_b = if y0 > 0 && ((y0 - 1) >> log2_ctb) != (y0 >> log2_ctb) {
        MODE_DC
    } else {
        cand_b
    };
    let _ = &mut cand_a;
    if cand_a == cand_b {
        if cand_a < 2 {
            [MODE_PLANAR, MODE_DC, 26]
        } else {
            [
                cand_a,
                2 + ((cand_a as i32 + 29) % 32) as u8,
                2 + ((cand_a as i32 - 2 + 1) % 32) as u8,
            ]
        }
    } else {
        let m0 = cand_a;
        let m1 = cand_b;
        let m2 = if m0 != MODE_PLANAR && m1 != MODE_PLANAR {
            MODE_PLANAR
        } else if m0 != MODE_DC && m1 != MODE_DC {
            MODE_DC
        } else {
            26
        };
        [m0, m1, m2]
    }
}

// ── transform tree + unit (impl continued) ─────────────────────────────────
impl<'a> FullDecoder<'a> {
    #[allow(clippy::too_many_arguments)]
    fn transform_tree(
        &mut self,
        x0: usize,
        y0: usize,
        xbase: usize,
        ybase: usize,
        log2_ts: u32,
        depth: u8,
        blk_idx: u8,
        luma_modes: &[u8; 4],
        chroma_mode: u8,
        intra_split: bool,
        max_depth: u32,
        parent_cbf_cb: bool,
        parent_cbf_cr: bool,
    ) {
        let split_allowed = log2_ts <= self.log2_max_tb
            && log2_ts > self.log2_min_tb
            && (depth as u32) < max_depth
            && !(intra_split && depth == 0);
        let split = if split_allowed {
            self.cab
                .decode_bin(&mut self.ctx.split_transform_flag[(5 - log2_ts) as usize])
                != 0
        } else {
            log2_ts > self.log2_max_tb || (intra_split && depth == 0)
        };

        // chroma cbf
        let chroma_present = !self.sps.chroma.is_monochrome();
        let _ = depth;
        let mut cbf_cb = parent_cbf_cb;
        let mut cbf_cr = parent_cbf_cr;
        if chroma_present && (log2_ts > 2 || self.sps.chroma_idc == 3) {
            if depth == 0 || parent_cbf_cb {
                cbf_cb = self
                    .cab
                    .decode_bin(&mut self.ctx.cbf_chroma[depth.min(4) as usize])
                    != 0;
            }
            if depth == 0 || parent_cbf_cr {
                cbf_cr = self
                    .cab
                    .decode_bin(&mut self.ctx.cbf_chroma[depth.min(4) as usize])
                    != 0;
            }
        }

        if split {
            let half = 1usize << (log2_ts - 1);
            self.transform_tree(
                x0,
                y0,
                x0,
                y0,
                log2_ts - 1,
                depth + 1,
                0,
                luma_modes,
                chroma_mode,
                intra_split,
                max_depth,
                cbf_cb,
                cbf_cr,
            );
            self.transform_tree(
                x0 + half,
                y0,
                x0,
                y0,
                log2_ts - 1,
                depth + 1,
                1,
                luma_modes,
                chroma_mode,
                intra_split,
                max_depth,
                cbf_cb,
                cbf_cr,
            );
            self.transform_tree(
                x0,
                y0 + half,
                x0,
                y0,
                log2_ts - 1,
                depth + 1,
                2,
                luma_modes,
                chroma_mode,
                intra_split,
                max_depth,
                cbf_cb,
                cbf_cr,
            );
            self.transform_tree(
                x0 + half,
                y0 + half,
                x0,
                y0,
                log2_ts - 1,
                depth + 1,
                3,
                luma_modes,
                chroma_mode,
                intra_split,
                max_depth,
                cbf_cb,
                cbf_cr,
            );
        } else {
            // cbf_luma always read for intra
            let cbf_luma = self
                .cab
                .decode_bin(&mut self.ctx.cbf_luma[if depth == 0 { 1 } else { 0 }])
                != 0;
            self.transform_unit(
                x0,
                y0,
                xbase,
                ybase,
                log2_ts,
                depth,
                blk_idx,
                luma_modes,
                chroma_mode,
                cbf_luma,
                cbf_cb,
                cbf_cr,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_unit(
        &mut self,
        x0: usize,
        y0: usize,
        xbase: usize,
        ybase: usize,
        log2_ts: u32,
        depth: u8,
        blk_idx: u8,
        luma_modes: &[u8; 4],
        chroma_mode: u8,
        cbf_luma: bool,
        cbf_cb: bool,
        cbf_cr: bool,
    ) {
        let chroma_present = !self.sps.chroma.is_monochrome();
        let _ = depth;
        let any_chroma = cbf_cb || cbf_cr;
        let need_qp = cbf_luma || any_chroma;

        // cu_qp_delta
        if self.pps.cu_qp_delta_enabled && need_qp && !self.is_cu_qp_delta_coded {
            self.cu_qp_delta_val = self.decode_cu_qp_delta();
            self.is_cu_qp_delta_coded = true;
            // recompute QpY for the QG
            let qp_bd = 0; // 8/10/12-bit luma offset is 0 here (QpBdOffsetY=6*(bd-8) but applied symmetrically)
            let off = 6 * (self.bd as i32 - 8);
            self.cur_qp =
                ((self.predict_qp_cur() + self.cu_qp_delta_val + 52 + 2 * off) % (52 + off)) - off
                    + qp_bd;
        }

        // luma residual + reconstruction
        let luma_mode = self.luma_mode_at(x0, y0, luma_modes, blk_idx);
        if cbf_luma {
            let scan = luma_scan(luma_mode, log2_ts);
            let ts_ctx = if self.pps.transform_skip_enabled && log2_ts == 2 {
                Some(0)
            } else {
                None
            };
            let (levels, _tskip) = residual_coding(
                &mut self.cab,
                &mut self.ctx,
                log2_ts,
                true,
                scan,
                self.sign_hiding,
                ts_ctx,
                false,
            );
            self.reconstruct_luma(x0, y0, log2_ts, luma_mode, &levels);
        } else {
            // prediction only (no residual) still needs to fill rec for neighbors
            self.predict_only_luma(x0, y0, log2_ts, luma_mode);
        }

        // chroma
        if chroma_present {
            if log2_ts > 2 || self.sps.chroma_idc == 3 {
                self.do_chroma(x0, y0, log2_ts, chroma_mode, cbf_cb, cbf_cr);
            } else if blk_idx == 3 {
                // 4×4 luma TUs: chroma coded once at parent 8×8 (log2=2 chroma)
                self.do_chroma(xbase, ybase, 3, chroma_mode, cbf_cb, cbf_cr);
            }
        }
    }

    fn predict_qp_cur(&self) -> i32 {
        // qPY_PRED was stored in cur_qp at QG entry (before delta).
        self.cur_qp
    }

    fn decode_cu_qp_delta(&mut self) -> i32 {
        // cu_qp_delta_abs: prefix TU (cMax=5) ctx[0] then ctx[1], then bypass EG0
        let mut abs_val;
        let mut prefix = 0;
        while prefix < 5 {
            let ci = if prefix == 0 { 0 } else { 1 };
            if self.cab.decode_bin(&mut self.ctx.cu_qp_delta_abs[ci]) == 0 {
                break;
            }
            prefix += 1;
        }
        abs_val = prefix;
        if prefix >= 5 {
            // EG0 suffix (bypass)
            let mut k = 0;
            while self.cab.decode_bypass() != 0 {
                k += 1;
                if k > 30 {
                    break;
                }
            }
            let mut suffix = 0i32;
            for _ in 0..k {
                suffix = (suffix << 1) | self.cab.decode_bypass() as i32;
            }
            abs_val += suffix + (1 << k) - 1;
        }
        if abs_val > 0 {
            let sign = self.cab.decode_bypass();
            if sign != 0 { -abs_val } else { abs_val }
        } else {
            0
        }
    }

    fn luma_mode_at(&self, x0: usize, y0: usize, _modes: &[u8; 4], _blk: u8) -> u8 {
        self.mode_y[(y0 / 4) * self.grid_w + x0 / 4]
    }
}

fn luma_scan(mode: u8, log2_ts: u32) -> u8 {
    if log2_ts == 2 || log2_ts == 3 {
        if (6..=14).contains(&mode) {
            SCAN_VERT
        } else if (22..=30).contains(&mode) {
            SCAN_HORIZ
        } else {
            SCAN_DIAG
        }
    } else {
        SCAN_DIAG
    }
}

fn chroma_scan(mode: u8, log2_ts: u32) -> u8 {
    if log2_ts == 2 {
        if (6..=14).contains(&mode) {
            SCAN_VERT
        } else if (22..=30).contains(&mode) {
            SCAN_HORIZ
        } else {
            SCAN_DIAG
        }
    } else {
        SCAN_DIAG
    }
}

// ── reconstruction helpers ──────────────────────────────────────────────────
impl<'a> FullDecoder<'a> {
    fn luma_avail(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return false;
        }
        self.decoded[(y as usize / 4) * self.grid_w + x as usize / 4]
    }
    fn chroma_avail(&self, cx: i32, cy: i32) -> bool {
        if cx < 0 || cy < 0 || cx as usize >= self.cw || cy as usize >= self.ch {
            return false;
        }
        let lx = cx as usize * self.sub_w;
        let ly = cy as usize * self.sub_h;
        if lx >= self.w || ly >= self.h {
            return false;
        }
        self.decoded[(ly / 4) * self.grid_w + lx / 4]
    }

    fn gather_luma_refs_into(
        &self,
        x0: usize,
        y0: usize,
        n: usize,
        above: &mut [Option<u16>],
        left: &mut [Option<u16>],
    ) -> Option<u16> {
        let corner = if self.luma_avail(x0 as i32 - 1, y0 as i32 - 1) {
            Some(self.y[(y0 - 1) * self.w + (x0 - 1)])
        } else {
            None
        };
        for i in 0..2 * n {
            let ax = x0 as i32 + i as i32;
            above[i] = if self.luma_avail(ax, y0 as i32 - 1) {
                Some(self.y[(y0 - 1) * self.w + ax as usize])
            } else {
                None
            };
            let ly = y0 as i32 + i as i32;
            left[i] = if self.luma_avail(x0 as i32 - 1, ly) {
                Some(self.y[ly as usize * self.w + (x0 - 1)])
            } else {
                None
            };
        }
        corner
    }

    fn reconstruct_luma(&mut self, x0: usize, y0: usize, log2_ts: u32, mode: u8, levels: &[i32]) {
        let n = 1usize << log2_ts;
        self.predict_luma_block_into(x0, y0, n, mode);
        let qp = self.cur_qp.clamp(0, 51) as u8;
        transform::dequantize_i32_into(levels, n, qp, self.bd, &mut self.deq_scratch[..n * n]);
        if n == 4 {
            transform::inv_transform_dst_into(
                &self.deq_scratch[..n * n],
                self.bd,
                &mut self.res_scratch[..n * n],
            );
        } else {
            transform::inv_transform_into(
                &self.deq_scratch[..n * n],
                n,
                self.bd,
                &mut self.res_scratch[..n * n],
            );
        }
        let max = (1i32 << self.bd) - 1;
        for yy in 0..n {
            for xx in 0..n {
                let v = (self.scratch.pred[yy * n + xx] as i32 + self.res_scratch[yy * n + xx])
                    .clamp(0, max);
                self.y[(y0 + yy) * self.w + (x0 + xx)] = v as u16;
            }
        }
        self.mark_decoded(x0, y0, n);
    }

    fn predict_only_luma(&mut self, x0: usize, y0: usize, log2_ts: u32, mode: u8) {
        let n = 1usize << log2_ts;
        self.predict_luma_block_into(x0, y0, n, mode);
        for yy in 0..n {
            for xx in 0..n {
                self.y[(y0 + yy) * self.w + (x0 + xx)] = self.scratch.pred[yy * n + xx];
            }
        }
        self.mark_decoded(x0, y0, n);
    }

    fn predict_luma_block_into(&mut self, x0: usize, y0: usize, n: usize, mode: u8) {
        let mut above = std::mem::take(&mut self.scratch.raw_above);
        let mut left = std::mem::take(&mut self.scratch.raw_left);
        let corner = self.gather_luma_refs_into(x0, y0, n, &mut above[..2 * n], &mut left[..2 * n]);
        let neutral = 1u16 << (self.bd - 1);
        let strong = self.strong_smoothing && self.sps.strong_intra_smoothing;
        let sc = &mut self.scratch;
        intra_full::substitute_refs_into(
            corner,
            &above[..2 * n],
            &left[..2 * n],
            n,
            neutral,
            &mut sc.sub_s,
            &mut sc.sub_avail,
            &mut sc.above,
            &mut sc.left,
        );
        intra_full::filter_refs_into(
            &sc.above[..2 * n + 1],
            &sc.left[..2 * n + 1],
            n,
            mode,
            true,
            strong,
            self.bd,
            &mut sc.fa,
            &mut sc.fl,
        );
        intra_full::predict_into(
            mode,
            &sc.fa[..2 * n + 1],
            &sc.fl[..2 * n + 1],
            n,
            true,
            self.bd,
            &mut sc.pred[..n * n],
            &mut sc.refs_ang,
        );
        self.scratch.raw_above = above;
        self.scratch.raw_left = left;
    }

    fn do_chroma(
        &mut self,
        lx: usize,
        ly: usize,
        luma_log2: u32,
        mode: u8,
        cbf_cb: bool,
        cbf_cr: bool,
    ) {
        let clog2 = if self.sps.chroma_idc == 3 {
            luma_log2
        } else {
            luma_log2 - 1
        };
        let cn = 1usize << clog2;
        let cx0 = lx / self.sub_w;
        let cy0 = ly / self.sub_h;
        let scan = chroma_scan(mode, clog2);

        // Cb
        let qp_cb = qpc(self.cur_qp + self.pps.cb_qp_offset, self.sps.chroma_idc);
        if cbf_cb {
            let (levels, _) = residual_coding(
                &mut self.cab,
                &mut self.ctx,
                clog2,
                false,
                scan,
                self.sign_hiding,
                None,
                false,
            );
            self.reconstruct_chroma(true, cx0, cy0, cn, mode, &levels, qp_cb);
        } else {
            self.predict_only_chroma(true, cx0, cy0, cn, mode);
        }
        // Cr
        let qp_cr = qpc(self.cur_qp + self.pps.cr_qp_offset, self.sps.chroma_idc);
        if cbf_cr {
            let (levels, _) = residual_coding(
                &mut self.cab,
                &mut self.ctx,
                clog2,
                false,
                scan,
                self.sign_hiding,
                None,
                false,
            );
            self.reconstruct_chroma(false, cx0, cy0, cn, mode, &levels, qp_cr);
        } else {
            self.predict_only_chroma(false, cx0, cy0, cn, mode);
        }
    }

    fn gather_chroma_refs_into(
        &self,
        is_cb: bool,
        cx0: usize,
        cy0: usize,
        n: usize,
        above: &mut [Option<u16>],
        left: &mut [Option<u16>],
    ) -> Option<u16> {
        let plane = if is_cb { &self.cb } else { &self.cr };
        let corner = if self.chroma_avail(cx0 as i32 - 1, cy0 as i32 - 1) {
            Some(plane[(cy0 - 1) * self.cw + (cx0 - 1)])
        } else {
            None
        };
        for i in 0..2 * n {
            let ax = cx0 as i32 + i as i32;
            above[i] = if self.chroma_avail(ax, cy0 as i32 - 1) {
                Some(plane[(cy0 - 1) * self.cw + ax as usize])
            } else {
                None
            };
            let ly = cy0 as i32 + i as i32;
            left[i] = if self.chroma_avail(cx0 as i32 - 1, ly) {
                Some(plane[ly as usize * self.cw + (cx0 - 1)])
            } else {
                None
            };
        }
        corner
    }

    fn predict_chroma_block_into(
        &mut self,
        is_cb: bool,
        cx0: usize,
        cy0: usize,
        n: usize,
        mode: u8,
    ) {
        let mut above = std::mem::take(&mut self.scratch.raw_above);
        let mut left = std::mem::take(&mut self.scratch.raw_left);
        let corner = self.gather_chroma_refs_into(
            is_cb,
            cx0,
            cy0,
            n,
            &mut above[..2 * n],
            &mut left[..2 * n],
        );
        let neutral = 1u16 << (self.bd_c - 1);
        let sc = &mut self.scratch;
        intra_full::substitute_refs_into(
            corner,
            &above[..2 * n],
            &left[..2 * n],
            n,
            neutral,
            &mut sc.sub_s,
            &mut sc.sub_avail,
            &mut sc.above,
            &mut sc.left,
        );
        // Chroma is not filtered for 4:2:0/4:2:2: skip filter step, use above/left directly
        intra_full::predict_into(
            mode,
            &sc.above[..2 * n + 1],
            &sc.left[..2 * n + 1],
            n,
            false,
            self.bd_c,
            &mut sc.pred[..n * n],
            &mut sc.refs_ang,
        );
        self.scratch.raw_above = above;
        self.scratch.raw_left = left;
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_chroma(
        &mut self,
        is_cb: bool,
        cx0: usize,
        cy0: usize,
        n: usize,
        mode: u8,
        levels: &[i32],
        qp: i32,
    ) {
        self.predict_chroma_block_into(is_cb, cx0, cy0, n, mode);
        let qp_c = qp.clamp(0, 51) as u8;
        transform::dequantize_i32_into(levels, n, qp_c, self.bd_c, &mut self.deq_scratch[..n * n]);
        transform::inv_transform_into(
            &self.deq_scratch[..n * n],
            n,
            self.bd_c,
            &mut self.res_scratch[..n * n],
        );
        let max = (1i32 << self.bd_c) - 1;
        // Copy scratch.pred out before mutable borrow of plane
        let n2 = n * n;
        let pred_tmp: [u16; 1024] = {
            // max chroma TB = 16×16 = 256 samples
            let mut buf = [0u16; 1024];
            buf[..n2].copy_from_slice(&self.scratch.pred[..n2]);
            buf
        };
        let plane = if is_cb { &mut self.cb } else { &mut self.cr };
        for yy in 0..n {
            for xx in 0..n {
                let v =
                    (pred_tmp[yy * n + xx] as i32 + self.res_scratch[yy * n + xx]).clamp(0, max);
                plane[(cy0 + yy) * self.cw + (cx0 + xx)] = v as u16;
            }
        }
    }

    fn predict_only_chroma(&mut self, is_cb: bool, cx0: usize, cy0: usize, n: usize, mode: u8) {
        self.predict_chroma_block_into(is_cb, cx0, cy0, n, mode);
        let n2 = n * n;
        let pred_tmp: [u16; 1024] = {
            let mut buf = [0u16; 1024];
            buf[..n2].copy_from_slice(&self.scratch.pred[..n2]);
            buf
        };
        let plane = if is_cb { &mut self.cb } else { &mut self.cr };
        for yy in 0..n {
            for xx in 0..n {
                plane[(cy0 + yy) * self.cw + (cx0 + xx)] = pred_tmp[yy * n + xx];
            }
        }
    }
}

/// Chroma QP mapping (Table 8-10). ChromaArrayType 1 (4:2:0) uses the table;
/// 2/3 clamp differently but share the <30 / table / -6 structure.
fn qpc(qpi: i32, chroma_idc: u8) -> i32 {
    let qpi = qpi.clamp(0, 57);
    if chroma_idc != 1 {
        // 4:2:2 / 4:4:4: QpC = min(qpi, 51)
        return qpi.min(51);
    }
    if qpi < 30 {
        qpi
    } else if qpi > 43 {
        qpi - 6
    } else {
        const T: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
        T[(qpi - 30) as usize]
    }
}

// ── Top-level entry point for lib.rs ────────────────────────────────────────

/// Parse a slice header from the RBSP (after 2-byte NAL header has been consumed
/// by the caller or is still in the byte slice — we consume it here).
/// Returns (slice_qp, sao_luma, sao_chroma, cabac_byte_offset).
pub(crate) fn parse_slice_header_full(
    rbsp: &[u8],
    sps: &crate::config::Sps,
    pps: &crate::config::Pps,
    nal_type: u8,
) -> Result<(i32, bool, bool, usize), crate::error::DecodeError> {
    let mut r = crate::bitreader::BitReader::new(rbsp);
    let e = |s: &'static str| crate::error::DecodeError::Bitstream(s.into());
    r.read_bits(16).map_err(|_| e("NAL header"))?; // consume 2-byte NAL header
    let _first = r.read_flag().map_err(|_| e("first_slice"))?;
    let is_irap = (16..=23).contains(&nal_type);
    if is_irap {
        r.read_flag().map_err(|_| e("no_prior_pics"))?;
    }
    let _pps_id = r.read_ue().map_err(|_| e("pps_id"))?;
    for _ in 0..pps.num_extra_slice_header_bits {
        r.read_bit().map_err(|_| e("extra_bits"))?;
    }
    let _slice_type = r.read_ue().map_err(|_| e("slice_type"))?;
    if pps.output_flag_present {
        r.read_flag().map_err(|_| e("pic_output_flag"))?;
    }
    if sps.separate_colour_plane {
        r.read_bits(2).map_err(|_| e("colour_plane"))?;
    }
    let is_idr = nal_type == 19 || nal_type == 20;
    if !is_idr { /* skip poc/ref-pic-set — not for IDR */ }
    let mut sao_luma = false;
    let mut sao_chroma = false;
    if sps.sao_enabled {
        sao_luma = r.read_flag().map_err(|_| e("sao_luma"))?;
        if !sps.chroma.is_monochrome() {
            sao_chroma = r.read_flag().map_err(|_| e("sao_chroma"))?;
        }
    }
    let slice_qp_delta = r.read_se().map_err(|_| e("qp_delta"))?;
    let slice_qp = pps.init_qp + slice_qp_delta;
    if pps.slice_chroma_qp_offsets_present {
        r.read_se().map_err(|_| e("cb_qp_off"))?;
        r.read_se().map_err(|_| e("cr_qp_off"))?;
    }
    let mut deblock_override = false;
    if pps.deblocking_filter_override_enabled {
        deblock_override = r.read_flag().map_err(|_| e("deblock_override"))?;
    }
    if deblock_override {
        let disabled = r.read_flag().map_err(|_| e("deblock_disabled"))?;
        if !disabled {
            r.read_se().map_err(|_| e("beta_off"))?;
            r.read_se().map_err(|_| e("tc_off"))?;
        }
    }
    if pps.loop_filter_across_slices && (sao_luma || sao_chroma || !pps.deblocking_filter_disabled)
    {
        r.read_flag()
            .map_err(|_| e("loop_filter_across_slices_flag"))?;
    }
    if pps.tiles_enabled || pps.entropy_coding_sync_enabled {
        let n = r.read_ue().map_err(|_| e("num_entry_points"))?;
        if n > 0 {
            let len = r.read_ue().map_err(|_| e("offset_len"))? + 1;
            for _ in 0..n {
                r.read_bits(len).map_err(|_| e("entry_point"))?;
            }
        }
    }
    if pps.slice_segment_header_extension_present {
        let l = r.read_ue().map_err(|_| e("ext_len"))?;
        for _ in 0..l {
            r.read_bits(8).map_err(|_| e("ext_byte"))?;
        }
    }
    r.read_bit().map_err(|_| e("alignment_bit"))?;
    while !r.bit_pos().is_multiple_of(8) {
        r.read_bit().map_err(|_| e("alignment_pad"))?;
    }
    Ok((slice_qp, sao_luma, sao_chroma, r.bit_pos() / 8))
}
