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
use crate::intra;
use crate::reconstruct;
use crate::transform;
use crate::yuv::YuvPlanes;

const MODE_PLANAR: u8 = 0;
const MODE_DC: u8 = 1;

/// Minimum total CTB count before the WPP wavefront is worth its fixed costs.
/// Below this the serial per-row decode is faster (spawn/coordination overhead
/// and the diagonal ramp dominate). ~64 CTBs ≈ a 512×512 picture at 64×64 CTBs.
const WAVEFRONT_MIN_CTBS: usize = 64;

#[derive(Clone, Default)]
struct SaoCtb {
    type_idx: [u8; 3], // 0=off,1=band,2=edge
    offsets: [[i32; 4]; 3],
    band_pos: [u8; 3],
    eo_class: [u8; 3],
}

pub(crate) struct FullDecoder {
    cab: CabacDecoder,
    ctx: ContextSet,
    ictx: IntraModeContexts,
    sps: Sps,
    pps: Pps,

    // Reconstruction planes (coded dimensions, CTB-aligned).
    y: crate::plane::Plane<u16>,
    cb: crate::plane::Plane<u16>,
    cr: crate::plane::Plane<u16>,
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
    mode_y: crate::plane::Plane<u8>,
    decoded: crate::plane::Plane<bool>, // per 4×4 luma block
    tqb: crate::plane::Plane<bool>,     // per 4×4 luma block: cu_transquant_bypass_flag (lossless)
    cu_tqb: bool,                       // current CU's cu_transquant_bypass_flag
    grid_w: usize,                      // ceil(w/4), one entry for every covered 4×4 luma grid cell
    #[allow(dead_code)]
    grid_h: usize, // ceil(h/4)
    ct_depth: crate::plane::Plane<u8>,  // per 4×4, coding-tree depth (for split_cu_flag ctx)

    // QP tracking
    slice_qp: i32,
    qp_y_prev: i32,
    qp_y_map: crate::plane::Plane<i16>, // per 4×4 luma (QpY ∈ −24..51, fits i16; halves a multi-MB buffer)
    cu_qp_delta_val: i32,
    is_cu_qp_delta_coded: bool,
    log2_qg: u32,
    cur_qp: i32,

    sao: crate::plane::Plane<SaoCtb>,
    ctb_cols: usize,
    ctb_rows: usize,
    sao_luma: bool,
    sao_chroma: bool,

    // Slice-level chroma QP offsets, added to the PPS offsets during chroma QP
    // derivation (§8.6.1).
    slice_cb_qp_offset: i32,
    slice_cr_qp_offset: i32,

    // Effective per-slice deblocking state (PPS values unless the slice header
    // overrode them).
    deblocking_disabled: bool,
    beta_offset_div2: i32,
    tc_offset_div2: i32,

    sign_hiding: bool,

    // WPP context snapshots
    wpp_ctx_snap: Vec<Option<ContextSet>>,
    wpp_ictx_snap: Vec<Option<IntraModeContexts>>,

    /// Pre-allocated scratch memory reused every TU to avoid per-block
    /// heap allocations on the hot path (~4–6 allocs per TU eliminated).
    scratch: intra::IntraScratch,
    /// Dequantised coefficient scratch (max 32×32 = 1024 values, clamped to ±32768 → i32)
    deq_scratch: Vec<i32>,
    /// Inverse-transform output scratch (max 32×32 = 1024 i32 values)
    res_scratch: Vec<i32>,
    /// i16 dequant/residual scratch, used on the 8-bit-depth path (half the width).
    deq_scratch16: Vec<i16>,
    res_scratch16: Vec<i16>,
    /// Parsed residual levels scratch (max 32×32), reused across TUs.
    coeff_scratch: Vec<i32>,
    /// Cached strong_intra_smoothing (avoids env-var lookup per TU)
    strong_smoothing: bool,
}

impl FullDecoder {
    /// Maximum allowed dimension per axis and pixel count.
    pub(crate) const MAX_DIM: usize = 16_384;
    pub(crate) const MAX_PIXELS: usize = 64 * 1024 * 1024; // 64 MP

    pub(crate) fn new(
        cabac: &[u8],
        sps: Sps,
        pps: Pps,
        hdr: &SliceHeader,
    ) -> Result<Self, DecodeError> {
        let slice_qp = hdr.slice_qp;
        let sao_luma = hdr.sao_luma;
        let sao_chroma = hdr.sao_chroma;
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
        // The pixel/output paths are implemented for 8/10/12-bit streams only.
        // Reject malformed SPS values before they reach shifts like `1 << (bd - 1)`.
        sps.bit_depth()?;
        match sps.bit_depth_chroma {
            8 | 10 | 12 => {}
            n => return Err(DecodeError::UnsupportedBitDepth(n)),
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
            w.div_ceil(sub_w)
        };
        let ch = if sps.chroma.is_monochrome() {
            0
        } else {
            h.div_ceil(sub_h)
        };
        let grid_w = w.div_ceil(4);
        let grid_h = h.div_ceil(4);
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
            y: crate::plane::Plane::owned(vec![0; w * h]),
            cb: crate::plane::Plane::owned(vec![0; cw * ch]),
            cr: crate::plane::Plane::owned(vec![0; cw * ch]),
            w,
            h,
            cw,
            ch,
            sub_w,
            sub_h,
            mode_y: crate::plane::Plane::owned(vec![MODE_DC; grid_w * grid_h]),
            decoded: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            tqb: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            cu_tqb: false,
            ct_depth: crate::plane::Plane::owned(vec![0; grid_w * grid_h]),
            grid_w,
            grid_h,
            slice_qp,
            qp_y_prev: slice_qp,
            qp_y_map: crate::plane::Plane::owned(vec![slice_qp as i16; grid_w * grid_h]),
            cu_qp_delta_val: 0,
            is_cu_qp_delta_coded: false,
            log2_qg,
            cur_qp: slice_qp,
            sao: crate::plane::Plane::owned(vec![SaoCtb::default(); ctb_cols * ctb_rows]),
            ctb_cols,
            ctb_rows,
            sao_luma,
            sao_chroma,
            slice_cb_qp_offset: hdr.cb_qp_offset,
            slice_cr_qp_offset: hdr.cr_qp_offset,
            deblocking_disabled: hdr.deblocking_disabled,
            beta_offset_div2: hdr.beta_offset_div2,
            tc_offset_div2: hdr.tc_offset_div2,
            sign_hiding: pps.sign_data_hiding_enabled,
            wpp_ctx_snap: vec![None; ctb_rows],
            wpp_ictx_snap: vec![None; ctb_rows],
            scratch: intra::IntraScratch::new(),
            deq_scratch: vec![0i32; 1024],
            res_scratch: vec![0i32; 1024],
            deq_scratch16: vec![0i16; 1024],
            res_scratch16: vec![0i16; 1024],
            coeff_scratch: vec![0i32; 1024],
            strong_smoothing: true,
            sps,
            pps,
        })
    }

    pub(crate) fn decode_segment(
        &mut self,
        cabac: &[u8],
        hdr: &SliceHeader,
    ) -> Result<(), DecodeError> {
        self.cab.reset_with(cabac)?;
        if !hdr.dependent_slice_segment {
            // Independent segment: reset entropy contexts and slice-level state.
            self.slice_qp = hdr.slice_qp;
            self.qp_y_prev = hdr.slice_qp;
            self.cur_qp = hdr.slice_qp;
            self.cu_qp_delta_val = 0;
            self.is_cu_qp_delta_coded = false;
            self.sao_luma = hdr.sao_luma;
            self.sao_chroma = hdr.sao_chroma;
            self.slice_cb_qp_offset = hdr.cb_qp_offset;
            self.slice_cr_qp_offset = hdr.cr_qp_offset;
            self.deblocking_disabled = hdr.deblocking_disabled;
            self.beta_offset_div2 = hdr.beta_offset_div2;
            self.tc_offset_div2 = hdr.tc_offset_div2;
            let qp = hdr.slice_qp.clamp(0, 51) as u8;
            self.ctx = ContextSet::init_islice(qp);
            self.ictx = IntraModeContexts::init_islice(qp);
        } else {
            // Dependent segment inherits contexts/state; only the QP predictor
            // resets at the segment's first quantization group (handled by the
            // QG logic), so continue with the retained context.
        }
        self.decode_slice(hdr.slice_segment_address)
    }

    /// Decode one CTB at grid position `(rx, ry)`: parse SAO params, run the
    /// coding quadtree (parse + reconstruct), take the WPP context snapshot after
    /// CTB column 1, and read `end_of_slice_segment_flag`. Returns `true` if the
    /// slice segment terminated at this CTB. Shared by the serial `decode_slice`
    /// and the parallel wavefront so both parse every CTB identically.
    #[inline]
    fn decode_one_ctb(&mut self, rx: usize, ry: usize, wpp: bool) -> bool {
        self.decode_one_ctb_inner(rx, ry, wpp, true)
    }

    /// As [`decode_one_ctb`], but `store_snap` controls whether the post-CTB-1
    /// context snapshot is written into `wpp_ctx_snap`/`wpp_ictx_snap`. The
    /// serial path needs it (rows share one decoder); the wavefront path sets
    /// `false` and captures the snapshot directly from `self.ctx`/`self.ictx`,
    /// avoiding the per-row `Vec<Option<..>>` allocations entirely.
    #[inline]
    fn decode_one_ctb_inner(&mut self, rx: usize, ry: usize, wpp: bool, store_snap: bool) -> bool {
        let ctb = 1usize << self.log2_ctb;
        if self.sps.sao_enabled {
            self.parse_sao(rx, ry);
        }
        // New CTB → QG reset handled inside coding_unit via QG tracking.
        self.coding_quadtree(rx * ctb, ry * ctb, self.log2_ctb, 0);

        // WPP: save context snapshot after the 2nd CTB of each row.
        if store_snap && wpp && rx == 1 {
            self.wpp_ctx_snap[ry] = Some(self.ctx.clone());
            self.wpp_ictx_snap[ry] = Some(self.ictx);
        }

        self.cab.decode_terminate() != 0
    }

    /// Attempt a parallel WPP-wavefront decode of a single independent slice
    /// segment covering the whole picture. Returns `Ok(true)` if the wavefront
    /// ran (and the picture is fully reconstructed up to the loop filters);
    /// `Ok(false)` if the stream is ineligible and the caller should fall back to
    /// the serial [`decode_slice`]. `Err` only on a genuine bitstream error.
    ///
    /// Eligibility: WPP enabled, the segment starts at CTB 0, entry points are
    /// present and number exactly `ctb_rows - 1`, more than one CTB row, and a
    /// multi-threaded pool. Anything else → `Ok(false)`.
    pub(crate) fn try_decode_wavefront(
        &mut self,
        rbsp: &[u8],
        nal_bytes: &[u8],
        hdr: &SliceHeader,
        pool: &crate::threadpool::ThreadPool,
    ) -> Result<bool, DecodeError> {
        if !self.pps.entropy_coding_sync_enabled
            || hdr.slice_segment_address != 0
            || self.ctb_rows <= 1
            || pool.threads() <= 1
            || hdr.entry_points.is_empty()
        {
            return Ok(false);
        }
        // Below a minimum amount of work the wavefront's fixed costs (runner
        // spawn, per-row decoder construction, atomic coordination) outweigh the
        // serial decode, and the diagonal ramp-up leaves most cores idle anyway.
        // Require enough CTB rows to fill the pool plus a couple of diagonals,
        // and a non-trivial total CTB count. Tuned conservatively; small stills
        // fall through to the serial path.
        let total_ctbs = self.ctb_cols * self.ctb_rows;
        let enough_rows = self.ctb_rows >= pool.threads().max(2) + 2;
        if total_ctbs < WAVEFRONT_MIN_CTBS || !enough_rows {
            return Ok(false);
        }
        // Only now (wavefront is actually going to run) build the NAL→RBSP
        // offset map — it costs ~8 bytes per RBSP byte, so we avoid it on the
        // common serial / non-WPP path entirely.
        let src_of = crate::bitreader::rbsp_src_map(nal_bytes);
        let rows = match crate::wpp::row_substreams(
            &src_of,
            hdr.cabac_offset,
            &hdr.entry_points,
            rbsp.len(),
            self.ctb_rows,
        ) {
            Some(r) => r,
            None => return Ok(false),
        };
        crate::wpp::run_wavefront(self, rbsp, &rows, pool)?;
        Ok(true)
    }

    /// Number of CTB rows (public accessor for the wavefront driver).
    pub(crate) fn ctb_rows_pub(&self) -> usize {
        self.ctb_rows
    }

    /// I-slice initial entropy contexts for seeding wavefront row 0.
    pub(crate) fn init_contexts_pub(&self) -> (ContextSet, IntraModeContexts) {
        let qp = self.slice_qp.clamp(0, 51) as u8;
        (
            ContextSet::init_islice(qp),
            IntraModeContexts::init_islice(qp),
        )
    }

    /// Build a [`RowFactory`] capturing raw aliasing pointers to this decoder's
    /// shared picture buffers plus all immutable config. The factory is `Send +
    /// Sync` and each wavefront worker uses it to construct its per-row decoder.
    ///
    /// SAFETY: the returned factory holds `*mut` into `self`'s buffers. The
    /// caller (the wavefront driver) must keep `self` alive for the whole scope
    /// and must not access `self`'s planes through `self` while row views built
    /// from the factory are alive. The 2-CTB lag keeps concurrent row writes
    /// disjoint.
    pub(crate) fn row_factory(&mut self) -> RowFactory {
        RowFactory {
            y: (self.y.as_mut_ptr(), self.y.len()),
            cb: (self.cb.as_mut_ptr(), self.cb.len()),
            cr: (self.cr.as_mut_ptr(), self.cr.len()),
            mode_y: (self.mode_y.as_mut_ptr(), self.mode_y.len()),
            decoded: (self.decoded.as_mut_ptr(), self.decoded.len()),
            tqb: (self.tqb.as_mut_ptr(), self.tqb.len()),
            ct_depth: (self.ct_depth.as_mut_ptr(), self.ct_depth.len()),
            qp_y_map: (self.qp_y_map.as_mut_ptr(), self.qp_y_map.len()),
            sao: (self.sao.as_mut_ptr(), self.sao.len()),
            sps: self.sps.clone(),
            pps: self.pps.clone(),
            w: self.w,
            h: self.h,
            cw: self.cw,
            ch: self.ch,
            sub_w: self.sub_w,
            sub_h: self.sub_h,
            bd: self.bd,
            bd_c: self.bd_c,
            log2_ctb: self.log2_ctb,
            log2_min_cb: self.log2_min_cb,
            log2_min_tb: self.log2_min_tb,
            log2_max_tb: self.log2_max_tb,
            max_trafo_depth_intra: self.max_trafo_depth_intra,
            grid_w: self.grid_w,
            grid_h: self.grid_h,
            slice_qp: self.slice_qp,
            log2_qg: self.log2_qg,
            ctb_cols: self.ctb_cols,
            ctb_rows: self.ctb_rows,
            sao_luma: self.sao_luma,
            sao_chroma: self.sao_chroma,
            slice_cb_qp_offset: self.slice_cb_qp_offset,
            slice_cr_qp_offset: self.slice_cr_qp_offset,
            deblocking_disabled: self.deblocking_disabled,
            beta_offset_div2: self.beta_offset_div2,
            tc_offset_div2: self.tc_offset_div2,
            sign_hiding: self.sign_hiding,
            strong_smoothing: self.strong_smoothing,
        }
    }

    /// Decode a single CTB row `ry` (all columns) for the wavefront.
    ///
    /// Gating: column `c` is processed only once the row above has completed
    /// column `c + 2` (the 2-CTB lag), enforced via `above_progress`. After
    /// finishing CTB column 1 this row publishes its CABAC context snapshot into
    /// `snapshot_out` so the row below can seed its engine. `progress` publishes
    /// this row's completed-column count. Stops at the row's terminate bin.
    ///
    /// The engine and contexts (`self.cab`, `self.ctx`, `self.ictx`) must already
    /// be seeded by the caller: row 0 from I-slice init, row `r>0` from row
    /// `r-1`'s published snapshot.
    pub(crate) fn decode_wavefront_row(
        &mut self,
        ry: usize,
        progress: &crate::threadpool::ProgressGate,
        above_progress: Option<&crate::threadpool::ProgressGate>,
        snapshot_out: &std::sync::OnceLock<(ContextSet, IntraModeContexts)>,
    ) -> Result<(), DecodeError> {
        let cols = self.ctb_cols;
        for rx in 0..cols {
            // Wavefront gate: wait until the row above is ≥ 2 CTBs ahead.
            if let Some(above) = above_progress {
                above.wait_at_least(rx + 2);
            }
            let terminated = self.decode_one_ctb_inner(rx, ry, true, false);

            // Publish the post-CTB-1 snapshot for the row below, straight from
            // the live contexts (no per-row snapshot array needed).
            if rx == 1 {
                let _ = snapshot_out.set((self.ctx.clone(), self.ictx));
            }

            // Publish progress *after* the CTB is fully reconstructed so a waiter
            // observing `rx+1` can safely read our columns ≤ rx.
            progress.publish(rx + 1);
            if terminated {
                break;
            }
        }
        // If the row terminated before CTB 1 (a 1-CTB-wide picture is excluded by
        // eligibility, but a mid-row end_of_slice could still occur on malformed
        // streams), make sure the row below gets *some* snapshot to avoid a hang.
        if snapshot_out.get().is_none() {
            let _ = snapshot_out.set((self.ctx.clone(), self.ictx));
        }
        // Ensure the row below can always advance past our last column.
        progress.publish(cols + 2);
        Ok(())
    }

    /// Reconstruct CTBs starting at raster address `start_ctb`, stopping at the
    /// slice's `end_of_slice_segment_flag`. Does not run loop filters — call
    /// [`finish`] once, after all segments of the picture are decoded.
    pub(crate) fn decode_slice(&mut self, start_ctb: usize) -> Result<(), DecodeError> {
        let _ctb = 1usize << self.log2_ctb;
        let wpp = self.pps.entropy_coding_sync_enabled;
        let total = self.ctb_cols * self.ctb_rows;
        let start_ctb = start_ctb.min(total);

        let start_ry = start_ctb.checked_div(self.ctb_cols).unwrap_or(0);
        let start_rx0 = if self.ctb_cols == 0 {
            0
        } else {
            start_ctb % self.ctb_cols
        };

        for ry in start_ry..self.ctb_rows {
            // WPP: at start of every non-first row, restore saved contexts and
            // reinitialize the CABAC engine from the current stream position
            // (which the previous row's sub-stream end already byte-aligned to).
            if wpp && ry > start_ry {
                if let (Some(ctx), Some(ictx)) = (
                    self.wpp_ctx_snap[ry - 1].take(),
                    self.wpp_ictx_snap[ry - 1].take(),
                ) {
                    self.ctx = ctx;
                    self.ictx = ictx;
                }
                self.cab.reinit_engine();
            }

            // Only the first row of this segment starts at its column offset;
            // subsequent rows start at column 0.
            let rx_start = if ry == start_ry { start_rx0 } else { 0 };

            let mut terminated = false;
            for rx in rx_start..self.ctb_cols {
                let end = self.decode_one_ctb(rx, ry, wpp);
                if end {
                    // end_of_slice_segment_flag: this slice segment is complete.
                    terminated = true;
                    break;
                }
            }

            if terminated {
                break;
            }

            // WPP: after the last CTB of each non-final row, the stream contains
            // an end_of_sub_stream_one_bit (= 1), then byte-alignment padding,
            // then the next row's sub-stream starts.
            if wpp && ry < self.ctb_rows - 1 {
                let eoss = self.cab.decode_terminate();
                if eoss != 1 {
                    return Err(DecodeError::Bitstream(
                        "WPP end_of_sub_stream_one_bit must be 1".into(),
                    ));
                }
                self.cab.byte_align();
                // Engine reinit happens at the top of the next loop iteration.
            }
        }
        Ok(())
    }

    /// Apply in-loop filters (deblocking then SAO) over the fully-reconstructed
    /// picture and return the planes. Call once, after all slice segments.
    pub(crate) fn finish(&mut self, pool: Option<&crate::threadpool::ThreadPool>) -> YuvPlanes {
        // In-loop filters run in HEVC order: deblocking first, then SAO. They
        // are independently gated: deblocking runs unless it is disabled (PPS or
        // a slice-level override), while SAO runs only when the SPS enables it.
        if !self.deblocking_disabled {
            match pool {
                Some(p) if p.threads() > 1 && self.ctb_rows > 1 => {
                    self.apply_deblocking_parallel(p)
                }
                _ => self.apply_deblocking(),
            }
        }
        if self.sps.sao_enabled {
            // Parallel SAO only when a pool is supplied (single-item path where
            // SAO is the only parallelism available). In the grid path each tile
            // already runs on a pool worker, so `None` keeps SAO serial there to
            // avoid oversubscription.
            match pool {
                Some(p) if p.threads() > 1 && self.ctb_rows > 1 => self.apply_sao_parallel(p),
                _ => self.apply_sao(),
            }
        }
        YuvPlanes {
            y: self.y.take_vec(),
            cb: self.cb.take_vec(),
            cr: self.cr.take_vec(),
            width: self.w,
            height: self.h,
            chroma: self.sps.chroma,
            bit_depth: self.sps.bit_depth().unwrap_or(BitDepth::Eight),
        }
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

    /// Parallel deblocking: dispatch CTB-aligned row bands across the pool.
    /// Bit-identical to [`Self::apply_deblocking`]; see
    /// [`crate::deblock::apply_deblocking_parallel`].
    fn apply_deblocking_parallel(&mut self, pool: &crate::threadpool::ThreadPool) {
        // Move planes out first so the immutable borrows in `ctx` (qp_y_map,
        // tqb) don't conflict with taking `&mut self.y/cb/cr`.
        let y = self.y.take_vec();
        let cb = self.cb.take_vec();
        let cr = self.cr.take_vec();
        let ctx = crate::deblock::DeblockCtx {
            w: self.w,
            h: self.h,
            cw: self.cw,
            ch: self.ch,
            gw: self.grid_w,
            gh: self.grid_h,
            sub_w: self.sub_w,
            sub_h: self.sub_h,
            bd: self.bd,
            bd_c: self.bd_c,
            beta_offset: self.beta_offset_div2 * 2,
            tc_offset: self.tc_offset_div2 * 2,
            qp_bd_offset_y: 6 * (self.bd as i32 - 8),
            qp_bd_offset_c: 6 * (self.bd_c as i32 - 8),
            default_qp: self.slice_qp as i16,
            qp_y_map: &self.qp_y_map[..],
            tqb: &self.tqb[..],
        };
        let out = crate::deblock::apply_deblocking_parallel(pool, &ctx, self.log2_ctb, y, cb, cr);
        self.y = crate::plane::Plane::owned(out.y);
        self.cb = crate::plane::Plane::owned(out.cb);
        self.cr = crate::plane::Plane::owned(out.cr);
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

        // Slice-effective deblocking offsets (PPS values unless the slice
        // header overrode them).
        let beta_offset = self.beta_offset_div2 * 2;
        let tc_offset = self.tc_offset_div2 * 2;
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
        let gh = self.grid_h;
        let default_qp = self.slice_qp as i16;

        // Helper: look up qp_y_map for a luma pixel (rounded to 4×4 grid).
        // Fuzzed / malformed pictures can be smaller than 4 pixels in one
        // dimension; avoid the old `w / 4 - 1` / `h / 4 - 1` underflow and
        // treat missing grid entries as the slice QP.
        let qp_at = |qp_map: &[i16], px: usize, py: usize| -> i32 {
            if qp_map.is_empty() || gw == 0 || gh == 0 {
                return default_qp as i32;
            }
            let gx = (px / 4).min(gw.saturating_sub(1));
            let gy = (py / 4).min(gh.saturating_sub(1));
            gy.checked_mul(gw)
                .and_then(|base| base.checked_add(gx))
                .and_then(|idx| qp_map.get(idx))
                .copied()
                .unwrap_or(default_qp) as i32
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
            // The luma filter reads/writes up to p3 and q3. On fuzzed edge
            // geometry the final nominal 8-pixel edge can be too close to the
            // picture boundary, so only run full filtering when both sides have
            // all required samples.
            let last_full_edge = edge_max.saturating_sub(4);
            while edge <= last_full_edge {
                // For each 4-pixel segment along the edge
                let mut scan = 0;
                while scan + 4 <= scan_max {
                    // p-side: pixels going into the block (edge-1, edge-2, edge-3, edge-4)
                    // q-side: pixels going out (edge, edge+1, edge+2, edge+3)
                    if pass == 0 {
                        // vertical edge at x=edge, rows scan..scan+3
                        let mid = scan + 1; // representative row
                        let qp_p = qp_at(&self.qp_y_map[..], edge - 1, mid);
                        let qp_q = qp_at(&self.qp_y_map[..], edge, mid);
                        // Lossless (transquant-bypass) CUs are exempt from deblocking
                        // (HEVC §8.7.2): if either side is bypass, skip this segment.
                        if self.tqb_at(edge - 1, mid) || self.tqb_at(edge, mid) {
                            scan += 4;
                            continue;
                        }
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
                        let qp_p = qp_at(&self.qp_y_map[..], mid, edge - 1);
                        let qp_q = qp_at(&self.qp_y_map[..], mid, edge);
                        if self.tqb_at(mid, edge - 1) || self.tqb_at(mid, edge) {
                            scan += 4;
                            continue;
                        }
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
            let chroma_edge_max = if pass == 0 { cw } else { ch };
            // Chroma filter reads p1 and q1 around the edge, so require at
            // least one q-side sample beyond the boundary.
            let last_full_chroma_edge = chroma_edge_max.saturating_sub(2);
            while edge <= last_full_chroma_edge {
                let mut scan = 0;
                while scan + 4 <= scan_max {
                    let mid = scan + 1;
                    // QP for chroma — use luma QP at corresponding position
                    let (qlx, qly) = if pass == 0 {
                        (edge * self.sub_w, mid * self.sub_h)
                    } else {
                        (mid * self.sub_w, edge * self.sub_h)
                    };
                    let avg_qp_l = qp_at(&self.qp_y_map[..], qlx.min(w - 1), qly.min(h - 1));
                    let tc_prime_c = (avg_qp_l + qp_bd_offset_c + 2 + tc_offset).clamp(0, 53);
                    let tc_c = TC[tc_prime_c as usize];
                    if tc_c == 0 {
                        scan += 4;
                        continue;
                    }
                    // Lossless CUs are exempt from chroma deblocking too.
                    let (px_p, py_p, px_q, py_q) = if pass == 0 {
                        (
                            (edge - 1) * self.sub_w,
                            mid * self.sub_h,
                            edge * self.sub_w,
                            mid * self.sub_h,
                        )
                    } else {
                        (
                            mid * self.sub_w,
                            (edge - 1) * self.sub_h,
                            mid * self.sub_w,
                            edge * self.sub_h,
                        )
                    };
                    if self.tqb_at(px_p, py_p) || self.tqb_at(px_q, py_q) {
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

    /// Parallel SAO: flatten per-CTB params and dispatch CTB-row bands across
    /// the pool. Bit-identical to [`Self::apply_sao`]; see
    /// [`crate::sao::apply_sao_parallel`].
    fn apply_sao_parallel(&mut self, pool: &crate::threadpool::ThreadPool) {
        let params: Vec<crate::sao::SaoCtbParams> = self
            .sao
            .iter()
            .map(|s| crate::sao::SaoCtbParams {
                type_idx: s.type_idx,
                offsets: s.offsets,
                band_pos: s.band_pos,
                eo_class: s.eo_class,
            })
            .collect();
        let ctx = crate::sao::SaoPlanesCtx {
            params: &params,
            ctb_cols: self.ctb_cols,
            ctb_rows: self.ctb_rows,
            log2_ctb: self.log2_ctb,
            w: self.w,
            h: self.h,
            cw: self.cw,
            ch: self.ch,
            sub_w: self.sub_w,
            sub_h: self.sub_h,
            bd: self.bd,
            bd_c: self.bd_c,
            sao_luma: self.sao_luma,
            sao_chroma: self.sao_chroma,
        };
        let y = self.y.take_vec();
        let cb = self.cb.take_vec();
        let cr = self.cr.take_vec();
        let (y, cb, cr) = crate::sao::apply_sao_parallel(pool, &ctx, y, cb, cr);
        self.y = crate::plane::Plane::owned(y);
        self.cb = crate::plane::Plane::owned(cb);
        self.cr = crate::plane::Plane::owned(cr);
    }

    fn apply_sao(&mut self) {
        let ctb = 1usize << self.log2_ctb;
        // Work on clones so EO neighbor lookups always use original values.
        let orig_y = self.y.to_vec_clone();
        let orig_cb = self.cb.to_vec_clone();
        let orig_cr = self.cr.to_vec_clone();

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
                        &mut self.y[..],
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
                            &mut self.cb[..],
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
                            &mut self.cr[..],
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
        crate::sao::apply_sao_plane(
            dst, src, w, h, x0, y0, x_end, y_end, type_idx, offsets, band_pos, eo_class, bd,
        );
    }

    fn coding_quadtree(&mut self, x0: usize, y0: usize, log2_cb: u32, depth: u8) {
        let cb_size = 1usize << log2_cb;
        let in_pic = x0 + cb_size <= self.w && y0 + cb_size <= self.h;
        let can_split = log2_cb > self.log2_min_cb;
        let split = if x0 + cb_size <= self.w && y0 + cb_size <= self.h && can_split {
            // read split_cu_flag with neighbor-depth context
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
        if x0 >= 4
            && let Some(g) = self.grid_idx(x0 - 1, y0)
            && self.decoded[g]
            && self.ct_depth[g] as usize > depth as usize
        {
            inc += 1;
        }
        if y0 >= 4
            && let Some(g) = self.grid_idx(x0, y0 - 1)
            && self.decoded[g]
            && self.ct_depth[g] as usize > depth as usize
        {
            inc += 1;
        }
        inc
    }

    fn set_ct_depth(&mut self, x0: usize, y0: usize, size: usize, depth: u8) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w
                    && yy < self.h
                    && let Some(g) = self.grid_idx(xx, yy)
                {
                    self.ct_depth[g] = depth;
                }
            }
        }
    }

    fn coding_unit(&mut self, x0: usize, y0: usize, log2_cb: u32) {
        // QG handling
        let qg_mask = !((1usize << self.log2_qg) - 1);
        let xqg = x0 & qg_mask;
        let yqg = y0 & qg_mask;
        if x0 == xqg && y0 == yqg {
            self.is_cu_qp_delta_coded = false;
            self.cu_qp_delta_val = 0;
            self.cur_qp = self.predict_qp(xqg, yqg);
        }

        // cu_transquant_bypass_flag (HEVC §7.3.8.5): first CU element, present only
        // when the PPS enables transquant bypass. When set, transform + quantization
        // are skipped and the parsed residual is used verbatim (lossless coding).
        self.cu_tqb = if self.pps.transquant_bypass_enabled {
            self.cab.decode_bin(&mut self.ctx.cu_transquant_bypass_flag) != 0
        } else {
            false
        };
        if self.cu_tqb {
            let cb = 1usize << log2_cb;
            self.set_tqb(x0, y0, cb);
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

        let npu_sqr = npu * npu;
        for (i, ((luma_mode, &prev_flag), &mpm_or_rem)) in luma_modes[..npu_sqr]
            .iter_mut()
            .zip(prev_flags[..npu_sqr].iter())
            .zip(mpm_or_rem[..npu_sqr].iter())
            .enumerate()
        {
            let pux = x0 + (i % npu) * pu_size;
            let puy = y0 + (i / npu) * pu_size;
            let mode = self.derive_luma_mode(pux, puy, prev_flag, mpm_or_rem);
            *luma_mode = mode;
            self.set_mode(pux, puy, pu_size, mode);
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
            [false; 2],
            [false; 2],
        );

        // mark decoded
        self.mark_decoded(x0, y0, cb_size);
        self.qp_y_prev = self.cur_qp;
        self.set_qp(x0, y0, cb_size, self.cur_qp);
    }

    #[inline]
    fn grid_idx(&self, x: usize, y: usize) -> Option<usize> {
        if x >= self.w || y >= self.h {
            return None;
        }
        let idx = (y / 4).checked_mul(self.grid_w)?.checked_add(x / 4)?;
        (idx < self.decoded.len()).then_some(idx)
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

        // qPY_A: left neighbor, must be in same CTB
        let qp_a = if xqg >= 1 && (xqg - 1) >= ctb_x {
            self.grid_idx(xqg - 1, yqg)
                .and_then(|g| self.qp_y_map.get(g))
                .copied()
                .map(i32::from)
                .unwrap_or(self.qp_y_prev)
        } else {
            self.qp_y_prev
        };
        // qPY_B: above neighbor, must be in same CTB
        let qp_b = if yqg >= 1 && (yqg - 1) >= ctb_y {
            self.grid_idx(xqg, yqg - 1)
                .and_then(|g| self.qp_y_map.get(g))
                .copied()
                .map(i32::from)
                .unwrap_or(self.qp_y_prev)
        } else {
            self.qp_y_prev
        };
        (qp_a + qp_b + 1) >> 1
    }

    fn set_qp(&mut self, x0: usize, y0: usize, size: usize, qp: i32) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w
                    && yy < self.h
                    && let Some(g) = self.grid_idx(xx, yy)
                {
                    self.qp_y_map[g] = qp as i16;
                }
            }
        }
    }

    fn set_mode(&mut self, x0: usize, y0: usize, size: usize, mode: u8) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w
                    && yy < self.h
                    && let Some(g) = self.grid_idx(xx, yy)
                {
                    self.mode_y[g] = mode;
                }
            }
        }
    }

    fn mark_decoded(&mut self, x0: usize, y0: usize, size: usize) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w
                    && yy < self.h
                    && let Some(g) = self.grid_idx(xx, yy)
                {
                    self.decoded[g] = true;
                }
            }
        }
    }

    fn set_tqb(&mut self, x0: usize, y0: usize, size: usize) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w
                    && yy < self.h
                    && let Some(g) = self.grid_idx(xx, yy)
                {
                    self.tqb[g] = true;
                }
            }
        }
    }

    /// cu_transquant_bypass_flag at a luma pixel (4×4 grid). Out-of-range → false.
    fn tqb_at(&self, px: usize, py: usize) -> bool {
        if px >= self.w || py >= self.h {
            return false;
        }
        self.grid_idx(px, py)
            .and_then(|g| self.tqb.get(g))
            .copied()
            .unwrap_or(false)
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
        self.grid_idx(x as usize, y as usize)
            .and_then(|g| self.mode_y.get(g))
            .copied()
            .unwrap_or(MODE_DC)
    }

    fn decode_chroma_mode(&mut self, luma_mode: u8) -> u8 {
        let bin0 = self.cab.decode_bin(&mut self.ictx.intra_chroma_pred_mode);
        let derived = if bin0 == 0 {
            luma_mode // DM
        } else {
            let mut idx = 0u8;
            for _ in 0..2 {
                idx = (idx << 1) | self.cab.decode_bypass();
            }
            let cand = [0u8, 26, 10, 1][idx as usize];
            if cand == luma_mode { 34 } else { cand }
        };
        // HEVC §8.4.3 / Table 8-3: for ChromaArrayType==2 (4:2:2) the derived chroma
        // intra mode is remapped (the asymmetric sampling rotates the angle). This
        // mode drives both the angular prediction and the mode-dependent coefficient
        // scan, so it must match the encoder exactly.
        if self.sps.chroma_idc == 2 {
            MODE_422_MAP[derived as usize]
        } else {
            derived
        }
    }
}

/// HEVC Table 8-3: derived-chroma-mode remap for 4:2:2 (ChromaArrayType==2).
static MODE_422_MAP: [u8; 35] = [
    0, 1, 2, 2, 2, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 23, 24, 24, 25, 25,
    26, 27, 27, 28, 28, 29, 29, 30, 31,
];

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

impl FullDecoder {
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
        parent_cbf_cb: [bool; 2],
        parent_cbf_cr: [bool; 2],
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

        // chroma cbf. For ChromaArrayType==2 (4:2:2) there are two stacked chroma
        // TBs, each with its own cbf_cb / cbf_cr, signaled cb[0],cb[1],cr[0],cr[1]
        // (HEVC §7.3.8.8). 4:2:0 / 4:4:4 have one of each.
        let chroma_present = !self.sps.chroma.is_monochrome();
        let _ = depth;
        let n_tb = if self.sps.chroma_idc == 2 { 2 } else { 1 };
        let mut cbf_cb = parent_cbf_cb;
        let mut cbf_cr = parent_cbf_cr;
        if chroma_present && (log2_ts > 2 || self.sps.chroma_idc == 3) {
            for t in 0..n_tb {
                if depth == 0 || parent_cbf_cb[t] {
                    cbf_cb[t] = self
                        .cab
                        .decode_bin(&mut self.ctx.cbf_chroma[depth.min(4) as usize])
                        != 0;
                }
            }
            for t in 0..n_tb {
                if depth == 0 || parent_cbf_cr[t] {
                    cbf_cr[t] = self
                        .cab
                        .decode_bin(&mut self.ctx.cbf_chroma[depth.min(4) as usize])
                        != 0;
                }
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
        cbf_cb: [bool; 2],
        cbf_cr: [bool; 2],
    ) {
        let chroma_present = !self.sps.chroma.is_monochrome();
        let _ = depth;
        let any_chroma = cbf_cb.iter().any(|&b| b) || cbf_cr.iter().any(|&b| b);
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
            let mut coeffs = std::mem::take(&mut self.coeff_scratch);
            let (_tskip, max_x, _last_y) = residual_coding(
                &mut self.cab,
                &mut self.ctx,
                log2_ts,
                true,
                scan,
                self.sign_hiding,
                ts_ctx,
                self.cu_tqb,
                &mut coeffs,
            );
            self.reconstruct_luma(x0, y0, log2_ts, luma_mode, &coeffs, max_x + 1);
            self.coeff_scratch = coeffs;
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
        self.grid_idx(x0, y0)
            .and_then(|g| self.mode_y.get(g))
            .copied()
            .unwrap_or(MODE_DC)
    }
}

#[inline]
fn plane_tail_mut(plane: &mut [u16], stride: usize, x0: usize, y0: usize) -> Option<&mut [u16]> {
    if stride == 0 {
        return None;
    }
    let off = y0.checked_mul(stride)?.checked_add(x0)?;
    plane.get_mut(off..)
}

fn copy_pred_block_clipped(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    n: usize,
    valid_w: usize,
    valid_h: usize,
) {
    if n == 0 || stride == 0 {
        return;
    }
    let Some(n2) = n.checked_mul(n) else {
        return;
    };
    if pred.len() < n2 {
        return;
    }
    let valid_w = valid_w.min(n).min(stride);
    let valid_h = valid_h.min(n);
    for y in 0..valid_h {
        let dst_off = y.saturating_mul(stride);
        if dst_off >= dst.len() {
            break;
        }
        let cols = valid_w.min(dst.len() - dst_off);
        if cols == 0 {
            break;
        }
        let row_off = y * n;
        dst[dst_off..dst_off + cols].copy_from_slice(&pred[row_off..row_off + cols]);
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

fn chroma_scan(mode: u8, log2_ts: u32, is_444: bool) -> u8 {
    // HEVC §6.5.3: scan is mode-dependent for 4×4, and for 8×8 when it's luma
    // (handled by luma_scan) or ChromaArrayType==3 (4:4:4). 4:2:0/4:2:2 chroma at
    // 8×8 stays diagonal. Mirrors the encoder's dct::scan_idx_for.
    let mode_dependent = log2_ts == 2 || (log2_ts == 3 && is_444);
    if mode_dependent {
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

impl FullDecoder {
    fn luma_avail(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return false;
        }
        self.grid_idx(x as usize, y as usize)
            .and_then(|g| self.decoded.get(g))
            .copied()
            .unwrap_or(false)
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
        self.grid_idx(lx, ly)
            .and_then(|g| self.decoded.get(g))
            .copied()
            .unwrap_or(false)
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
        for (i, (above, left)) in above[..2 * n].iter_mut().zip(left.iter_mut()).enumerate() {
            let ax = x0 as i32 + i as i32;
            *above = if self.luma_avail(ax, y0 as i32 - 1) {
                Some(self.y[(y0 - 1) * self.w + ax as usize])
            } else {
                None
            };
            let ly = y0 as i32 + i as i32;
            *left = if self.luma_avail(x0 as i32 - 1, ly) {
                Some(self.y[ly as usize * self.w + (x0 - 1)])
            } else {
                None
            };
        }
        corner
    }

    fn reconstruct_luma(
        &mut self,
        x0: usize,
        y0: usize,
        log2_ts: u32,
        mode: u8,
        levels: &[i32],
        nx: usize,
    ) {
        let n = 1usize << log2_ts;
        self.predict_luma_block_into(x0, y0, n, mode);
        let stride = self.w;
        let valid_w = self.w.saturating_sub(x0).min(n);
        let valid_h = self.h.saturating_sub(y0).min(n);
        // 8-bit depth: residuals fit i16, halving memory traffic and widening SIMD.
        if self.bd <= 8 {
            if self.cu_tqb {
                for (o, &l) in self.res_scratch16[..n * n].iter_mut().zip(levels.iter()) {
                    *o = l.clamp(-32768, 32767) as i16;
                }
            } else {
                let qp = self.cur_qp.clamp(0, 51) as u8;
                transform::dequantize_into(
                    levels,
                    n,
                    qp,
                    self.bd,
                    &mut self.deq_scratch16[..n * n],
                );
                if n == 4 {
                    transform::inv_transform_dst_into16(
                        &self.deq_scratch16[..n * n],
                        self.bd,
                        &mut self.res_scratch16[..n * n],
                    );
                } else {
                    transform::inv_transform_into16(
                        &self.deq_scratch16[..n * n],
                        n,
                        self.bd,
                        nx,
                        &mut self.res_scratch16[..n * n],
                    );
                }
            }
            let pred = &self.scratch.pred[..n * n];
            let res = &self.res_scratch16[..n * n];
            if valid_w != 0
                && valid_h != 0
                && let Some(dst) = plane_tail_mut(&mut self.y, stride, x0, y0)
            {
                reconstruct::add_residual_into16(
                    dst, stride, pred, res, n, valid_w, valid_h, self.bd,
                );
            }
            self.mark_decoded(x0, y0, n);
            return;
        }
        if self.cu_tqb {
            // Lossless: residual is the parsed level array verbatim (row-major),
            // no scaling or inverse transform (HEVC §8.6.5).
            self.res_scratch[..n * n].copy_from_slice(&levels[..n * n]);
        } else {
            let qp = self.cur_qp.clamp(0, 51) as u8;
            transform::dequantize_into(levels, n, qp, self.bd, &mut self.deq_scratch[..n * n]);
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
                    nx,
                    &mut self.res_scratch[..n * n],
                );
            }
        }
        let pred = &self.scratch.pred[..n * n];
        let res = &self.res_scratch[..n * n];
        if valid_w != 0
            && valid_h != 0
            && let Some(dst) = plane_tail_mut(&mut self.y, stride, x0, y0)
        {
            reconstruct::add_residual_into(dst, stride, pred, res, n, valid_w, valid_h, self.bd);
        }
        self.mark_decoded(x0, y0, n);
    }

    fn predict_only_luma(&mut self, x0: usize, y0: usize, log2_ts: u32, mode: u8) {
        let n = 1usize << log2_ts;
        self.predict_luma_block_into(x0, y0, n, mode);
        let pred = &self.scratch.pred[..n * n];
        let stride = self.w;
        let valid_w = self.w.saturating_sub(x0).min(n);
        let valid_h = self.h.saturating_sub(y0).min(n);
        if valid_w != 0
            && valid_h != 0
            && let Some(dst) = plane_tail_mut(&mut self.y, stride, x0, y0)
        {
            copy_pred_block_clipped(dst, stride, pred, n, valid_w, valid_h);
        }
        self.mark_decoded(x0, y0, n);
    }

    fn predict_luma_block_into(&mut self, x0: usize, y0: usize, n: usize, mode: u8) {
        let mut above = std::mem::take(&mut self.scratch.raw_above);
        let mut left = std::mem::take(&mut self.scratch.raw_left);
        let corner = self.gather_luma_refs_into(x0, y0, n, &mut above[..2 * n], &mut left[..2 * n]);
        let neutral = 1u16
            .checked_shl((self.bd.saturating_sub(1)) as u32)
            .unwrap_or(0);
        let strong = self.strong_smoothing && self.sps.strong_intra_smoothing;
        let sc = &mut self.scratch;
        intra::substitute_refs_into(
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
        intra::filter_refs_into(
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
        intra::predict_into(
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
        cbf_cb: [bool; 2],
        cbf_cr: [bool; 2],
    ) {
        let idc = self.sps.chroma_idc;
        let clog2 = if idc == 3 { luma_log2 } else { luma_log2 - 1 };
        let cn = 1usize << clog2;
        let cx0 = lx / self.sub_w;
        let cy0 = ly / self.sub_h;
        // 4:2:2 stacks two square chroma TBs vertically per luma TB (ChromaArrayType
        // 2); 4:2:0 and 4:4:4 have a single chroma TB. The bitstream codes them
        // component-major: all Cb TBs, then all Cr TBs (HEVC §7.3.8.11). Each TB is
        // reconstructed before the next so a lower stacked TB can use the upper one
        // as its intra above-reference.
        let n_tb = if idc == 2 { 2 } else { 1 };
        let scan = chroma_scan(mode, clog2, idc == 3);

        let qp_cb = qpc(
            self.cur_qp + self.pps.cb_qp_offset + self.slice_cb_qp_offset,
            idc,
        );
        for (t, &cb) in cbf_cb[0..n_tb].iter().enumerate() {
            let ty = cy0 + t * cn;
            if cb {
                let mut coeffs = std::mem::take(&mut self.coeff_scratch);
                let (_, max_x, _) = residual_coding(
                    &mut self.cab,
                    &mut self.ctx,
                    clog2,
                    false,
                    scan,
                    self.sign_hiding,
                    None,
                    self.cu_tqb,
                    &mut coeffs,
                );
                self.reconstruct_chroma(true, cx0, ty, cn, mode, &coeffs, qp_cb, max_x + 1);
                self.coeff_scratch = coeffs;
            } else {
                self.predict_only_chroma(true, cx0, ty, cn, mode);
            }
        }
        let qp_cr = qpc(
            self.cur_qp + self.pps.cr_qp_offset + self.slice_cr_qp_offset,
            idc,
        );
        for (t, &cr) in cbf_cr[..n_tb].iter().enumerate() {
            let ty = cy0 + t * cn;
            if cr {
                let mut coeffs = std::mem::take(&mut self.coeff_scratch);
                let (_, max_x, _) = residual_coding(
                    &mut self.cab,
                    &mut self.ctx,
                    clog2,
                    false,
                    scan,
                    self.sign_hiding,
                    None,
                    self.cu_tqb,
                    &mut coeffs,
                );
                self.reconstruct_chroma(false, cx0, ty, cn, mode, &coeffs, qp_cr, max_x + 1);
                self.coeff_scratch = coeffs;
            } else {
                self.predict_only_chroma(false, cx0, ty, cn, mode);
            }
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
        let neutral = 1u16
            .checked_shl((self.bd_c.saturating_sub(1)) as u32)
            .unwrap_or(0);
        let sc = &mut self.scratch;
        intra::substitute_refs_into(
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
        // Reference filtering: 4:2:0/4:2:2 chroma TBs are 4×4 and never filtered.
        // 4:4:4 chroma (≥8×8) filters references with the same [1 2 1] rule as luma
        // (HEVC: cIdx>0 filters only when ChromaArrayType==3), but without the luma
        // strong-intra-smoothing path. The DC/H/V prediction edge filter stays off
        // for chroma (is_luma=false in predict_into).
        if self.sps.chroma_idc == 3 {
            intra::filter_refs_into(
                &sc.above[..2 * n + 1],
                &sc.left[..2 * n + 1],
                n,
                mode,
                true,  // apply the luma [1 2 1] filtering decision
                false, // no strong intra smoothing for chroma
                self.bd_c,
                &mut sc.fa,
                &mut sc.fl,
            );
            intra::predict_into(
                mode,
                &sc.fa[..2 * n + 1],
                &sc.fl[..2 * n + 1],
                n,
                false,
                self.bd_c,
                &mut sc.pred[..n * n],
                &mut sc.refs_ang,
            );
        } else {
            intra::predict_into(
                mode,
                &sc.above[..2 * n + 1],
                &sc.left[..2 * n + 1],
                n,
                false,
                self.bd_c,
                &mut sc.pred[..n * n],
                &mut sc.refs_ang,
            );
        }
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
        nx: usize,
    ) {
        self.predict_chroma_block_into(is_cb, cx0, cy0, n, mode);
        let n2 = n * n;
        let stride = self.cw;
        let valid_w = self.cw.saturating_sub(cx0).min(n);
        let valid_h = self.ch.saturating_sub(cy0).min(n);
        if self.bd_c <= 8 {
            if self.cu_tqb {
                for (o, &l) in self.res_scratch16[..n2].iter_mut().zip(levels.iter()) {
                    *o = l.clamp(-32768, 32767) as i16;
                }
            } else {
                let qp_c = qp.clamp(0, 51) as u8;
                transform::dequantize_into(
                    levels,
                    n,
                    qp_c,
                    self.bd_c,
                    &mut self.deq_scratch16[..n2],
                );
                transform::inv_transform_into16(
                    &self.deq_scratch16[..n2],
                    n,
                    self.bd_c,
                    nx,
                    &mut self.res_scratch16[..n2],
                );
            }
            let pred = &self.scratch.pred[..n2];
            let res = &self.res_scratch16[..n2];
            if valid_w != 0 && valid_h != 0 {
                let plane = if is_cb { &mut self.cb } else { &mut self.cr };
                if let Some(dst) = plane_tail_mut(plane, stride, cx0, cy0) {
                    reconstruct::add_residual_into16(
                        dst, stride, pred, res, n, valid_w, valid_h, self.bd_c,
                    );
                }
            }
            return;
        }
        if self.cu_tqb {
            // Lossless: chroma residual is the parsed levels verbatim.
            self.res_scratch[..n2].copy_from_slice(&levels[..n2]);
        } else {
            let qp_c = qp.clamp(0, 51) as u8;
            transform::dequantize_into(levels, n, qp_c, self.bd_c, &mut self.deq_scratch[..n2]);
            transform::inv_transform_into(
                &self.deq_scratch[..n2],
                n,
                self.bd_c,
                nx,
                &mut self.res_scratch[..n2],
            );
        }
        let pred = &self.scratch.pred[..n2];
        let res = &self.res_scratch[..n2];
        if valid_w != 0 && valid_h != 0 {
            let plane = if is_cb { &mut self.cb } else { &mut self.cr };
            if let Some(dst) = plane_tail_mut(plane, stride, cx0, cy0) {
                reconstruct::add_residual_into(
                    dst, stride, pred, res, n, valid_w, valid_h, self.bd_c,
                );
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
        let stride = self.cw;
        let valid_w = self.cw.saturating_sub(cx0).min(n);
        let valid_h = self.ch.saturating_sub(cy0).min(n);
        if valid_w != 0 && valid_h != 0 {
            let plane = if is_cb { &mut self.cb } else { &mut self.cr };
            let pred = &pred_tmp[..n2];
            if let Some(dst) = plane_tail_mut(plane, stride, cx0, cy0) {
                copy_pred_block_clipped(dst, stride, pred, n, valid_w, valid_h);
            }
        }
    }
}

/// Captured aliasing pointers + immutable config for building per-row decoders
/// in the WPP wavefront. Holds `*mut` into a live [`FullDecoder`]'s shared
/// picture buffers; sound to share across threads under the wavefront lag.
pub(crate) struct RowFactory {
    y: (*mut u16, usize),
    cb: (*mut u16, usize),
    cr: (*mut u16, usize),
    mode_y: (*mut u8, usize),
    decoded: (*mut bool, usize),
    tqb: (*mut bool, usize),
    ct_depth: (*mut u8, usize),
    qp_y_map: (*mut i16, usize),
    sao: (*mut SaoCtb, usize),
    sps: Sps,
    pps: Pps,
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    sub_w: usize,
    sub_h: usize,
    bd: u8,
    bd_c: u8,
    log2_ctb: u32,
    log2_min_cb: u32,
    log2_min_tb: u32,
    log2_max_tb: u32,
    max_trafo_depth_intra: u32,
    grid_w: usize,
    grid_h: usize,
    slice_qp: i32,
    log2_qg: u32,
    ctb_cols: usize,
    ctb_rows: usize,
    sao_luma: bool,
    sao_chroma: bool,
    slice_cb_qp_offset: i32,
    slice_cr_qp_offset: i32,
    deblocking_disabled: bool,
    beta_offset_div2: i32,
    tc_offset_div2: i32,
    sign_hiding: bool,
    strong_smoothing: bool,
}

// SAFETY: the raw pointers address buffers kept alive by the template decoder
// for the whole wavefront scope. Concurrent access from row workers is disjoint
// by the 2-CTB lag, so sharing/sending the factory is sound.
unsafe impl Send for RowFactory {}
unsafe impl Sync for RowFactory {}

impl RowFactory {
    /// Build a per-row [`FullDecoder`] whose picture buffers alias the shared
    /// storage and whose engine is seeded from `row_cabac` + the given contexts.
    ///
    /// SAFETY: caller upholds the lag discipline so this row's writes never race
    /// another live row's, and the backing buffers outlive the returned decoder.
    pub(crate) unsafe fn make(
        &self,
        row_cabac: &[u8],
        ctx: ContextSet,
        ictx: IntraModeContexts,
    ) -> Result<FullDecoder, DecodeError> {
        let cab = CabacDecoder::new(row_cabac)
            .map_err(|_| DecodeError::Bitstream("row cabac init".into()))?;
        let mk = |p: (*mut u16, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        let mk8 = |p: (*mut u8, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        let mkb = |p: (*mut bool, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        let mki = |p: (*mut i16, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        let mks = |p: (*mut SaoCtb, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        Ok(FullDecoder {
            cab,
            ctx,
            ictx,
            sps: self.sps.clone(),
            pps: self.pps.clone(),
            y: mk(self.y),
            cb: mk(self.cb),
            cr: mk(self.cr),
            w: self.w,
            h: self.h,
            cw: self.cw,
            ch: self.ch,
            sub_w: self.sub_w,
            sub_h: self.sub_h,
            bd: self.bd,
            bd_c: self.bd_c,
            log2_ctb: self.log2_ctb,
            log2_min_cb: self.log2_min_cb,
            log2_min_tb: self.log2_min_tb,
            log2_max_tb: self.log2_max_tb,
            max_trafo_depth_intra: self.max_trafo_depth_intra,
            mode_y: mk8(self.mode_y),
            decoded: mkb(self.decoded),
            tqb: mkb(self.tqb),
            cu_tqb: false,
            grid_w: self.grid_w,
            grid_h: self.grid_h,
            ct_depth: mk8(self.ct_depth),
            slice_qp: self.slice_qp,
            qp_y_prev: self.slice_qp,
            qp_y_map: mki(self.qp_y_map),
            cu_qp_delta_val: 0,
            is_cu_qp_delta_coded: false,
            log2_qg: self.log2_qg,
            cur_qp: self.slice_qp,
            sao: mks(self.sao),
            ctb_cols: self.ctb_cols,
            ctb_rows: self.ctb_rows,
            sao_luma: self.sao_luma,
            sao_chroma: self.sao_chroma,
            slice_cb_qp_offset: self.slice_cb_qp_offset,
            slice_cr_qp_offset: self.slice_cr_qp_offset,
            deblocking_disabled: self.deblocking_disabled,
            beta_offset_div2: self.beta_offset_div2,
            tc_offset_div2: self.tc_offset_div2,
            sign_hiding: self.sign_hiding,
            // Row-views capture the WPP snapshot directly from live contexts, so
            // these arrays are never indexed here — keep them empty (no O(rows)
            // allocation per row).
            wpp_ctx_snap: Vec::new(),
            wpp_ictx_snap: Vec::new(),
            scratch: intra::IntraScratch::new(),
            deq_scratch: vec![0i32; 1024],
            res_scratch: vec![0i32; 1024],
            deq_scratch16: vec![0i16; 1024],
            res_scratch16: vec![0i16; 1024],
            coeff_scratch: vec![0i32; 1024],
            strong_smoothing: self.strong_smoothing,
        })
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

/// Parsed slice-segment header fields the reconstruction path needs.
#[derive(Clone, Debug)]
pub(crate) struct SliceHeader {
    /// SliceQpY = init_qp + slice_qp_delta.
    pub(crate) slice_qp: i32,
    /// Per-slice SAO enable for luma / chroma.
    pub(crate) sao_luma: bool,
    pub(crate) sao_chroma: bool,
    /// Byte offset in the RBSP where CABAC slice data begins.
    pub(crate) cabac_offset: usize,
    /// slice_cb_qp_offset / slice_cr_qp_offset (0 when not present). Added to the
    /// PPS chroma offsets during chroma QP derivation.
    pub(crate) cb_qp_offset: i32,
    pub(crate) cr_qp_offset: i32,
    /// Effective deblocking state for this slice: the PPS values unless the slice
    /// header overrides them (`deblocking_filter_override_flag`).
    pub(crate) deblocking_disabled: bool,
    pub(crate) beta_offset_div2: i32,
    pub(crate) tc_offset_div2: i32,
    /// CTB raster address where this slice segment starts (0 for the first).
    pub(crate) slice_segment_address: usize,
    /// True when this is the first slice segment of the picture.
    pub(crate) first_slice_in_pic: bool,
    /// True for a dependent slice segment (inherits the previous segment's
    /// header state and CABAC contexts rather than re-initialising).
    pub(crate) dependent_slice_segment: bool,
    /// WPP/tiles entry-point sub-stream byte lengths, i.e.
    /// `entry_point_offset_minus1[i] + 1` for each `i`. For WPP these are the
    /// byte lengths of every CTB-row sub-stream except the last (whose length is
    /// implied by the end of the CABAC payload). Empty when the stream carries
    /// no entry points. Used to position an independent CABAC engine per row for
    /// the parallel wavefront decode.
    pub(crate) entry_points: Vec<u32>,
}

/// Parse a slice header from the RBSP (after 2-byte NAL header has been consumed
/// by the caller or is still in the byte slice — we consume it here).
pub(crate) fn parse_slice_header_full(
    rbsp: &[u8],
    sps: &Sps,
    pps: &Pps,
    nal_type: u8,
) -> Result<SliceHeader, DecodeError> {
    let mut r = crate::bitreader::BitReader::new(rbsp);
    let e = |s: &'static str| DecodeError::Bitstream(s.into());
    r.read_bits(16).map_err(|_| e("NAL header"))?; // consume 2-byte NAL header
    let first_slice = r.read_flag().map_err(|_| e("first_slice"))?;
    let is_irap = (16..=23).contains(&nal_type);
    if is_irap {
        r.read_flag().map_err(|_| e("no_prior_pics"))?;
    }
    let _pps_id = r.read_ue().map_err(|_| e("pps_id"))?;

    // Number of CTBs in the picture, used to size slice_segment_address.
    let ctb = 1usize << sps.log2_ctb;
    let ctb_cols = (sps.width as usize).div_ceil(ctb);
    let ctb_rows = (sps.height as usize).div_ceil(ctb);
    let pic_size_in_ctbs = ctb_cols * ctb_rows;

    let mut dependent_slice_segment = false;
    let mut slice_segment_address = 0usize;
    if !first_slice {
        if pps.dependent_slice_segments_enabled {
            dependent_slice_segment = r.read_flag().map_err(|_| e("dep_slice_flag"))?;
        }
        // slice_segment_address is Ceil(Log2(PicSizeInCtbsY)) bits.
        let addr_bits = ceil_log2(pic_size_in_ctbs as u64);
        slice_segment_address = r.read_bits(addr_bits).map_err(|_| e("slice_addr"))? as usize;
    }

    // A dependent slice segment inherits all header state from the preceding
    // independent segment; its header carries only the (optional) extension and
    // the byte-alignment. The reconstruction fields returned here are filled by
    // the caller from the retained independent-segment header.
    if dependent_slice_segment {
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
        return Ok(SliceHeader {
            slice_qp: pps.init_qp,
            sao_luma: false,
            sao_chroma: false,
            cabac_offset: r.bit_pos() / 8,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            deblocking_disabled: pps.deblocking_filter_disabled,
            beta_offset_div2: pps.beta_offset_div2,
            tc_offset_div2: pps.tc_offset_div2,
            slice_segment_address,
            first_slice_in_pic: false,
            dependent_slice_segment: true,
            entry_points: Vec::new(),
        });
    }

    for _ in 0..pps.num_extra_slice_header_bits {
        r.read_bit().map_err(|_| e("extra_bits"))?;
    }
    let _slice_type = r.read_ue().map_err(|_| e("slice_type"))?;
    if pps.output_flag_present {
        r.read_flag().map_err(|_| e("pic_output_flag"))?;
    }
    if sps.separate_color_plane {
        r.read_bits(2).map_err(|_| e("color_plane"))?;
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
    // slice_cb_qp_offset / slice_cr_qp_offset (§7.3.6.1): added to the PPS-level
    // offsets when deriving chroma QP. Previously parsed and dropped.
    let mut cb_qp_offset = 0;
    let mut cr_qp_offset = 0;
    if pps.slice_chroma_qp_offsets_present {
        cb_qp_offset = r.read_se().map_err(|_| e("cb_qp_off"))?;
        cr_qp_offset = r.read_se().map_err(|_| e("cr_qp_off"))?;
    }
    // Deblocking: default to the PPS state, overridden per-slice if signalled.
    let mut deblocking_disabled = pps.deblocking_filter_disabled;
    let mut beta_offset_div2 = pps.beta_offset_div2;
    let mut tc_offset_div2 = pps.tc_offset_div2;
    let mut deblock_override = false;
    if pps.deblocking_filter_override_enabled {
        deblock_override = r.read_flag().map_err(|_| e("deblock_override"))?;
    }
    if deblock_override {
        deblocking_disabled = r.read_flag().map_err(|_| e("deblock_disabled"))?;
        if !deblocking_disabled {
            beta_offset_div2 = r.read_se().map_err(|_| e("beta_off"))?;
            tc_offset_div2 = r.read_se().map_err(|_| e("tc_off"))?;
        }
    }
    if pps.loop_filter_across_slices && (sao_luma || sao_chroma || !deblocking_disabled) {
        r.read_flag()
            .map_err(|_| e("loop_filter_across_slices_flag"))?;
    }
    let mut entry_points: Vec<u32> = Vec::new();
    if pps.tiles_enabled || pps.entropy_coding_sync_enabled {
        let n = r.read_ue().map_err(|_| e("num_entry_points"))?;
        if n > 0 {
            let len = r.read_ue().map_err(|_| e("offset_len"))? + 1;
            entry_points.reserve(n as usize);
            for _ in 0..n {
                // entry_point_offset_minus1[i] → sub-stream byte length.
                let off = r.read_bits(len).map_err(|_| e("entry_point"))?;
                entry_points.push(off + 1);
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
    Ok(SliceHeader {
        slice_qp,
        sao_luma,
        sao_chroma,
        cabac_offset: r.bit_pos() / 8,
        cb_qp_offset,
        cr_qp_offset,
        deblocking_disabled,
        beta_offset_div2,
        tc_offset_div2,
        slice_segment_address,
        first_slice_in_pic: first_slice,
        dependent_slice_segment: false,
        entry_points,
    })
}

/// Ceil(log2(n)) — the number of bits needed to represent values 0..n-1.
/// Returns 0 for n <= 1 (a single-CTB picture needs no address bits).
fn ceil_log2(n: u64) -> u32 {
    if n <= 1 {
        0
    } else {
        64 - (n - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::ceil_log2;

    #[test]
    fn ceil_log2_matches_slice_address_widths() {
        // Bits needed to hold values 0..n-1 (== slice_segment_address width).
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0); // single CTB: no address bits
        assert_eq!(ceil_log2(2), 1); // 0..1
        assert_eq!(ceil_log2(3), 2); // 0..2 needs 2 bits
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
        assert_eq!(ceil_log2(1023), 10);
        assert_eq!(ceil_log2(1024), 10);
        assert_eq!(ceil_log2(1025), 11);
    }
}
