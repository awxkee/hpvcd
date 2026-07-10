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
use crate::config::{Pps, ScalingList, Sps};
use crate::error::DecodeError;
use crate::exec::ExecContext;
use crate::fast_divide::FastDivU32;
use crate::fmt::BitDepth;
use crate::inter::MotionInfo;
use crate::intra;
use crate::transform;
use crate::yuv::YuvPlanes;

const MODE_PLANAR: u8 = 0;
const MODE_DC: u8 = 1;

/// Minimum total CTB count before the WPP wavefront is worth its fixed costs.
/// Below this the serial per-row decode is faster (spawn/coordination overhead
/// and the diagonal ramp dominate). ~64 CTBs ≈ a 512×512 picture at 64×64 CTBs.
const WAVEFRONT_MIN_CTBS: usize = 64;

#[inline(always)]
fn sort3_u8(mut v: [u8; 3]) -> [u8; 3] {
    if v[1] < v[0] {
        v.swap(0, 1);
    }
    if v[2] < v[1] {
        v.swap(1, 2);
    }
    if v[1] < v[0] {
        v.swap(0, 1);
    }
    v
}

#[derive(Clone, Copy, Default)]
struct SaoCtb {
    type_idx: [u8; 3], // 0=off,1=band,2=edge
    offsets: [[i32; 4]; 3],
    band_pos: [u8; 3],
    eo_class: [u8; 3],
}

pub(crate) struct FullDecoder<'cab> {
    cab: CabacDecoder<'cab>,
    ctx: ContextSet,
    ictx: IntraModeContexts,
    sps: Sps,
    pps: Pps,
    exec: ExecContext,

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
    sub_w_div: FastDivU32,
    sub_h_div: FastDivU32,
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
    pcm: crate::plane::Plane<bool>,     // per 4×4 luma block: pcm_flag (I_PCM CU)
    /// Per-4×4-luma deblock edge flags. `edge_v[g]` is set when a TU/PU/CU
    /// vertical boundary lies on the left side of the 4×4 block at `g`;
    /// `edge_h[g]` likewise for a horizontal boundary on its top side. Only
    /// edges on the 8×8 sample grid are ever filtered (§8.7.2).
    edge_v: crate::plane::Plane<bool>,
    edge_h: crate::plane::Plane<bool>,
    /// Per-4×4-luma boundary-strength for the left (`bs_v`) and top (`bs_h`)
    /// edge of each block. For an all-intra picture this is 2 at a real
    /// TU/PU/CU boundary and 0 elsewhere (§8.7.2.4).
    bs_v: crate::plane::Plane<u8>,
    bs_h: crate::plane::Plane<u8>,
    /// Per-4×4: the block's TU carries nonzero *luma* coefficients (§8.7.2.4
    /// coefficient condition, luma cbf only — mirrors de265 tu_info NONZERO).
    nz_coeff: crate::plane::Plane<bool>,
    /// Per-4×4: a transform-block boundary runs along this cell's left/top edge.
    /// Distinct from `edge_v`/`edge_h`, which also carry prediction edges: the
    /// coefficient BS-1 condition applies only at transform edges.
    tu_edge_v: crate::plane::Plane<bool>,
    tu_edge_h: crate::plane::Plane<bool>,
    /// Per-4×4 prediction-unit boundary flags. Inter PU edges that are not also
    /// TU edges still get a boundary-strength evaluation (§8.7.2.4); marking them
    /// separately lets `finalize_coeff_bs` cover PU-only edges.
    pu_edge_v: crate::plane::Plane<bool>,
    pu_edge_h: crate::plane::Plane<bool>,
    /// Per-4×4-luma slice index, used to gate cross-slice filtering when
    /// `loop_filter_across_slices` is disabled.
    slice_idx: crate::plane::Plane<u16>,
    /// Current slice index being decoded (incremented per independent segment).
    cur_slice_idx: u16,
    /// slice_loop_filter_across_slices_enabled_flag per slice index. Index 0 is
    /// unused (slice indices start at 1); grows as slices are decoded so a
    /// boundary can consult the flag of the slice that owns it (§8.7.1).
    slice_lf_across: Vec<bool>,
    cu_tqb: bool,  // current CU's cu_transquant_bypass_flag
    grid_w: usize, // ceil(w/4), one entry for every covered 4×4 luma grid cell
    #[allow(dead_code)]
    grid_h: usize, // ceil(h/4)
    ct_depth: crate::plane::Plane<u8>, // per 4×4, coding-tree depth (for split_cu_flag ctx)

    // QP tracking
    slice_qp: i32,
    qp_y_prev: i32,
    qp_y_map: crate::plane::Plane<i16>, // per 4×4 luma (QpY ∈ -QpBdOffsetY..=51, fits i16)
    cu_qp_delta_val: i32,
    is_cu_qp_delta_coded: bool,
    log2_qg: u32,
    cur_qp: i32,

    sao: crate::plane::Plane<SaoCtb>,
    ctb_cols: usize,
    ctb_rows: usize,
    sao_luma: bool,
    sao_chroma: bool,

    /// Resolved in-picture tile geometry (§6.5.1). `None` when the PPS does not
    /// enable tiles (plain raster scan, single implicit tile).
    tiles: Option<crate::tiles::TileGrid>,

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

    /// TileId of the CTB currently being decoded (0 when tiles are disabled).
    /// Used to reject intra reference neighbors that lie in a different tile
    /// (§6.4.1: a neighbor in a different tile is unavailable).
    cur_tile_id: usize,

    /// Pre-allocated scratch memory reused every TU to avoid per-block
    /// heap allocations on the hot path (~4–6 allocs per TU eliminated).
    scratch: intra::IntraScratch,
    /// Dequantised coefficient scratch (max 32×32 = 1024 values, clamped to ±32768 → i32)
    deq_scratch: Box<[i32; 1024]>,
    /// Inverse-transform output scratch (max 32×32 = 1024 i32 values)
    res_scratch: Box<[i32; 1024]>,
    /// i16 dequant/residual scratch, used on the 8-bit-depth path (half the width).
    deq_scratch16: Box<[i16; 1024]>,
    res_scratch16: Box<[i16; 1024]>,
    /// Parsed residual levels scratch (max 32×32), reused across TUs.
    coeff_scratch: Vec<i32>,
    /// Cached strong_intra_smoothing (avoids env-var lookup per TU)
    strong_smoothing: bool,

    // ---- Inter-prediction state (video decoding) ----
    /// Current slice type (0=B, 1=P, 2=I).
    slice_type: u8,
    /// cabac_init_flag for the current slice (swaps P/B context init tables).
    cabac_init: bool,
    /// Explicit weighted-prediction table for the current slice, when weighted
    /// prediction is enabled by the PPS for this slice type (§7.4.7.3).
    pred_weights: Option<crate::inter::PredWeightTable>,
    /// Per-4×4-luma motion field for the current picture.
    motion: crate::plane::Plane<MotionInfo>,
    /// Reference picture lists for the current slice (planes + POC).
    ref_list0: Vec<crate::dpb::RefEntry>,
    ref_list1: Vec<crate::dpb::RefEntry>,
    /// POC of the current picture.
    cur_poc: i32,
    /// Slice-level inter parameters captured from the header.
    mvd_l1_zero: bool,
    temporal_mvp: bool,
    max_num_merge_cand: usize,
    /// True while decoding an inter CU: the residual reconstruction must not
    /// re-run intra prediction (the MC prediction is already in the planes).
    cur_cu_inter: bool,
    /// Whether the most recently decoded PU used merge mode (for the 2Nx2N
    /// merge rqt_root_cbf inference, §7.3.8.5).
    last_pu_merge: bool,
    /// Per-4×4-luma cu_skip_flag for the skip-flag context increment (§9.3.4.2.2).
    cu_skip_map: Vec<bool>,
    /// Collocated picture selection for temporal MVP (from the slice header).
    collocated_from_l0: bool,
    collocated_ref_idx: usize,
    /// Reference frame planes (cloned Y/Cb/Cr) indexed by DPB index used in
    /// ref lists, supplied by the driver before decoding the slice.
    ref_frames: Vec<crate::inter::RefFramePlanes>,
    /// Motion-compensation scratch reused for L0/L1 intermediates and separable
    /// interpolation temporaries. Capacity grows to the largest PU seen and then
    /// stays hot for subsequent inter blocks.
    mc_pred0: Vec<i16>,
    mc_pred1: Vec<i16>,
    mc_tmp: Vec<i32>,
    chroma_scratch: Box<[u16; 1024]>,
}

impl FullDecoder<'static> {
    /// Maximum allowed dimension per axis and pixel count.
    pub(crate) const MAX_DIM: usize = 16_384;
    pub(crate) const MAX_PIXELS: usize = 64 * 1024 * 1024; // 64 MP

    pub(crate) fn new(
        cabac: &[u8],
        sps: Sps,
        pps: Pps,
        hdr: &SliceHeader,
    ) -> Result<Self, DecodeError> {
        let hdr_slice_qp = hdr.slice_qp;
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

        let slice_qp = clamp_qpy(hdr_slice_qp, sps.bit_depth_luma);

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
        // CABAC init type (§9.3.2.2): I=0; P=1 or 2 (swapped by cabac_init_flag);
        // B=2 or 1. cabac_init_flag toggles P<->B tables.
        let init_type = match hdr.slice_type {
            crate::inter::SLICE_I => 0u8,
            crate::inter::SLICE_P => {
                if hdr.cabac_init {
                    2
                } else {
                    1
                }
            }
            _ => {
                if hdr.cabac_init {
                    1
                } else {
                    2
                }
            }
        };
        let qp = ContextSet::init(init_type, slice_qp.clamp(0, 51) as u8);
        let ictx = IntraModeContexts::init(init_type, slice_qp.clamp(0, 51) as u8);
        let ctb_cols = w.div_ceil(ctb);
        let ctb_rows = h.div_ceil(ctb);
        let tiles = crate::tiles::TileGrid::from_pps(&pps, ctb_cols, ctb_rows);
        let log2_qg = sps.log2_ctb - pps.diff_cu_qp_delta_depth;
        Ok(FullDecoder {
            cab,
            ctx: qp,
            ictx,
            exec: ExecContext::new(),
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
            sub_w_div: FastDivU32::new(sub_w as u32),
            sub_h_div: FastDivU32::new(sub_h as u32),
            mode_y: crate::plane::Plane::owned(vec![MODE_DC; grid_w * grid_h]),
            decoded: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            tqb: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            pcm: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            edge_v: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            edge_h: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            bs_v: crate::plane::Plane::owned(vec![0u8; grid_w * grid_h]),
            bs_h: crate::plane::Plane::owned(vec![0u8; grid_w * grid_h]),
            nz_coeff: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            tu_edge_v: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            tu_edge_h: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            pu_edge_v: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            pu_edge_h: crate::plane::Plane::owned(vec![false; grid_w * grid_h]),
            slice_idx: crate::plane::Plane::owned(vec![0u16; grid_w * grid_h]),
            cur_slice_idx: 1,
            // Index 0 unused; index 1 is the first (independent) slice.
            slice_lf_across: vec![
                hdr.slice_loop_filter_across_slices,
                hdr.slice_loop_filter_across_slices,
            ],
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
            tiles,
            slice_cb_qp_offset: hdr.cb_qp_offset,
            slice_cr_qp_offset: hdr.cr_qp_offset,
            deblocking_disabled: hdr.deblocking_disabled,
            beta_offset_div2: hdr.beta_offset_div2,
            tc_offset_div2: hdr.tc_offset_div2,
            sign_hiding: pps.sign_data_hiding_enabled,
            wpp_ctx_snap: vec![None; ctb_rows],
            wpp_ictx_snap: vec![None; ctb_rows],
            scratch: intra::IntraScratch::new(),
            deq_scratch: Box::new([0i32; 1024]),
            res_scratch: Box::new([0i32; 1024]),
            deq_scratch16: Box::new([0i16; 1024]),
            res_scratch16: Box::new([0i16; 1024]),
            coeff_scratch: vec![0i32; 1024],
            cur_tile_id: 0,
            strong_smoothing: true,
            slice_type: hdr.slice_type,
            cabac_init: hdr.cabac_init,
            pred_weights: hdr.pred_weights.clone(),
            motion: crate::plane::Plane::owned(vec![MotionInfo::intra(); grid_w * grid_h]),
            ref_list0: Vec::new(),
            ref_list1: Vec::new(),
            cur_poc: 0,
            mvd_l1_zero: hdr.mvd_l1_zero,
            temporal_mvp: hdr.temporal_mvp,
            max_num_merge_cand: hdr.max_num_merge_cand.clamp(1, 5),
            cur_cu_inter: false,
            last_pu_merge: false,
            cu_skip_map: vec![false; grid_w * grid_h],
            collocated_from_l0: hdr.collocated_from_l0,
            collocated_ref_idx: hdr.collocated_ref_idx,
            ref_frames: Vec::new(),
            mc_pred0: Vec::new(),
            mc_pred1: Vec::new(),
            mc_tmp: Vec::new(),
            sps,
            pps,
            chroma_scratch: Box::new([0; 1024]),
        })
    }
}

impl<'cab> FullDecoder<'cab> {
    pub(crate) fn decode_segment(
        &mut self,
        cabac: &[u8],
        hdr: &SliceHeader,
        sub_starts: &[usize],
    ) -> Result<(), DecodeError> {
        self.cab.reset_with(cabac)?;
        self.slice_type = hdr.slice_type;
        self.cabac_init = hdr.cabac_init;
        self.pred_weights = hdr.pred_weights.clone();
        if !hdr.dependent_slice_segment {
            // Independent segment: reset entropy contexts and slice-level state.
            self.cur_slice_idx = self.cur_slice_idx.wrapping_add(1);
            // Record this slice's loop_filter_across_slices flag so boundary
            // filtering can consult the owning slice (§8.7.1).
            let idx = self.cur_slice_idx as usize;
            if self.slice_lf_across.len() <= idx {
                self.slice_lf_across
                    .resize(idx + 1, hdr.slice_loop_filter_across_slices);
            }
            self.slice_lf_across[idx] = hdr.slice_loop_filter_across_slices;
            let slice_qp = clamp_qpy(hdr.slice_qp, self.bd);
            self.slice_qp = slice_qp;
            self.qp_y_prev = slice_qp;
            self.cur_qp = slice_qp;
            self.cu_qp_delta_val = 0;
            self.is_cu_qp_delta_coded = false;
            self.sao_luma = hdr.sao_luma;
            self.sao_chroma = hdr.sao_chroma;
            self.slice_cb_qp_offset = hdr.cb_qp_offset;
            self.slice_cr_qp_offset = hdr.cr_qp_offset;
            self.deblocking_disabled = hdr.deblocking_disabled;
            self.beta_offset_div2 = hdr.beta_offset_div2;
            self.tc_offset_div2 = hdr.tc_offset_div2;
            let qp = slice_qp.clamp(0, 51) as u8;
            // CABAC init type (§9.3.2.2): I=0; P=1/2 and B=2/1, swapped by
            // cabac_init_flag. Using the I-slice tables for every slice leaves
            // the inter contexts mis-initialised and eventually desyncs CABAC.
            let init_type = match hdr.slice_type {
                crate::inter::SLICE_I => 0u8,
                crate::inter::SLICE_P => {
                    if hdr.cabac_init {
                        2
                    } else {
                        1
                    }
                }
                _ => {
                    if hdr.cabac_init {
                        1
                    } else {
                        2
                    }
                }
            };
            self.ctx = ContextSet::init(init_type, qp);
            self.ictx = IntraModeContexts::init(init_type, qp);
        } else {
            // Dependent segment inherits contexts/state; only the QP predictor
            // resets at the segment's first quantization group (handled by the
            // QG logic), so continue with the retained context.
        }
        let starts = if sub_starts.is_empty() {
            None
        } else {
            Some((cabac, sub_starts))
        };
        self.decode_slice_ctx(hdr.slice_segment_address, starts)
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
        // Record the current CTB's tile so intra availability can reject
        // neighbors that fall in a different tile (§6.4.1).
        self.cur_tile_id = match &self.tiles {
            Some(g) => g.tile_id_at(rx, ry),
            None => 0,
        };
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
            || self.tiles.is_some()
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
        // CABAC init type (§9.3.2.2): I=0; P=1/2 and B=2/1, swapped by
        // cabac_init_flag. Must match the serial start-of-slice init so the
        // wavefront row 0 seeds identical contexts.
        let init_type = match self.slice_type {
            crate::inter::SLICE_I => 0u8,
            crate::inter::SLICE_P => {
                if self.cabac_init {
                    2
                } else {
                    1
                }
            }
            _ => {
                if self.cabac_init {
                    1
                } else {
                    2
                }
            }
        };
        (
            ContextSet::init(init_type, qp),
            IntraModeContexts::init(init_type, qp),
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
            pcm: (self.pcm.as_mut_ptr(), self.pcm.len()),
            edge_v: (self.edge_v.as_mut_ptr(), self.edge_v.len()),
            edge_h: (self.edge_h.as_mut_ptr(), self.edge_h.len()),
            bs_v: (self.bs_v.as_mut_ptr(), self.bs_v.len()),
            bs_h: (self.bs_h.as_mut_ptr(), self.bs_h.len()),
            nz_coeff: (self.nz_coeff.as_mut_ptr(), self.nz_coeff.len()),
            tu_edge_v: (self.tu_edge_v.as_mut_ptr(), self.tu_edge_v.len()),
            tu_edge_h: (self.tu_edge_h.as_mut_ptr(), self.tu_edge_h.len()),
            slice_idx: (self.slice_idx.as_mut_ptr(), self.slice_idx.len()),
            ct_depth: (self.ct_depth.as_mut_ptr(), self.ct_depth.len()),
            qp_y_map: (self.qp_y_map.as_mut_ptr(), self.qp_y_map.len()),
            sao: (self.sao.as_mut_ptr(), self.sao.len()),
            sps: self.sps.clone(),
            pps: self.pps.clone(),
            exec: self.exec.clone(),
            w: self.w,
            h: self.h,
            cw: self.cw,
            ch: self.ch,
            sub_w: self.sub_w,
            sub_h: self.sub_h,
            sub_w_div: self.sub_w_div,
            sub_h_div: self.sub_h_div,
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
            motion: (self.motion.as_mut_ptr(), self.motion.len()),
            slice_type: self.slice_type,
            cabac_init: self.cabac_init,
            cur_poc: self.cur_poc,
            mvd_l1_zero: self.mvd_l1_zero,
            temporal_mvp: self.temporal_mvp,
            max_num_merge_cand: self.max_num_merge_cand,
            collocated_from_l0: self.collocated_from_l0,
            collocated_ref_idx: self.collocated_ref_idx,
            ref_list0: self.ref_list0.clone(),
            ref_list1: self.ref_list1.clone(),
            ref_frames: self.ref_frames.clone(),
            pred_weights: self.pred_weights.clone(),
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

    /// As [`decode_slice`], but with the slice segment's CABAC payload and
    /// entry-point sub-stream lengths available so the tiled path can seek to
    /// each tile's sub-stream. `cabac_and_entries` is `(cabac_bytes,
    /// entry_point_lengths)` where the lengths are `entry_point_offset_minus1+1`
    /// (byte lengths of sub-streams `0..n-1`; the last is implied).
    pub(crate) fn decode_slice_ctx(
        &mut self,
        start_ctb: usize,
        cabac_and_starts: Option<(&[u8], &[usize])>,
    ) -> Result<(), DecodeError> {
        if self.tiles.is_some() {
            return self.decode_slice_tiled(start_ctb, cabac_and_starts);
        }
        self.decode_slice_raster(start_ctb)
    }

    /// Tile-scan CTB decode. Advances through tile-scan addresses `ts`, mapping
    /// each to its raster `(rx, ry)`. At each tile's first CTB the CABAC engine
    /// is repositioned to that tile's entry-point sub-stream (when entry points
    /// are available) and the contexts + QP predictor are re-initialised.
    fn decode_slice_tiled(
        &mut self,
        start_ctb: usize,
        cabac_and_starts: Option<(&[u8], &[usize])>,
    ) -> Result<(), DecodeError> {
        let grid = self.tiles.clone().unwrap();
        let total = self.ctb_cols * self.ctb_rows;
        let start_ctb = start_ctb.min(total);
        let start_ts = grid.rs_to_ts(start_ctb);

        // `sub_starts[i]` is the RBSP offset (relative to the CABAC payload) of
        // tile sub-stream `i`; index 0 is 0. Precomputed by the caller through
        // the NAL→RBSP map so emulation-prevention bytes are handled correctly.
        let empty: &[usize] = &[];
        let sub_starts: &[usize] = cabac_and_starts.map(|(_, s)| s).unwrap_or(empty);

        // Counter of how many tile sub-streams we've entered so far (indexes
        // `sub_starts`). The slice segment begins in sub-stream 0.
        let mut substream = 0usize;

        let mut ts = start_ts;
        while ts < total {
            let rs = grid.ts_to_rs(ts);
            let rx = rs % self.ctb_cols;
            let ry = rs / self.ctb_cols;

            // On entering a new tile (not the slice segment's first CTB), seek
            // to that tile's sub-stream and re-initialise contexts (§9.3.1).
            if ts != start_ts && grid.is_tile_start_rs(rs) {
                substream += 1;
                let mut seeked = false;
                if let Some((cabac, _)) = cabac_and_starts
                    && let Some(&off) = sub_starts.get(substream)
                    && off <= cabac.len()
                {
                    // Reposition the arithmetic engine at the tile sub-stream.
                    if self.cab.reset_with(&cabac[off..]).is_ok() {
                        seeked = true;
                    }
                }
                if !seeked {
                    // Fallback: contiguous sub-streams (byte-aligned after the
                    // previous tile's end_of_sub_stream_one_bit).
                    self.cab.reinit_engine();
                }
                let qp = self.slice_qp.clamp(0, 51) as u8;
                let init_type = match self.slice_type {
                    crate::inter::SLICE_I => 0u8,
                    crate::inter::SLICE_P => {
                        if self.cabac_init {
                            2
                        } else {
                            1
                        }
                    }
                    _ => {
                        if self.cabac_init {
                            1
                        } else {
                            2
                        }
                    }
                };
                self.ctx = ContextSet::init(init_type, qp);
                self.ictx = IntraModeContexts::init(init_type, qp);
                self.qp_y_prev = self.slice_qp;
            }

            let terminated = self.decode_one_ctb(rx, ry, false);
            if terminated {
                break;
            }
            ts += 1;
        }
        Ok(())
    }

    fn decode_slice_raster(&mut self, start_ctb: usize) -> Result<(), DecodeError> {
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
        self.finish_with(pool, pool)
    }

    /// As [`Self::finish`], but with independent pools for deblocking and SAO so
    /// a caller can keep one filter serial. The video path passes `None` for
    /// `deblock_pool` because the parallel deblock's chroma kernel is not yet
    /// bit-identical to the serial reference; SAO stays parallel.
    pub(crate) fn finish_with(
        &mut self,
        deblock_pool: Option<&crate::threadpool::ThreadPool>,
        sao_pool: Option<&crate::threadpool::ThreadPool>,
    ) -> YuvPlanes {
        // In-loop filters run in HEVC order: deblocking first, then SAO. They
        // are independently gated: deblocking runs unless it is disabled (PPS or
        // a slice-level override), while SAO runs only when the SPS enables it.
        // Derive the coefficient-based deblock BS (§8.7.2.4) now that the whole
        // picture's transform edges and nonzero-coefficient flags are known. A
        // no-op for all-intra pictures (their edges already hold BS 2).
        self.finalize_coeff_bs();
        if !self.deblocking_disabled {
            match deblock_pool {
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
            // avoid oversubscription. The parallel EO kernels don't consult
            // slice/tile/PCM boundary maps, so a boundary-restricted picture
            // (§8.7.3.2) always takes the serial gated path.
            let restricted = self.sao_boundary_restricted();
            match sao_pool {
                Some(p) if p.threads() > 1 && self.ctb_rows > 1 && !restricted => {
                    self.apply_sao_parallel(p)
                }
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
        // §7.3.8.3: merge flags are only coded when the neighboring CTB is
        // available — same tile and same slice. At a tile/slice boundary the
        // flag is absent, so reading it would desync CABAC.
        let left_avail = rx > 0 && self.ctb_merge_avail(rx, ry, rx - 1, ry);
        let up_avail = ry > 0 && self.ctb_merge_avail(rx, ry, rx, ry - 1);
        let mut merge_left = false;
        let mut merge_up = false;
        if left_avail {
            merge_left = self.cab.decode_bin(&mut self.ctx.sao_merge_flag) != 0;
        }
        if !merge_left && up_avail {
            merge_up = self.cab.decode_bin(&mut self.ctx.sao_merge_flag) != 0;
        }
        if merge_left {
            self.sao[idx] = self.sao[idx - 1];
            return;
        }
        if merge_up {
            self.sao[idx] = self.sao[idx - self.ctb_cols];
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
            exec: self.exec.clone(),
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
            log2_ctb: self.log2_ctb,
            qp_y_map: &self.qp_y_map[..],
            tqb: &self.tqb[..],
            edge_v: &self.edge_v[..],
            edge_h: &self.edge_h[..],
            bs_v: &self.bs_v[..],
            bs_h: &self.bs_h[..],
            pcm: &self.pcm[..],
            slice_idx: &self.slice_idx[..],
            pcm_loop_filter_disabled: self.sps.pcm_loop_filter_disabled,
            loop_filter_across_slices: self.pps.loop_filter_across_slices
                && self.slice_lf_across.iter().all(|&f| f),
            tile_grid: match &self.tiles {
                Some(g) if !g.loop_filter_across_tiles => Some(g.clone()),
                _ => None,
            },
        };
        let out = crate::deblock::apply_deblocking_parallel(pool, &ctx, self.log2_ctb, y, cb, cr);
        self.y = crate::plane::Plane::owned(out.y);
        self.cb = crate::plane::Plane::owned(out.cb);
        self.cr = crate::plane::Plane::owned(out.cr);
    }

    /// Boundary strength for the vertical edge to the *left* of luma pixel
    /// `(px, py)` (px on the 8-sample grid). Returns 0 when the edge should not
    /// be filtered: not a real TU/PU/CU boundary, a disabled cross-slice or
    /// cross-tile boundary, or a PCM/transquant-bypass exemption (§8.7.2).
    #[inline]
    fn deblock_bs_v(&self, px: usize, py: usize) -> u8 {
        if px == 0 || px >= self.w || py >= self.h {
            return 0;
        }
        let g = match self.grid_idx(px, py) {
            Some(g) => g,
            None => return 0,
        };
        if !self.edge_v.get(g).copied().unwrap_or(false) {
            return 0;
        }
        if !self.filter_across_boundary(px - 1, py, px, py) {
            return 0;
        }
        self.bs_v.get(g).copied().unwrap_or(0)
    }

    /// Boundary strength for the horizontal edge *above* luma pixel `(px, py)`.
    #[inline]
    fn deblock_bs_h(&self, px: usize, py: usize) -> u8 {
        if py == 0 || px >= self.w || py >= self.h {
            return 0;
        }
        let g = match self.grid_idx(px, py) {
            Some(g) => g,
            None => return 0,
        };
        if !self.edge_h.get(g).copied().unwrap_or(false) {
            return 0;
        }
        if !self.filter_across_boundary(px, py - 1, px, py) {
            return 0;
        }
        self.bs_h.get(g).copied().unwrap_or(0)
    }

    /// Whether the boundary between P pixel `(pxp, pyp)` and Q pixel
    /// `(pxq, pyq)` may be filtered given slice/tile/PCM/TQB exemptions.
    #[inline]
    fn filter_across_boundary(&self, pxp: usize, pyp: usize, pxq: usize, pyq: usize) -> bool {
        // Transquant-bypass (lossless) blocks are never deblocked on the side
        // that is bypass (§8.7.2 restore_tqb behaviour ≈ skip).
        if self.tqb_at(pxp, pyp) || self.tqb_at(pxq, pyq) {
            return false;
        }
        // I_PCM blocks are exempt when pcm_loop_filter_disabled_flag is set.
        if self.sps.pcm_loop_filter_disabled && (self.pcm_at(pxp, pyp) || self.pcm_at(pxq, pyq)) {
            return false;
        }
        // Cross-slice filtering: disabled when the current (q-side) slice's
        // slice_loop_filter_across_slices_enabled_flag is 0 and the two sides
        // are in different slices (§8.7.1). The flag is per-slice, so consult
        // the slice that owns the q sample rather than the PPS default.
        let sq = self.slice_idx_at(pxq, pyq);
        let q_across = self
            .slice_lf_across
            .get(sq as usize)
            .copied()
            .unwrap_or(self.pps.loop_filter_across_slices);
        if !q_across {
            let sp = self.slice_idx_at(pxp, pyp);
            if sp != sq {
                return false;
            }
        }
        // Cross-tile filtering: if disabled and the two sides are in different
        // tiles, do not filter.
        if let Some(g) = &self.tiles
            && !g.loop_filter_across_tiles
        {
            let ctb = self.log2_ctb;
            let tp = g.tile_id_at(pxp >> ctb, pyp >> ctb);
            let tq = g.tile_id_at(pxq >> ctb, pyq >> ctb);
            if tp != tq {
                return false;
            }
        }
        true
    }

    #[inline]
    fn pcm_at(&self, px: usize, py: usize) -> bool {
        if px >= self.w || py >= self.h {
            return false;
        }
        self.grid_idx(px, py)
            .and_then(|g| self.pcm.get(g))
            .copied()
            .unwrap_or(false)
    }

    #[inline]
    fn slice_idx_at(&self, px: usize, py: usize) -> u16 {
        if px >= self.w || py >= self.h {
            return 0;
        }
        self.grid_idx(px, py)
            .and_then(|g| self.slice_idx.get(g))
            .copied()
            .unwrap_or(0)
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
                        // Only a real TU/PU/CU boundary with Bs>0 that isn't a
                        // disabled slice/tile/PCM/TQB boundary is filtered
                        // (§8.7.2). For intra pictures Bs is 2 at such edges.
                        let bs_v = self.deblock_bs_v(edge, mid);
                        if bs_v == 0 {
                            scan += 4;
                            continue;
                        }
                        let qp_p = qp_at(&self.qp_y_map[..], edge - 1, mid);
                        let qp_q = qp_at(&self.qp_y_map[..], edge, mid);
                        let avg_qp = (qp_p + qp_q + 1) >> 1;
                        let beta_prime = (avg_qp + qp_bd_offset_y + beta_offset).clamp(0, 51);
                        let tc_prime =
                            (avg_qp + qp_bd_offset_y + 2 * (bs_v as i32 - 1) + tc_offset)
                                .clamp(0, 53);
                        let beta = BETA[beta_prime as usize];
                        let tc = TC[tc_prime as usize];
                        if tc == 0 {
                            scan += 4;
                            continue;
                        }

                        // Vertical edge at x=edge; the 4 lines are rows
                        // scan..scan+4. tap<0 selects the p side (edge-1-..),
                        // tap>=0 the q side (edge+..).
                        if scan + 4 <= h {
                            let maxv = (1i32 << self.bd) - 1;
                            crate::deblock::deblock_luma_segment(
                                &mut self.y[..],
                                beta,
                                tc,
                                maxv,
                                |line, tap| {
                                    let col = (edge as i32 + tap) as usize;
                                    (scan + line) * w + col
                                },
                            );
                        }
                        scan += 4;
                        continue;
                    } else {
                        // horizontal edge at y=edge, cols scan..scan+3
                        let mid = scan + 1;
                        let bs_h = self.deblock_bs_h(mid, edge);
                        if bs_h == 0 {
                            scan += 4;
                            continue;
                        }
                        let qp_p = qp_at(&self.qp_y_map[..], mid, edge - 1);
                        let qp_q = qp_at(&self.qp_y_map[..], mid, edge);
                        let avg_qp = (qp_p + qp_q + 1) >> 1;
                        let beta_prime = (avg_qp + qp_bd_offset_y + beta_offset).clamp(0, 51);
                        let tc_prime =
                            (avg_qp + qp_bd_offset_y + 2 * (bs_h as i32 - 1) + tc_offset)
                                .clamp(0, 53);
                        let beta = BETA[beta_prime as usize];
                        let tc = TC[tc_prime as usize];
                        if tc == 0 {
                            scan += 4;
                            continue;
                        }

                        // Horizontal edge at y=edge; the 4 lines are columns
                        // scan..scan+4. tap<0 selects the p side (rows above),
                        // tap>=0 the q side (rows below).
                        if scan + 4 <= w && edge + 4 <= h {
                            let maxv = (1i32 << self.bd) - 1;
                            crate::deblock::deblock_luma_segment(
                                &mut self.y[..],
                                beta,
                                tc,
                                maxv,
                                |line, tap| {
                                    let row = (edge as i32 + tap) as usize;
                                    row * w + (scan + line)
                                },
                            );
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
                    // Chroma is filtered only where the co-located luma edge has
                    // Bs == 2 and is a real, non-disabled boundary (§8.7.2.4).
                    let (lex, ley) = if pass == 0 {
                        (edge * self.sub_w, mid * self.sub_h)
                    } else {
                        (mid * self.sub_w, edge * self.sub_h)
                    };
                    let bs = if pass == 0 {
                        self.deblock_bs_v(lex, ley)
                    } else {
                        self.deblock_bs_h(lex, ley)
                    };
                    if bs < 2 {
                        scan += 4;
                        continue;
                    }
                    // Chroma QP derivation (§8.7.2.5.5): average the co-located
                    // luma QpY of the two sides of the edge. The P sample is to
                    // the left for a vertical edge and above for a horizontal
                    // edge; Q is at the edge. Add the per-plane chroma QP offset
                    // (Cb vs Cr differ), then map through QpC and derive tC.
                    let (px_p, py_p) = if pass == 0 {
                        (lex.saturating_sub(1).min(w - 1), ley.min(h - 1))
                    } else {
                        (lex.min(w - 1), ley.saturating_sub(1).min(h - 1))
                    };
                    let qp_p_l = qp_at(&self.qp_y_map[..], px_p, py_p);
                    let qp_q_l = qp_at(&self.qp_y_map[..], lex.min(w - 1), ley.min(h - 1));
                    let avg_qp_l = (qp_p_l + qp_q_l + 1) >> 1;

                    for plane in 0..2usize {
                        let cqp_offset = if plane == 0 {
                            self.pps.cb_qp_offset
                        } else {
                            self.pps.cr_qp_offset
                        };
                        let qp_c = qpc(avg_qp_l + cqp_offset, self.sps.chroma_idc, self.bd_c);
                        let tc_prime_c = (qp_c + qp_bd_offset_c + 2 + tc_offset).clamp(0, 53);
                        let tc_c = TC[tc_prime_c as usize];
                        if tc_c == 0 {
                            continue;
                        }
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
            exec: self.exec.clone(),
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

    fn sao_usage(&self) -> ([bool; 3], [bool; 3]) {
        let mut active = [false; 3];
        let mut needs_src = [false; 3];
        for sao in self.sao.iter() {
            if self.sao_luma {
                active[0] |= sao.type_idx[0] != 0;
                needs_src[0] |= sao.type_idx[0] == 2;
            }
            if self.sao_chroma && self.cw != 0 && self.ch != 0 {
                active[1] |= sao.type_idx[1] != 0;
                active[2] |= sao.type_idx[2] != 0;
                needs_src[1] |= sao.type_idx[1] == 2;
                needs_src[2] |= sao.type_idx[2] == 2;
            }
        }
        (active, needs_src)
    }

    /// Whether SAO edge-offset must honour slice/tile/PCM/TQB boundary
    /// availability (§8.7.3.2). False for the common single-slice, no-tile,
    /// no-PCM picture, where the fast SIMD/scalar EO path is exact.
    fn sao_boundary_restricted(&self) -> bool {
        // Any slice (not just the PPS default) may disable cross-slice
        // filtering; if so, SAO must take the boundary-aware path.
        if self.slice_lf_across.iter().any(|&f| !f) {
            return true;
        }
        if !self.pps.loop_filter_across_slices {
            return true;
        }
        if let Some(g) = &self.tiles
            && !g.loop_filter_across_tiles
        {
            return true;
        }
        if self.sps.pcm_loop_filter_disabled && self.pcm.iter().any(|&b| b) {
            return true;
        }
        // Transquant-bypass samples are always loop-filter exempt.
        if self.tqb.iter().any(|&b| b) {
            return true;
        }
        false
    }

    /// Build the boundary map view used by the gated SAO edge-offset path.
    fn sao_boundary(&self) -> crate::sao::SaoBoundary<'_> {
        crate::sao::SaoBoundary {
            gw: self.grid_w,
            log2_ctb: self.log2_ctb,
            sub_w: self.sub_w,
            sub_h: self.sub_h,
            slice_idx: &self.slice_idx[..],
            tqb: &self.tqb[..],
            pcm: &self.pcm[..],
            loop_filter_across_slices: self.pps.loop_filter_across_slices
                && self.slice_lf_across.iter().all(|&f| f),
            pcm_loop_filter_disabled: self.sps.pcm_loop_filter_disabled,
            tile_grid: self.tiles.as_ref(),
        }
    }

    fn apply_sao(&mut self) {
        let ctb = 1usize << self.log2_ctb;
        let (active, needs_src) = self.sao_usage();
        if !active.iter().any(|&x| x) {
            return;
        }

        // Only EO needs an untouched source snapshot. BO is pointwise and can
        // run in place, avoiding full-plane clones for common BO-only pictures.
        let orig_y = needs_src[0].then(|| self.y.to_vec_clone());
        let orig_cb = needs_src[1].then(|| self.cb.to_vec_clone());
        let orig_cr = needs_src[2].then(|| self.cr.to_vec_clone());

        let restricted = self.sao_boundary_restricted();
        // Take the planes out so the boundary map borrows (slice_idx/tqb/pcm/
        // tiles) don't conflict with the &mut plane writes on the gated path.
        let mut y = self.y.take_vec();
        let mut cb = self.cb.take_vec();
        let mut cr = self.cr.take_vec();
        let bnd = self.sao_boundary();
        let (w, h, cw, ch) = (self.w, self.h, self.cw, self.ch);
        let (bd, bd_c) = (self.bd, self.bd_c);
        let (sub_w, sub_h) = (self.sub_w, self.sub_h);
        let exec = self.exec.clone();

        for ry in 0..self.ctb_rows {
            for rx in 0..self.ctb_cols {
                let idx = ry * self.ctb_cols + rx;
                let sao = self.sao[idx];
                let x0 = rx * ctb;
                let y0 = ry * ctb;

                // Luma
                if self.sao_luma && sao.type_idx[0] != 0 {
                    let x_end = (x0 + ctb).min(w);
                    let y_end = (y0 + ctb).min(h);
                    match sao.type_idx[0] {
                        1 => (exec.sao_band_offset_inplace)(
                            &mut y[..],
                            w,
                            x0,
                            y0,
                            x_end,
                            y_end,
                            &sao.offsets[0],
                            sao.band_pos[0],
                            bd,
                        ),
                        2 => {
                            let src = orig_y.as_deref().expect("SAO EO requires luma snapshot");
                            if restricted {
                                crate::sao::apply_sao_edge_offset_gated(
                                    &mut y[..],
                                    src,
                                    w,
                                    h,
                                    x0,
                                    y0,
                                    x_end,
                                    y_end,
                                    &sao.offsets[0],
                                    sao.eo_class[0],
                                    bd,
                                    &|cx, cy, nx, ny| bnd.luma_neighbor_ok(cx, cy, nx, ny),
                                );
                            } else {
                                (exec.sao_plane)(
                                    &mut y[..],
                                    src,
                                    w,
                                    h,
                                    x0,
                                    y0,
                                    x_end,
                                    y_end,
                                    2,
                                    &sao.offsets[0],
                                    sao.band_pos[0],
                                    sao.eo_class[0],
                                    bd,
                                );
                            }
                        }
                        _ => {}
                    }
                }
                // Chroma (Cb, Cr share eo_class)
                if self.sao_chroma {
                    let cx0 = x0 / sub_w;
                    let cy0 = y0 / sub_h;
                    let cx_end = ((x0 + ctb) / sub_w).min(cw);
                    let cy_end = ((y0 + ctb) / sub_h).min(ch);

                    for (plane_i, (buf, src_opt)) in
                        [(&mut cb, orig_cb.as_deref()), (&mut cr, orig_cr.as_deref())]
                            .into_iter()
                            .enumerate()
                    {
                        let ci = 1 + plane_i;
                        match sao.type_idx[ci] {
                            1 => (exec.sao_band_offset_inplace)(
                                &mut buf[..],
                                cw,
                                cx0,
                                cy0,
                                cx_end,
                                cy_end,
                                &sao.offsets[ci],
                                sao.band_pos[ci],
                                bd_c,
                            ),
                            2 => {
                                let src = src_opt.expect("SAO EO requires chroma snapshot");
                                if restricted {
                                    crate::sao::apply_sao_edge_offset_gated(
                                        &mut buf[..],
                                        src,
                                        cw,
                                        ch,
                                        cx0,
                                        cy0,
                                        cx_end,
                                        cy_end,
                                        &sao.offsets[ci],
                                        sao.eo_class[ci],
                                        bd_c,
                                        &|ccx, ccy, ncx, ncy| {
                                            bnd.chroma_neighbor_ok(ccx, ccy, ncx, ncy)
                                        },
                                    );
                                } else {
                                    (exec.sao_plane)(
                                        &mut buf[..],
                                        src,
                                        cw,
                                        ch,
                                        cx0,
                                        cy0,
                                        cx_end,
                                        cy_end,
                                        2,
                                        &sao.offsets[ci],
                                        sao.band_pos[ci],
                                        sao.eo_class[ci],
                                        bd_c,
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        self.y = crate::plane::Plane::owned(y);
        self.cb = crate::plane::Plane::owned(cb);
        self.cr = crate::plane::Plane::owned(cr);
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
            && self.same_tile(x0 - 1, y0)
            && self.same_slice(x0 - 1, y0)
            && let Some(g) = self.grid_idx(x0 - 1, y0)
            && self.decoded[g]
            && self.ct_depth[g] as usize > depth as usize
        {
            inc += 1;
        }
        if y0 >= 4
            && self.same_tile(x0, y0 - 1)
            && self.same_slice(x0, y0 - 1)
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
            self.cur_qp = clamp_qpy(self.predict_qp(xqg, yqg), self.bd);
        }

        // For P/B slices, cu_skip_flag then (if not skipped) pred_mode_flag select
        // between inter and intra. I-slices are always intra (no flags coded).
        if self.slice_type != crate::inter::SLICE_I {
            let skip = self.decode_cu_skip_flag(x0, y0);
            if skip {
                self.decode_inter_cu(x0, y0, log2_cb, true);
                return;
            }
            let pred_mode_intra = self.cab.decode_bin(&mut self.ctx.pred_mode_flag) != 0;
            if !pred_mode_intra {
                self.decode_inter_cu(x0, y0, log2_cb, false);
                return;
            }
            // else fall through to the intra CU path below.
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

        // pcm_flag (§7.3.8.5): present for an intra CU when I_PCM is enabled and
        // the CB size is within [Log2MinIpcmCbSize, Log2MaxIpcmCbSize]. When set,
        // the CU carries uncompressed samples instead of a prediction+residual.
        if self.sps.pcm_enabled
            && log2_cb >= self.sps.log2_min_pcm_cb
            && log2_cb <= self.sps.log2_max_pcm_cb
            && self.cab.decode_terminate() != 0
        {
            self.decode_pcm_cu(x0, y0, log2_cb);
            return;
        }

        let cb_size = 1usize << log2_cb;
        // part_mode: NxN only at min CB
        let nxn = if log2_cb == self.log2_min_cb {
            self.cab.decode_bin(&mut self.ctx.part_mode[0]) == 0
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
        // The CU's outer boundary is already marked by its constituent TUs'
        // left/top edges, so only the NxN internal PU split lines need adding.
        if nxn {
            let half = cb_size / 2;
            self.mark_block_edges(x0 + half, y0, half, 2); // internal vertical PU edge
            self.mark_block_edges(x0, y0 + half, half, 2); // internal horizontal PU edge
        }
        self.set_slice_idx(x0, y0, cb_size);
        let cur_qp = clamp_qpy(self.cur_qp, self.bd);
        self.qp_y_prev = cur_qp;
        self.cur_qp = cur_qp;
        self.set_qp(x0, y0, cb_size, cur_qp);
    }

    /// Decode an I_PCM coding unit (§7.3.8.5 pcm_sample, §8.4.5.2 reconstruction).
    /// The arithmetic engine has just returned 1 for `pcm_flag`; the bitstream is
    /// byte-aligned, raw fixed-length luma then chroma samples are read, scaled
    /// up to the coded bit depth, and written directly into the planes. The
    /// arithmetic engine is re-initialised afterward.
    fn decode_pcm_cu(&mut self, x0: usize, y0: usize, log2_cb: u32) {
        let n = 1usize << log2_cb;
        // Align the raw bit pointer (pcm_alignment_zero_bit padding) — same
        // terminate→align sequence as a WPP/tile sub-stream boundary.
        self.cab.byte_align();

        // Luma samples: n×n, each pcm_bit_depth_luma bits, scaled to bd.
        let pbd_y = (self.sps.pcm_bit_depth_luma as u32).min(self.bd as u32);
        let shift_y = self.bd as u32 - pbd_y;
        let w = self.w;
        for yy in 0..n {
            let py = y0 + yy;
            if py >= self.h {
                // Still must consume the bits to stay in sync.
                for _ in 0..n {
                    self.cab.read_pcm_bits(pbd_y);
                }
                continue;
            }
            for xx in 0..n {
                let s = self.cab.read_pcm_bits(pbd_y);
                let px = x0 + xx;
                if px < w {
                    self.y[py * w + px] = (s << shift_y) as u16;
                }
            }
        }

        // Chroma samples (skipped for monochrome).
        if !self.sps.chroma.is_monochrome() {
            let cn_w = n / self.sub_w;
            let cn_h = n / self.sub_h;
            let cx0 = x0 / self.sub_w;
            let cy0 = y0 / self.sub_h;
            let pbd_c = (self.sps.pcm_bit_depth_chroma as u32).min(self.bd_c as u32);
            let shift_c = self.bd_c as u32 - pbd_c;
            let cw = self.cw;
            for (is_cr, plane_is_cr) in [false, true].into_iter().enumerate() {
                let _ = is_cr;
                for yy in 0..cn_h {
                    let py = cy0 + yy;
                    let in_pic = py < self.ch;
                    for xx in 0..cn_w {
                        let s = self.cab.read_pcm_bits(pbd_c);
                        if !in_pic {
                            continue;
                        }
                        let px = cx0 + xx;
                        if px < cw {
                            let v = (s << shift_c) as u16;
                            if plane_is_cr {
                                self.cr[py * cw + px] = v;
                            } else {
                                self.cb[py * cw + px] = v;
                            }
                        }
                    }
                }
            }
        }

        // Re-prime the arithmetic engine from the now byte-aligned position.
        self.cab.reinit_engine();

        // Bookkeeping: I_PCM CUs are intra (mode irrelevant for prediction but
        // needed for neighbor availability), lossless-like for the deblocking
        // filter when pcm_loop_filter_disabled, and always "decoded".
        self.set_mode(x0, y0, n, MODE_DC);
        self.set_pcm(x0, y0, n);
        self.mark_decoded(x0, y0, n);
        self.mark_block_edges(x0, y0, n, 2);
        self.set_slice_idx(x0, y0, n);
        // PCM CUs carry no delta QP; QpY stays at the predicted value.
        let cur_qp = clamp_qpy(self.cur_qp, self.bd);
        self.qp_y_prev = cur_qp;
        self.cur_qp = cur_qp;
        self.set_qp(x0, y0, n, cur_qp);
    }

    fn set_pcm(&mut self, x0: usize, y0: usize, size: usize) {
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w
                    && yy < self.h
                    && let Some(g) = self.grid_idx(xx, yy)
                {
                    self.pcm[g] = true;
                }
            }
        }
    }

    /// Mark the left (vertical) and top (horizontal) boundaries of the block
    /// `[x0,x0+size) × [y0,y0+size)` as deblock filter edges with the given
    /// boundary strength. Only edges on the 8-sample grid are ever *filtered*
    /// (checked at filter time), but every TU/PU/CU boundary is recorded so the
    /// filter can distinguish real block edges from transform-interior 8-grid
    /// lines. For an all-intra picture `bs` is 2 (§8.7.2.4 rule: a boundary with
    /// an intra block on either side has Bs = 2).
    /// Mark the left/top boundaries of a transform block on the 4×4 grid.
    /// Mirrors de265's `markTransformBlockBoundary` leaf case: every TU leaf
    /// (and, for CUs without a transform tree, the CB itself) contributes its
    /// left and top edge as a transform edge.
    fn mark_tu_edges(&mut self, x0: usize, y0: usize, size: usize) {
        let gw = self.grid_w;
        if x0 > 0 && x0 < self.w {
            let gx = x0 / 4;
            let y_end = (y0 + size).min(self.h);
            let mut g = (y0 / 4) * gw + gx;
            let mut yy = y0;
            while yy < y_end {
                self.tu_edge_v[g] = true;
                g += gw;
                yy += 4;
            }
        }
        if y0 > 0 && y0 < self.h {
            let x_end = (x0 + size).min(self.w);
            let mut g = (y0 / 4) * gw + x0 / 4;
            let mut xx = x0;
            while xx < x_end {
                self.tu_edge_h[g] = true;
                g += 1;
                xx += 4;
            }
        }
    }

    /// Mark an internal prediction-unit vertical boundary (left edge of a PU
    /// that starts inside its CU) so `finalize_coeff_bs` evaluates its inter
    /// boundary strength even if it is not a transform edge.
    fn mark_pu_edge_v(&mut self, x0: usize, y0: usize, height: usize) {
        if x0 == 0 || x0 >= self.w {
            return;
        }
        let gw = self.grid_w;
        let gx = x0 / 4;
        let y_end = (y0 + height).min(self.h);
        let mut g = (y0 / 4) * gw + gx;
        let mut yy = y0;
        while yy < y_end {
            self.pu_edge_v[g] = true;
            g += gw;
            yy += 4;
        }
    }

    /// Mark an internal prediction-unit horizontal boundary (top edge of a PU
    /// that starts inside its CU).
    fn mark_pu_edge_h(&mut self, x0: usize, y0: usize, width: usize) {
        if y0 == 0 || y0 >= self.h {
            return;
        }
        let gw = self.grid_w;
        let x_end = (x0 + width).min(self.w);
        let mut g = (y0 / 4) * gw + x0 / 4;
        let mut xx = x0;
        while xx < x_end {
            self.pu_edge_h[g] = true;
            g += 1;
            xx += 4;
        }
    }

    /// Record that the transform block at (x0,y0) carries nonzero luma
    /// coefficients, over its whole 4×4-grid footprint.
    fn set_nz_coeff(&mut self, x0: usize, y0: usize, size: usize) {
        let gw = self.grid_w;
        let y_end = (y0 + size).min(self.h);
        let x_end = (x0 + size).min(self.w);
        let mut yy = y0;
        while yy < y_end {
            let base = (yy / 4) * gw;
            let mut xx = x0;
            while xx < x_end {
                self.nz_coeff[base + xx / 4] = true;
                xx += 4;
            }
            yy += 4;
        }
    }

    /// Final deblock BS derivation (§8.7.2.4), run once per picture before the
    /// filter, mirroring de265's `derive_boundaryStrength`:
    /// - either side of an edge intra-predicted → BS 2 (this catches edges where
    ///   the intra block is the P side, which per-TU marking cannot see);
    /// - else, a transform edge with nonzero luma coefficients on either side →
    ///   BS 1.
    fn finalize_coeff_bs(&mut self) {
        let gw = self.grid_w;
        let gh = self.grid_h;
        let have_motion = self.motion.len() >= gw * gh;
        let is_intra = |m: &[crate::inter::MotionInfo], g: usize| have_motion && m[g].is_intra;
        // Inter boundary strength (§8.7.2.4): bS = 1 between two inter blocks
        // that use different reference pictures, a different number of MVs, or
        // whose motion vectors differ by ≥ 4 quarter-samples in any component.
        let inter_bs1 = |p: &crate::inter::MotionInfo, q: &crate::inter::MotionInfo| -> bool {
            let rp = |m: &crate::inter::MotionInfo, l: usize| {
                if m.pred_used(l) { m.ref_poc[l] } else { -1 }
            };
            let (rp0, rp1) = (rp(p, 0), rp(p, 1));
            let (rq0, rq1) = (rp(q, 0), rp(q, 1));
            let same_pics = (rp0 == rq0 && rp1 == rq1) || (rp0 == rq1 && rp1 == rq0);
            if !same_pics {
                return true;
            }
            let mv = |m: &crate::inter::MotionInfo, l: usize| {
                if m.pred_used(l) {
                    m.mv[l]
                } else {
                    crate::inter::Mv::default()
                }
            };
            let (mp0, mp1, mq0, mq1) = (mv(p, 0), mv(p, 1), mv(q, 0), mv(q, 1));
            let far = |a: crate::inter::Mv, b: crate::inter::Mv| {
                (a.x - b.x).abs() >= 4 || (a.y - b.y).abs() >= 4
            };
            if rp0 != rp1 {
                if rp0 == rq0 {
                    far(mp0, mq0) || far(mp1, mq1)
                } else {
                    far(mp0, mq1) || far(mp1, mq0)
                }
            } else {
                (far(mp0, mq0) || far(mp1, mq1)) && (far(mp0, mq1) || far(mp1, mq0))
            }
        };
        for gy in 0..gh {
            let row = gy * gw;
            for gx in 0..gw {
                let g = row + gx;
                let v_edge = self.tu_edge_v[g] || self.pu_edge_v[g];
                if gx > 0 && v_edge {
                    if self.bs_v[g] < 2
                        && (is_intra(&self.motion[..], g) || is_intra(&self.motion[..], g - 1))
                    {
                        self.bs_v[g] = 2;
                        self.edge_v[g] = true;
                    } else if self.bs_v[g] < 1
                        && self.tu_edge_v[g]
                        && (self.nz_coeff[g] || self.nz_coeff[g - 1])
                    {
                        // Coefficient-based bS is a transform-edge property.
                        self.bs_v[g] = 1;
                        self.edge_v[g] = true;
                    } else if self.bs_v[g] < 1
                        && have_motion
                        && inter_bs1(&self.motion[g], &self.motion[g - 1])
                    {
                        self.bs_v[g] = 1;
                        self.edge_v[g] = true;
                    }
                }
                let h_edge = self.tu_edge_h[g] || self.pu_edge_h[g];
                if gy > 0 && h_edge {
                    if self.bs_h[g] < 2
                        && (is_intra(&self.motion[..], g) || is_intra(&self.motion[..], g - gw))
                    {
                        self.bs_h[g] = 2;
                        self.edge_h[g] = true;
                    } else if self.bs_h[g] < 1
                        && self.tu_edge_h[g]
                        && (self.nz_coeff[g] || self.nz_coeff[g - gw])
                    {
                        self.bs_h[g] = 1;
                        self.edge_h[g] = true;
                    } else if self.bs_h[g] < 1
                        && have_motion
                        && inter_bs1(&self.motion[g], &self.motion[g - gw])
                    {
                        self.bs_h[g] = 1;
                        self.edge_h[g] = true;
                    }
                }
            }
        }
    }

    fn mark_block_edges(&mut self, x0: usize, y0: usize, size: usize, bs: u8) {
        let gw = self.grid_w;
        // Left vertical edge: the column of 4×4 cells at x0, for each row. The
        // grid column is fixed (x0/4), so step the row index directly instead of
        // recomputing `grid_idx` (bounds + mul) every iteration.
        if x0 > 0 && x0 < self.w {
            let gx = x0 / 4;
            let y_end = (y0 + size).min(self.h);
            let mut g = (y0 / 4) * gw + gx;
            let mut yy = y0;
            while yy < y_end {
                self.edge_v[g] = true;
                if bs > self.bs_v[g] {
                    self.bs_v[g] = bs;
                }
                g += gw;
                yy += 4;
            }
        }
        // Top horizontal edge: the grid row is fixed (y0/4), step columns.
        if y0 > 0 && y0 < self.h {
            let x_end = (x0 + size).min(self.w);
            let base = (y0 / 4) * gw;
            let mut g = base + x0 / 4;
            let mut xx = x0;
            while xx < x_end {
                self.edge_h[g] = true;
                if bs > self.bs_h[g] {
                    self.bs_h[g] = bs;
                }
                g += 1;
                xx += 4;
            }
        }
    }

    /// Record the slice index over a block (for cross-slice filter gating).
    fn set_slice_idx(&mut self, x0: usize, y0: usize, size: usize) {
        // Single independent slice: the array stays uniformly 0, so any pair of
        // cells compares equal (same slice). Writing per-CU is pure overhead in
        // the common case, so skip it until a second slice segment appears.
        if self.cur_slice_idx <= 1 {
            return;
        }
        let s = self.cur_slice_idx;
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w
                    && yy < self.h
                    && let Some(g) = self.grid_idx(xx, yy)
                {
                    self.slice_idx[g] = s;
                }
            }
        }
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
        debug_assert!((qpy_min(self.bd)..=51).contains(&qp_a));
        debug_assert!((qpy_min(self.bd)..=51).contains(&qp_b));
        (qp_a + qp_b + 1) >> 1
    }

    fn set_qp(&mut self, x0: usize, y0: usize, size: usize, qp: i32) {
        let qp = clamp_qpy(qp, self.bd) as i16;
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w
                    && yy < self.h
                    && let Some(g) = self.grid_idx(xx, yy)
                {
                    self.qp_y_map[g] = qp;
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
        let mark_slice = self.cur_slice_idx > 1;
        let s = self.cur_slice_idx;
        for yy in (y0..y0 + size).step_by(4) {
            for xx in (x0..x0 + size).step_by(4) {
                if xx < self.w
                    && yy < self.h
                    && let Some(g) = self.grid_idx(xx, yy)
                {
                    self.decoded[g] = true;
                    // Keep slice ownership in lock-step with `decoded` so that
                    // intra neighbors decoded earlier in this same slice (but
                    // before the enclosing CU's set_slice_idx runs) are seen as
                    // same-slice rather than defaulting to the initial 0.
                    if mark_slice {
                        self.slice_idx[g] = s;
                    }
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
            let sorted = sort3_u8(mpm);
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
        // §8.4.2: a neighbor in a different tile is unavailable and contributes
        // candIntraPredMode = DC. The z-scan/decoded availability of the left and
        // above neighbors is already guaranteed by raster decode order, and the
        // above-CTB-row boundary case is handled by `mpm_list`, so only the tile
        // boundary needs checking here — using full `luma_avail` (which also
        // tests the `decoded` map) would wrongly zero valid candidates and
        // desync CABAC.
        if !self.same_tile(x as usize, y as usize) {
            return MODE_DC;
        }
        if !self.same_slice(x as usize, y as usize) {
            return MODE_DC;
        }
        if !self.constrained_intra_ok(x as usize, y as usize) {
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

impl<'cab> FullDecoder<'cab> {
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
            // cbf_luma: for inter, at the root TU (depth 0) with no chroma
            // residual, cbf_luma is inferred = 1 (an inter block with rqt_root_cbf
            // set must have luma residual); otherwise it is coded (§7.3.8.8).
            let any_chroma = cbf_cb.iter().any(|&b| b) || cbf_cr.iter().any(|&b| b);
            let cbf_luma = if self.cur_cu_inter && depth == 0 && !any_chroma {
                true
            } else {
                self.cab
                    .decode_bin(&mut self.ctx.cbf_luma[if depth == 0 { 1 } else { 0 }])
                    != 0
            };
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

        // Record this TU's boundaries for the deblocking filter (§8.7.2.4).
        // Intra TU boundaries are BS 2 directly. For inter, BS 1 from
        // coefficients is derived after the slice in `finalize_coeff_bs`, which
        // needs (a) transform-edge positions and (b) the per-block nonzero *luma*
        // coefficient flag — either side of a transform edge having coefficients
        // raises BS to 1.
        let tu_size = 1usize << log2_ts;
        self.mark_tu_edges(x0, y0, tu_size);
        if cbf_luma {
            self.set_nz_coeff(x0, y0, tu_size);
        }
        if !self.cur_cu_inter {
            self.mark_block_edges(x0, y0, tu_size, 2);
        }

        // cu_qp_delta
        if self.pps.cu_qp_delta_enabled && need_qp && !self.is_cu_qp_delta_coded {
            self.cu_qp_delta_val = self.decode_cu_qp_delta();
            self.is_cu_qp_delta_coded = true;
            // Recompute QpY for the QG. Keep the arithmetic in i64 so a
            // malformed/fuzzer delta cannot overflow before the final QpY clamp.
            self.cur_qp =
                derive_qpy_from_delta(self.predict_qp_cur(), self.cu_qp_delta_val, self.bd);
        }

        // luma residual + reconstruction
        let luma_mode = self.luma_mode_at(x0, y0, luma_modes, blk_idx);
        if cbf_luma {
            let scan = luma_scan(luma_mode, log2_ts);
            let ts_ctx = if self.pps.transform_skip_enabled && !self.cu_tqb && log2_ts == 2 {
                Some(0)
            } else {
                None
            };
            let mut coeffs = std::mem::take(&mut self.coeff_scratch);
            let (transform_skip, max_x, _last_y, max_abs_level) = residual_coding(
                &mut self.cab,
                &mut self.ctx,
                self.exec.residual_scans,
                log2_ts,
                true,
                scan,
                self.sign_hiding,
                ts_ctx,
                self.cu_tqb,
                &mut coeffs,
            );
            self.reconstruct_luma(
                x0,
                y0,
                log2_ts,
                luma_mode,
                &coeffs,
                max_x + 1,
                max_abs_level,
                transform_skip,
            );
            self.coeff_scratch = coeffs;
        } else if !self.cur_cu_inter {
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
        // cu_qp_delta_abs: prefix TU (cMax=5) ctx[0] then ctx[1], then bypass EG0.
        // Use wide/saturating arithmetic so malformed EG0 suffixes cannot poison
        // QpY state through debug-overflow panics before derive_qpy_from_delta().
        let mut prefix = 0;
        while prefix < 5 {
            let ci = if prefix == 0 { 0 } else { 1 };
            if self.cab.decode_bin(&mut self.ctx.cu_qp_delta_abs[ci]) == 0 {
                break;
            }
            prefix += 1;
        }
        let mut abs_val = i64::from(prefix);
        if prefix >= 5 {
            // EG0 suffix (bypass). A valid stream should stay tiny here; cap the
            // fuzz-only runaway case at an i32-representable delta.
            let mut k = 0u32;
            while self.cab.decode_bypass() != 0 {
                if k >= 30 {
                    break;
                }
                k += 1;
            }
            let mut suffix = 0i64;
            for _ in 0..k {
                suffix = (suffix << 1) | i64::from(self.cab.decode_bypass());
            }
            abs_val = abs_val
                .saturating_add(suffix)
                .saturating_add((1i64 << k).saturating_sub(1));
        }
        let abs_val = abs_val.min(i64::from(i32::MAX)) as i32;
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

impl<'cab> FullDecoder<'cab> {
    /// True when luma pixel `(x, y)` lies in the same tile as the CTB currently
    /// being decoded. Always true when tiles are disabled.
    #[inline]
    /// Whether the neighboring CTB `(nrx, nry)` is available for SAO merge from
    /// the current CTB `(rx, ry)`: it must be in the same tile and the same
    /// slice (§6.4.1). Uses per-CTB tile ids and the per-4×4 slice-index map.
    fn ctb_merge_avail(&self, rx: usize, ry: usize, nrx: usize, nry: usize) -> bool {
        // Fast path: no tiles and a single slice segment → the neighbor is
        // always available (the raster/tile-scan caller already guarantees it
        // was decoded). Avoids the per-CTB tile/slice map lookups.
        if self.tiles.is_none() && self.cur_slice_idx <= 1 {
            return true;
        }
        if let Some(g) = &self.tiles
            && g.tile_id_at(nrx, nry) != g.tile_id_at(rx, ry)
        {
            return false;
        }
        // Same slice: the neighbor must belong to the slice segment currently
        // being decoded. Its slice index was recorded when it was decoded; the
        // current CTB's own index is `cur_slice_idx` (its samples aren't marked
        // until its CUs are decoded, after SAO parsing).
        let ctb = 1usize << self.log2_ctb;
        let nbr = self.slice_idx_at(nrx * ctb, nry * ctb);
        nbr == self.cur_slice_idx
    }

    fn same_tile(&self, x: usize, y: usize) -> bool {
        match &self.tiles {
            None => true,
            Some(g) => {
                let ctb = self.log2_ctb;
                g.tile_id_at(x >> ctb, y >> ctb) == self.cur_tile_id
            }
        }
    }

    /// Whether luma sample (x, y) belongs to the current slice segment's slice.
    /// A neighbor in an earlier slice is unavailable for intra prediction and
    /// context derivation (§6.4.1). Cheap no-op for single-slice pictures where
    /// the whole `slice_idx` map stays 0 and `cur_slice_idx <= 1`.
    #[inline]
    fn same_slice(&self, x: usize, y: usize) -> bool {
        if self.cur_slice_idx <= 1 {
            return true;
        }
        self.slice_idx_at(x, y) == self.cur_slice_idx
    }

    /// Constrained intra prediction (§8.4.4.2.1): when the PPS flag is set, a
    /// neighbor coded in an inter mode is treated as unavailable for intra
    /// reference gathering and MPM derivation. No-op when the flag is off.
    #[inline]
    fn constrained_intra_ok(&self, x: usize, y: usize) -> bool {
        if !self.pps._constrained_intra_pred {
            return true;
        }
        self.grid_idx(x, y)
            .and_then(|g| self.motion.get(g))
            .map(|m| m.is_intra)
            .unwrap_or(false)
    }

    fn luma_avail(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return false;
        }
        if !self.same_tile(x as usize, y as usize) {
            return false;
        }
        if !self.same_slice(x as usize, y as usize) {
            return false;
        }
        if !self.constrained_intra_ok(x as usize, y as usize) {
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
        if !self.same_tile(lx, ly) {
            return false;
        }
        if !self.same_slice(lx, ly) {
            return false;
        }
        if !self.constrained_intra_ok(lx, ly) {
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

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_luma(
        &mut self,
        x0: usize,
        y0: usize,
        log2_ts: u32,
        mode: u8,
        levels: &[i32],
        nx: usize,
        max_abs_level: i32,
        transform_skip: bool,
    ) {
        let n = 1usize << log2_ts;
        if !self.cur_cu_inter {
            self.predict_luma_block_into(x0, y0, n, mode);
        } else {
            // Inter: the MC prediction is already in the plane. Copy it into the
            // prediction scratch so the shared residual-add path reconstructs
            // pred+res correctly instead of using a stale intra prediction.
            self.load_plane_pred_luma(x0, y0, n);
        }
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
                let qp_prime_y = qp_prime(self.cur_qp, self.bd);
                let scaling = scaling_matrix_from_lists(
                    self.pps.scaling_list.as_ref(),
                    self.sps.scaling_list.as_ref(),
                    0,
                    n,
                );
                if transform_skip {
                    dequantize_transform_skip_scaled_into_i16(
                        &self.exec,
                        levels,
                        n,
                        qp_prime_y,
                        self.bd,
                        max_abs_level,
                        scaling,
                        &mut self.res_scratch16[..n * n],
                    );
                } else {
                    dequantize_scaled_into_i16(
                        &self.exec,
                        levels,
                        n,
                        qp_prime_y,
                        self.bd,
                        max_abs_level,
                        scaling,
                        &mut self.deq_scratch16[..n * n],
                    );
                    if n == 4 && !self.cur_cu_inter {
                        (self.exec.inv_transform_dst4_16)(
                            &self.deq_scratch16[..n * n],
                            self.bd,
                            &mut self.res_scratch16[..n * n],
                        );
                    } else {
                        (self.exec.inv_transform16)(
                            &self.deq_scratch16[..n * n],
                            n,
                            self.bd,
                            nx,
                            &mut self.res_scratch16[..n * n],
                        );
                    }
                }
            }
            let pred = &self.scratch.pred[..n * n];
            let res = &self.res_scratch16[..n * n];
            if valid_w != 0
                && valid_h != 0
                && let Some(dst) = plane_tail_mut(&mut self.y, stride, x0, y0)
            {
                add_residual_into_i16(
                    &self.exec, dst, stride, pred, res, n, valid_w, valid_h, self.bd,
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
            let qp_prime_y = qp_prime(self.cur_qp, self.bd);
            let scaling = scaling_matrix_from_lists(
                self.pps.scaling_list.as_ref(),
                self.sps.scaling_list.as_ref(),
                0,
                n,
            );
            if transform_skip {
                dequantize_transform_skip_scaled_into_i32(
                    &self.exec,
                    levels,
                    n,
                    qp_prime_y,
                    self.bd,
                    max_abs_level,
                    scaling,
                    &mut self.res_scratch[..n * n],
                );
            } else {
                dequantize_scaled_into_i32(
                    &self.exec,
                    levels,
                    n,
                    qp_prime_y,
                    self.bd,
                    max_abs_level,
                    scaling,
                    &mut self.deq_scratch[..n * n],
                );
                if n == 4 && !self.cur_cu_inter {
                    (self.exec.inv_transform_dst4)(
                        &self.deq_scratch[..n * n],
                        self.bd,
                        &mut self.res_scratch[..n * n],
                    );
                } else {
                    (self.exec.inv_transform)(
                        &self.deq_scratch[..n * n],
                        n,
                        self.bd,
                        nx,
                        &mut self.res_scratch[..n * n],
                    );
                }
            }
        }
        let pred = &self.scratch.pred[..n * n];
        let res = &self.res_scratch[..n * n];
        if valid_w != 0
            && valid_h != 0
            && let Some(dst) = plane_tail_mut(&mut self.y, stride, x0, y0)
        {
            add_residual_into_i32(
                &self.exec, dst, stride, pred, res, n, valid_w, valid_h, self.bd,
            );
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

    /// Copy the current luma plane content at (x0,y0) into the prediction
    /// scratch (row-major n×n), zero-padding out-of-frame samples. Used for
    /// inter reconstruction where the MC prediction is already in the plane.
    fn load_plane_pred_luma(&mut self, x0: usize, y0: usize, n: usize) {
        let stride = self.w;
        let pred = &mut self.scratch.pred[..n * n];
        for row in 0..n {
            let py = y0 + row;
            for col in 0..n {
                let px = x0 + col;
                pred[row * n + col] = if px < self.w && py < self.h {
                    self.y[py * stride + px]
                } else {
                    0
                };
            }
        }
    }

    /// Chroma counterpart of [`load_plane_pred_luma`].
    fn load_plane_pred_chroma(&mut self, is_cb: bool, cx0: usize, cy0: usize, n: usize) {
        let stride = self.cw;
        let pred = &mut self.scratch.pred[..n * n];
        for row in 0..n {
            let py = cy0 + row;
            for col in 0..n {
                let px = cx0 + col;
                pred[row * n + col] = if px < self.cw && py < self.ch {
                    if is_cb {
                        self.cb[py * stride + px]
                    } else {
                        self.cr[py * stride + px]
                    }
                } else {
                    0
                };
            }
        }
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
        (self.exec.predict)(
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
        let cx0 = ((lx as u32) / self.sub_w_div) as usize;
        let cy0 = ((ly as u32) / self.sub_h_div) as usize;
        // 4:2:2 stacks two square chroma TBs vertically per luma TB (ChromaArrayType
        // 2); 4:2:0 and 4:4:4 have a single chroma TB. The bitstream codes them
        // component-major: all Cb TBs, then all Cr TBs (HEVC §7.3.8.11). Each TB is
        // reconstructed before the next so a lower stacked TB can use the upper one
        // as its intra above-reference.
        let n_tb = if idc == 2 { 2 } else { 1 };
        let scan = chroma_scan(mode, clog2, idc == 3);

        let qp_prime_cb = qp_prime(
            qpc(
                self.cur_qp + self.pps.cb_qp_offset + self.slice_cb_qp_offset,
                idc,
                self.bd_c,
            ),
            self.bd_c,
        );
        for (t, &cb) in cbf_cb[..n_tb].iter().enumerate() {
            let ty = cy0 + t * cn;
            if cb {
                let mut coeffs = std::mem::take(&mut self.coeff_scratch);
                let ts_ctx = if self.pps.transform_skip_enabled && !self.cu_tqb && clog2 == 2 {
                    Some(1)
                } else {
                    None
                };
                let (transform_skip, max_x, _, max_abs_level) = residual_coding(
                    &mut self.cab,
                    &mut self.ctx,
                    self.exec.residual_scans,
                    clog2,
                    false,
                    scan,
                    self.sign_hiding,
                    ts_ctx,
                    self.cu_tqb,
                    &mut coeffs,
                );
                self.reconstruct_chroma(
                    true,
                    cx0,
                    ty,
                    cn,
                    mode,
                    &coeffs,
                    qp_prime_cb,
                    max_x + 1,
                    max_abs_level,
                    transform_skip,
                );
                self.coeff_scratch = coeffs;
            } else if !self.cur_cu_inter {
                self.predict_only_chroma(true, cx0, ty, cn, mode);
            }
        }
        let qp_prime_cr = qp_prime(
            qpc(
                self.cur_qp + self.pps.cr_qp_offset + self.slice_cr_qp_offset,
                idc,
                self.bd_c,
            ),
            self.bd_c,
        );
        for (t, &cr) in cbf_cr[..n_tb].iter().enumerate() {
            let ty = cy0 + t * cn;
            if cr {
                let mut coeffs = std::mem::take(&mut self.coeff_scratch);
                let ts_ctx = if self.pps.transform_skip_enabled && !self.cu_tqb && clog2 == 2 {
                    Some(1)
                } else {
                    None
                };
                let (transform_skip, max_x, _, max_abs_level) = residual_coding(
                    &mut self.cab,
                    &mut self.ctx,
                    self.exec.residual_scans,
                    clog2,
                    false,
                    scan,
                    self.sign_hiding,
                    ts_ctx,
                    self.cu_tqb,
                    &mut coeffs,
                );
                self.reconstruct_chroma(
                    false,
                    cx0,
                    ty,
                    cn,
                    mode,
                    &coeffs,
                    qp_prime_cr,
                    max_x + 1,
                    max_abs_level,
                    transform_skip,
                );
                self.coeff_scratch = coeffs;
            } else if !self.cur_cu_inter {
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
            .checked_shl(self.bd_c.saturating_sub(1) as u32)
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
            (self.exec.predict)(
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
            (self.exec.predict)(
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
        qp_prime: i32,
        nx: usize,
        max_abs_level: i32,
        transform_skip: bool,
    ) {
        if !self.cur_cu_inter {
            self.predict_chroma_block_into(is_cb, cx0, cy0, n, mode);
        } else {
            self.load_plane_pred_chroma(is_cb, cx0, cy0, n);
        }
        let n2 = n * n;
        let component = if is_cb { 1 } else { 2 };
        let scaling = scaling_matrix_from_lists(
            self.pps.scaling_list.as_ref(),
            self.sps.scaling_list.as_ref(),
            component,
            n,
        );
        let stride = self.cw;
        let valid_w = self.cw.saturating_sub(cx0).min(n);
        let valid_h = self.ch.saturating_sub(cy0).min(n);
        if self.bd_c <= 8 {
            if self.cu_tqb {
                for (o, &l) in self.res_scratch16[..n2].iter_mut().zip(levels.iter()) {
                    *o = l.clamp(-32768, 32767) as i16;
                }
            } else {
                if transform_skip {
                    dequantize_transform_skip_scaled_into_i16(
                        &self.exec,
                        levels,
                        n,
                        qp_prime,
                        self.bd_c,
                        max_abs_level,
                        scaling,
                        &mut self.res_scratch16[..n2],
                    );
                } else {
                    dequantize_scaled_into_i16(
                        &self.exec,
                        levels,
                        n,
                        qp_prime,
                        self.bd_c,
                        max_abs_level,
                        scaling,
                        &mut self.deq_scratch16[..n2],
                    );
                    (self.exec.inv_transform16)(
                        &self.deq_scratch16[..n2],
                        n,
                        self.bd_c,
                        nx,
                        &mut self.res_scratch16[..n2],
                    );
                }
            }
            let pred = &self.scratch.pred[..n2];
            let res = &self.res_scratch16[..n2];
            if valid_w != 0 && valid_h != 0 {
                let plane = if is_cb { &mut self.cb } else { &mut self.cr };
                if let Some(dst) = plane_tail_mut(plane, stride, cx0, cy0) {
                    add_residual_into_i16(
                        &self.exec, dst, stride, pred, res, n, valid_w, valid_h, self.bd_c,
                    );
                }
            }
            return;
        }
        if self.cu_tqb {
            // Lossless: chroma residual is the parsed levels verbatim.
            self.res_scratch[..n2].copy_from_slice(&levels[..n2]);
        } else {
            if transform_skip {
                dequantize_transform_skip_scaled_into_i32(
                    &self.exec,
                    levels,
                    n,
                    qp_prime,
                    self.bd_c,
                    max_abs_level,
                    scaling,
                    &mut self.res_scratch[..n2],
                );
            } else {
                dequantize_scaled_into_i32(
                    &self.exec,
                    levels,
                    n,
                    qp_prime,
                    self.bd_c,
                    max_abs_level,
                    scaling,
                    &mut self.deq_scratch[..n2],
                );
                (self.exec.inv_transform)(
                    &self.deq_scratch[..n2],
                    n,
                    self.bd_c,
                    nx,
                    &mut self.res_scratch[..n2],
                );
            }
        }
        let pred = &self.scratch.pred[..n2];
        let res = &self.res_scratch[..n2];
        if valid_w != 0 && valid_h != 0 {
            let plane = if is_cb { &mut self.cb } else { &mut self.cr };
            if let Some(dst) = plane_tail_mut(plane, stride, cx0, cy0) {
                add_residual_into_i32(
                    &self.exec, dst, stride, pred, res, n, valid_w, valid_h, self.bd_c,
                );
            }
        }
    }

    fn predict_only_chroma(&mut self, is_cb: bool, cx0: usize, cy0: usize, n: usize, mode: u8) {
        self.predict_chroma_block_into(is_cb, cx0, cy0, n, mode);
        let n2 = n * n;
        self.chroma_scratch[..n2].copy_from_slice(&self.scratch.pred[..n2]);
        let stride = self.cw;
        let valid_w = self.cw.saturating_sub(cx0).min(n);
        let valid_h = self.ch.saturating_sub(cy0).min(n);
        if valid_w != 0 && valid_h != 0 {
            let plane = if is_cb { &mut self.cb } else { &mut self.cr };
            let pred = &self.chroma_scratch[..n2];
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
    pcm: (*mut bool, usize),
    edge_v: (*mut bool, usize),
    edge_h: (*mut bool, usize),
    bs_v: (*mut u8, usize),
    bs_h: (*mut u8, usize),
    nz_coeff: (*mut bool, usize),
    tu_edge_v: (*mut bool, usize),
    tu_edge_h: (*mut bool, usize),
    slice_idx: (*mut u16, usize),
    ct_depth: (*mut u8, usize),
    qp_y_map: (*mut i16, usize),
    sao: (*mut SaoCtb, usize),
    sps: Sps,
    pps: Pps,
    exec: ExecContext,
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    sub_w: usize,
    sub_h: usize,
    sub_w_div: FastDivU32,
    sub_h_div: FastDivU32,
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
    // ---- Inter-prediction state (shared across wavefront rows) --------------
    /// Per-4×4 motion field, shared like the picture planes: the 2-CTB lag makes
    /// the above row's motion visible before the current row reads it for MVP.
    motion: (*mut MotionInfo, usize),
    slice_type: u8,
    cabac_init: bool,
    cur_poc: i32,
    mvd_l1_zero: bool,
    temporal_mvp: bool,
    max_num_merge_cand: usize,
    collocated_from_l0: bool,
    collocated_ref_idx: usize,
    ref_list0: Vec<crate::dpb::RefEntry>,
    ref_list1: Vec<crate::dpb::RefEntry>,
    ref_frames: Vec<crate::inter::RefFramePlanes>,
    pred_weights: Option<crate::inter::PredWeightTable>,
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
    pub(crate) unsafe fn make<'row>(
        &self,
        row_cabac: &'row [u8],
        ctx: ContextSet,
        ictx: IntraModeContexts,
    ) -> Result<FullDecoder<'row>, DecodeError> {
        let cab = CabacDecoder::new_borrowed(row_cabac)
            .map_err(|_| DecodeError::Bitstream("row cabac init".into()))?;
        let mk = |p: (*mut u16, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        let mk8 = |p: (*mut u8, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        let mkb = |p: (*mut bool, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        let mki = |p: (*mut i16, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        let mku16 = |p: (*mut u16, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        let mks = |p: (*mut SaoCtb, usize)| unsafe { crate::plane::Plane::shared(p.0, p.1) };
        Ok(FullDecoder {
            cab,
            ctx,
            ictx,
            sps: self.sps.clone(),
            pps: self.pps.clone(),
            exec: self.exec.clone(),
            y: mk(self.y),
            cb: mk(self.cb),
            cr: mk(self.cr),
            w: self.w,
            h: self.h,
            cw: self.cw,
            ch: self.ch,
            sub_w: self.sub_w,
            sub_h: self.sub_h,
            sub_w_div: self.sub_w_div,
            sub_h_div: self.sub_h_div,
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
            pcm: mkb(self.pcm),
            edge_v: mkb(self.edge_v),
            edge_h: mkb(self.edge_h),
            bs_v: mk8(self.bs_v),
            bs_h: mk8(self.bs_h),
            nz_coeff: mkb(self.nz_coeff),
            tu_edge_v: mkb(self.tu_edge_v),
            tu_edge_h: mkb(self.tu_edge_h),
            // WPP rows are intra I-slices with no inter PU edges.
            pu_edge_v: crate::plane::Plane::owned(vec![false; self.grid_w * self.grid_h]),
            pu_edge_h: crate::plane::Plane::owned(vec![false; self.grid_w * self.grid_h]),
            slice_idx: mku16(self.slice_idx),
            cur_slice_idx: 1,
            // WPP rows belong to a single slice; cross-slice filtering is not
            // gated within a wavefront. Default both entries to "allow".
            slice_lf_across: vec![true, true],
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
            // WPP and tiles are mutually exclusive; row decoders never see tiles.
            tiles: None,
            cur_tile_id: 0,
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
            deq_scratch: Box::new([0i32; 1024]),
            res_scratch: Box::new([0i32; 1024]),
            deq_scratch16: Box::new([0i16; 1024]),
            res_scratch16: Box::new([0i16; 1024]),
            coeff_scratch: vec![0i32; 1024],
            strong_smoothing: self.strong_smoothing,
            slice_type: self.slice_type,
            cabac_init: self.cabac_init,
            pred_weights: self.pred_weights.clone(),
            motion: unsafe { crate::plane::Plane::shared(self.motion.0, self.motion.1) },
            ref_list0: self.ref_list0.clone(),
            ref_list1: self.ref_list1.clone(),
            cur_poc: self.cur_poc,
            mvd_l1_zero: self.mvd_l1_zero,
            temporal_mvp: self.temporal_mvp,
            max_num_merge_cand: self.max_num_merge_cand,
            cur_cu_inter: false,
            last_pu_merge: false,
            cu_skip_map: Vec::new(),
            collocated_from_l0: self.collocated_from_l0,
            collocated_ref_idx: self.collocated_ref_idx,
            ref_frames: self.ref_frames.clone(),
            mc_pred0: Vec::new(),
            mc_pred1: Vec::new(),
            mc_tmp: Vec::new(),
            chroma_scratch: Box::new([0; 1024]),
        })
    }
}

#[inline]
fn qp_bd_offset(bit_depth: u8) -> i32 {
    6 * (bit_depth as i32 - 8)
}

#[inline]
fn qpy_min(bit_depth: u8) -> i32 {
    -qp_bd_offset(bit_depth)
}

#[inline]
fn clamp_qpy(qp: i32, bit_depth: u8) -> i32 {
    qp.clamp(qpy_min(bit_depth), 51)
}

#[inline]
fn derive_qpy_from_delta(prev: i32, delta: i32, bit_depth: u8) -> i32 {
    let off = i64::from(qp_bd_offset(bit_depth));
    let modulus = 52 + off;
    let qp = (i64::from(prev) + i64::from(delta) + 52 + 2 * off).rem_euclid(modulus) - off;
    clamp_qpy(qp as i32, bit_depth)
}

#[inline]
fn qp_prime(qp: i32, bit_depth: u8) -> i32 {
    let off = qp_bd_offset(bit_depth);
    (qp + off).clamp(0, 51 + off)
}

#[inline]
fn scaling_matrix_from_lists<'a>(
    pps_scaling_list: Option<&'a ScalingList>,
    sps_scaling_list: Option<&'a ScalingList>,
    component: usize,
    n: usize,
) -> Option<transform::ScalingMatrix<'a>> {
    let lists = pps_scaling_list.or(sps_scaling_list)?;
    let size_id = (n as u32).trailing_zeros().saturating_sub(2) as usize;
    let matrix_id = component.min(2);
    let (coeffs, dc, flat_16) = lists.matrix(size_id, matrix_id);
    Some(transform::ScalingMatrix::new(coeffs, dc, n, flat_16))
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn dequantize_scaled_into_i32(
    exec: &ExecContext,
    levels: &[i32],
    n: usize,
    qp_prime: i32,
    bit_depth: u8,
    max_abs_level: i32,
    scaling: Option<transform::ScalingMatrix<'_>>,
    out: &mut [i32],
) {
    let params = transform::dequant_params(n, qp_prime, bit_depth);
    match scaling {
        Some(scaling) if !scaling.is_flat_16() => {
            (exec.dequant_scaled)(levels, n, params, scaling, max_abs_level, out)
        }
        _ => (exec.dequant)(levels, n, params, max_abs_level, out),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn dequantize_scaled_into_i16(
    exec: &ExecContext,
    levels: &[i32],
    n: usize,
    qp_prime: i32,
    bit_depth: u8,
    max_abs_level: i32,
    scaling: Option<transform::ScalingMatrix<'_>>,
    out: &mut [i16],
) {
    let params = transform::dequant_params(n, qp_prime, bit_depth);
    match scaling {
        Some(scaling) if !scaling.is_flat_16() => {
            (exec.dequant_scaled16)(levels, n, params, scaling, max_abs_level, out)
        }
        _ => (exec.dequant16)(levels, n, params, max_abs_level, out),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn dequantize_transform_skip_scaled_into_i32(
    exec: &ExecContext,
    levels: &[i32],
    n: usize,
    qp_prime: i32,
    bit_depth: u8,
    max_abs_level: i32,
    scaling: Option<transform::ScalingMatrix<'_>>,
    out: &mut [i32],
) {
    debug_assert!(
        n == 4,
        "HEVC transform_skip_flag is only signalled for 4x4 TUs"
    );
    let params = transform::transform_skip_params(n, qp_prime, bit_depth);
    match scaling {
        Some(scaling) if !scaling.is_flat_16() => {
            (exec.dequant_skip_scaled)(levels, n, params, scaling, max_abs_level, out)
        }
        _ => (exec.dequant_skip)(levels, n, params, max_abs_level, out),
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn dequantize_transform_skip_scaled_into_i16(
    exec: &ExecContext,
    levels: &[i32],
    n: usize,
    qp_prime: i32,
    bit_depth: u8,
    max_abs_level: i32,
    scaling: Option<transform::ScalingMatrix<'_>>,
    out: &mut [i16],
) {
    debug_assert!(
        n == 4,
        "HEVC transform_skip_flag is only signalled for 4x4 TUs"
    );
    let params = transform::transform_skip_params(n, qp_prime, bit_depth);
    match scaling {
        Some(scaling) if !scaling.is_flat_16() => {
            (exec.dequant_skip_scaled16)(levels, n, params, scaling, max_abs_level, out)
        }
        _ => (exec.dequant_skip16)(levels, n, params, max_abs_level, out),
    }
}

/// Inter CU partition modes (§7.4.9.5).
#[derive(Clone, Copy, PartialEq, Eq)]
enum InterPartMode {
    P2Nx2N,
    P2NxN,
    PNx2N,
    PNxN,
    P2NxnU,
    P2NxnD,
    PnLx2N,
    PnRx2N,
}

impl InterPartMode {
    /// PU rectangles (x, y, w, h) covering the CU at (x0,y0) size cb.
    fn pu_rects(self, x0: usize, y0: usize, cb: usize) -> Vec<(usize, usize, usize, usize)> {
        let h = cb / 2;
        let q = cb / 4;
        match self {
            InterPartMode::P2Nx2N => vec![(x0, y0, cb, cb)],
            InterPartMode::P2NxN => vec![(x0, y0, cb, h), (x0, y0 + h, cb, h)],
            InterPartMode::PNx2N => vec![(x0, y0, h, cb), (x0 + h, y0, h, cb)],
            InterPartMode::PNxN => vec![
                (x0, y0, h, h),
                (x0 + h, y0, h, h),
                (x0, y0 + h, h, h),
                (x0 + h, y0 + h, h, h),
            ],
            InterPartMode::P2NxnU => vec![(x0, y0, cb, q), (x0, y0 + q, cb, cb - q)],
            InterPartMode::P2NxnD => vec![(x0, y0, cb, cb - q), (x0, y0 + cb - q, cb, q)],
            InterPartMode::PnLx2N => vec![(x0, y0, q, cb), (x0 + q, y0, cb - q, cb)],
            InterPartMode::PnRx2N => vec![(x0, y0, cb - q, cb), (x0 + cb - q, y0, q, cb)],
        }
    }
}

/// Neighbor-motion accessor bridging the FullDecoder's per-4x4 motion field and
/// reference frames to the `motion` derivation module.
struct DecoderNeighbors<'a> {
    motion: &'a [MotionInfo],
    decoded: &'a crate::plane::Plane<bool>,
    slice_idx: &'a crate::plane::Plane<u16>,
    cur_slice: u16,
    grid_w: usize,
    w: usize,
    h: usize,
    cur_poc: i32,
    ref_frames: &'a [crate::inter::RefFramePlanes],
    ref_list0: &'a [crate::dpb::RefEntry],
    ref_list1: &'a [crate::dpb::RefEntry],
    collocated_from_l0: bool,
    collocated_ref_idx: usize,
    log2_ctb: u32,
}

impl<'a> DecoderNeighbors<'a> {
    #[inline]
    fn grid(&self, x: usize, y: usize) -> Option<usize> {
        if x >= self.w || y >= self.h {
            return None;
        }
        Some((y >> 2) * self.grid_w + (x >> 2))
    }
}

impl<'a> crate::motion::Neighbors for DecoderNeighbors<'a> {
    fn available(&self, x: isize, y: isize) -> bool {
        if x < 0 || y < 0 {
            return false;
        }
        let (x, y) = (x as usize, y as usize);
        match self.grid(x, y) {
            Some(g) => {
                self.decoded.get(g).copied().unwrap_or(false)
                    && (self.cur_slice <= 1
                        || self.slice_idx.get(g).copied().unwrap_or(u16::MAX) == self.cur_slice)
            }
            None => false,
        }
    }

    fn motion_at(&self, x: isize, y: isize) -> Option<MotionInfo> {
        if !self.available(x, y) {
            return None;
        }
        let g = self.grid(x as usize, y as usize)?;
        self.motion.get(g).copied()
    }

    fn temporal(
        &self,
        x: usize,
        y: usize,
        list: usize,
        ref_poc: i32,
        cur_poc: i32,
    ) -> Option<crate::inter::Mv> {
        // Use the collocated reference picture (collocated_from_l0 + collocated_ref_idx).
        let col = if self.collocated_from_l0 {
            self.ref_list0.get(self.collocated_ref_idx)
        } else {
            self.ref_list1.get(self.collocated_ref_idx)
        }?;
        let frame = self.ref_frames.iter().find(|f| f.poc == col.poc)?;
        if frame.width4 == 0 {
            return None;
        }
        // The collocated luma position is masked to a 16×16 grid (§8.5.3.2.8):
        // xColPb = (x >> 4) << 4, yColPb = (y >> 4) << 4. The bottom-right
        // candidate is only valid when it lies inside the picture and within the
        // current CTB row; otherwise the caller must fall back to the center. We
        // therefore reject (return None) instead of clamping to the edge.
        if x >= self.w || y >= self.h {
            return None;
        }
        let xcol = (x >> 4) << 4;
        let ycol = (y >> 4) << 4;
        let bx = xcol >> 2;
        let by = ycol >> 2;
        if bx >= frame.width4 || by >= frame.height4 {
            return None;
        }
        let m = frame.motion.get(by * frame.width4 + bx)?;
        if m.is_intra {
            return None;
        }
        // Collocated MV list selection (§8.5.3.2.9). Pick which of the
        // collocated block's motion vectors to use:
        //  - if it used only L1, take L1; only L0, take L0;
        //  - if bi-predicted: when every active reference of the current slice
        //    precedes the current picture, use the target list `list`;
        //    otherwise use collocated_from_l0_flag.
        let (mv_col, col_ref_poc) = if !m.pred.l0 {
            (m.mv[1], m.ref_poc[1])
        } else if !m.pred.l1 {
            (m.mv[0], m.ref_poc[0])
        } else {
            let all_before = self.ref_list0.iter().all(|r| r.poc < cur_poc)
                && self.ref_list1.iter().all(|r| r.poc < cur_poc);
            // §8.5.3.2.9: bi-predicted collocated block. When every active
            // reference precedes the current picture, use the target list;
            // otherwise use collocated_from_l0_flag as the list index directly.
            let sel = if all_before {
                list
            } else {
                self.collocated_from_l0 as usize
            };
            (m.mv[sel], m.ref_poc[sel])
        };
        // §8.5.3.2.9: when the collocated and current reference distances are
        // equal (or a long-term ref is involved), use the MV unscaled.
        let col_dist = col.poc - col_ref_poc;
        let cur_dist = cur_poc - ref_poc;
        if col_dist == cur_dist {
            return Some(mv_col);
        }
        Some(crate::motion::scale_mv(mv_col, col_dist, cur_dist))
    }

    fn cur_poc(&self) -> i32 {
        self.cur_poc
    }

    fn ctb_log2(&self) -> u32 {
        self.log2_ctb
    }
}

impl<'cab> FullDecoder<'cab> {
    /// cu_skip_flag with the neighbor-based context increment (§9.3.4.2.2).
    fn decode_cu_skip_flag(&mut self, x0: usize, y0: usize) -> bool {
        let mut inc = 0usize;
        if x0 >= 4
            && let Some(g) = self.grid_idx(x0 - 1, y0)
            && self.decoded[g]
            && self.motion_is_skip(g)
        {
            inc += 1;
        }
        if y0 >= 4
            && let Some(g) = self.grid_idx(x0, y0 - 1)
            && self.decoded[g]
            && self.motion_is_skip(g)
        {
            inc += 1;
        }
        self.cab.decode_bin(&mut self.ctx.cu_skip_flag[inc]) != 0
    }

    #[inline]
    fn motion_is_skip(&self, g: usize) -> bool {
        self.cu_skip_map.get(g).copied().unwrap_or(false)
    }

    /// Store a PU's motion across its covered 4x4 blocks.
    fn store_motion(
        &mut self,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        mi: crate::inter::MotionInfo,
    ) {
        // Motion-based deblock boundary strength (§8.7.2.4): before overwriting
        // the motion field, mark BS=1 on this PU's left/top edges where the
        // already-decoded neighbor differs enough (different ref picture,
        // different number of MVs, or |ΔMV| ≥ 4 quarter-pel in x or y).
        self.mark_motion_bs(x0, y0, w, h, &mi);
        for yy in (y0..y0 + h).step_by(4) {
            for xx in (x0..x0 + w).step_by(4) {
                if let Some(g) = self.grid_idx(xx, yy)
                    && g < self.motion.len()
                {
                    self.motion[g] = mi;
                }
            }
        }
    }

    /// Mark BS=1 on a PU's left (vertical) and top (horizontal) 8-grid edges
    /// where the neighboring block's motion differs enough to require filtering.
    fn mark_motion_bs(
        &mut self,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        mi: &crate::inter::MotionInfo,
    ) {
        let gw = self.grid_w;
        // Left edge at x0 (vertical boundary), on the 8×8 deblock grid.
        if x0 > 0 && x0.is_multiple_of(8) {
            let gx = x0 / 4;
            let mut yy = y0;
            while yy < (y0 + h).min(self.h) {
                let g = (yy / 4) * gw + gx;
                if let Some(gl) = self.grid_idx(x0 - 1, yy)
                    && self.motion_differs(gl, mi)
                    && self.bs_v[g] < 1
                {
                    self.bs_v[g] = 1;
                    self.edge_v[g] = true;
                }
                yy += 4;
            }
        }
        // Top edge at y0 (horizontal boundary).
        if y0 > 0 && y0.is_multiple_of(8) {
            let base = (y0 / 4) * gw;
            let mut xx = x0;
            while xx < (x0 + w).min(self.w) {
                let g = base + xx / 4;
                if let Some(gt) = self.grid_idx(xx, y0 - 1)
                    && self.motion_differs(gt, mi)
                    && self.bs_h[g] < 1
                {
                    self.bs_h[g] = 1;
                    self.edge_h[g] = true;
                }
                xx += 4;
            }
        }
    }

    /// Whether stored motion at grid `g` differs from `mi` enough for BS=1.
    fn motion_differs(&self, g: usize, mi: &MotionInfo) -> bool {
        let n = match self.motion.get(g) {
            Some(m) => *m,
            None => return false,
        };
        if n.is_intra {
            return true;
        }
        let count_a = n.pred.l0 as u8 + n.pred.l1 as u8;
        let count_b = mi.pred.l0 as u8 + mi.pred.l1 as u8;
        if count_a != count_b {
            return true;
        }
        // Compare L0 (P-slice common case); different reference → BS 1.
        if n.pred.l0 && (n.ref_poc[0] != mi.ref_poc[0]) {
            return true;
        }
        let dx = (n.mv[0].x as i32 - mi.mv[0].x as i32).abs();
        let dy = (n.mv[0].y as i32 - mi.mv[0].y as i32).abs();
        dx >= 4 || dy >= 4
    }

    /// Decode an inter (or skip) coding unit: partition, per-PU motion, MC, and
    /// (for non-skip) the inter residual. Prototype: supports PART_2Nx2N and the
    /// symmetric 2NxN/Nx2N partitions; AMP falls back to 2Nx2N geometry.
    fn decode_inter_cu(&mut self, x0: usize, y0: usize, log2_cb: u32, skip: bool) {
        let cb = 1usize << log2_cb;
        self.cur_cu_inter = true;
        // The coding-block boundary is always a transform edge (§8.7.2, mirrored
        // from de265's root markTransformBlockBoundary call): this covers skip
        // CUs and rqt_root_cbf=0 CUs, for which no transform_unit runs.
        self.mark_tu_edges(x0, y0, cb);
        for yy in (y0..y0 + cb).step_by(4) {
            for xx in (x0..x0 + cb).step_by(4) {
                if let Some(g) = self.grid_idx(xx, yy)
                    && g < self.cu_skip_map.len()
                {
                    self.cu_skip_map[g] = skip;
                }
            }
        }

        if skip {
            // Skip: single 2Nx2N merge PU, no residual.
            let mi = self.decode_merge_pu(x0, y0, cb, cb, x0, y0, cb, cb, 0);
            self.motion_compensate(x0, y0, cb, cb, &mi);
            self.store_motion(x0, y0, cb, cb, mi);
            self.finish_inter_cu(x0, y0, cb);
            return;
        }

        // Inter part_mode (§7.4.9.5 / 9.3.4.2). Determines PU geometry.
        let part = self.decode_part_mode_inter(log2_cb);
        let pus = part.pu_rects(x0, y0, cb);
        let npus = pus.len();
        // Mark the internal prediction-unit split boundaries so their boundary
        // strength is evaluated even when they don't coincide with a TU edge
        // (§8.7.2.4 applies at PU *and* TU edges).
        for &(px, py, pw, ph) in pus.iter() {
            if px > x0 {
                self.mark_pu_edge_v(px, py, ph);
            }
            if py > y0 {
                self.mark_pu_edge_h(px, py, pw);
            }
        }
        // Parse all PUs' motion first (bins are contiguous), then reconstruct.
        let mut motions: [crate::inter::MotionInfo; 4] = Default::default();
        for (i, &(px, py, pw, ph)) in pus.iter().enumerate() {
            let mi = self.decode_prediction_unit(px, py, pw, ph, x0, y0, cb, cb, i);
            motions[i] = mi;
        }
        for (i, &(px, py, pw, ph)) in pus.iter().enumerate() {
            self.motion_compensate(px, py, pw, ph, &motions[i]);
            self.store_motion(px, py, pw, ph, motions[i]);
        }
        let _ = npus;

        // rqt_root_cbf (§7.3.8.5): decoded unless the CU is a 2Nx2N merge, in
        // which case it is inferred = true (a 2Nx2N merge with no residual would
        // have been signalled as SKIP instead, so residual must be present).
        let is_2nx2n_merge = part == InterPartMode::P2Nx2N && self.last_pu_merge;
        let rqt_root_cbf = if is_2nx2n_merge {
            true
        } else {
            self.cab.decode_bin(&mut self.ctx.rqt_root_cbf) != 0
        };
        if rqt_root_cbf {
            // MaxTrafoDepth for inter (§7.4.9.8): max_transform_hierarchy_depth_inter
            // plus interSplitFlag (1 when the residual tree must split because the
            // partition is not 2Nx2N and the hierarchy depth is 0).
            let inter_split_flag = (self.sps.max_transform_hierarchy_inter == 0
                && part != InterPartMode::P2Nx2N) as u32;
            let max_depth = self.sps.max_transform_hierarchy_inter + inter_split_flag;
            // Reuse the transform tree; inter uses no intra prediction (modes
            // unused), residual added on top of the MC prediction already in
            // the planes. intra_split=false; the interSplitFlag rule (part!=2Nx2N
            // forcing a split at depth 0) is approximated via max_depth.
            self.transform_tree(
                x0,
                y0,
                x0,
                y0,
                log2_cb,
                0,
                0,
                &[MODE_DC; 4],
                MODE_DC,
                inter_split_flag != 0,
                max_depth,
                [false; 2],
                [false; 2],
            );
        }

        self.finish_inter_cu(x0, y0, cb);
    }

    /// Decode inter `part_mode` (§9.3.4.2.4). Returns the partition shape.
    fn decode_part_mode_inter(&mut self, log2_cb: u32) -> InterPartMode {
        let bit0 = self.cab.decode_bin(&mut self.ctx.part_mode[0]) != 0;
        if bit0 {
            return InterPartMode::P2Nx2N;
        }
        let bit1 = self.cab.decode_bin(&mut self.ctx.part_mode[1]) != 0;
        if log2_cb > self.log2_min_cb {
            if !self.sps.amp_enabled {
                return if bit1 {
                    InterPartMode::P2NxN
                } else {
                    InterPartMode::PNx2N
                };
            }
            let bit3 = self.cab.decode_bin(&mut self.ctx.part_mode[3]) != 0;
            if bit3 {
                return if bit1 {
                    InterPartMode::P2NxN
                } else {
                    InterPartMode::PNx2N
                };
            }
            let bit4 = self.cab.decode_bypass() != 0;
            match (bit1, bit4) {
                (true, true) => InterPartMode::P2NxnD,
                (true, false) => InterPartMode::P2NxnU,
                (false, false) => InterPartMode::PnLx2N,
                (false, true) => InterPartMode::PnRx2N,
            }
        } else {
            if bit1 {
                return InterPartMode::P2NxN;
            }
            if log2_cb == 3 {
                return InterPartMode::PNx2N;
            }
            let bit2 = self.cab.decode_bin(&mut self.ctx.part_mode[2]) != 0;
            if bit2 {
                InterPartMode::PNx2N
            } else {
                InterPartMode::PNxN
            }
        }
    }

    /// Common post-CU bookkeeping shared with the intra path.
    fn finish_inter_cu(&mut self, x0: usize, y0: usize, cb: usize) {
        self.cur_cu_inter = false;
        self.mark_decoded(x0, y0, cb);
        self.set_slice_idx(x0, y0, cb);
        let cur_qp = clamp_qpy(self.cur_qp, self.bd);
        self.qp_y_prev = cur_qp;
        self.cur_qp = cur_qp;
        self.set_qp(x0, y0, cb, cur_qp);
    }

    /// Build the merge candidate list and select `merge_idx` for a skip/merge PU.
    #[allow(clippy::too_many_arguments)]
    fn decode_merge_pu(
        &mut self,
        px: usize,
        py: usize,
        pw: usize,
        ph: usize,
        cux: usize,
        cuy: usize,
        cuw: usize,
        cuh: usize,
        part_idx: usize,
    ) -> crate::inter::MotionInfo {
        let merge_idx = self.decode_merge_idx();
        self.derive_pu_merge(px, py, pw, ph, cux, cuy, cuw, cuh, part_idx, merge_idx)
    }

    fn decode_merge_idx(&mut self) -> usize {
        if self.max_num_merge_cand <= 1 {
            return 0;
        }
        // TR-coded: first bin context, rest bypass, cMax = max_num_merge_cand-1.
        if self.cab.decode_bin(&mut self.ctx.merge_idx) == 0 {
            return 0;
        }
        let mut idx = 1usize;
        while idx < self.max_num_merge_cand - 1 && self.cab.decode_bypass() != 0 {
            idx += 1;
        }
        idx
    }

    #[allow(clippy::too_many_arguments)]
    fn derive_pu_merge(
        &self,
        px: usize,
        py: usize,
        pw: usize,
        ph: usize,
        cux: usize,
        cuy: usize,
        cuw: usize,
        cuh: usize,
        part_idx: usize,
        merge_idx: usize,
    ) -> MotionInfo {
        let nb = self.neighbors();
        let par_mrg_level = self.pps.log2_parallel_merge_level;
        // singleMCLFlag (§8.5.3.2.1): for 8×8 CUs with a parallel merge level > 2
        // every PU derives its merge list as if it were the whole 2Nx2N CU.
        let single_mcl = par_mrg_level > 2 && cuw == 8;
        let (px, py, pw, ph, part_idx) = if single_mcl {
            (cux, cuy, cuw, cuh, 0)
        } else {
            (px, py, pw, ph, part_idx)
        };
        let pu = crate::motion::PuGeom {
            x: px,
            y: py,
            w: pw,
            h: ph,
            is_b: self.slice_type == crate::inter::SLICE_B,
            part_idx,
            cu_w: cuw,
            cu_h: cuh,
            par_mrg_level,
        };
        let cand = crate::motion::derive_merge(
            &nb,
            &pu,
            merge_idx,
            self.max_num_merge_cand,
            self.temporal_mvp,
            &self.ref_list0,
            &self.ref_list1,
        );
        self.cand_to_motion(&cand)
    }

    fn cand_to_motion(&self, c: &crate::motion::MergeCand) -> MotionInfo {
        let mut mi = MotionInfo {
            pred: c.pred,
            mv: c.mv,
            ref_idx: c.ref_idx,
            ..Default::default()
        };
        if c.pred.l0 && (c.ref_idx[0] as usize) < self.ref_list0.len() {
            mi.ref_poc[0] = self.ref_list0[c.ref_idx[0] as usize].poc;
        }
        if c.pred.l1 && (c.ref_idx[1] as usize) < self.ref_list1.len() {
            mi.ref_poc[1] = self.ref_list1[c.ref_idx[1] as usize].poc;
        }
        mi
    }

    fn neighbors(&self) -> DecoderNeighbors<'_> {
        DecoderNeighbors {
            motion: &self.motion,
            log2_ctb: self.log2_ctb,
            decoded: &self.decoded,
            slice_idx: &self.slice_idx,
            cur_slice: self.cur_slice_idx,
            grid_w: self.grid_w,
            w: self.w,
            h: self.h,
            cur_poc: self.cur_poc,
            ref_frames: &self.ref_frames,
            ref_list0: &self.ref_list0,
            ref_list1: &self.ref_list1,
            collocated_from_l0: self.collocated_from_l0,
            collocated_ref_idx: self.collocated_ref_idx,
        }
    }

    /// Decode a non-merge prediction unit: merge_flag, then either merge or the
    /// AMVP path (inter_pred_idc, ref_idx, mvd, mvp_flag).
    #[allow(clippy::too_many_arguments)]
    fn decode_prediction_unit(
        &mut self,
        px: usize,
        py: usize,
        pw: usize,
        ph: usize,
        cux: usize,
        cuy: usize,
        cuw: usize,
        cuh: usize,
        part_idx: usize,
    ) -> MotionInfo {
        let merge = self.cab.decode_bin(&mut self.ctx.merge_flag) != 0;
        self.last_pu_merge = merge;
        if merge {
            return self.decode_merge_pu(px, py, pw, ph, cux, cuy, cuw, cuh, part_idx);
        }
        let is_b = self.slice_type == crate::inter::SLICE_B;
        // inter_pred_idc: PRED_L0 / PRED_L1 / PRED_BI.
        let (use_l0, use_l1) = if is_b {
            let ct_depth = self.log2_ctb.saturating_sub(cuw.trailing_zeros()) as usize;
            self.decode_inter_pred_idc(pw, ph, ct_depth)
        } else {
            (true, false)
        };

        let mut mi = MotionInfo {
            pred: crate::inter::PredFlags {
                l0: use_l0,
                l1: use_l1,
            },
            ..Default::default()
        };

        if use_l0 {
            let (mv, ridx, poc) =
                self.decode_mvd_amvp(px, py, pw, ph, cuw, cuh, part_idx, 0, false);
            mi.mv[0] = mv;
            mi.ref_idx[0] = ridx as i8;
            mi.ref_poc[0] = poc;
        } else {
            mi.ref_idx[0] = -1;
        }
        if use_l1 {
            let zero = self.mvd_l1_zero && use_l0;
            let (mv, ridx, poc) = self.decode_mvd_amvp(px, py, pw, ph, cuw, cuh, part_idx, 1, zero);
            mi.mv[1] = mv;
            mi.ref_idx[1] = ridx as i8;
            mi.ref_poc[1] = poc;
        } else {
            mi.ref_idx[1] = -1;
        }
        mi
    }

    fn decode_inter_pred_idc(&mut self, pw: usize, ph: usize, ct_depth: usize) -> (bool, bool) {
        // §7.3.8.6 / §9.3.4.2.2: for an 8×4 or 4×8 PU (nPbW+nPbH == 12) bi-
        // prediction is disallowed, so only the L0/L1 selection bin (ctx 4) is
        // coded. Otherwise the first bin (ctx = CtDepth of the CU) chooses BI vs
        // uni; on uni, ctx 4 selects the list.
        if pw + ph == 12 {
            let l1 = self.cab.decode_bin(&mut self.ctx.inter_pred_idc[4]) != 0;
            return if l1 { (false, true) } else { (true, false) };
        }
        let bi = self
            .cab
            .decode_bin(&mut self.ctx.inter_pred_idc[ct_depth.min(4)])
            != 0;
        if bi {
            return (true, true);
        }
        let l1 = self.cab.decode_bin(&mut self.ctx.inter_pred_idc[4]) != 0;
        if l1 { (false, true) } else { (true, false) }
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_mvd_amvp(
        &mut self,
        px: usize,
        py: usize,
        pw: usize,
        ph: usize,
        cuw: usize,
        cuh: usize,
        part_idx: usize,
        list: usize,
        mvd_zero: bool,
    ) -> (crate::inter::Mv, usize, i32) {
        let list_len = if list == 0 {
            self.ref_list0.len()
        } else {
            self.ref_list1.len()
        };
        let ref_idx = self.decode_ref_idx(list_len);
        let mvd = if mvd_zero {
            crate::inter::Mv::default()
        } else {
            self.decode_mvd()
        };
        let mvp_flag = self.cab.decode_bin(&mut self.ctx.mvp_flag) != 0;

        let ref_poc = if list == 0 {
            self.ref_list0
                .get(ref_idx)
                .map(|r| r.poc)
                .unwrap_or(self.cur_poc)
        } else {
            self.ref_list1
                .get(ref_idx)
                .map(|r| r.poc)
                .unwrap_or(self.cur_poc)
        };
        let nb = self.neighbors();
        let pu = crate::motion::PuGeom {
            x: px,
            y: py,
            w: pw,
            h: ph,
            is_b: self.slice_type == crate::inter::SLICE_B,
            part_idx,
            cu_w: cuw,
            cu_h: cuh,
            par_mrg_level: self.pps.log2_parallel_merge_level,
        };
        let preds = crate::motion::derive_amvp(
            &nb,
            &pu,
            list,
            ref_poc,
            self.temporal_mvp,
            &self.ref_list0,
            &self.ref_list1,
        );
        let mvp = preds[mvp_flag as usize];
        let mv = crate::inter::Mv::new(mvp.x.wrapping_add(mvd.x), mvp.y.wrapping_add(mvd.y));
        (mv, ref_idx, ref_poc)
    }

    fn decode_ref_idx(&mut self, list_len: usize) -> usize {
        if list_len <= 1 {
            return 0;
        }
        if self.cab.decode_bin(&mut self.ctx.ref_idx[0]) == 0 {
            return 0;
        }
        if list_len == 2 {
            return 1;
        }
        let mut idx = 1usize;
        if self.cab.decode_bin(&mut self.ctx.ref_idx[1]) == 0 {
            return idx;
        }
        idx = 2;
        // Remaining bins are bypass (TR), cMax = list_len-1.
        while idx < list_len - 1 && self.cab.decode_bypass() != 0 {
            idx += 1;
        }
        idx
    }

    /// Decode a motion vector difference (§7.3.8.9).
    fn decode_mvd(&mut self) -> crate::inter::Mv {
        let g0x = self.cab.decode_bin(&mut self.ctx.abs_mvd_greater01[0]) != 0;
        let g0y = self.cab.decode_bin(&mut self.ctx.abs_mvd_greater01[0]) != 0;
        let g1x = if g0x {
            self.cab.decode_bin(&mut self.ctx.abs_mvd_greater01[1]) != 0
        } else {
            false
        };
        let g1y = if g0y {
            self.cab.decode_bin(&mut self.ctx.abs_mvd_greater01[1]) != 0
        } else {
            false
        };
        let x = self.decode_mvd_component(g0x, g1x);
        let y = self.decode_mvd_component(g0y, g1y);
        crate::inter::Mv::new(x as i16, y as i16)
    }

    fn decode_mvd_component(&mut self, greater0: bool, greater1: bool) -> i32 {
        if !greater0 {
            return 0;
        }
        let mut abs = 1i32;
        if greater1 {
            // abs_mvd_minus2 EG1-coded (bypass).
            abs = 2 + self.decode_eg1_bypass() as i32;
        }
        let sign = self.cab.decode_bypass();
        if sign != 0 { -abs } else { abs }
    }

    /// Exp-Golomb order-1 bypass decode (used for abs_mvd_minus2).
    fn decode_eg1_bypass(&mut self) -> u32 {
        let k = 1u32; // order 1
        let mut value = 0u32;
        // prefix: count leading 1s
        let mut lead = 0u32;
        while self.cab.decode_bypass() != 0 {
            lead += 1;
            if lead > 31 {
                break;
            }
        }
        for _ in 0..(lead + k) {
            value = (value << 1) | self.cab.decode_bypass() as u32;
        }
        value + ((1u32 << (lead + k)) - (1u32 << k))
    }

    /// Motion-compensate a PU into the reconstruction planes.
    /// Return the active weighted-prediction table iff the PPS enables it for
    /// this slice type (weighted_pred for P, weighted_bipred for B). Otherwise
    /// default (non-weighted) averaging is used.
    fn luma_weighted(&self) -> Option<&crate::inter::PredWeightTable> {
        let enabled = match self.slice_type {
            crate::inter::SLICE_P => self.pps.weighted_pred,
            crate::inter::SLICE_B => self.pps.weighted_bipred,
            _ => false,
        };
        if enabled {
            self.pred_weights.as_ref()
        } else {
            None
        }
    }

    fn motion_compensate(&mut self, x0: usize, y0: usize, w: usize, h: usize, mi: &MotionInfo) {
        let bd = self.bd;
        let bd_c = self.bd_c;
        let weighted = self.luma_weighted().cloned();

        let len = w.saturating_mul(h);
        if len != 0 {
            self.mc_pred0.resize(len, 0);
            self.mc_pred1.resize(len, 0);

            let luma0 = if mi.pred.l0 {
                if let Some(frame) = ref_frame_for_lists(
                    &self.ref_frames,
                    &self.ref_list0,
                    &self.ref_list1,
                    mi.ref_idx[0],
                    0,
                ) {
                    let rp = crate::mc::RefPlane {
                        data: &frame.y,
                        stride: frame.w,
                        width: frame.w,
                        height: frame.h,
                    };
                    let ix = x0 as isize + (mi.mv[0].x >> 2) as isize;
                    let iy = y0 as isize + (mi.mv[0].y >> 2) as isize;
                    let fx = (mi.mv[0].x & 3) as usize;
                    let fy = (mi.mv[0].y & 3) as usize;
                    (self.exec.motion_luma_interp)(
                        &rp,
                        ix,
                        iy,
                        fx,
                        fy,
                        w,
                        h,
                        bd,
                        &mut self.mc_pred0[..len],
                        &mut self.mc_tmp,
                    );
                    true
                } else {
                    false
                }
            } else {
                false
            };

            let luma1 = if mi.pred.l1 {
                if let Some(frame) = ref_frame_for_lists(
                    &self.ref_frames,
                    &self.ref_list0,
                    &self.ref_list1,
                    mi.ref_idx[1],
                    1,
                ) {
                    let rp = crate::mc::RefPlane {
                        data: &frame.y,
                        stride: frame.w,
                        width: frame.w,
                        height: frame.h,
                    };
                    let ix = x0 as isize + (mi.mv[1].x >> 2) as isize;
                    let iy = y0 as isize + (mi.mv[1].y >> 2) as isize;
                    let fx = (mi.mv[1].x & 3) as usize;
                    let fy = (mi.mv[1].y & 3) as usize;
                    (self.exec.motion_luma_interp)(
                        &rp,
                        ix,
                        iy,
                        fx,
                        fy,
                        w,
                        h,
                        bd,
                        &mut self.mc_pred1[..len],
                        &mut self.mc_tmp,
                    );
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if x0 < self.w && y0 < self.h {
                let valid_w = w.min(self.w - x0);
                let valid_h = h.min(self.h - y0);
                let dst_off = y0 * self.w + x0;
                let dst_stride = self.w;
                let uni_mc = self.exec.motion_uni_mc;
                let bi_mc = self.exec.motion_bi_mc;
                let uni_mc_weighted = self.exec.motion_uni_mc_weighted;
                let bi_mc_weighted = self.exec.motion_bi_mc_weighted;
                let dst = &mut self.y[dst_off..];

                match (luma0, luma1) {
                    (true, true) => {
                        if let Some(wt) = &weighted {
                            let (w0, o0) = wt.luma(0, mi.ref_idx[0]);
                            let (w1, o1) = wt.luma(1, mi.ref_idx[1]);
                            bi_mc_weighted(
                                &self.mc_pred0[..len],
                                &self.mc_pred1[..len],
                                w,
                                h,
                                valid_w,
                                valid_h,
                                bd,
                                w0,
                                o0,
                                w1,
                                o1,
                                wt.luma_log2_denom,
                                dst,
                                dst_stride,
                            );
                        } else {
                            bi_mc(
                                &self.mc_pred0[..len],
                                &self.mc_pred1[..len],
                                w,
                                h,
                                valid_w,
                                valid_h,
                                bd,
                                dst,
                                dst_stride,
                            );
                        }
                    }
                    (true, false) | (false, true) => {
                        let (list, ridx, src) = if luma0 {
                            (0, mi.ref_idx[0], &self.mc_pred0[..len])
                        } else {
                            (1, mi.ref_idx[1], &self.mc_pred1[..len])
                        };
                        if let Some(wt) = &weighted {
                            let (wgt, off) = wt.luma(list, ridx);
                            uni_mc_weighted(
                                src,
                                w,
                                h,
                                valid_w,
                                valid_h,
                                bd,
                                wgt,
                                off,
                                wt.luma_log2_denom,
                                dst,
                                dst_stride,
                            );
                        } else {
                            uni_mc(src, w, h, valid_w, valid_h, bd, dst, dst_stride);
                        }
                    }
                    (false, false) => {
                        // No usable reference (e.g. a reference missing during
                        // random access). Write the defined mid-grey
                        // "unavailable reference" value (§8.3.3) instead of
                        // leaving stale/zero luma so output is deterministic.
                        let gray = 1u16 << bd.saturating_sub(1);
                        for r in 0..valid_h {
                            let row = r * dst_stride;
                            for c in 0..valid_w {
                                dst[row + c] = gray;
                            }
                        }
                    }
                }
            }
        }

        // Chroma motion compensation (chroma MV derived per SubWidthC/SubHeightC).
        if !self.sps.chroma.is_monochrome() {
            let cw = w / self.sub_w;
            let ch = h / self.sub_h;
            let clen = cw.saturating_mul(ch);
            if clen == 0 {
                return;
            }
            let cx = x0 / self.sub_w;
            let cy = y0 / self.sub_h;
            self.mc_pred0.resize(clen, 0);
            self.mc_pred1.resize(clen, 0);

            for plane in 0..2 {
                let c0 = if mi.pred.l0 {
                    if let Some(frame) = ref_frame_for_lists(
                        &self.ref_frames,
                        &self.ref_list0,
                        &self.ref_list1,
                        mi.ref_idx[0],
                        0,
                    ) {
                        let data = if plane == 0 { &frame.cb } else { &frame.cr };
                        let rp = crate::mc::RefPlane {
                            data,
                            stride: frame.cw,
                            width: frame.cw,
                            height: frame.ch,
                        };
                        let mvcx = mi.mv[0].x as isize * 2 / self.sub_w as isize;
                        let mvcy = mi.mv[0].y as isize * 2 / self.sub_h as isize;
                        let ix = cx as isize + (mvcx >> 3);
                        let iy = cy as isize + (mvcy >> 3);
                        let fx = (mvcx & 7) as usize;
                        let fy = (mvcy & 7) as usize;
                        (self.exec.motion_chroma_interp)(
                            &rp,
                            ix,
                            iy,
                            fx,
                            fy,
                            cw,
                            ch,
                            bd_c,
                            &mut self.mc_pred0[..clen],
                            &mut self.mc_tmp,
                        );
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                let c1 = if mi.pred.l1 {
                    if let Some(frame) = ref_frame_for_lists(
                        &self.ref_frames,
                        &self.ref_list0,
                        &self.ref_list1,
                        mi.ref_idx[1],
                        1,
                    ) {
                        let data = if plane == 0 { &frame.cb } else { &frame.cr };
                        let rp = crate::mc::RefPlane {
                            data,
                            stride: frame.cw,
                            width: frame.cw,
                            height: frame.ch,
                        };
                        let mvcx = mi.mv[1].x as isize * 2 / self.sub_w as isize;
                        let mvcy = mi.mv[1].y as isize * 2 / self.sub_h as isize;
                        let ix = cx as isize + (mvcx >> 3);
                        let iy = cy as isize + (mvcy >> 3);
                        let fx = (mvcx & 7) as usize;
                        let fy = (mvcy & 7) as usize;
                        (self.exec.motion_chroma_interp)(
                            &rp,
                            ix,
                            iy,
                            fx,
                            fy,
                            cw,
                            ch,
                            bd_c,
                            &mut self.mc_pred1[..clen],
                            &mut self.mc_tmp,
                        );
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if cx >= self.cw || cy >= self.ch {
                    continue;
                }
                let valid_w = cw.min(self.cw - cx);
                let valid_h = ch.min(self.ch - cy);
                let c_stride = self.cw;
                let dst_off = cy * c_stride + cx;
                let uni_mc = self.exec.motion_uni_mc;
                let bi_mc = self.exec.motion_bi_mc;
                let uni_mc_weighted = self.exec.motion_uni_mc_weighted;
                let bi_mc_weighted = self.exec.motion_bi_mc_weighted;
                let dst = if plane == 0 {
                    &mut self.cb[dst_off..]
                } else {
                    &mut self.cr[dst_off..]
                };

                match (c0, c1) {
                    (true, true) => {
                        if let Some(wt) = &weighted {
                            let (w0, o0) = wt.chroma(0, mi.ref_idx[0], plane);
                            let (w1, o1) = wt.chroma(1, mi.ref_idx[1], plane);
                            bi_mc_weighted(
                                &self.mc_pred0[..clen],
                                &self.mc_pred1[..clen],
                                cw,
                                ch,
                                valid_w,
                                valid_h,
                                bd_c,
                                w0,
                                o0,
                                w1,
                                o1,
                                wt.chroma_log2_denom,
                                dst,
                                c_stride,
                            );
                        } else {
                            bi_mc(
                                &self.mc_pred0[..clen],
                                &self.mc_pred1[..clen],
                                cw,
                                ch,
                                valid_w,
                                valid_h,
                                bd_c,
                                dst,
                                c_stride,
                            );
                        }
                    }
                    (true, false) | (false, true) => {
                        let (list, ridx, src) = if c0 {
                            (0, mi.ref_idx[0], &self.mc_pred0[..clen])
                        } else {
                            (1, mi.ref_idx[1], &self.mc_pred1[..clen])
                        };
                        if let Some(wt) = &weighted {
                            let (wgt, off) = wt.chroma(list, ridx, plane);
                            uni_mc_weighted(
                                src,
                                cw,
                                ch,
                                valid_w,
                                valid_h,
                                bd_c,
                                wgt,
                                off,
                                wt.chroma_log2_denom,
                                dst,
                                c_stride,
                            );
                        } else {
                            uni_mc(src, cw, ch, valid_w, valid_h, bd_c, dst, c_stride);
                        }
                    }
                    (false, false) => {
                        // Missing chroma reference: defined mid-grey (§8.3.3).
                        let gray = 1u16 << bd_c.saturating_sub(1);
                        for r in 0..valid_h {
                            let row = r * c_stride;
                            for cc in 0..valid_w {
                                dst[row + cc] = gray;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Inject the current picture's inter state (POC + reference lists + frames)
    /// before decoding. Called by the video driver after DPB list construction.
    pub(crate) fn set_inter_state(
        &mut self,
        cur_poc: i32,
        ref_list0: Vec<crate::dpb::RefEntry>,
        ref_list1: Vec<crate::dpb::RefEntry>,
        ref_frames: Vec<crate::inter::RefFramePlanes>,
    ) {
        self.cur_poc = cur_poc;
        self.ref_list0 = ref_list0;
        self.ref_list1 = ref_list1;
        self.ref_frames = ref_frames;
    }

    /// Extract the coded picture's motion field and grid dimensions for storage
    /// in the DPB (temporal MV prediction of later pictures).
    pub(crate) fn take_motion(&mut self) -> (Vec<MotionInfo>, usize, usize) {
        let m = std::mem::take(&mut self.motion);
        (m.to_vec(), self.grid_w, self.grid_h)
    }
}

#[inline]
fn ref_frame_for_lists<'a>(
    ref_frames: &'a [crate::inter::RefFramePlanes],
    ref_list0: &[crate::dpb::RefEntry],
    ref_list1: &[crate::dpb::RefEntry],
    ref_idx: i8,
    list: usize,
) -> Option<&'a crate::inter::RefFramePlanes> {
    if ref_idx < 0 {
        return None;
    }
    let entry = if list == 0 {
        ref_list0.get(ref_idx as usize)
    } else {
        ref_list1.get(ref_idx as usize)
    }?;
    ref_frames.iter().find(|f| f.poc == entry.poc)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn add_residual_into_i32(
    exec: &ExecContext,
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i32],
    n: usize,
    valid_w: usize,
    valid_h: usize,
    bit_depth: u8,
) {
    if !crate::reconstruct::can_reconstruct_full_block(
        dst, stride, pred, res, n, valid_w, valid_h, bit_depth,
    ) {
        crate::reconstruct::add_residual_into_scalar(
            dst, stride, pred, res, n, valid_w, valid_h, bit_depth,
        );
        return;
    }
    (exec.reconstruct)(dst, stride, pred, res, n, valid_w, valid_h, bit_depth);
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn add_residual_into_i16(
    exec: &ExecContext,
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i16],
    n: usize,
    valid_w: usize,
    valid_h: usize,
    bit_depth: u8,
) {
    if !crate::reconstruct::can_reconstruct_full_block(
        dst, stride, pred, res, n, valid_w, valid_h, bit_depth,
    ) {
        crate::reconstruct::add_residual_into_scalar16(
            dst, stride, pred, res, n, valid_w, valid_h, bit_depth,
        );
        return;
    }
    (exec.reconstruct16)(dst, stride, pred, res, n, valid_w, valid_h, bit_depth);
}

/// Chroma QP mapping (Table 8-10). The input qPi is a nominal signed QP, so
/// high-bit-depth streams may legitimately pass negative values. The returned
/// QpC is still nominal/signed; callers add QpBdOffsetC with `qp_prime`.
fn qpc(qpi: i32, chroma_idc: u8, bit_depth: u8) -> i32 {
    let qpi = qpi.clamp(-qp_bd_offset(bit_depth), 57);
    if chroma_idc != 1 {
        // 4:2:2 / 4:4:4: QpC = min(qPi, 51), preserving negative QP.
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
    /// 0 = B, 1 = P, 2 = I (matches HEVC `slice_type`).
    pub(crate) slice_type: u8,
    /// Picture order count LSB (`slice_pic_order_cnt_lsb`); 0 for IDR.
    pub(crate) poc_lsb: i32,
    /// Resolved short-term RPS for the current picture (from SPS or inline).
    pub(crate) cur_rps: crate::rps::ShortTermRps,
    /// Long-term reference POCs for this picture. Each is (poc_or_lsb,
    /// used_by_curr, has_msb): when `has_msb` the value is a full POC, otherwise
    /// only the POC LSB is known and matching is by LSB.
    pub(crate) lt_refs: Vec<(i32, bool, bool, i32)>,
    /// num_ref_idx_l0/l1 active minus nothing (already the active counts).
    pub(crate) num_ref_idx_l0: usize,
    pub(crate) num_ref_idx_l1: usize,
    /// Reference list modification indices (empty = identity mapping).
    pub(crate) list_mod_l0: Vec<u32>,
    pub(crate) list_mod_l1: Vec<u32>,
    /// mvd_l1_zero_flag: L1 MVDs are inferred zero for bi-pred.
    pub(crate) mvd_l1_zero: bool,
    /// cabac_init_flag: swap P/B context init tables.
    pub(crate) cabac_init: bool,
    /// slice_temporal_mvp_enabled_flag.
    pub(crate) temporal_mvp: bool,
    /// collocated_from_l0_flag and collocated_ref_idx.
    pub(crate) collocated_from_l0: bool,
    pub(crate) collocated_ref_idx: usize,
    /// five_minus_max_num_merge_cand -> MaxNumMergeCand.
    pub(crate) max_num_merge_cand: usize,
    /// Weighted-prediction table (luma/chroma weights+offsets), if present.
    pub(crate) pred_weights: Option<crate::inter::PredWeightTable>,
    /// slice_loop_filter_across_slices_enabled_flag (inherits the PPS default
    /// when absent). Controls whether the in-loop filters cross this slice's
    /// boundaries.
    pub(crate) slice_loop_filter_across_slices: bool,
    /// pic_output_flag: false suppresses this picture from output.
    pub(crate) pic_output_flag: bool,
    /// WPP/tiles entry-point sub-stream byte lengths, i.e.
    /// `entry_point_offset_minus1[i] + 1` for each `i`. For WPP these are the
    /// byte lengths of every CTB-row sub-stream except the last (whose length is
    /// implied by the end of the CABAC payload). Empty when the stream carries
    /// no entry points. Used to position an independent CABAC engine per row for
    /// the parallel wavefront decode.
    pub(crate) entry_points: Vec<u32>,
}

/// Read just the leading fields of a slice segment header to recover the
/// pic_parameter_set_id it activates, so the caller can select the matching
/// PPS/SPS before the full parse (§7.3.6.1). Returns None on a short read.
pub(crate) fn peek_slice_pps_id(rbsp: &[u8], nal_type: u8) -> Option<u32> {
    let mut r = crate::bitreader::BitReader::new(rbsp);
    r.read_bits(16).ok()?; // NAL header
    r.read_flag().ok()?; // first_slice_segment_in_pic_flag
    if (16..=23).contains(&nal_type) {
        r.read_flag().ok()?; // no_output_of_prior_pics_flag
    }
    r.read_ue().ok()
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
            slice_qp: clamp_qpy(pps.init_qp, sps.bit_depth_luma),
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
            slice_type: crate::inter::SLICE_I,
            poc_lsb: 0,
            cur_rps: crate::rps::ShortTermRps::default(),
            lt_refs: Vec::new(),
            num_ref_idx_l0: 0,
            num_ref_idx_l1: 0,
            list_mod_l0: Vec::new(),
            list_mod_l1: Vec::new(),
            mvd_l1_zero: false,
            cabac_init: false,
            temporal_mvp: false,
            collocated_from_l0: true,
            collocated_ref_idx: 0,
            max_num_merge_cand: 5,
            pred_weights: None,
            slice_loop_filter_across_slices: pps.loop_filter_across_slices,
            pic_output_flag: true,
        });
    }

    for _ in 0..pps.num_extra_slice_header_bits {
        r.read_bit().map_err(|_| e("extra_bits"))?;
    }
    let slice_type = r.read_ue().map_err(|_| e("slice_type"))? as u8;
    // pic_output_flag: when 0 the picture is decoded but not output (§7.4.7.1).
    let mut pic_output_flag = true;
    if pps.output_flag_present {
        pic_output_flag = r.read_flag().map_err(|_| e("pic_output_flag"))?;
    }
    if sps.separate_color_plane {
        r.read_bits(2).map_err(|_| e("color_plane"))?;
    }
    let is_idr = crate::demux::nal::is_idr(nal_type);
    let is_irap = crate::demux::nal::is_irap(nal_type);

    // POC LSB + reference picture set (§7.3.6.1). IDR pictures have POC 0 and no RPS.
    let mut poc_lsb = 0i32;
    let mut cur_rps = crate::rps::ShortTermRps::default();
    let mut lt_refs: Vec<(i32, bool, bool, i32)> = Vec::new();
    if !is_idr {
        poc_lsb = r
            .read_bits(sps.log2_max_poc_lsb)
            .map_err(|_| e("poc_lsb"))? as i32;
        let short_term_sps_flag = r.read_flag().map_err(|_| e("short_term_rps_sps_flag"))?;
        if !short_term_sps_flag {
            // Inline RPS: index equals num sets in SPS.
            let n = sps.short_term_rps.len();
            cur_rps = crate::rps::parse_short_term_rps(&mut r, n, n, &sps.short_term_rps)?;
        } else {
            let num_sets = sps.short_term_rps.len();
            if num_sets > 1 {
                let bits = ceil_log2(num_sets as u64);
                let idx = r.read_bits(bits).map_err(|_| e("short_term_rps_idx"))? as usize;
                cur_rps = sps.short_term_rps.get(idx).cloned().unwrap_or_default();
            } else if num_sets == 1 {
                cur_rps = sps.short_term_rps[0].clone();
            }
        }
        // long_term reference pictures (§7.3.6.1). Collect each LT ref's POC
        // (or POC LSB when the MSB delta is not signalled) and used_by_curr flag.
        if long_term_ref_pics_present(sps) {
            let num_lt_sps = if !sps.lt_ref_poc_lsb.is_empty() {
                r.read_ue().map_err(|_| e("num_lt_sps"))? as usize
            } else {
                0
            };
            let num_lt_pics = r.read_ue().map_err(|_| e("num_lt_pics"))? as usize;
            let num_lt = num_lt_sps + num_lt_pics;
            let mut prev_delta_msb = 0i32;
            for i in 0..num_lt {
                let (poc_lsb_lt, used) = if i < num_lt_sps {
                    let idx = if sps.lt_ref_poc_lsb.len() > 1 {
                        let bits = ceil_log2(sps.lt_ref_poc_lsb.len() as u64);
                        r.read_bits(bits).map_err(|_| e("lt_idx_sps"))? as usize
                    } else {
                        0
                    };
                    (
                        *sps.lt_ref_poc_lsb.get(idx).unwrap_or(&0) as i32,
                        *sps.lt_used_by_curr.get(idx).unwrap_or(&false),
                    )
                } else {
                    let lsb = r
                        .read_bits(sps.log2_max_poc_lsb)
                        .map_err(|_| e("poc_lsb_lt"))? as i32;
                    let used = r.read_flag().map_err(|_| e("used_by_curr_lt"))?;
                    (lsb, used)
                };
                let delta_msb_present = r.read_flag().map_err(|_| e("delta_msb_present"))?;
                if delta_msb_present {
                    let delta_msb = r.read_ue().map_err(|_| e("delta_poc_msb_cycle_lt"))? as i32;
                    // delta_poc_msb_cycle_lt is delta-coded for entries past the
                    // first in each group (§7.4.8).
                    let cycle = if i == 0 || i == num_lt_sps {
                        delta_msb
                    } else {
                        delta_msb + prev_delta_msb
                    };
                    prev_delta_msb = cycle;
                    // The full long-term POC (§8.3.2) needs the current
                    // picture's full POC, which is not known at parse time; only
                    // the LSB is. Carry poc_lsb_lt and the delta MSB cycle so the
                    // DPB can resolve it once the current POC is derived.
                    lt_refs.push((poc_lsb_lt, used, true, cycle));
                } else {
                    // No MSB: only the LSB is known; match references by LSB.
                    lt_refs.push((poc_lsb_lt, used, false, 0));
                }
            }
        }
    }

    // slice_temporal_mvp_enabled_flag.
    let mut temporal_mvp = false;
    if sps.temporal_mvp_enabled && !is_idr {
        temporal_mvp = r.read_flag().map_err(|_| e("slice_temporal_mvp"))?;
    }

    let mut sao_luma = false;
    let mut sao_chroma = false;
    if sps.sao_enabled {
        sao_luma = r.read_flag().map_err(|_| e("sao_luma"))?;
        if !sps.chroma.is_monochrome() {
            sao_chroma = r.read_flag().map_err(|_| e("sao_chroma"))?;
        }
    }

    // Reference list configuration for P/B slices (§7.3.6.1).
    let mut num_ref_idx_l0 = pps.num_ref_idx_l0_default;
    let mut num_ref_idx_l1 = pps.num_ref_idx_l1_default;
    let mut list_mod_l0 = Vec::new();
    let mut list_mod_l1 = Vec::new();
    let mut mvd_l1_zero = false;
    let mut cabac_init = false;
    let mut collocated_from_l0 = true;
    let mut collocated_ref_idx = 0usize;
    let mut max_num_merge_cand = 5usize;
    let mut pred_weights = None;
    let is_inter = slice_type == crate::inter::SLICE_P || slice_type == crate::inter::SLICE_B;
    let is_b = slice_type == crate::inter::SLICE_B;
    if is_inter {
        let num_ref_override = r.read_flag().map_err(|_| e("num_ref_idx_override"))?;
        if num_ref_override {
            num_ref_idx_l0 = r.read_ue().map_err(|_| e("num_ref_l0"))? as usize + 1;
            if is_b {
                num_ref_idx_l1 = r.read_ue().map_err(|_| e("num_ref_l1"))? as usize + 1;
            }
        }
        if !is_b {
            num_ref_idx_l1 = 0;
        }
        // NumPicTotalCurr (§7.4.7.2): the number of reference pictures marked
        // used_by_curr_pic — the short-term S0/S1 entries whose used flag is set
        // plus the long-term entries used by the current picture. This (not the
        // total delta-POC count) sizes the list_entry_lX fixed-length codes.
        let num_pics_total = cur_rps.used_s0.iter().filter(|&&u| u).count()
            + cur_rps.used_s1.iter().filter(|&&u| u).count()
            + lt_refs.iter().filter(|&&(_, used, _, _)| used).count();
        if pps.lists_modification_present && num_pics_total > 1 {
            let bits = ceil_log2(num_pics_total as u64);
            if r.read_flag().map_err(|_| e("ref_pic_list_mod_l0"))? {
                for _ in 0..num_ref_idx_l0 {
                    list_mod_l0.push(r.read_bits(bits).map_err(|_| e("list_entry_l0"))?);
                }
            }
            if is_b && r.read_flag().map_err(|_| e("ref_pic_list_mod_l1"))? {
                for _ in 0..num_ref_idx_l1 {
                    list_mod_l1.push(r.read_bits(bits).map_err(|_| e("list_entry_l1"))?);
                }
            }
        }
        if is_b {
            mvd_l1_zero = r.read_flag().map_err(|_| e("mvd_l1_zero"))?;
        }
        if pps.cabac_init_present {
            cabac_init = r.read_flag().map_err(|_| e("cabac_init_flag"))?;
        }
        if temporal_mvp {
            if is_b {
                collocated_from_l0 = r.read_flag().map_err(|_| e("collocated_from_l0"))?;
            }
            let active = if collocated_from_l0 {
                num_ref_idx_l0
            } else {
                num_ref_idx_l1
            };
            if active > 1 {
                collocated_ref_idx = r.read_ue().map_err(|_| e("collocated_ref_idx"))? as usize;
            }
        }
        let weighted = (slice_type == crate::inter::SLICE_P && pps.weighted_pred)
            || (is_b && pps.weighted_bipred);
        if weighted {
            pred_weights = Some(crate::inter::PredWeightTable::parse(
                &mut r,
                [num_ref_idx_l0, num_ref_idx_l1],
                is_b,
                !sps.chroma.is_monochrome(),
                sps.bit_depth_luma,
                sps.bit_depth_chroma,
            )?);
        }
        let five_minus = r.read_ue().map_err(|_| e("five_minus_max_merge"))? as usize;
        max_num_merge_cand = 5usize.saturating_sub(five_minus);
    }
    let _ = is_irap;
    let slice_qp_delta = r.read_se().map_err(|_| e("qp_delta"))?;
    let slice_qp = clamp_qpy(
        pps.init_qp.saturating_add(slice_qp_delta),
        sps.bit_depth_luma,
    );
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
    // slice_loop_filter_across_slices_enabled_flag: per-slice override of the
    // PPS default; when absent it inherits the PPS value (§7.4.7.1).
    let mut slice_loop_filter_across_slices = pps.loop_filter_across_slices;
    if pps.loop_filter_across_slices && (sao_luma || sao_chroma || !deblocking_disabled) {
        slice_loop_filter_across_slices = r
            .read_flag()
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
        slice_type,
        poc_lsb,
        cur_rps,
        lt_refs,
        num_ref_idx_l0,
        num_ref_idx_l1,
        list_mod_l0,
        list_mod_l1,
        mvd_l1_zero,
        cabac_init,
        temporal_mvp,
        collocated_from_l0,
        collocated_ref_idx,
        max_num_merge_cand,
        pred_weights,
        slice_loop_filter_across_slices,
        pic_output_flag,
    })
}

/// Whether the SPS signals long-term reference pictures (a non-empty SPS LT set
/// or the presence flag). Used to decide if the slice header carries LT syntax.
fn long_term_ref_pics_present(sps: &Sps) -> bool {
    sps.long_term_ref_pics_present
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
    use super::{
        ceil_log2, clamp_qpy, derive_qpy_from_delta, peek_slice_pps_id, qp_prime, qpc, qpy_min,
    };

    // Build a slice-header prefix: 2-byte NAL header, first_slice flag,
    // [no_output_of_prior_pics if IRAP], then pps_id as ue(v). Values are packed
    // MSB-first to mirror the bitstream.
    fn slice_prefix(is_irap: bool, first_slice: bool, pps_id_ue_bits: &[u8]) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        for _ in 0..16 {
            bits.push(0);
        } // NAL header (value irrelevant to peek)
        bits.push(first_slice as u8);
        if is_irap {
            bits.push(0);
        } // no_output_of_prior_pics_flag
        bits.extend_from_slice(pps_id_ue_bits);
        // pack to bytes
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, b) in bits.iter().enumerate() {
            if *b != 0 {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        out
    }

    #[test]
    fn peek_pps_id_zero() {
        // ue(0) = '1'
        let buf = slice_prefix(false, true, &[1]);
        assert_eq!(peek_slice_pps_id(&buf, 1), Some(0));
    }

    #[test]
    fn peek_pps_id_nonzero_and_irap_offset() {
        // ue(1) = '0 1 0'; with an IRAP the extra no_output flag must be skipped.
        let buf = slice_prefix(true, true, &[0, 1, 0]);
        assert_eq!(peek_slice_pps_id(&buf, 19), Some(1)); // 19 = IDR_W_RADL
        // ue(2) = '0 1 1'
        let buf2 = slice_prefix(false, false, &[0, 1, 1]);
        assert_eq!(peek_slice_pps_id(&buf2, 1), Some(2));
    }

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

    #[test]
    fn qp_prime_preserves_negative_high_bit_depth_qp() {
        assert_eq!(qp_prime(-12, 10), 0);
        assert_eq!(qp_prime(-24, 12), 0);
        assert_eq!(qp_prime(0, 10), 12);
        assert_eq!(qp_prime(51, 12), 75);
    }

    #[test]
    fn chroma_qpc_preserves_negative_high_bit_depth_qp() {
        assert_eq!(qpc(-12, 1, 10), -12);
        assert_eq!(qpc(-24, 1, 12), -24);
        assert_eq!(qp_prime(qpc(-12, 1, 10), 10), 0);
        assert_eq!(qp_prime(qpc(-24, 3, 12), 12), 0);
    }

    #[test]
    fn luma_qpy_clamp_uses_bit_depth_offset() {
        assert_eq!(clamp_qpy(-1, 8), 0);
        assert_eq!(clamp_qpy(-12, 10), -12);
        assert_eq!(clamp_qpy(-13, 10), -12);
        assert_eq!(clamp_qpy(-24, 12), -24);
        assert_eq!(clamp_qpy(90, 12), 51);
    }

    #[test]
    fn luma_qpy_delta_derivation_is_overflow_hardened() {
        assert_eq!(derive_qpy_from_delta(51, 0, 8), 51);
        assert!((qpy_min(12)..=51).contains(&derive_qpy_from_delta(i32::MAX, i32::MAX, 12)));
        assert!((qpy_min(12)..=51).contains(&derive_qpy_from_delta(i32::MIN, i32::MIN, 12)));
    }
}
