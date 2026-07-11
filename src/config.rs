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

use crate::bitreader::BitReader;
use crate::error::DecodeError;
use crate::fmt::{BitDepth, ChromaFormat};

#[derive(Debug, Clone)]
pub(crate) struct Sps {
    /// seq_parameter_set_id (0..15).
    pub(crate) id: u32,
    pub(crate) chroma_idc: u8,
    pub(crate) chroma: ChromaFormat,
    pub(crate) separate_color_plane: bool,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Conformance-window crop offsets in *luma* samples (left, right, top, bottom).
    pub(crate) crop_left: u32,
    pub(crate) crop_right: u32,
    pub(crate) crop_top: u32,
    pub(crate) crop_bottom: u32,
    pub(crate) bit_depth_luma: u8,
    pub(crate) bit_depth_chroma: u8,
    pub(crate) log2_ctb: u32,
    pub(crate) log2_min_cb: u32,
    pub(crate) log2_min_tb: u32,
    pub(crate) log2_max_tb: u32,
    pub(crate) max_transform_hierarchy_intra: u32,
    pub(crate) max_transform_hierarchy_inter: u32,
    /// log2 of MaxPicOrderCntLsb (§7.4.3.2), 4..=16.
    pub(crate) log2_max_poc_lsb: u32,
    /// sps_max_num_reorder_pics for the highest sub-layer (§7.4.3.2.1): the
    /// maximum number of pictures that can precede any picture in decode order
    /// and follow it in output order. Bounds output-reorder latency.
    pub(crate) max_num_reorder_pics: u32,
    /// sps_max_dec_pic_buffering_minus1 + 1 for the highest sub-layer: the DPB
    /// size in pictures. Used to size the output-bumping threshold.
    pub(crate) max_dec_pic_buffering: u32,
    /// Asymmetric motion partitions enabled (inter).
    pub(crate) amp_enabled: bool,
    /// SPS-level temporal motion vector prediction enabled.
    pub(crate) temporal_mvp_enabled: bool,
    /// Short-term reference picture sets parsed from the SPS.
    pub(crate) short_term_rps: Vec<crate::rps::ShortTermRps>,
    /// Long-term reference POC LSBs signalled at SPS level.
    pub(crate) lt_ref_poc_lsb: Vec<u32>,
    pub(crate) lt_used_by_curr: Vec<bool>,
    /// long_term_ref_pics_present_flag (§7.4.3.2). LT refs may still be signalled
    /// per-slice even when the SPS candidate list is empty.
    pub(crate) long_term_ref_pics_present: bool,
    pub(crate) scaling_list_enabled: bool,
    pub(crate) scaling_list: Option<ScalingList>,
    pub(crate) sao_enabled: bool,
    pub(crate) pcm_enabled: bool,
    pub(crate) pcm_bit_depth_luma: u8,
    pub(crate) pcm_bit_depth_chroma: u8,
    pub(crate) log2_min_pcm_cb: u32,
    pub(crate) log2_max_pcm_cb: u32,
    pub(crate) pcm_loop_filter_disabled: bool,
    pub(crate) strong_intra_smoothing: bool,
    /// Whether the SPS carried a VUI and its nested colour signalling fields.
    /// These are retained separately from the code-point defaults so callers can
    /// distinguish “unspecified/absent” from an explicitly signalled value.
    #[allow(dead_code)]
    pub(crate) vui_parameters_present: bool,
    #[allow(dead_code)]
    pub(crate) video_signal_type_present: bool,
    pub(crate) colour_description_present: bool,
    pub(crate) video_full_range: bool,
    pub(crate) color_primaries: u8, // ISO/IEC 23091-2 Table 2; 2 = unspecified
    pub(crate) transfer_characteristics: u8, // ISO/IEC 23091-2 Table 3; 2 = unspecified
    pub(crate) matrix_coefficients: u8, // ISO/IEC 23091-2 Table 4; 2 = unspecified
    /// VUI timing: frame rate = time_scale / num_units_in_tick (0 = absent).
    pub(crate) vui_num_units_in_tick: u32,
    pub(crate) vui_time_scale: u32,

    // ---- Range extension (§7.3.2.2.2). Parsed so the bitstream is navigated
    // correctly; the corresponding residual-coding features are not decoded, so
    // these are retained for completeness / future use rather than consumed. ----
    #[allow(dead_code)]
    pub(crate) transform_skip_rotation_enabled: bool,
    #[allow(dead_code)]
    pub(crate) transform_skip_context_enabled: bool,
    #[allow(dead_code)]
    pub(crate) implicit_rdpcm_enabled: bool,
    #[allow(dead_code)]
    pub(crate) explicit_rdpcm_enabled: bool,
    #[allow(dead_code)]
    pub(crate) extended_precision_processing: bool,
    #[allow(dead_code)]
    pub(crate) intra_smoothing_disabled: bool,
    #[allow(dead_code)]
    pub(crate) high_precision_offsets_enabled: bool,
    #[allow(dead_code)]
    pub(crate) persistent_rice_adaptation_enabled: bool,
    #[allow(dead_code)]
    pub(crate) cabac_bypass_alignment_enabled: bool,

    // ---- SCC extension (§7.3.2.2.3) ----
    /// curr_pic_ref: current picture is inserted into RefPicList0 as an IBC ref.
    pub(crate) curr_pic_ref_enabled: bool,
    pub(crate) palette_mode_enabled: bool,
    pub(crate) palette_max_size: u32,
    pub(crate) palette_max_predictor_size: u32,
    /// Per-slice-reset predictor seed (component-major: [comp][entry]).
    pub(crate) palette_predictor_initializers: Vec<Vec<u16>>,
    /// 0 = adaptive/off, 2 = force integer block-vector MVs. IBC block vectors
    /// are always integerized, so this is currently informational.
    #[allow(dead_code)]
    pub(crate) motion_vector_resolution_control_idc: u8,
    #[allow(dead_code)]
    pub(crate) intra_boundary_filtering_disabled: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Pps {
    /// pic_parameter_set_id (0..63) and the seq_parameter_set_id it references.
    pub(crate) id: u32,
    pub(crate) sps_id: u32,
    pub(crate) dependent_slice_segments_enabled: bool,
    pub(crate) output_flag_present: bool,
    pub(crate) num_extra_slice_header_bits: u32,
    pub(crate) sign_data_hiding_enabled: bool,
    pub(crate) cabac_init_present: bool,
    pub(crate) init_qp: i32,
    pub(crate) _constrained_intra_pred: bool,
    pub(crate) transform_skip_enabled: bool,
    pub(crate) cu_qp_delta_enabled: bool,
    pub(crate) diff_cu_qp_delta_depth: u32,
    pub(crate) cb_qp_offset: i32,
    pub(crate) cr_qp_offset: i32,
    pub(crate) slice_chroma_qp_offsets_present: bool,
    pub(crate) weighted_pred: bool,
    pub(crate) weighted_bipred: bool,
    pub(crate) transquant_bypass_enabled: bool,
    pub(crate) tiles_enabled: bool,
    /// Tile column/row structure (§7.4.3.3). Only meaningful when `tiles_enabled`.
    /// `num_tile_columns`/`num_tile_rows` are the counts (≥1); `uniform_spacing`
    /// selects even division; when non-uniform, `column_widths`/`row_heights`
    /// hold the explicit sizes in CTBs for all but the last column/row (the last
    /// is implied). Resolved to pixel boundaries per-picture in the decoder.
    pub(crate) num_tile_columns: u32,
    pub(crate) num_tile_rows: u32,
    pub(crate) tile_uniform_spacing: bool,
    pub(crate) tile_column_widths: Vec<u32>,
    pub(crate) tile_row_heights: Vec<u32>,
    /// loop_filter_across_tiles_enabled_flag (§7.4.3.3). When false, in-loop
    /// filters do not cross tile boundaries.
    pub(crate) loop_filter_across_tiles: bool,
    pub(crate) entropy_coding_sync_enabled: bool,
    pub(crate) loop_filter_across_slices: bool,
    pub(crate) _deblocking_filter_control_present: bool,
    pub(crate) deblocking_filter_override_enabled: bool,
    pub(crate) deblocking_filter_disabled: bool,
    pub(crate) beta_offset_div2: i32,
    pub(crate) tc_offset_div2: i32,
    pub(crate) scaling_list: Option<ScalingList>,
    pub(crate) lists_modification_present: bool,
    pub(crate) log2_parallel_merge_level: u32,
    pub(crate) num_ref_idx_l0_default: usize,
    pub(crate) num_ref_idx_l1_default: usize,
    pub(crate) slice_segment_header_extension_present: bool,

    // ---- Range extension (§7.3.2.3.2). Parsed for correct bitstream navigation;
    // the associated features are not decoded, so these are reserved. ----
    #[allow(dead_code)]
    pub(crate) log2_max_transform_skip_block_size: u32,
    #[allow(dead_code)]
    pub(crate) cross_component_prediction_enabled: bool,
    #[allow(dead_code)]
    pub(crate) chroma_qp_offset_list_enabled: bool,
    #[allow(dead_code)]
    pub(crate) diff_cu_chroma_qp_offset_depth: u32,
    /// (cb, cr) offset pairs indexed by cu_chroma_qp_offset_idx.
    #[allow(dead_code)]
    pub(crate) chroma_qp_offset_list: Vec<(i32, i32)>,
    #[allow(dead_code)]
    pub(crate) log2_sao_offset_scale_luma: u32,
    #[allow(dead_code)]
    pub(crate) log2_sao_offset_scale_chroma: u32,

    // ---- SCC extension (§7.3.2.3.3) ----
    /// curr_pic_ref: PPS-level enable for IBC (both SPS and PPS must be set).
    pub(crate) curr_pic_ref_enabled: bool,
    pub(crate) residual_adaptive_colour_transform_enabled: bool,
    /// ACT slice-level QP override present flag.
    pub(crate) pps_slice_act_qp_offsets_present: bool,
    /// ACT component QP offsets (Y/Cg/Co), added to the −5/−5/−3 base.
    pub(crate) pps_act_y_qp_offset: i32,
    pub(crate) pps_act_cb_qp_offset: i32,
    pub(crate) pps_act_cr_qp_offset: i32,
    /// Distinguishes an absent PPS initializer table (fall back to SPS) from
    /// a present table with zero entries (explicitly reset the predictor empty).
    pub(crate) palette_predictor_initializer_present: bool,
    pub(crate) palette_predictor_initializers: Vec<Vec<u16>>,
    // Bit-depth metadata for the PPS palette initialisers: used while parsing
    // the initialiser entries; not re-consulted afterward.
    #[allow(dead_code)]
    pub(crate) monochrome_palette: bool,
    #[allow(dead_code)]
    pub(crate) luma_bit_depth_entry: u32,
    #[allow(dead_code)]
    pub(crate) chroma_bit_depth_entry: u32,
}

impl Sps {
    pub(crate) fn bit_depth(&self) -> Result<BitDepth, DecodeError> {
        match self.bit_depth_luma {
            8 => Ok(BitDepth::Eight),
            10 => Ok(BitDepth::Ten),
            12 => Ok(BitDepth::Twelve),
            n => Err(DecodeError::UnsupportedBitDepth(n)),
        }
    }
}

#[cfg(test)]
impl Pps {
    /// All-defaults PPS for unit tests (tiles disabled, no offsets).
    pub(crate) fn test_default() -> Pps {
        Pps {
            id: 0,
            sps_id: 0,
            dependent_slice_segments_enabled: false,
            output_flag_present: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled: false,
            cabac_init_present: false,
            init_qp: 26,
            _constrained_intra_pred: false,
            transform_skip_enabled: false,
            cu_qp_delta_enabled: false,
            diff_cu_qp_delta_depth: 0,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            slice_chroma_qp_offsets_present: false,
            weighted_pred: false,
            weighted_bipred: false,
            transquant_bypass_enabled: false,
            tiles_enabled: false,
            num_tile_columns: 1,
            num_tile_rows: 1,
            tile_uniform_spacing: true,
            tile_column_widths: Vec::new(),
            tile_row_heights: Vec::new(),
            loop_filter_across_tiles: true,
            entropy_coding_sync_enabled: false,
            loop_filter_across_slices: true,
            _deblocking_filter_control_present: false,
            deblocking_filter_override_enabled: false,
            deblocking_filter_disabled: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            scaling_list: None,
            lists_modification_present: false,
            log2_parallel_merge_level: 2,
            num_ref_idx_l0_default: 1,
            num_ref_idx_l1_default: 1,
            slice_segment_header_extension_present: false,
            log2_max_transform_skip_block_size: 2,
            cross_component_prediction_enabled: false,
            chroma_qp_offset_list_enabled: false,
            diff_cu_chroma_qp_offset_depth: 0,
            chroma_qp_offset_list: Vec::new(),
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
            curr_pic_ref_enabled: false,
            residual_adaptive_colour_transform_enabled: false,
            pps_slice_act_qp_offsets_present: false,
            pps_act_y_qp_offset: 0,
            pps_act_cb_qp_offset: 0,
            pps_act_cr_qp_offset: 0,
            palette_predictor_initializer_present: false,
            palette_predictor_initializers: Vec::new(),
            monochrome_palette: false,
            luma_bit_depth_entry: 0,
            chroma_bit_depth_entry: 0,
        }
    }
}

fn e(s: &'static str) -> DecodeError {
    DecodeError::ParamSet(s.into())
}

/// Profile-tier-level (§7.3.3), variable length depending on sub-layers.
fn parse_ptl(r: &mut BitReader, max_sub_layers_minus1: u32) -> Result<(), DecodeError> {
    r.read_bits(8).map_err(|_| e("ptl profile byte"))?; // space/tier/profile_idc
    r.read_bits(32).map_err(|_| e("ptl compat"))?; // compatibility flags
    r.read_bits(32).map_err(|_| e("ptl constraint hi"))?; // 48 constraint bits...
    r.read_bits(16).map_err(|_| e("ptl constraint lo"))?;
    r.read_bits(8).map_err(|_| e("ptl level"))?; // general_level_idc

    let n = max_sub_layers_minus1 as usize;
    let mut prof = vec![false; n];
    let mut lvl = vec![false; n];
    for i in 0..n {
        prof[i] = r.read_flag().map_err(|_| e("sub prof flag"))?;
        lvl[i] = r.read_flag().map_err(|_| e("sub lvl flag"))?;
    }
    if n > 0 {
        for _ in n..8 {
            r.read_bits(2).map_err(|_| e("ptl reserved"))?;
        }
    }
    for i in 0..n {
        if prof[i] {
            r.read_bits(8)?;
            r.read_bits(32)?;
            r.read_bits(32)?;
            r.read_bits(16)?;
        }
        if lvl[i] {
            r.read_bits(8)?;
        }
    }
    Ok(())
}

const SCALING_LIST_NUM_SIZES: usize = 4;
const SCALING_LIST_NUM_LISTS: usize = 6;
const SCALING_LIST_DC: u8 = 16;

static QUANT_TS_DEFAULT_4X4: [u8; 16] = [16; 16];

static QUANT_INTRA_DEFAULT_8X8: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 21, 24, 16, 16, 16, 16, 17, 19, 22, 25, 16, 16, 17, 18, 20, 22, 25, 29,
    16, 16, 18, 21, 24, 27, 31, 36, 17, 17, 20, 24, 30, 35, 41, 47, 18, 19, 22, 27, 35, 44, 54, 65,
    21, 22, 25, 31, 41, 54, 70, 88, 24, 25, 29, 36, 47, 65, 88, 115,
];

static QUANT_INTER_DEFAULT_8X8: [u8; 64] = [
    16, 16, 16, 16, 17, 18, 20, 24, 16, 16, 16, 17, 18, 20, 24, 25, 16, 16, 17, 18, 20, 24, 25, 28,
    16, 17, 18, 20, 24, 25, 28, 33, 17, 18, 20, 24, 25, 28, 33, 41, 18, 20, 24, 25, 28, 33, 41, 54,
    20, 24, 25, 28, 33, 41, 54, 71, 24, 25, 28, 33, 41, 54, 71, 91,
];

#[derive(Debug, Clone)]
pub(crate) struct ScalingList {
    matrices: [[[u8; 64]; SCALING_LIST_NUM_LISTS]; SCALING_LIST_NUM_SIZES],
    dc: [[u8; SCALING_LIST_NUM_LISTS]; SCALING_LIST_NUM_SIZES],
    flat_16: [[bool; SCALING_LIST_NUM_LISTS]; SCALING_LIST_NUM_SIZES],
    max_coeff: [[u8; SCALING_LIST_NUM_LISTS]; SCALING_LIST_NUM_SIZES],
}

impl Default for ScalingList {
    fn default() -> Self {
        let mut out = ScalingList {
            matrices: [[[16; 64]; SCALING_LIST_NUM_LISTS]; SCALING_LIST_NUM_SIZES],
            dc: [[SCALING_LIST_DC; SCALING_LIST_NUM_LISTS]; SCALING_LIST_NUM_SIZES],
            flat_16: [[true; SCALING_LIST_NUM_LISTS]; SCALING_LIST_NUM_SIZES],
            max_coeff: [[16; SCALING_LIST_NUM_LISTS]; SCALING_LIST_NUM_SIZES],
        };

        for list_id in 0..SCALING_LIST_NUM_LISTS {
            out.matrices[0][list_id][..16].copy_from_slice(&QUANT_TS_DEFAULT_4X4);
            out.refresh_stats(0, list_id);
            let default_8x8 = if list_id < 3 {
                &QUANT_INTRA_DEFAULT_8X8
            } else {
                &QUANT_INTER_DEFAULT_8X8
            };
            for size_id in 1..SCALING_LIST_NUM_SIZES {
                out.matrices[size_id][list_id].copy_from_slice(default_8x8);
                out.refresh_stats(size_id, list_id);
            }
        }

        out
    }
}

impl ScalingList {
    #[inline]
    fn refresh_stats(&mut self, size_id: usize, matrix_id: usize) {
        let used = if size_id == 0 { 16 } else { 64 };
        self.flat_16[size_id][matrix_id] = (size_id < 2
            || self.dc[size_id][matrix_id] == SCALING_LIST_DC)
            && self.matrices[size_id][matrix_id][..used]
                .iter()
                .all(|&v| v == 16);

        let mut max_coeff = self.matrices[size_id][matrix_id][..used]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        if size_id >= 2 {
            max_coeff = max_coeff.max(self.dc[size_id][matrix_id]);
        }
        self.max_coeff[size_id][matrix_id] = max_coeff;
    }

    #[inline]
    fn set_matrix(&mut self, size_id: usize, matrix_id: usize, matrix: [u8; 64], dc: u8) {
        self.matrices[size_id][matrix_id] = matrix;
        self.dc[size_id][matrix_id] = dc;
        self.refresh_stats(size_id, matrix_id);
    }

    #[inline]
    fn copy_matrix(&mut self, size_id: usize, matrix_id: usize, pred_id: usize) {
        self.matrices[size_id][matrix_id] = self.matrices[size_id][pred_id];
        self.dc[size_id][matrix_id] = self.dc[size_id][pred_id];
        self.flat_16[size_id][matrix_id] = self.flat_16[size_id][pred_id];
        self.max_coeff[size_id][matrix_id] = self.max_coeff[size_id][pred_id];
    }

    #[inline]
    pub(crate) fn matrix(&self, size_id: usize, matrix_id: usize) -> (&[u8; 64], u8, bool, u8) {
        let size_id = size_id.min(SCALING_LIST_NUM_SIZES - 1);
        let matrix_id = matrix_id.min(SCALING_LIST_NUM_LISTS - 1);
        (
            &self.matrices[size_id][matrix_id],
            self.dc[size_id][matrix_id],
            self.flat_16[size_id][matrix_id],
            self.max_coeff[size_id][matrix_id],
        )
    }

    #[cfg(test)]
    pub(crate) fn test_constant_lists(values: [u8; 6]) -> Self {
        let mut out = Self::default();
        for size_id in 0..SCALING_LIST_NUM_SIZES {
            for (matrix_id, &value) in values.iter().enumerate() {
                out.matrices[size_id][matrix_id].fill(value);
                out.dc[size_id][matrix_id] = value;
                out.refresh_stats(size_id, matrix_id);
            }
        }
        out
    }
}

fn scaling_list_scan_pos(size: usize, scan_idx: usize) -> usize {
    let (mut x, mut y) = (0i32, 0i32);
    let mut seen = 0usize;
    loop {
        while y >= 0 {
            if (x as usize) < size && (y as usize) < size {
                if seen == scan_idx {
                    return y as usize * size + x as usize;
                }
                seen += 1;
            }
            y -= 1;
            x += 1;
        }
        y = x;
        x = 0;
    }
}

fn default_scaling_list_matrix(size_id: usize, matrix_id: usize) -> ([u8; 64], u8) {
    let mut matrix = [16u8; 64];
    if size_id == 0 {
        matrix[..16].copy_from_slice(&QUANT_TS_DEFAULT_4X4);
    } else {
        let default_8x8 = if matrix_id < 3 {
            &QUANT_INTRA_DEFAULT_8X8
        } else {
            &QUANT_INTER_DEFAULT_8X8
        };
        matrix.copy_from_slice(default_8x8);
    }
    (matrix, SCALING_LIST_DC)
}

/// Parse scaling_list_data (§7.3.4) into raster-order matrices
fn parse_scaling_list_data(r: &mut BitReader) -> Result<ScalingList, DecodeError> {
    let mut out = ScalingList::default();

    for size_id in 0..SCALING_LIST_NUM_SIZES {
        let mut matrix_id = 0usize;
        while matrix_id < SCALING_LIST_NUM_LISTS {
            let pred_mode_flag = r
                .read_flag()
                .map_err(|_| e("scaling_list_pred_mode_flag"))?;
            if !pred_mode_flag {
                let delta = r
                    .read_ue()
                    .map_err(|_| e("scaling_list_pred_matrix_id_delta"))?
                    as usize;
                if delta == 0 {
                    let (matrix, dc) = default_scaling_list_matrix(size_id, matrix_id);
                    out.set_matrix(size_id, matrix_id, matrix, dc);
                } else {
                    let pred_delta = if size_id == 3 { delta * 3 } else { delta };
                    let pred_id = matrix_id
                        .checked_sub(pred_delta)
                        .ok_or_else(|| e("scaling_list_pred_matrix_id_delta"))?;
                    out.copy_matrix(size_id, matrix_id, pred_id);
                }
            } else {
                let scan_size = if size_id == 0 { 4 } else { 8 };
                let coef_num = scan_size * scan_size;
                let mut next_coef = 8i32;
                if size_id > 1 {
                    next_coef = r.read_se().map_err(|_| e("scaling_list_dc_coef"))? + 8;
                    if !(0..=255).contains(&next_coef) {
                        return Err(e("scaling_list_dc_coef range"));
                    }
                    out.dc[size_id][matrix_id] = next_coef as u8;
                } else {
                    out.dc[size_id][matrix_id] = SCALING_LIST_DC;
                }

                for scan_idx in 0..coef_num {
                    let delta_coef = r.read_se().map_err(|_| e("scaling_list_delta_coef"))?;
                    next_coef = (next_coef + delta_coef + 256) & 255;
                    let pos = scaling_list_scan_pos(scan_size, scan_idx);
                    // Zero is a legal 8-bit scaling-list entry. Clamping it to
                    // one changes the inverse-quantisation matrix.
                    out.matrices[size_id][matrix_id][pos] = next_coef as u8;
                }
                out.refresh_stats(size_id, matrix_id);
            }
            matrix_id += if size_id == 3 { 3 } else { 1 };
        }
    }

    for &matrix_id in &[1usize, 2, 4, 5] {
        out.matrices[3][matrix_id] = out.matrices[2][matrix_id];
        out.dc[3][matrix_id] = out.dc[2][matrix_id];
        out.flat_16[3][matrix_id] = out.flat_16[2][matrix_id];
        out.max_coeff[3][matrix_id] = out.max_coeff[2][matrix_id];
    }

    Ok(out)
}

pub(crate) fn parse_sps(rbsp: &[u8]) -> Result<Sps, DecodeError> {
    let mut r = BitReader::new(rbsp);
    r.read_bits(4).map_err(|_| e("sps_vps_id"))?;
    let max_sub_layers_minus1 = r.read_bits(3).map_err(|_| e("max_sub_layers"))?;
    r.read_bit().map_err(|_| e("temporal_id_nesting"))?;
    parse_ptl(&mut r, max_sub_layers_minus1)?;

    let sps_id = r.read_ue().map_err(|_| e("sps_id"))?;
    let chroma_idc = r.read_ue().map_err(|_| e("chroma_idc"))? as u8;
    let separate_color_plane = if chroma_idc == 3 {
        r.read_flag().map_err(|_| e("separate_color_plane"))?
    } else {
        false
    };
    let width = r.read_ue().map_err(|_| e("pic_width"))?;
    let height = r.read_ue().map_err(|_| e("pic_height"))?;
    let (mut crop_left, mut crop_right, mut crop_top, mut crop_bottom) = (0, 0, 0, 0);
    if r.read_flag().map_err(|_| e("conf_win"))? {
        crop_left = r.read_ue().map_err(|_| e("cw_left"))?;
        crop_right = r.read_ue().map_err(|_| e("cw_right"))?;
        crop_top = r.read_ue().map_err(|_| e("cw_top"))?;
        crop_bottom = r.read_ue().map_err(|_| e("cw_bottom"))?;
    }
    let bit_depth_luma = r
        .read_ue()
        .map_err(|_| e("bd_luma"))?
        .checked_add(8)
        .filter(|&v| v <= u8::MAX as u32)
        .map(|v| v as u8)
        .ok_or_else(|| e("bd_luma"))?;
    if !matches!(bit_depth_luma, 8 | 10 | 12) {
        return Err(DecodeError::UnsupportedBitDepth(bit_depth_luma));
    }
    let bit_depth_chroma = r
        .read_ue()
        .map_err(|_| e("bd_chroma"))?
        .checked_add(8)
        .filter(|&v| v <= u8::MAX as u32)
        .map(|v| v as u8)
        .ok_or_else(|| e("bd_chroma"))?;
    if !matches!(bit_depth_chroma, 8 | 10 | 12) {
        return Err(DecodeError::UnsupportedBitDepth(bit_depth_chroma));
    }
    let log2_max_poc_lsb = r.read_ue().map_err(|_| e("log2_max_poc"))? + 4;
    let log2_max_poc = log2_max_poc_lsb;

    let sub_layer_ordering = r.read_flag().map_err(|_| e("sub_layer_ordering"))?;
    let start = if sub_layer_ordering {
        0
    } else {
        max_sub_layers_minus1
    };
    let mut max_dec_pic_buffering = 1u32;
    let mut max_num_reorder_pics = 0u32;
    for _ in start..=max_sub_layers_minus1 {
        // Only the highest sub-layer's values are retained; they govern the
        // output DPB size and reorder latency (§C.5.2.2).
        max_dec_pic_buffering = r.read_ue().map_err(|_| e("max_dec_pic_buffering"))? + 1;
        max_num_reorder_pics = r.read_ue().map_err(|_| e("num_reorder_pics"))?;
        r.read_ue().map_err(|_| e("max_latency"))?;
    }

    let log2_min_cb = r
        .read_ue()
        .map_err(|_| e("log2_min_cb"))?
        .checked_add(3)
        .ok_or_else(|| e("log2_min_cb overflow"))?;
    let log2_diff_max_min_cb = r.read_ue().map_err(|_| e("log2_diff_max_min_cb"))?;
    let log2_ctb = log2_min_cb
        .checked_add(log2_diff_max_min_cb)
        .ok_or_else(|| e("log2_ctb overflow"))?;
    let log2_min_tb = r
        .read_ue()
        .map_err(|_| e("log2_min_tb"))?
        .checked_add(2)
        .ok_or_else(|| e("log2_min_tb overflow"))?;
    let log2_diff_max_min_tb = r.read_ue().map_err(|_| e("log2_diff_max_min_tb"))?;
    let log2_max_tb = log2_min_tb
        .checked_add(log2_diff_max_min_tb)
        .ok_or_else(|| e("log2_max_tb overflow"))?;
    if !(3..=6).contains(&log2_min_cb) || log2_ctb < log2_min_cb || log2_ctb > 6 {
        return Err(e("log2_ctb"));
    }
    if !(2..=5).contains(&log2_min_tb) || log2_max_tb < log2_min_tb || log2_max_tb > 5 {
        return Err(e("log2_tb"));
    }
    let max_transform_hierarchy_inter = r.read_ue().map_err(|_| e("mth_inter"))?;
    let max_transform_hierarchy_intra = r.read_ue().map_err(|_| e("mth_intra"))?;

    let scaling_list_enabled = r.read_flag().map_err(|_| e("scaling_list_enabled"))?;
    let mut scaling_list = None;
    if scaling_list_enabled {
        let present = r.read_flag().map_err(|_| e("sps_scaling_list_present"))?;
        scaling_list = Some(if present {
            parse_scaling_list_data(&mut r)?
        } else {
            ScalingList::default()
        });
    }
    let amp_enabled = r.read_flag().map_err(|_| e("amp"))?;
    let sao_enabled = r.read_flag().map_err(|_| e("sao"))?;
    let pcm_enabled = r.read_flag().map_err(|_| e("pcm"))?;
    let mut pcm_bit_depth_luma = 0;
    let mut pcm_bit_depth_chroma = 0;
    let mut log2_min_pcm_cb = 0;
    let mut log2_max_pcm_cb = 0;
    let mut pcm_loop_filter_disabled = false;
    if pcm_enabled {
        pcm_bit_depth_luma = r.read_bits(4).map_err(|_| e("pcm_bd_luma"))? as u8 + 1;
        pcm_bit_depth_chroma = r.read_bits(4).map_err(|_| e("pcm_bd_chroma"))? as u8 + 1;
        log2_min_pcm_cb = r.read_ue().map_err(|_| e("log2_min_pcm"))? + 3;
        let diff = r.read_ue().map_err(|_| e("log2_diff_pcm"))?;
        log2_max_pcm_cb = log2_min_pcm_cb + diff;
        pcm_loop_filter_disabled = r.read_flag().map_err(|_| e("pcm_loop_filter"))?;
    }

    let num_st_rps = r.read_ue().map_err(|_| e("num_short_term_rps"))? as usize;
    let mut short_term_rps: Vec<crate::rps::ShortTermRps> = Vec::with_capacity(num_st_rps);
    for i in 0..num_st_rps {
        let rps = crate::rps::parse_short_term_rps(&mut r, i, num_st_rps, &short_term_rps)?;
        short_term_rps.push(rps);
    }

    let mut lt_ref_poc_lsb = Vec::new();
    let mut lt_used_by_curr = Vec::new();
    let long_term_present = r.read_flag().map_err(|_| e("long_term_present"))?;
    if long_term_present {
        let num_lt = r.read_ue().map_err(|_| e("num_long_term"))? as usize;
        for _ in 0..num_lt {
            let poc = r.read_bits(log2_max_poc).map_err(|_| e("lt_ref_poc"))?;
            let used = r.read_flag().map_err(|_| e("used_by_curr_lt"))?;
            lt_ref_poc_lsb.push(poc);
            lt_used_by_curr.push(used);
        }
    }

    let temporal_mvp_enabled = r.read_flag().map_err(|_| e("temporal_mvp"))?;
    let strong_intra_smoothing = r.read_flag().map_err(|_| e("strong_intra_smoothing"))?;
    // VUI parameters — extract matrix_coefficients, video_full_range_flag, and
    // the timing info (frame rate) if present.
    let mut video_full_range = false;
    let mut color_primaries = 2u8; // unspecified
    let mut transfer_characteristics = 2u8; // unspecified
    let mut matrix_coefficients = 2u8; // unspecified
    let mut vui_num_units_in_tick = 0u32;
    let mut vui_time_scale = 0u32;
    let vui_parameters_present = r.read_flag().map_err(|_| e("vui_present"))?;
    let mut video_signal_type_present = false;
    let mut colour_description_present = false;
    if vui_parameters_present {
        // aspect_ratio_info_present_flag
        if r.read_flag().map_err(|_| e("ar_present"))? {
            let ar_idc = r.read_bits(8).map_err(|_| e("ar_idc"))?;
            if ar_idc == 255 {
                // Extended_SAR
                r.read_bits(16).map_err(|_| e("sar_w"))?;
                r.read_bits(16).map_err(|_| e("sar_h"))?;
            }
        }
        // overscan_info_present_flag
        if r.read_flag().map_err(|_| e("overscan_present"))? {
            r.read_flag().map_err(|_| e("overscan"))?;
        }
        // video_signal_type_present_flag
        video_signal_type_present = r.read_flag().map_err(|_| e("vst_present"))?;
        if video_signal_type_present {
            r.read_bits(3).map_err(|_| e("video_format"))?; // video_format
            video_full_range = r.read_flag().map_err(|_| e("full_range"))?;
            // color_description_present_flag
            colour_description_present = r.read_flag().map_err(|_| e("color_desc"))?;
            if colour_description_present {
                color_primaries = r.read_bits(8).map_err(|_| e("color_primaries"))? as u8;
                transfer_characteristics = r.read_bits(8).map_err(|_| e("transfer_char"))? as u8;
                matrix_coefficients = r.read_bits(8).map_err(|_| e("matrix_coeff"))? as u8;
            }
        }
        // chroma_loc_info_present_flag
        if r.read_flag().map_err(|_| e("chroma_loc_present"))? {
            r.read_ue().map_err(|_| e("chroma_sample_loc_top"))?;
            r.read_ue().map_err(|_| e("chroma_sample_loc_bottom"))?;
        }
        r.read_flag().map_err(|_| e("neutral_chroma"))?; // neutral_chroma_indication_flag
        r.read_flag().map_err(|_| e("field_seq"))?; // field_seq_flag
        r.read_flag().map_err(|_| e("frame_field_info"))?; // frame_field_info_present_flag
        // default_display_window_flag
        if r.read_flag().map_err(|_| e("def_disp_win"))? {
            r.read_ue().map_err(|_| e("ddw_left"))?;
            r.read_ue().map_err(|_| e("ddw_right"))?;
            r.read_ue().map_err(|_| e("ddw_top"))?;
            r.read_ue().map_err(|_| e("ddw_bottom"))?;
        }
        // vui_timing_info_present_flag — the frame-rate source.
        if r.read_flag().map_err(|_| e("timing_present"))? {
            vui_num_units_in_tick = r.read_bits(32).map_err(|_| e("num_units_in_tick"))?;
            vui_time_scale = r.read_bits(32).map_err(|_| e("time_scale"))?;
            if r.read_flag().map_err(|_| e("poc_proportional"))? {
                r.read_ue().map_err(|_| e("num_ticks_poc_diff_one"))?;
            }
            if r.read_flag().map_err(|_| e("hrd_present"))? {
                skip_hrd_parameters(&mut r, true, max_sub_layers_minus1)?;
            }
        }
        // bitstream_restriction (§E.2.1): present after the timing info; the
        // extension flags follow the VUI, so every field must be consumed.
        if r.read_flag().map_err(|_| e("bitstream_restriction"))? {
            r.read_flag().map_err(|_| e("tiles_fixed"))?;
            r.read_flag().map_err(|_| e("mc_over_boundaries"))?;
            r.read_flag().map_err(|_| e("restricted_ref_lists"))?;
            r.read_ue().map_err(|_| e("min_spatial_seg"))?;
            r.read_ue().map_err(|_| e("max_bytes_per_pic"))?;
            r.read_ue().map_err(|_| e("max_bits_per_min_cu"))?;
            r.read_ue().map_err(|_| e("log2_max_mv_h"))?;
            r.read_ue().map_err(|_| e("log2_max_mv_v"))?;
        }
    }

    // sps_extension_present_flag → range/multilayer/3d/scc/4bits (§7.3.2.2.1).
    let mut re = RangeExt::default();
    let mut scc = SccExt::default();
    if r.read_flag().map_err(|_| e("sps_ext_present"))? {
        let range_ext = r.read_flag().map_err(|_| e("sps_range_ext"))?;
        let multilayer_ext = r.read_flag().map_err(|_| e("sps_multilayer_ext"))?;
        let ext_3d = r.read_flag().map_err(|_| e("sps_3d_ext"))?;
        let scc_ext = r.read_flag().map_err(|_| e("sps_scc_ext"))?;
        r.read_bits(4).map_err(|_| e("sps_ext_4bits"))?;
        if range_ext {
            re = parse_sps_range_ext(&mut r)?;
        }
        if multilayer_ext {
            r.read_flag().map_err(|_| e("inter_view_mv_vert"))?;
        }
        if ext_3d {
            return Err(e("sps 3d extension unsupported"));
        }
        if scc_ext {
            scc = parse_sps_scc_ext(&mut r, chroma_idc, bit_depth_luma, bit_depth_chroma)?;
        }
    }

    let chroma = match chroma_idc {
        0 => ChromaFormat::Monochrome,
        1 => ChromaFormat::Yuv420,
        2 => ChromaFormat::Yuv422,
        3 => ChromaFormat::Yuv444,
        n => return Err(DecodeError::UnsupportedChroma(n)),
    };

    // HEVC stores conformance-window offsets in crop units, not pixels.
    // Convert them once here so all users of `Sps::crop_*` see luma-sample
    // offsets as documented above.
    let (crop_unit_x, crop_unit_y): (u32, u32) = if separate_color_plane {
        (1, 1)
    } else {
        match chroma {
            ChromaFormat::Monochrome | ChromaFormat::Yuv444 => (1, 1),
            ChromaFormat::Yuv422 => (2, 1),
            ChromaFormat::Yuv420 => (2, 2),
        }
    };
    let crop_left = crop_left.saturating_mul(crop_unit_x);
    let crop_right = crop_right.saturating_mul(crop_unit_x);
    let crop_top = crop_top.saturating_mul(crop_unit_y);
    let crop_bottom = crop_bottom.saturating_mul(crop_unit_y);

    Ok(Sps {
        id: sps_id,
        chroma_idc,
        chroma,
        separate_color_plane,
        width,
        height,
        crop_left,
        crop_right,
        crop_top,
        crop_bottom,
        bit_depth_luma,
        bit_depth_chroma,
        log2_ctb,
        log2_min_cb,
        log2_min_tb,
        log2_max_tb,
        max_transform_hierarchy_intra,
        max_transform_hierarchy_inter,
        log2_max_poc_lsb,
        max_num_reorder_pics,
        max_dec_pic_buffering,
        amp_enabled,
        temporal_mvp_enabled,
        short_term_rps,
        lt_ref_poc_lsb,
        lt_used_by_curr,
        long_term_ref_pics_present: long_term_present,
        scaling_list,
        scaling_list_enabled,
        sao_enabled,
        pcm_enabled,
        pcm_bit_depth_luma,
        pcm_bit_depth_chroma,
        log2_min_pcm_cb,
        log2_max_pcm_cb,
        pcm_loop_filter_disabled,
        strong_intra_smoothing,
        vui_parameters_present,
        video_signal_type_present,
        colour_description_present,
        video_full_range,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        vui_num_units_in_tick,
        vui_time_scale,
        transform_skip_rotation_enabled: re.transform_skip_rotation,
        transform_skip_context_enabled: re.transform_skip_context,
        implicit_rdpcm_enabled: re.implicit_rdpcm,
        explicit_rdpcm_enabled: re.explicit_rdpcm,
        extended_precision_processing: re.extended_precision,
        intra_smoothing_disabled: re.intra_smoothing_disabled,
        high_precision_offsets_enabled: re.high_precision_offsets,
        persistent_rice_adaptation_enabled: re.persistent_rice,
        cabac_bypass_alignment_enabled: re.cabac_bypass_alignment,
        curr_pic_ref_enabled: scc.curr_pic_ref,
        palette_mode_enabled: scc.palette_mode,
        palette_max_size: scc.palette_max_size,
        palette_max_predictor_size: scc.palette_max_predictor_size,
        palette_predictor_initializers: scc.palette_predictor_initializers,
        motion_vector_resolution_control_idc: scc.mv_resolution_control_idc,
        intra_boundary_filtering_disabled: scc.intra_boundary_filtering_disabled,
    })
}

#[derive(Default)]
struct RangeExt {
    transform_skip_rotation: bool,
    transform_skip_context: bool,
    implicit_rdpcm: bool,
    explicit_rdpcm: bool,
    extended_precision: bool,
    intra_smoothing_disabled: bool,
    high_precision_offsets: bool,
    persistent_rice: bool,
    cabac_bypass_alignment: bool,
}

fn parse_sps_range_ext(r: &mut BitReader) -> Result<RangeExt, DecodeError> {
    Ok(RangeExt {
        transform_skip_rotation: r.read_flag().map_err(|_| e("ts_rotation"))?,
        transform_skip_context: r.read_flag().map_err(|_| e("ts_context"))?,
        implicit_rdpcm: r.read_flag().map_err(|_| e("implicit_rdpcm"))?,
        explicit_rdpcm: r.read_flag().map_err(|_| e("explicit_rdpcm"))?,
        extended_precision: r.read_flag().map_err(|_| e("extended_precision"))?,
        intra_smoothing_disabled: r.read_flag().map_err(|_| e("intra_smoothing_dis"))?,
        high_precision_offsets: r.read_flag().map_err(|_| e("high_prec_offsets"))?,
        persistent_rice: r.read_flag().map_err(|_| e("persistent_rice"))?,
        cabac_bypass_alignment: r.read_flag().map_err(|_| e("cabac_bypass_align"))?,
    })
}

#[derive(Default)]
struct SccExt {
    curr_pic_ref: bool,
    palette_mode: bool,
    palette_max_size: u32,
    palette_max_predictor_size: u32,
    palette_predictor_initializers: Vec<Vec<u16>>,
    mv_resolution_control_idc: u8,
    intra_boundary_filtering_disabled: bool,
}

fn parse_sps_scc_ext(
    r: &mut BitReader,
    chroma_idc: u8,
    bd_luma: u8,
    bd_chroma: u8,
) -> Result<SccExt, DecodeError> {
    let curr_pic_ref = r.read_flag().map_err(|_| e("curr_pic_ref"))?;
    let palette_mode = r.read_flag().map_err(|_| e("palette_mode"))?;
    let mut palette_max_size = 0;
    let mut palette_max_predictor_size = 0;
    let mut palette_predictor_initializers = Vec::new();
    if palette_mode {
        palette_max_size = r.read_ue().map_err(|_| e("palette_max_size"))?;
        let delta = r.read_ue().map_err(|_| e("delta_palette_max_pred"))?;
        palette_max_predictor_size = palette_max_size.saturating_add(delta);
        if palette_max_size > 4096 || palette_max_predictor_size > 8192 {
            return Err(e("palette size out of range"));
        }
        if r.read_flag().map_err(|_| e("pal_pred_init_present"))? {
            let num = r
                .read_ue()
                .map_err(|_| e("num_pal_pred_init"))?
                .saturating_add(1);
            if num > palette_max_predictor_size {
                return Err(e("too many palette initializers"));
            }
            // numComps = (ChromaArrayType == 0) ? 1 : 3 — every non-monochrome
            // format stores three components.
            let num_comps = if chroma_idc == 0 { 1 } else { 3 };
            for c in 0..num_comps {
                let bits = if c == 0 { bd_luma } else { bd_chroma } as u32;
                let mut col = Vec::with_capacity(num as usize);
                for _ in 0..num {
                    col.push(r.read_bits(bits).map_err(|_| e("pal_pred_init"))? as u16);
                }
                palette_predictor_initializers.push(col);
            }
        }
    }
    let mv_resolution_control_idc = r.read_bits(2).map_err(|_| e("mv_res_ctrl"))? as u8;
    let intra_boundary_filtering_disabled = r.read_flag().map_err(|_| e("intra_bound_filt_dis"))?;
    Ok(SccExt {
        curr_pic_ref,
        palette_mode,
        palette_max_size,
        palette_max_predictor_size,
        palette_predictor_initializers,
        mv_resolution_control_idc,
        intra_boundary_filtering_disabled,
    })
}

pub(crate) fn parse_pps(rbsp: &[u8], _scaling_list_enabled: bool) -> Result<Pps, DecodeError> {
    let mut r = BitReader::new(rbsp);
    let pps_id = r.read_ue().map_err(|_| e("pps_id"))?;
    let pps_sps_id = r.read_ue().map_err(|_| e("pps_sps_id"))?;
    let dependent_slice_segments_enabled = r.read_flag().map_err(|_| e("dep_slices"))?;
    let output_flag_present = r.read_flag().map_err(|_| e("output_flag_present"))?;
    let num_extra_slice_header_bits = r.read_bits(3).map_err(|_| e("extra_hdr_bits"))?;
    let sign_data_hiding_enabled = r.read_flag().map_err(|_| e("sign_hiding"))?;
    let cabac_init_present = r.read_flag().map_err(|_| e("cabac_init"))?;
    let num_ref_idx_l0_default = r.read_ue().map_err(|_| e("nref0"))? as usize + 1;
    let num_ref_idx_l1_default = r.read_ue().map_err(|_| e("nref1"))? as usize + 1;
    let init_qp = 26i32
        .saturating_add(r.read_se().map_err(|_| e("init_qp"))?)
        .clamp(0, 51);
    let constrained_intra_pred = r.read_flag().map_err(|_| e("constrained_intra"))?;
    let transform_skip_enabled = r.read_flag().map_err(|_| e("transform_skip"))?;
    let cu_qp_delta_enabled = r.read_flag().map_err(|_| e("cu_qp_delta"))?;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled {
        r.read_ue().map_err(|_| e("diff_cu_qp_delta_depth"))?
    } else {
        0
    };
    let cb_qp_offset = r.read_se().map_err(|_| e("cb_qp_offset"))?;
    let cr_qp_offset = r.read_se().map_err(|_| e("cr_qp_offset"))?;
    let slice_chroma_qp_offsets_present = r.read_flag().map_err(|_| e("slice_chroma_qp"))?;
    let weighted_pred = r.read_flag().map_err(|_| e("weighted_pred"))?;
    let weighted_bipred = r.read_flag().map_err(|_| e("weighted_bipred"))?;
    let transquant_bypass_enabled = r.read_flag().map_err(|_| e("transquant_bypass"))?;
    let tiles_enabled = r.read_flag().map_err(|_| e("tiles"))?;
    let entropy_coding_sync_enabled = r.read_flag().map_err(|_| e("entropy_sync"))?;
    // Tile structure (§7.3.2.3.1). num_tile_columns_minus1 / num_tile_rows_minus1
    // are ue(v); the retained counts are the +1 values. When present but not
    // uniform, the explicit per-column/row sizes (in CTBs) are for all but the
    // last, which is implied by the picture size. Bound the counts to keep a
    // malformed PPS from allocating unboundedly.
    let mut num_tile_columns = 1u32;
    let mut num_tile_rows = 1u32;
    let mut tile_uniform_spacing = true;
    let mut tile_column_widths: Vec<u32> = Vec::new();
    let mut tile_row_heights: Vec<u32> = Vec::new();
    let mut loop_filter_across_tiles = true;
    if tiles_enabled {
        num_tile_columns = r
            .read_ue()
            .map_err(|_| e("num_tile_cols"))?
            .saturating_add(1);
        num_tile_rows = r
            .read_ue()
            .map_err(|_| e("num_tile_rows"))?
            .saturating_add(1);
        if num_tile_columns > 1024 || num_tile_rows > 1024 {
            return Err(e("tile count out of range"));
        }
        tile_uniform_spacing = r.read_flag().map_err(|_| e("uniform_spacing"))?;
        if !tile_uniform_spacing {
            for _ in 0..num_tile_columns.saturating_sub(1) {
                tile_column_widths.push(r.read_ue().map_err(|_| e("col_width"))?.saturating_add(1));
            }
            for _ in 0..num_tile_rows.saturating_sub(1) {
                tile_row_heights.push(r.read_ue().map_err(|_| e("row_height"))?.saturating_add(1));
            }
        }
        loop_filter_across_tiles = r.read_flag().map_err(|_| e("loop_filter_across_tiles"))?;
    }
    let loop_filter_across_slices = r.read_flag().map_err(|_| e("loop_filter_across_slices"))?;
    let deblocking_filter_control_present = r.read_flag().map_err(|_| e("deblock_control"))?;
    let mut deblocking_filter_override_enabled = false;
    let mut deblocking_filter_disabled = false;
    let mut beta_offset_div2 = 0;
    let mut tc_offset_div2 = 0;
    if deblocking_filter_control_present {
        deblocking_filter_override_enabled = r.read_flag().map_err(|_| e("deblock_override"))?;
        deblocking_filter_disabled = r.read_flag().map_err(|_| e("deblock_disabled"))?;
        if !deblocking_filter_disabled {
            beta_offset_div2 = r.read_se().map_err(|_| e("beta_offset"))?;
            tc_offset_div2 = r.read_se().map_err(|_| e("tc_offset"))?;
        }
    }
    // pps_scaling_list_data_present_flag is always present (1 bit), regardless
    // of the SPS scaling_list_enabled flag. Reading it conditionally desyncs the
    // rest of the PPS by one bit.
    let scaling_list = if r.read_flag().map_err(|_| e("pps_scaling_list_present"))? {
        Some(parse_scaling_list_data(&mut r)?)
    } else {
        None
    };
    let lists_modification_present = r.read_flag().map_err(|_| e("lists_modification"))?;
    let log2_parallel_merge_level = r.read_ue().map_err(|_| e("log2_parallel_merge"))? + 2;
    let slice_segment_header_extension_present = r.read_flag().map_err(|_| e("slice_hdr_ext"))?;

    let mut pre = PpsRangeExt::default();
    let mut pscc = PpsSccExt::default();
    if r.read_flag().map_err(|_| e("pps_ext_present"))? {
        let range_ext = r.read_flag().map_err(|_| e("pps_range_ext"))?;
        let multilayer_ext = r.read_flag().map_err(|_| e("pps_multilayer_ext"))?;
        let ext_3d = r.read_flag().map_err(|_| e("pps_3d_ext"))?;
        let scc_ext = r.read_flag().map_err(|_| e("pps_scc_ext"))?;
        r.read_bits(4).map_err(|_| e("pps_ext_4bits"))?;
        if range_ext {
            pre = parse_pps_range_ext(&mut r, transform_skip_enabled)?;
        }
        if multilayer_ext || ext_3d {
            return Err(e("pps multilayer/3d extension unsupported"));
        }
        if scc_ext {
            pscc = parse_pps_scc_ext(&mut r)?;
        }
    }

    Ok(Pps {
        id: pps_id,
        sps_id: pps_sps_id,
        dependent_slice_segments_enabled,
        output_flag_present,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled,
        cabac_init_present,
        init_qp,
        _constrained_intra_pred: constrained_intra_pred,
        transform_skip_enabled,
        cu_qp_delta_enabled,
        diff_cu_qp_delta_depth,
        cb_qp_offset,
        cr_qp_offset,
        slice_chroma_qp_offsets_present,
        weighted_pred,
        weighted_bipred,
        transquant_bypass_enabled,
        tiles_enabled,
        num_tile_columns,
        num_tile_rows,
        tile_uniform_spacing,
        tile_column_widths,
        tile_row_heights,
        loop_filter_across_tiles,
        entropy_coding_sync_enabled,
        loop_filter_across_slices,
        _deblocking_filter_control_present: deblocking_filter_control_present,
        deblocking_filter_override_enabled,
        deblocking_filter_disabled,
        beta_offset_div2,
        tc_offset_div2,
        scaling_list,
        lists_modification_present,
        log2_parallel_merge_level,
        num_ref_idx_l0_default,
        num_ref_idx_l1_default,
        slice_segment_header_extension_present,
        log2_max_transform_skip_block_size: pre.log2_max_transform_skip_block_size,
        cross_component_prediction_enabled: pre.cross_component_prediction,
        chroma_qp_offset_list_enabled: pre.chroma_qp_offset_list_enabled,
        diff_cu_chroma_qp_offset_depth: pre.diff_cu_chroma_qp_offset_depth,
        chroma_qp_offset_list: pre.chroma_qp_offset_list,
        log2_sao_offset_scale_luma: pre.log2_sao_offset_scale_luma,
        log2_sao_offset_scale_chroma: pre.log2_sao_offset_scale_chroma,
        curr_pic_ref_enabled: pscc.curr_pic_ref,
        residual_adaptive_colour_transform_enabled: pscc.residual_act_enabled,
        pps_slice_act_qp_offsets_present: pscc.slice_act_qp_offsets_present,
        pps_act_y_qp_offset: pscc.act_y_qp_offset,
        pps_act_cb_qp_offset: pscc.act_cb_qp_offset,
        pps_act_cr_qp_offset: pscc.act_cr_qp_offset,
        palette_predictor_initializer_present: pscc.palette_predictor_initializer_present,
        palette_predictor_initializers: pscc.palette_predictor_initializers,
        monochrome_palette: pscc.monochrome_palette,
        luma_bit_depth_entry: pscc.luma_bit_depth_entry,
        chroma_bit_depth_entry: pscc.chroma_bit_depth_entry,
    })
}

struct PpsRangeExt {
    log2_max_transform_skip_block_size: u32,
    cross_component_prediction: bool,
    chroma_qp_offset_list_enabled: bool,
    diff_cu_chroma_qp_offset_depth: u32,
    chroma_qp_offset_list: Vec<(i32, i32)>,
    log2_sao_offset_scale_luma: u32,
    log2_sao_offset_scale_chroma: u32,
}

impl Default for PpsRangeExt {
    fn default() -> Self {
        Self {
            // §7.4.3.3.3: log2_max_transform_skip_block_size_minus2 defaults to
            // 0 → block size 4×4 (log2 = 2) when the PPS range extension is absent.
            log2_max_transform_skip_block_size: 2,
            cross_component_prediction: false,
            chroma_qp_offset_list_enabled: false,
            diff_cu_chroma_qp_offset_depth: 0,
            chroma_qp_offset_list: Vec::new(),
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
        }
    }
}

fn parse_pps_range_ext(
    r: &mut BitReader,
    transform_skip_enabled: bool,
) -> Result<PpsRangeExt, DecodeError> {
    let mut out = PpsRangeExt {
        log2_max_transform_skip_block_size: 2,
        ..Default::default()
    };
    if transform_skip_enabled {
        out.log2_max_transform_skip_block_size =
            r.read_ue().map_err(|_| e("log2_max_ts_block"))? + 2;
    }
    out.cross_component_prediction = r.read_flag().map_err(|_| e("cross_comp_pred"))?;
    out.chroma_qp_offset_list_enabled = r.read_flag().map_err(|_| e("chroma_qp_off_list_en"))?;
    if out.chroma_qp_offset_list_enabled {
        out.diff_cu_chroma_qp_offset_depth = r.read_ue().map_err(|_| e("diff_cu_chroma_qp"))?;
        let len = r
            .read_ue()
            .map_err(|_| e("chroma_qp_off_list_len"))?
            .saturating_add(1);
        if len > 6 {
            return Err(e("chroma_qp_offset_list_len out of range"));
        }
        for _ in 0..len {
            let cb = r.read_se().map_err(|_| e("cb_qp_off_list"))?;
            let cr = r.read_se().map_err(|_| e("cr_qp_off_list"))?;
            out.chroma_qp_offset_list.push((cb, cr));
        }
    }
    out.log2_sao_offset_scale_luma = r.read_ue().map_err(|_| e("sao_scale_luma"))?;
    out.log2_sao_offset_scale_chroma = r.read_ue().map_err(|_| e("sao_scale_chroma"))?;
    Ok(out)
}

#[derive(Default)]
struct PpsSccExt {
    curr_pic_ref: bool,
    residual_act_enabled: bool,
    slice_act_qp_offsets_present: bool,
    act_y_qp_offset: i32,
    act_cb_qp_offset: i32,
    act_cr_qp_offset: i32,
    palette_predictor_initializer_present: bool,
    palette_predictor_initializers: Vec<Vec<u16>>,
    monochrome_palette: bool,
    luma_bit_depth_entry: u32,
    chroma_bit_depth_entry: u32,
}

fn parse_pps_scc_ext(r: &mut BitReader) -> Result<PpsSccExt, DecodeError> {
    let mut out = PpsSccExt {
        curr_pic_ref: r.read_flag().map_err(|_| e("pps_curr_pic_ref"))?,
        residual_act_enabled: r.read_flag().map_err(|_| e("residual_act_en"))?,
        ..Default::default()
    };
    if out.residual_act_enabled {
        // §7.3.2.3.3: the slice-present flag precedes the three QP offsets.
        out.slice_act_qp_offsets_present = r.read_flag().map_err(|_| e("slice_act_qp_present"))?;
        // The bitstream syntax elements are named *_plus5 / *_plus3; the
        // derived PPS ACT offsets subtract those biases (§7.4.3.3.3).
        out.act_y_qp_offset = r.read_se().map_err(|_| e("act_y_qp"))? - 5;
        out.act_cb_qp_offset = r.read_se().map_err(|_| e("act_cb_qp"))? - 5;
        out.act_cr_qp_offset = r.read_se().map_err(|_| e("act_cr_qp"))? - 3;
    }
    out.palette_predictor_initializer_present =
        r.read_flag().map_err(|_| e("pps_pal_pred_init_present"))?;
    if out.palette_predictor_initializer_present {
        let num = r.read_ue().map_err(|_| e("pps_num_pal_pred"))?;
        if num > 0 {
            out.monochrome_palette = r.read_flag().map_err(|_| e("mono_palette"))?;
            out.luma_bit_depth_entry = r.read_ue().map_err(|_| e("luma_bd_entry"))? + 8;
            if !out.monochrome_palette {
                out.chroma_bit_depth_entry = r.read_ue().map_err(|_| e("chroma_bd_entry"))? + 8;
            }
            let num_comps = if out.monochrome_palette { 1 } else { 3 };
            for c in 0..num_comps {
                let bits = if c == 0 {
                    out.luma_bit_depth_entry
                } else {
                    out.chroma_bit_depth_entry
                };
                let mut col = Vec::with_capacity(num as usize);
                for _ in 0..num {
                    col.push(r.read_bits(bits).map_err(|_| e("pps_pal_pred"))? as u16);
                }
                out.palette_predictor_initializers.push(col);
            }
        }
    }
    Ok(out)
}

/// Demux SPS+PPS from an hvcC and parse both fully.
pub(crate) fn parse_hvcc_full(hvcc: &[u8]) -> Result<(Sps, Pps), DecodeError> {
    if hvcc.len() < 23 {
        return Err(DecodeError::ParamSet("hvcC too short".into()));
    }
    let num_arrays = hvcc[22] as usize;
    let mut pos = 23usize;
    let mut sps_rbsp = None;
    let mut pps_rbsp = None;
    for _ in 0..num_arrays {
        if pos + 3 > hvcc.len() {
            break;
        }
        let nal_type = hvcc[pos] & 0x3f;
        pos += 1;
        let count = u16::from_be_bytes([hvcc[pos], hvcc[pos + 1]]) as usize;
        pos += 2;
        for _ in 0..count {
            if pos + 2 > hvcc.len() {
                break;
            }
            let nlen = u16::from_be_bytes([hvcc[pos], hvcc[pos + 1]]) as usize;
            pos += 2;
            if pos + nlen > hvcc.len() {
                break;
            }
            let nalu = &hvcc[pos..pos + nlen];
            pos += nlen;
            if nalu.len() < 2 {
                continue;
            }
            let rbsp = crate::bitreader::unescape_rbsp(&nalu[2..]);
            match nal_type {
                33 => sps_rbsp = Some(rbsp),
                34 => pps_rbsp = Some(rbsp),
                _ => {}
            }
        }
    }
    let sps = parse_sps(&sps_rbsp.ok_or_else(|| e("no SPS"))?)?;
    let pps = parse_pps(
        &pps_rbsp.ok_or_else(|| e("no PPS"))?,
        sps.scaling_list_enabled,
    )?;
    Ok((sps, pps))
}

/// Skip hrd_parameters (§E.2.2) so following SPS/VUI syntax stays aligned.
fn skip_hrd_parameters(
    r: &mut BitReader,
    common_inf: bool,
    max_sub_layers_minus1: u32,
) -> Result<(), DecodeError> {
    let mut nal_hrd = false;
    let mut vcl_hrd = false;
    let mut sub_pic = false;
    if common_inf {
        nal_hrd = r.read_flag().map_err(|_| e("nal_hrd"))?;
        vcl_hrd = r.read_flag().map_err(|_| e("vcl_hrd"))?;
        if nal_hrd || vcl_hrd {
            sub_pic = r.read_flag().map_err(|_| e("sub_pic_hrd"))?;
            if sub_pic {
                r.read_bits(8).map_err(|_| e("tick_divisor"))?;
                r.read_bits(5).map_err(|_| e("du_cpb_len"))?;
                r.read_flag().map_err(|_| e("sub_pic_cpb_in_pt"))?;
                r.read_bits(5).map_err(|_| e("dpb_du_len"))?;
            }
            r.read_bits(4).map_err(|_| e("bit_rate_scale"))?;
            r.read_bits(4).map_err(|_| e("cpb_size_scale"))?;
            if sub_pic {
                r.read_bits(4).map_err(|_| e("cpb_du_scale"))?;
            }
            r.read_bits(5).map_err(|_| e("init_cpb_len"))?;
            r.read_bits(5).map_err(|_| e("au_cpb_len"))?;
            r.read_bits(5).map_err(|_| e("dpb_out_len"))?;
        }
    }
    for _ in 0..=max_sub_layers_minus1 {
        let fixed_general = r.read_flag().map_err(|_| e("fixed_rate_gen"))?;
        let fixed_cvs = if !fixed_general {
            r.read_flag().map_err(|_| e("fixed_rate_cvs"))?
        } else {
            true
        };
        let low_delay = if fixed_cvs {
            r.read_ue().map_err(|_| e("elemental_duration"))?;
            false
        } else {
            r.read_flag().map_err(|_| e("low_delay"))?
        };
        let cpb_cnt = if !low_delay {
            r.read_ue().map_err(|_| e("cpb_cnt"))? as usize
        } else {
            0
        };
        for hrd_on in [nal_hrd, vcl_hrd] {
            if hrd_on {
                for _ in 0..=cpb_cnt {
                    r.read_ue().map_err(|_| e("bit_rate_value"))?;
                    r.read_ue().map_err(|_| e("cpb_size_value"))?;
                    if sub_pic {
                        r.read_ue().map_err(|_| e("cpb_size_du"))?;
                        r.read_ue().map_err(|_| e("bit_rate_du"))?;
                    }
                    r.read_flag().map_err(|_| e("cbr"))?;
                }
            }
        }
    }
    Ok(())
}
