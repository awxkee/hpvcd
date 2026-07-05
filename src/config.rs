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
    pub(crate) _max_transform_hierarchy_inter: u32,
    pub(crate) _scaling_list_enabled: bool,
    pub(crate) _amp_enabled: bool,
    pub(crate) sao_enabled: bool,
    pub(crate) _pcm_enabled: bool,
    pub(crate) _pcm_bit_depth_luma: u8,
    pub(crate) _pcm_bit_depth_chroma: u8,
    pub(crate) _log2_min_pcm_cb: u32,
    pub(crate) _log2_max_pcm_cb: u32,
    pub(crate) _pcm_loop_filter_disabled: bool,
    pub(crate) strong_intra_smoothing: bool,
    pub(crate) video_full_range: bool,
    pub(crate) color_primaries: u8, // ISO/IEC 23091-2 Table 2; 2 = unspecified
    pub(crate) transfer_characteristics: u8, // ISO/IEC 23091-2 Table 3; 2 = unspecified
    pub(crate) matrix_coefficients: u8, // ISO/IEC 23091-2 Table 4; 2 = unspecified
}

#[derive(Debug, Clone)]
pub(crate) struct Pps {
    pub(crate) dependent_slice_segments_enabled: bool,
    pub(crate) output_flag_present: bool,
    pub(crate) num_extra_slice_header_bits: u32,
    pub(crate) sign_data_hiding_enabled: bool,
    pub(crate) _cabac_init_present: bool,
    pub(crate) init_qp: i32,
    pub(crate) _constrained_intra_pred: bool,
    pub(crate) transform_skip_enabled: bool,
    pub(crate) cu_qp_delta_enabled: bool,
    pub(crate) diff_cu_qp_delta_depth: u32,
    pub(crate) cb_qp_offset: i32,
    pub(crate) cr_qp_offset: i32,
    pub(crate) slice_chroma_qp_offsets_present: bool,
    pub(crate) _weighted_pred: bool,
    pub(crate) _weighted_bipred: bool,
    pub(crate) transquant_bypass_enabled: bool,
    pub(crate) tiles_enabled: bool,
    pub(crate) entropy_coding_sync_enabled: bool,
    pub(crate) loop_filter_across_slices: bool,
    pub(crate) _deblocking_filter_control_present: bool,
    pub(crate) deblocking_filter_override_enabled: bool,
    pub(crate) deblocking_filter_disabled: bool,
    pub(crate) beta_offset_div2: i32,
    pub(crate) tc_offset_div2: i32,
    pub(crate) _scaling_list_data_present: bool,
    pub(crate) _lists_modification_present: bool,
    pub(crate) _log2_parallel_merge_level: u32,
    pub(crate) slice_segment_header_extension_present: bool,
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

/// Parse scaling_list_data (§7.3.4) — we only need to skip past it.
fn skip_scaling_list_data(r: &mut BitReader) -> Result<(), DecodeError> {
    for size_id in 0..4 {
        let mut matrix_id = 0;
        while matrix_id < 6 {
            let pred_mode_flag = r
                .read_flag()
                .map_err(|_| e("scaling_list_pred_mode_flag"))?;
            if !pred_mode_flag {
                r.read_ue()
                    .map_err(|_| e("scaling_list_pred_matrix_id_delta"))?;
            } else {
                let coef_num = std::cmp::min(64, 1 << (4 + (size_id << 1)));
                if size_id > 1 {
                    r.read_se().map_err(|_| e("scaling_list_dc_coef"))?;
                }
                for _ in 0..coef_num {
                    r.read_se().map_err(|_| e("scaling_list_delta_coef"))?;
                }
            }
            matrix_id += if size_id == 3 { 3 } else { 1 };
        }
    }
    Ok(())
}

/// Parse a short_term_ref_pic_set (§7.3.7). Returns NumDeltaPocs for this set.
fn parse_st_rps(
    r: &mut BitReader,
    idx: usize,
    num_sets: usize,
    num_delta_pocs: &mut Vec<u32>,
) -> Result<(), DecodeError> {
    let mut inter_pred = false;
    if idx != 0 {
        inter_pred = r.read_flag().map_err(|_| e("inter_ref_pic_set_pred"))?;
    }
    if inter_pred {
        if idx == num_sets {
            r.read_ue().map_err(|_| e("delta_idx_minus1"))?;
        }
        r.read_bit().map_err(|_| e("delta_rps_sign"))?;
        r.read_ue().map_err(|_| e("abs_delta_rps_minus1"))?;
        let ref_idx = idx - 1; // simplification (delta_idx assumed 1)
        let n = num_delta_pocs.get(ref_idx).copied().unwrap_or(0);
        let mut count = 0u32;
        for _ in 0..=n {
            let used = r.read_flag().map_err(|_| e("used_by_curr_pic"))?;
            if !used {
                let use_delta = r.read_flag().map_err(|_| e("use_delta_flag"))?;
                if use_delta {
                    count += 1;
                }
            } else {
                count += 1;
            }
        }
        num_delta_pocs.push(count);
    } else {
        let num_neg = r.read_ue().map_err(|_| e("num_negative_pics"))?;
        let num_pos = r.read_ue().map_err(|_| e("num_positive_pics"))?;
        for _ in 0..num_neg {
            r.read_ue().map_err(|_| e("delta_poc_s0"))?;
            r.read_bit().map_err(|_| e("used_by_curr_s0"))?;
        }
        for _ in 0..num_pos {
            r.read_ue().map_err(|_| e("delta_poc_s1"))?;
            r.read_bit().map_err(|_| e("used_by_curr_s1"))?;
        }
        num_delta_pocs.push(num_neg + num_pos);
    }
    Ok(())
}

pub(crate) fn parse_sps(rbsp: &[u8]) -> Result<Sps, DecodeError> {
    let mut r = BitReader::new(rbsp);
    r.read_bits(4).map_err(|_| e("sps_vps_id"))?;
    let max_sub_layers_minus1 = r.read_bits(3).map_err(|_| e("max_sub_layers"))?;
    r.read_bit().map_err(|_| e("temporal_id_nesting"))?;
    parse_ptl(&mut r, max_sub_layers_minus1)?;

    let _sps_id = r.read_ue().map_err(|_| e("sps_id"))?;
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
    let log2_max_poc = r.read_ue().map_err(|_| e("log2_max_poc"))? + 4;
    let _ = log2_max_poc;

    let sub_layer_ordering = r.read_flag().map_err(|_| e("sub_layer_ordering"))?;
    let start = if sub_layer_ordering {
        0
    } else {
        max_sub_layers_minus1
    };
    for _ in start..=max_sub_layers_minus1 {
        r.read_ue().map_err(|_| e("max_dec_pic_buffering"))?;
        r.read_ue().map_err(|_| e("num_reorder_pics"))?;
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
    if scaling_list_enabled {
        let present = r.read_flag().map_err(|_| e("sps_scaling_list_present"))?;
        if present {
            skip_scaling_list_data(&mut r)?;
        }
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
    let mut num_delta_pocs = Vec::with_capacity(num_st_rps);
    for i in 0..num_st_rps {
        parse_st_rps(&mut r, i, num_st_rps, &mut num_delta_pocs)?;
    }

    let long_term_present = r.read_flag().map_err(|_| e("long_term_present"))?;
    if long_term_present {
        let num_lt = r.read_ue().map_err(|_| e("num_long_term"))? as usize;
        for _ in 0..num_lt {
            r.read_bits(log2_max_poc).map_err(|_| e("lt_ref_poc"))?;
            r.read_bit().map_err(|_| e("used_by_curr_lt"))?;
        }
    }

    let _temporal_mvp = r.read_flag().map_err(|_| e("temporal_mvp"))?;
    let strong_intra_smoothing = r.read_flag().map_err(|_| e("strong_intra_smoothing"))?;
    // VUI parameters — extract matrix_coefficients and video_full_range_flag.
    let mut video_full_range = false;
    let mut color_primaries = 2u8; // unspecified
    let mut transfer_characteristics = 2u8; // unspecified
    let mut matrix_coefficients = 2u8; // unspecified
    if r.read_flag().map_err(|_| e("vui_present"))? {
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
        if r.read_flag().map_err(|_| e("vst_present"))? {
            r.read_bits(3).map_err(|_| e("video_format"))?; // video_format
            video_full_range = r.read_flag().map_err(|_| e("full_range"))?;
            // color_description_present_flag
            if r.read_flag().map_err(|_| e("color_desc"))? {
                color_primaries = r.read_bits(8).map_err(|_| e("color_primaries"))? as u8;
                transfer_characteristics = r.read_bits(8).map_err(|_| e("transfer_char"))? as u8;
                matrix_coefficients = r.read_bits(8).map_err(|_| e("matrix_coeff"))? as u8;
            }
        }
        // We only need the above; skip the rest of the VUI.
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
        _max_transform_hierarchy_inter: max_transform_hierarchy_inter,
        _scaling_list_enabled: scaling_list_enabled,
        _amp_enabled: amp_enabled,
        sao_enabled,
        _pcm_enabled: pcm_enabled,
        _pcm_bit_depth_luma: pcm_bit_depth_luma,
        _pcm_bit_depth_chroma: pcm_bit_depth_chroma,
        _log2_min_pcm_cb: log2_min_pcm_cb,
        _log2_max_pcm_cb: log2_max_pcm_cb,
        _pcm_loop_filter_disabled: pcm_loop_filter_disabled,
        strong_intra_smoothing,
        video_full_range,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
    })
}

pub(crate) fn parse_pps(rbsp: &[u8]) -> Result<Pps, DecodeError> {
    let mut r = BitReader::new(rbsp);
    let _pps_id = r.read_ue().map_err(|_| e("pps_id"))?;
    let _sps_id = r.read_ue().map_err(|_| e("pps_sps_id"))?;
    let dependent_slice_segments_enabled = r.read_flag().map_err(|_| e("dep_slices"))?;
    let output_flag_present = r.read_flag().map_err(|_| e("output_flag_present"))?;
    let num_extra_slice_header_bits = r.read_bits(3).map_err(|_| e("extra_hdr_bits"))?;
    let sign_data_hiding_enabled = r.read_flag().map_err(|_| e("sign_hiding"))?;
    let cabac_init_present = r.read_flag().map_err(|_| e("cabac_init"))?;
    let _nref0 = r.read_ue().map_err(|_| e("nref0"))?;
    let _nref1 = r.read_ue().map_err(|_| e("nref1"))?;
    let init_qp = 26 + r.read_se().map_err(|_| e("init_qp"))?;
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
    if tiles_enabled {
        let cols = r.read_ue().map_err(|_| e("num_tile_cols"))?;
        let rows = r.read_ue().map_err(|_| e("num_tile_rows"))?;
        let uniform = r.read_flag().map_err(|_| e("uniform_spacing"))?;
        if !uniform {
            for _ in 0..cols {
                r.read_ue().map_err(|_| e("col_width"))?;
            }
            for _ in 0..rows {
                r.read_ue().map_err(|_| e("row_height"))?;
            }
        }
        r.read_flag().map_err(|_| e("loop_filter_across_tiles"))?;
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
    let scaling_list_data_present = r.read_flag().map_err(|_| e("pps_scaling_list_present"))?;
    if scaling_list_data_present {
        skip_scaling_list_data(&mut r)?;
    }
    let lists_modification_present = r.read_flag().map_err(|_| e("lists_modification"))?;
    let log2_parallel_merge_level = r.read_ue().map_err(|_| e("log2_parallel_merge"))? + 2;
    let slice_segment_header_extension_present = r.read_flag().map_err(|_| e("slice_hdr_ext"))?;

    Ok(Pps {
        dependent_slice_segments_enabled,
        output_flag_present,
        num_extra_slice_header_bits,
        sign_data_hiding_enabled,
        _cabac_init_present: cabac_init_present,
        init_qp,
        _constrained_intra_pred: constrained_intra_pred,
        transform_skip_enabled,
        cu_qp_delta_enabled,
        diff_cu_qp_delta_depth,
        cb_qp_offset,
        cr_qp_offset,
        slice_chroma_qp_offsets_present,
        _weighted_pred: weighted_pred,
        _weighted_bipred: weighted_bipred,
        transquant_bypass_enabled,
        tiles_enabled,
        entropy_coding_sync_enabled,
        loop_filter_across_slices,
        _deblocking_filter_control_present: deblocking_filter_control_present,
        deblocking_filter_override_enabled,
        deblocking_filter_disabled,
        beta_offset_div2,
        tc_offset_div2,
        _scaling_list_data_present: scaling_list_data_present,
        _lists_modification_present: lists_modification_present,
        _log2_parallel_merge_level: log2_parallel_merge_level,
        slice_segment_header_extension_present,
    })
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
    let pps = parse_pps(&pps_rbsp.ok_or_else(|| e("no PPS"))?)?;
    Ok((sps, pps))
}
