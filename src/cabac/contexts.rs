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
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(crate) struct CtxModel {
    // HEVC CABAC context in one byte: bits 0..5 are pStateIdx, bit 6 is valMPS.
    // Keeping it packed cuts context-set traffic/cloning and gives decode_bin one
    // context load + one context store instead of touching two independent bytes.
    pub(crate) state: u8,
}

impl CtxModel {
    #[inline(always)]
    pub(crate) fn new(p_state_idx: u8, val_mps: u8) -> Self {
        debug_assert!(p_state_idx < 64);
        debug_assert!(val_mps <= 1);
        Self {
            state: (p_state_idx & 63) | ((val_mps & 1) << 6),
        }
    }

    /// Formula is identical to hpvca CabacEncoder.
    pub(crate) fn init(init_value: u8, qp: u8) -> Self {
        let slope_idx = (init_value >> 4) as i32;
        let offset_idx = (init_value & 0x0F) as i32;
        let m = slope_idx * 5 - 45;
        let n = (offset_idx << 3) - 16;
        let qpc = (qp as i32).clamp(0, 51);
        let pre = (((m * qpc) >> 4) + n).clamp(1, 126);
        if pre >= 64 {
            CtxModel::new((pre - 64) as u8, 1)
        } else {
            CtxModel::new((63 - pre) as u8, 0)
        }
    }
}

/// All context models for I-slice residual coding, keyed by ffmpeg's
/// init_values[0] (I-slice column).
#[derive(Clone)]
pub(crate) struct ContextSet {
    pub(crate) _qp: u8,

    // CU-level
    pub(crate) split_cu_flag: [CtxModel; 3],

    // split_transform_flag[trafoDepth]: 3 contexts (initValues: 153,138,138)
    pub(crate) split_transform_flag: [CtxModel; 3],

    // CBF: cbf_luma[0..1], cbf_chroma[0..4]
    pub(crate) cbf_luma: [CtxModel; 2],
    pub(crate) cbf_chroma: [CtxModel; 5], // cb_trafoDepth0, cb_tD1, cr_tD0, cr_tD1, cr_tD2

    // last_sig_coeff_x/y prefix (18 contexts each)
    pub(crate) last_sig_coeff_x_prefix: [CtxModel; 18],
    pub(crate) last_sig_coeff_y_prefix: [CtxModel; 18],

    // sig_coeff_flag (44 contexts, luma+chroma merged)
    pub(crate) sig_coeff_flag: [CtxModel; 44],

    // coded_sub_block_flag (4 contexts: 2 luma + 2 chroma)
    pub(crate) coded_sub_block_flag: [CtxModel; 4],

    // coeff_abs_level_greater1 (24 contexts)
    pub(crate) coeff_abs_level_greater1: [CtxModel; 24],

    // coeff_abs_level_greater2 (6 contexts)
    pub(crate) coeff_abs_level_greater2: [CtxModel; 6],

    // SAO: sao_merge_left/up flag (1 ctx, shared) and sao_type_idx (1 ctx).
    // initType0: sao_merge=153, sao_type_idx=200.
    pub(crate) sao_merge_flag: CtxModel,
    pub(crate) sao_type_idx: CtxModel,

    // transform_skip_flag[luma/chroma], initType0 = {139, 139}
    pub(crate) transform_skip_flag: [CtxModel; 2],
    // cu_qp_delta_abs (2 ctx), initType0 = {154, 154}
    pub(crate) cu_qp_delta_abs: [CtxModel; 2],
    // cu_transquant_bypass_flag (1 ctx), initValue 154 (all init types).
    pub(crate) cu_transquant_bypass_flag: CtxModel,

    // ---- Inter-only contexts (present for P/B slices) ----
    pub(crate) cu_skip_flag: [CtxModel; 3],
    pub(crate) pred_mode_flag: CtxModel,
    pub(crate) merge_flag: CtxModel,
    pub(crate) merge_idx: CtxModel,
    pub(crate) inter_pred_idc: [CtxModel; 5],
    pub(crate) ref_idx: [CtxModel; 2],
    pub(crate) abs_mvd_greater01: [CtxModel; 2],
    pub(crate) mvp_flag: CtxModel,
    pub(crate) rqt_root_cbf: CtxModel,
    /// Inter part_mode contexts (4), §9.3.4.2.
    pub(crate) part_mode: [CtxModel; 4],
}

impl ContextSet {
    pub(crate) fn init_islice(qp: u8) -> Self {
        fn c(iv: u8, qp: u8) -> CtxModel {
            CtxModel::init(iv, qp)
        }
        fn arr<const N: usize>(ivs: [u8; N], qp: u8) -> [CtxModel; N] {
            ivs.map(|iv| CtxModel::init(iv, qp))
        }

        // I-slice initValues (initType=0) — authoritative, from libde265 contextmodel.cc.
        Self {
            _qp: qp,
            // split_cu_flag initType0: {139,141,157}
            split_cu_flag: arr([139, 141, 157], qp),

            // split_transform_flag initType0 (×3): {153,138,138}
            split_transform_flag: arr([153, 138, 138], qp),

            // cbf_luma initType0 (×2): {111,141}
            cbf_luma: arr([111, 141], qp),

            // cbf_chroma initType0 (×4): {94,138,182,154}; 5th slot reuses last.
            cbf_chroma: arr([94, 138, 182, 154, 154], qp),

            // last_significant_coeff_x_prefix (18) — initType0
            last_sig_coeff_x_prefix: arr(
                [
                    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108,
                    123, 63,
                ],
                qp,
            ),

            // last_significant_coeff_y_prefix (18) — initType0 (same table)
            last_sig_coeff_y_prefix: arr(
                [
                    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108,
                    123, 63,
                ],
                qp,
            ),

            // significant_coeff_flag — initType0 row (42 values + 2 pad)
            sig_coeff_flag: arr(
                [
                    111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107,
                    125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152,
                    136, 152, 136, 153, 136, 139, 111, 136, 139, 111, 141, 111,
                ],
                qp,
            ),

            // coded_sub_block_flag initType0 (×4): {91,171,134,141}
            coded_sub_block_flag: arr([91, 171, 134, 141], qp),

            // coeff_abs_level_greater1_flag (24) — initType0
            coeff_abs_level_greater1: arr(
                [
                    140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152,
                    140, 179, 166, 182, 140, 227, 122, 197,
                ],
                qp,
            ),

            // coeff_abs_level_greater2_flag (6) — initType0: {138,153,136,167,152,152}
            coeff_abs_level_greater2: arr([138, 153, 136, 167, 152, 152], qp),

            // SAO initType0: sao_merge_flag=153, sao_type_idx=200 (libde265).
            sao_merge_flag: c(153, qp),
            sao_type_idx: c(200, qp),

            transform_skip_flag: arr([139, 139], qp),
            cu_qp_delta_abs: arr([154, 154], qp),
            cu_transquant_bypass_flag: c(154, qp),
            // Inter contexts are unused in I-slices; initialise to initType-0-ish
            // placeholders (never read).
            cu_skip_flag: arr([197, 185, 201], qp),
            pred_mode_flag: c(149, qp),
            merge_flag: c(110, qp),
            merge_idx: c(122, qp),
            inter_pred_idc: arr([95, 79, 63, 31, 31], qp),
            ref_idx: arr([153, 153], qp),
            abs_mvd_greater01: arr([140, 198], qp),
            mvp_flag: c(168, qp),
            rqt_root_cbf: c(79, qp),
            part_mode: arr([184, 154, 139, 154], qp),
        }
    }

    /// Table-driven init for any slice type. `init_type` is 0 (I), 1 (P) or 2
    /// (B) per §9.3.2.2. All init values from libde265 `contextmodel.cc`.
    pub(crate) fn init(init_type: u8, qp: u8) -> Self {
        if init_type == 0 {
            return Self::init_islice(qp);
        }
        fn c(iv: u8, qp: u8) -> CtxModel {
            CtxModel::init(iv, qp)
        }
        fn arr<const N: usize>(ivs: [u8; N], qp: u8) -> [CtxModel; N] {
            ivs.map(|iv| CtxModel::init(iv, qp))
        }
        let it = init_type as usize; // 1 or 2
        // Row selectors into the multi-initType tables.
        let split_cu = [[139u8, 141, 157], [107, 139, 126], [107, 139, 126]][it];
        let split_tf = [[153u8, 138, 138], [124, 138, 94], [224, 167, 122]][it];
        let cbf_luma = [[111u8, 141], [153, 111], [153, 111]][it];
        let cbf_chroma_all: [[u8; 4]; 3] = [
            [94, 138, 182, 154],
            [149, 107, 167, 154],
            [149, 92, 167, 154],
        ];
        let cbfc = cbf_chroma_all[it];
        let last_sig: [u8; 18] = {
            let base = it * 18;
            let t = LAST_SIG_PREFIX;
            core::array::from_fn(|i| t[base + i])
        };
        let sig_row = SIG_COEFF_FLAG[it];
        let csb = [
            [91u8, 171, 134, 141],
            [121, 140, 61, 154],
            [121, 140, 61, 154],
        ][it];
        let g1: [u8; 24] = {
            let base = it * 24;
            core::array::from_fn(|i| COEFF_G1[base + i])
        };
        let g2: [u8; 6] = {
            let base = it * 6;
            core::array::from_fn(|i| COEFF_G2[base + i])
        };
        // Inter tables: index [it-1] (two entries: P then B).
        let inter = it - 1;
        Self {
            _qp: qp,
            split_cu_flag: arr(split_cu, qp),
            split_transform_flag: arr(split_tf, qp),
            cbf_luma: arr(cbf_luma, qp),
            cbf_chroma: arr([cbfc[0], cbfc[1], cbfc[2], cbfc[3], cbfc[3]], qp),
            last_sig_coeff_x_prefix: arr(last_sig, qp),
            last_sig_coeff_y_prefix: arr(last_sig, qp),
            sig_coeff_flag: {
                let mut a = [CtxModel::init(0, qp); 44];
                for i in 0..42 {
                    a[i] = CtxModel::init(sig_row[i], qp);
                }
                a[42] = CtxModel::init(sig_row[41], qp);
                a[43] = CtxModel::init(sig_row[41], qp);
                a
            },
            coded_sub_block_flag: arr(csb, qp),
            coeff_abs_level_greater1: arr(g1, qp),
            coeff_abs_level_greater2: arr(g2, qp),
            sao_merge_flag: c(153, qp),
            sao_type_idx: c([200u8, 185, 160][it], qp),
            transform_skip_flag: arr([139, 139], qp),
            cu_qp_delta_abs: arr([154, 154], qp),
            cu_transquant_bypass_flag: c(154, qp),
            cu_skip_flag: arr([[197u8, 185, 201], [197, 185, 201]][inter], qp),
            pred_mode_flag: c([149u8, 134][inter], qp),
            merge_flag: c([110u8, 154][inter], qp),
            merge_idx: c([122u8, 137][inter], qp),
            inter_pred_idc: arr([95, 79, 63, 31, 31], qp),
            ref_idx: arr([153, 153], qp),
            abs_mvd_greater01: arr([[140u8, 198], [169, 198]][inter], qp),
            mvp_flag: c(168, qp),
            rqt_root_cbf: c(79, qp),
            part_mode: {
                let pm = [184u8, 154, 139, 154, 154, 154, 139, 154, 154];
                let off = if it != 2 { it } else { 5 };
                core::array::from_fn(|i| CtxModel::init(pm[off + i], qp))
            },
        }
    }
}

// Authoritative multi-initType tables (libde265 contextmodel.cc).
const LAST_SIG_PREFIX: [u8; 54] = [
    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63, 125,
    110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108, 125, 110, 124,
    110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93,
];
const SIG_COEFF_FLAG: [[u8; 42]; 3] = [
    [
        111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141,
        179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153,
        136, 139, 111, 136, 139, 111,
    ],
    [
        155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140,
        136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167,
        151, 183, 140, 151, 183, 140,
    ],
    [
        170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140,
        136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121, 167,
        151, 183, 140, 151, 183, 140,
    ],
];
const COEFF_G1: [u8; 72] = [
    140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166,
    182, 140, 227, 122, 197, 154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121,
    136, 137, 169, 194, 166, 167, 154, 167, 137, 182, 154, 196, 167, 167, 154, 152, 167, 182, 182,
    134, 149, 136, 153, 121, 136, 122, 169, 208, 166, 167, 154, 152, 167, 182,
];
const COEFF_G2: [u8; 18] = [
    138, 153, 136, 167, 152, 152, 107, 167, 91, 122, 107, 167, 107, 167, 91, 107, 107, 167,
];

/// Intra-mode contexts (prev_intra_luma_pred_flag, intra_chroma_pred_mode).
/// I-slice (initType=0) init values from libde265:
///   part_mode = 184, prev_intra_luma_pred_flag = 184, intra_chroma_pred_mode = 63
#[derive(Clone, Copy, Debug)]
pub(crate) struct IntraModeContexts {
    pub(crate) _part_mode: CtxModel,
    pub(crate) prev_intra_luma_pred_flag: CtxModel,
    pub(crate) intra_chroma_pred_mode: CtxModel,
}

impl IntraModeContexts {
    pub(crate) fn init_islice(qp: u8) -> Self {
        Self {
            _part_mode: CtxModel::init(184, qp),
            prev_intra_luma_pred_flag: CtxModel::init(184, qp),
            intra_chroma_pred_mode: CtxModel::init(63, qp),
        }
    }

    /// initType 0/1/2. part_mode has 4 contexts across init types; for the
    /// single CABAC-context part_mode bin the P/B init values are 154.
    pub(crate) fn init(init_type: u8, qp: u8) -> Self {
        if init_type == 0 {
            return Self::init_islice(qp);
        }
        let it = init_type as usize;
        Self {
            _part_mode: CtxModel::init(154, qp),
            prev_intra_luma_pred_flag: CtxModel::init([184u8, 154, 183][it], qp),
            intra_chroma_pred_mode: CtxModel::init([63u8, 152, 152][it], qp),
        }
    }
}
