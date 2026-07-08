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

use crate::{cabac, deblock, intra, reconstruct, sao, transform};

#[derive(Clone)]
pub(crate) struct ExecContext {
    pub(crate) predict: intra::PredictFn,
    pub(crate) residual_scans: &'static cabac::ResidualScanTables,

    pub(crate) dequant: transform::DequantFn,
    pub(crate) dequant16: transform::DequantFn16,
    pub(crate) dequant_skip: transform::DequantSkipFn,
    pub(crate) dequant_skip16: transform::DequantSkipFn16,
    pub(crate) dequant_scaled: transform::DequantScaledFn,
    pub(crate) dequant_scaled16: transform::DequantScaledFn16,
    pub(crate) dequant_skip_scaled: transform::DequantSkipScaledFn,
    pub(crate) dequant_skip_scaled16: transform::DequantSkipScaledFn16,

    pub(crate) inv_transform: transform::InvTransformFn,
    pub(crate) inv_transform_dst4: transform::InvTransform4Fn,
    pub(crate) inv_transform16: transform::InvTransformFn16,
    pub(crate) inv_transform_dst4_16: transform::InvTransform4Fn16,

    pub(crate) reconstruct: reconstruct::ReconstructFn,
    pub(crate) reconstruct16: reconstruct::ReconstructFn16,

    pub(crate) sao_plane: sao::SaoPlaneFn,
    pub(crate) sao_plane_banded: sao::SaoPlaneBandedFn,

    pub(crate) luma_deblock_vertical: deblock::LumaDeblockPlaneFn,
    pub(crate) luma_deblock_horizontal: deblock::LumaDeblockPlaneFn,
    pub(crate) chroma_deblock_vertical: deblock::ChromaDeblockPlaneFn,
    pub(crate) chroma_deblock_horizontal: deblock::ChromaDeblockPlaneFn,
}

impl Default for ExecContext {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ExecContext {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            predict: intra::resolve_predict(),
            residual_scans: cabac::resolve_residual_scan_tables(),

            dequant: transform::resolve_dequant(),
            dequant16: transform::resolve_dequant16(),
            dequant_skip: transform::resolve_dequant_skip(),
            dequant_skip16: transform::resolve_dequant_skip16(),
            dequant_scaled: transform::resolve_dequant_scaled(),
            dequant_scaled16: transform::resolve_dequant_scaled16(),
            dequant_skip_scaled: transform::resolve_dequant_skip_scaled(),
            dequant_skip_scaled16: transform::resolve_dequant_skip_scaled16(),

            inv_transform: transform::resolve_inv_transform(),
            inv_transform_dst4: transform::resolve_inv_transform_dst4(),
            inv_transform16: transform::resolve_inv_transform16(),
            inv_transform_dst4_16: transform::resolve_inv_transform_dst4_16(),

            reconstruct: reconstruct::resolve_reconstruct_add_clip(),
            reconstruct16: reconstruct::resolve_reconstruct_add_clip16(),

            sao_plane: sao::resolve_apply_sao_plane(),
            sao_plane_banded: sao::resolve_apply_sao_plane_banded(),

            luma_deblock_vertical: deblock::resolve_luma_vertical_plane(),
            luma_deblock_horizontal: deblock::resolve_luma_horizontal_plane(),
            chroma_deblock_vertical: deblock::resolve_chroma_vertical_plane(),
            chroma_deblock_horizontal: deblock::resolve_chroma_horizontal_plane(),
        }
    }
}
