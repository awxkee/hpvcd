/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * // BSD-3-Clause OR Apache-2.0
 */

//! Inter-prediction data types: motion vectors, per-PU motion info, and the
//! weighted-prediction table. Motion derivation and compensation live in
//! `motion.rs` / `mc.rs`; this module holds the shared plain-data structures.

use crate::bitreader::BitReader;
use crate::error::DecodeError;

fn e(s: &'static str) -> DecodeError {
    DecodeError::Bitstream(s.into())
}

/// A quarter-pel motion vector.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(crate) struct Mv {
    pub(crate) x: i16,
    pub(crate) y: i16,
}

impl Mv {
    #[inline]
    pub(crate) fn new(x: i16, y: i16) -> Self {
        Mv { x, y }
    }
}

/// Prediction direction flags for a PU.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(crate) struct PredFlags {
    pub(crate) l0: bool,
    pub(crate) l1: bool,
}

/// Per-PU (or per-4x4 block) motion information stored in the picture's motion
/// field, used by spatial/temporal MV prediction of later blocks and by the
/// collocated temporal predictor of later pictures.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct MotionInfo {
    pub(crate) pred: PredFlags,
    pub(crate) mv: [Mv; 2],
    /// Reference index into RefPicList0 / RefPicList1 (-1 = unused).
    pub(crate) ref_idx: [i8; 2],
    /// POC of the referenced pictures (for temporal scaling), valid where used.
    pub(crate) ref_poc: [i32; 2],
    /// True when the block is intra-coded (not a valid MV predictor).
    pub(crate) is_intra: bool,
}

impl MotionInfo {
    #[inline]
    pub(crate) fn intra() -> Self {
        MotionInfo {
            is_intra: true,
            ref_idx: [-1, -1],
            ..Default::default()
        }
    }
}

/// Weighted prediction parameters (§7.4.7.3). Weights are applied as
/// `(sample * w + (1 << (shift-1))) >> shift + o` with `shift = log2_denom`.
#[derive(Clone, Debug)]
pub(crate) struct PredWeightTable {
    pub(crate) luma_log2_denom: u8,
    pub(crate) chroma_log2_denom: u8,
    /// [list][ref] weight/offset. Luma.
    pub(crate) luma_weight: [Vec<i32>; 2],
    pub(crate) luma_offset: [Vec<i32>; 2],
    /// [list][ref][cb=0/cr=1].
    pub(crate) chroma_weight: [Vec<[i32; 2]>; 2],
    pub(crate) chroma_offset: [Vec<[i32; 2]>; 2],
    /// Whether an explicit weight was signalled for each entry.
    pub(crate) luma_flag: [Vec<bool>; 2],
    pub(crate) chroma_flag: [Vec<bool>; 2],
}

impl PredWeightTable {
    /// Parse `pred_weight_table()`. `num_ref` = active ref counts per list;
    /// `has_l1` false for P slices. `bd_*` are luma/chroma bit depths.
    pub(crate) fn parse(
        r: &mut BitReader,
        num_ref: [usize; 2],
        has_l1: bool,
        chroma: bool,
        bd_luma: u8,
        bd_chroma: u8,
    ) -> Result<Self, DecodeError> {
        let luma_log2_denom = r.read_ue().map_err(|_| e("luma_log2_denom"))? as u8;
        let chroma_log2_denom = if chroma {
            let d = r.read_se().map_err(|_| e("delta_chroma_log2_denom"))?;
            (luma_log2_denom as i32 + d).clamp(0, 7) as u8
        } else {
            0
        };
        let default_luma_w = 1i32 << luma_log2_denom;
        let default_chroma_w = 1i32 << chroma_log2_denom;
        let wp_off_half_luma = 1i32 << (bd_luma - 1);
        let wp_off_half_chroma = 1i32 << (bd_chroma - 1);

        let mut t = PredWeightTable {
            luma_log2_denom,
            chroma_log2_denom,
            luma_weight: [Vec::new(), Vec::new()],
            luma_offset: [Vec::new(), Vec::new()],
            chroma_weight: [Vec::new(), Vec::new()],
            chroma_offset: [Vec::new(), Vec::new()],
            luma_flag: [Vec::new(), Vec::new()],
            chroma_flag: [Vec::new(), Vec::new()],
        };

        let lists = if has_l1 { 2 } else { 1 };
        for (list, &n) in num_ref[..lists].iter().enumerate() {
            let mut luma_flags = Vec::with_capacity(n);
            for _ in 0..n {
                luma_flags.push(r.read_flag().map_err(|_| e("luma_weight_flag"))?);
            }
            let mut chroma_flags = vec![false; n];
            if chroma {
                for dst in chroma_flags[..n].iter_mut() {
                    *dst = r.read_flag().map_err(|_| e("chroma_weight_flag"))?;
                }
            }
            for i in 0..n {
                if luma_flags[i] {
                    let dw = r.read_se().map_err(|_| e("delta_luma_weight"))?;
                    let o = r.read_se().map_err(|_| e("luma_offset"))?;
                    t.luma_weight[list].push(default_luma_w + dw);
                    t.luma_offset[list].push(o);
                } else {
                    t.luma_weight[list].push(default_luma_w);
                    t.luma_offset[list].push(0);
                }
                if chroma && chroma_flags[i] {
                    let mut w = [0i32; 2];
                    let mut o = [0i32; 2];
                    for c in 0..2 {
                        let dw = r.read_se().map_err(|_| e("delta_chroma_weight"))?;
                        let doff = r.read_se().map_err(|_| e("delta_chroma_offset"))?;
                        w[c] = default_chroma_w + dw;
                        // offset reconstruction per spec.
                        let pred =
                            wp_off_half_chroma - ((wp_off_half_chroma * w[c]) >> chroma_log2_denom);
                        o[c] = (pred + doff).clamp(-wp_off_half_chroma, wp_off_half_chroma - 1);
                    }
                    t.chroma_weight[list].push(w);
                    t.chroma_offset[list].push(o);
                } else {
                    t.chroma_weight[list].push([default_chroma_w, default_chroma_w]);
                    t.chroma_offset[list].push([0, 0]);
                }
            }
            t.luma_flag[list] = luma_flags;
            t.chroma_flag[list] = chroma_flags;
        }
        let _ = wp_off_half_luma;
        Ok(t)
    }
}

/// Slice types.
pub(crate) const SLICE_B: u8 = 0;
pub(crate) const SLICE_P: u8 = 1;
pub(crate) const SLICE_I: u8 = 2;

/// An owned reference picture the motion-compensation path reads from. Planes
/// are stored at coded (CTB-aligned) dimensions matching the current picture.
#[derive(Clone)]
pub(crate) struct RefFramePlanes {
    pub(crate) poc: i32,
    pub(crate) y: Vec<u16>,
    pub(crate) cb: Vec<u16>,
    pub(crate) cr: Vec<u16>,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) cw: usize,
    pub(crate) ch: usize,
    /// Per-4×4 motion field of the reference (for temporal MVP).
    pub(crate) motion: Vec<MotionInfo>,
    pub(crate) width4: usize,
    pub(crate) height4: usize,
}
