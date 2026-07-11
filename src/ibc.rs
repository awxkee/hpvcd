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

//! Intra Block Copy (SCC §8.6.3 / §8.4.5.1 IBC path). The current picture is
//! referenced through a block vector (integer-pel, quarter-pel units with zero
//! fractional bits). Prediction is an unfiltered copy of already-reconstructed
//! samples from the same picture; no interpolation, no in-loop filtering on the
//! read. Block-vector candidates reuse the inter merge/AMVP field but resolve
//! against the current-picture reference entry.

use crate::inter::Mv;
use std::convert::TryFrom;

/// Geometry needed by the current-picture reference conformance checks in
/// §8.5.3.2.1. `pb_span_*` are nPbSw/nPbSh from equations 8-86 and 8-87;
/// unlike `pu_w`/`pu_h`, they describe the partition-wide span used by the CTB
/// availability constraint for asymmetric partitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PuGeometry {
    pub(crate) cu_x: usize,
    pub(crate) cu_y: usize,
    pub(crate) cu_size: usize,
    pub(crate) pu_x: usize,
    pub(crate) pu_y: usize,
    pub(crate) pu_w: usize,
    pub(crate) pu_h: usize,
    pub(crate) pb_span_w: usize,
    pub(crate) pb_span_h: usize,
}

/// Inclusive luma rectangle whose z-scan availability must be checked for a
/// current-picture reference. The margins account for a fractional chroma
/// position caused by an integer luma BV on a subsampled chroma grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SourceArea {
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) x1: usize,
    pub(crate) y1: usize,
}

/// Force a block vector to integer-pel. `mv_res_ctrl_idc == 2` (and the IBC
/// path in general) codes integer BVs; any residual fractional bits are cleared
/// so the copy reads whole samples (§8.4.5.1: bv is in integer luma units).
#[inline]
pub(crate) fn integerize_bv(mv: Mv) -> Mv {
    Mv::new(mv.x & !3, mv.y & !3)
}

/// Return the top-left luma source sample for a current-picture copy when the
/// complete prediction block is memory-safe. This is deliberately separate
/// from the normative BV conformance checks: a conservative conformance
/// validator must not turn a safely addressable reference into a grey block.
#[inline]
pub(crate) fn source_origin(
    pu_x: usize,
    pu_y: usize,
    pu_w: usize,
    pu_h: usize,
    mv: Mv,
    pic_w: usize,
    pic_h: usize,
) -> Option<(usize, usize)> {
    if pu_w == 0 || pu_h == 0 {
        return None;
    }
    let pu_x = isize::try_from(pu_x).ok()?;
    let pu_y = isize::try_from(pu_y).ok()?;
    let pic_w = isize::try_from(pic_w).ok()?;
    let pic_h = isize::try_from(pic_h).ok()?;
    let pu_w = isize::try_from(pu_w).ok()?;
    let pu_h = isize::try_from(pu_h).ok()?;
    let sx = pu_x.checked_add((mv.x >> 2) as isize)?;
    let sy = pu_y.checked_add((mv.y >> 2) as isize)?;
    let x1 = sx.checked_add(pu_w)?;
    let y1 = sy.checked_add(pu_h)?;
    if sx < 0 || sy < 0 || x1 > pic_w || y1 > pic_h {
        return None;
    }
    Some((sx as usize, sy as usize))
}

/// Validate the geometry-only current-picture MV constraints from §8.5.3.2.1
/// and return the two source corners that must additionally pass the §6.4.1
/// z-scan/tile/slice availability process. `(bvx, bvy)` are integer luma
/// offsets and `(offset_x, offset_y)` are the chroma interpolation margins from
/// equations 8-104 and 8-105 (each is either 0 or 2).
#[allow(clippy::too_many_arguments)]
pub(crate) fn source_area(
    geom: PuGeometry,
    bvx: isize,
    bvy: isize,
    offset_x: isize,
    offset_y: isize,
    pic_w: usize,
    pic_h: usize,
    ctb_log2: u32,
) -> Option<SourceArea> {
    if geom.pu_w == 0
        || geom.pu_h == 0
        || geom.pb_span_w == 0
        || geom.pb_span_h == 0
        || offset_x < 0
        || offset_y < 0
        || ctb_log2 >= usize::BITS
    {
        return None;
    }

    let pu_x = isize::try_from(geom.pu_x).ok()?;
    let pu_y = isize::try_from(geom.pu_y).ok()?;
    let cu_x = isize::try_from(geom.cu_x).ok()?;
    let cu_y = isize::try_from(geom.cu_y).ok()?;
    let pu_w = isize::try_from(geom.pu_w).ok()?;
    let pu_h = isize::try_from(geom.pu_h).ok()?;
    let span_w = isize::try_from(geom.pb_span_w).ok()?;
    let span_h = isize::try_from(geom.pb_span_h).ok()?;
    let pic_w_i = isize::try_from(pic_w).ok()?;
    let pic_h_i = isize::try_from(pic_h).ok()?;

    let src_x = pu_x.checked_add(bvx)?;
    let src_y = pu_y.checked_add(bvy)?;
    let x0 = src_x.checked_sub(offset_x)?;
    let y0 = src_y.checked_sub(offset_y)?;
    let x1 = src_x
        .checked_add(pu_w.checked_sub(1)?)?
        .checked_add(offset_x)?;
    let y1 = src_y
        .checked_add(pu_h.checked_sub(1)?)?
        .checked_add(offset_y)?;
    if x0 < 0 || y0 < 0 || x1 >= pic_w_i || y1 >= pic_h_i {
        return None;
    }

    // The source and destination prediction blocks must not overlap. xBl/yBl
    // are the PU offsets relative to the coding block.
    let x_bl = pu_x.checked_sub(cu_x)?;
    let y_bl = pu_y.checked_sub(cu_y)?;
    let left_of_current = bvx
        .checked_add(pu_w)?
        .checked_add(x_bl)?
        .checked_add(offset_x)?
        <= 0;
    let above_current = bvy
        .checked_add(pu_h)?
        .checked_add(y_bl)?
        .checked_add(offset_y)?
        <= 0;
    if !left_of_current && !above_current {
        return None;
    }

    // Equation 8-106 limits how far right an above-row reference may reach,
    // matching the wavefront of reconstructed CTBs. This uses nPbSw/nPbSh,
    // not necessarily the dimensions of the current asymmetric PU.
    let ctb = isize::try_from(1usize << ctb_log2).ok()?;
    let span_right = src_x
        .checked_add(span_w.checked_sub(1)?)?
        .checked_add(offset_x)?;
    let span_bottom = src_y
        .checked_add(span_h.checked_sub(1)?)?
        .checked_add(offset_y)?;
    if span_right < 0 || span_bottom < 0 {
        return None;
    }
    let left = span_right / ctb - cu_x / ctb;
    let right = cu_y / ctb - span_bottom / ctb;
    if left > right {
        return None;
    }

    Some(SourceArea {
        x0: x0 as usize,
        y0: y0 as usize,
        x1: x1 as usize,
        y1: y1 as usize,
    })
}

/// Compute MinTbAddrZs for a luma sample position when the containing CTB's
/// tile-scan address is already known (§6.5.2). Keeping this as a small pure
/// helper makes the current-picture availability check auditable and testable.
pub(crate) fn min_tb_addr_zs(
    x: usize,
    y: usize,
    log2_ctb: u32,
    log2_min_tb: u32,
    ctb_addr_ts: usize,
) -> Option<usize> {
    let delta = log2_ctb.checked_sub(log2_min_tb)?;
    let shift = delta.checked_mul(2)?;
    let mut addr = ctb_addr_ts.checked_shl(shift)?;
    let min_x = x >> log2_min_tb;
    let min_y = y >> log2_min_tb;
    for i in 0..delta {
        let m = 1usize.checked_shl(i)?;
        let square = m.checked_mul(m)?;
        if min_x & m != 0 {
            addr = addr.checked_add(square)?;
        }
        if min_y & m != 0 {
            addr = addr.checked_add(square.checked_mul(2)?)?;
        }
    }
    Some(addr)
}

/// Load an already-reconstructed current-picture block into the motion-
/// compensation intermediate precision. Interpolation kernels represent a
/// full-pel sample as `sample << (14 - bit_depth)`; current-picture prediction
/// must use the same representation before uni/bi weighting and averaging.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_internal_block(
    plane: &[u16],
    stride: usize,
    sx: usize,
    sy: usize,
    w: usize,
    h: usize,
    bit_depth: u8,
    dst: &mut [i16],
) -> bool {
    let Some(src_start) = sy.checked_mul(stride) else {
        return false;
    };
    let Some(src_end) = sy.checked_add(h).and_then(|rows| rows.checked_mul(stride)) else {
        return false;
    };
    let Some(row_end) = sx.checked_add(w) else {
        return false;
    };
    let Some(len) = w.checked_mul(h) else {
        return false;
    };
    if w == 0
        || h == 0
        || !(8..=14).contains(&bit_depth)
        || row_end > stride
        || src_end > plane.len()
        || dst.len() < len
    {
        return false;
    }

    let shift = u32::from(14 - bit_depth);
    let src = &plane[src_start..src_end];
    let dst = &mut dst[..len];
    for (src_row, dst_row) in src
        .chunks_exact(stride)
        .zip(dst.chunks_exact_mut(w))
        .take(h)
    {
        for (&sample, out) in src_row[sx..sx + w].iter().zip(dst_row) {
            *out = ((sample as i32) << shift) as i16;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integerize_clears_fractional() {
        assert_eq!(integerize_bv(Mv::new(-13, 7)), Mv::new(-16, 4));
        assert_eq!(integerize_bv(Mv::new(0, 0)), Mv::new(0, 0));
    }

    #[test]
    fn source_origin_checks_the_complete_block() {
        assert_eq!(
            source_origin(32, 16, 16, 8, Mv::new(-64, -32), 128, 64),
            Some((16, 8))
        );
        assert_eq!(source_origin(0, 0, 8, 8, Mv::new(-4, 0), 64, 64), None);
        assert_eq!(source_origin(60, 0, 8, 8, Mv::new(0, 0), 64, 64), None);
        assert_eq!(source_origin(0, 60, 8, 8, Mv::new(0, 0), 64, 64), None);
        assert_eq!(source_origin(0, 0, 0, 8, Mv::new(0, 0), 64, 64), None);
    }

    #[test]
    fn source_area_constraints() {
        let ctb = 6; // 64x64
        let geom = |cu_x, cu_y, pu_x, pu_y| PuGeometry {
            cu_x,
            cu_y,
            cu_size: 16,
            pu_x,
            pu_y,
            pu_w: 16,
            pu_h: 16,
            pb_span_w: 16,
            pb_span_h: 16,
        };
        // Left within the same CTB row, fully to the left: valid.
        assert!(source_area(geom(32, 32, 32, 32), -16, 0, 0, 0, 128, 128, ctb).is_some());
        // Above (previous CTB row): valid.
        assert!(source_area(geom(0, 64, 0, 64), 0, -16, 0, 0, 128, 128, ctb).is_some());
        // Pointing into a not-yet-decoded future CTB row: invalid.
        assert!(source_area(geom(0, 0, 0, 0), 0, 64, 0, 0, 128, 128, ctb).is_none());
        // Pointing right, at/after the current block in the same row: invalid.
        assert!(source_area(geom(32, 32, 32, 32), 16, 0, 0, 0, 128, 128, ctb).is_none());
        // Source outside the picture (negative): invalid.
        assert!(source_area(geom(0, 0, 0, 0), -16, 0, 0, 0, 128, 128, ctb).is_none());
        // Source overruns the right picture edge: invalid.
        assert!(source_area(geom(112, 0, 112, 0), 16, 0, 0, 0, 128, 128, ctb).is_none());
        // Overlapping source (small BV): invalid.
        assert!(source_area(geom(32, 32, 32, 32), -8, -8, 0, 0, 128, 128, ctb).is_none());
        // A source two CTBs to the right but only one row above violates the
        // equation 8-106 reconstruction wavefront even though it is in bounds.
        assert!(source_area(geom(0, 64, 0, 64), 128, -64, 0, 0, 256, 128, ctb).is_none());
        // Chroma fractional margins participate in both bounds and overlap.
        assert_eq!(
            source_area(geom(32, 32, 32, 32), -18, 0, 2, 0, 128, 128, ctb),
            Some(SourceArea {
                x0: 12,
                y0: 32,
                x1: 31,
                y1: 47,
            })
        );
    }

    #[test]
    fn min_tb_address_uses_morton_order_inside_ctb() {
        // 64x64 CTB, 16x16 minimum TB: four minimum blocks per dimension.
        assert_eq!(min_tb_addr_zs(0, 0, 6, 4, 0), Some(0));
        assert_eq!(min_tb_addr_zs(16, 0, 6, 4, 0), Some(1));
        assert_eq!(min_tb_addr_zs(0, 16, 6, 4, 0), Some(2));
        assert_eq!(min_tb_addr_zs(16, 16, 6, 4, 0), Some(3));
        assert_eq!(min_tb_addr_zs(32, 0, 6, 4, 0), Some(4));
        // A CTB at tile-scan address 3 starts after 3 * 16 minimum blocks.
        assert_eq!(min_tb_addr_zs(64, 0, 6, 4, 3), Some(48));
    }

    #[test]
    fn current_picture_samples_use_mc_internal_precision() {
        let plane: Vec<u16> = (0..32).collect();
        let mut dst = [0i16; 6];
        assert!(load_internal_block(&plane, 8, 2, 1, 3, 2, 8, &mut dst));
        assert_eq!(dst, [10 << 6, 11 << 6, 12 << 6, 18 << 6, 19 << 6, 20 << 6]);

        let mut dst10 = [0i16; 2];
        assert!(load_internal_block(&plane, 8, 4, 0, 2, 1, 10, &mut dst10));
        assert_eq!(dst10, [4 << 4, 5 << 4]);
    }

    #[test]
    fn current_picture_loader_rejects_out_of_range_blocks() {
        let plane = [0u16; 16];
        let mut dst = [0i16; 4];
        assert!(!load_internal_block(&plane, 4, 3, 0, 2, 1, 8, &mut dst));
        assert!(!load_internal_block(&plane, 4, 0, 4, 2, 1, 8, &mut dst));
        assert!(!load_internal_block(
            &plane,
            4,
            0,
            0,
            2,
            2,
            8,
            &mut dst[..3]
        ));
    }
}
