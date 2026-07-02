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

type SaoPlaneFn =
    fn(&mut [u16], &[u16], usize, usize, usize, usize, usize, usize, u8, &[i32; 4], u8, u8, u8);

type SaoPlaneBandedFn = fn(
    &mut [u16],
    &[u16],
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    u8,
    &[i32; 4],
    u8,
    u8,
    u8,
);

static APPLY_SAO_PLANE: std::sync::OnceLock<SaoPlaneFn> = std::sync::OnceLock::new();
static APPLY_SAO_PLANE_BANDED: std::sync::OnceLock<SaoPlaneBandedFn> = std::sync::OnceLock::new();

#[inline]
fn resolve_apply_sao_plane() -> SaoPlaneFn {
    *APPLY_SAO_PLANE.get_or_init(|| {
        let mut _f: SaoPlaneFn = apply_sao_plane_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::apply_sao_plane_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::apply_sao_plane_sse41;
            }
        }

        _f
    })
}

#[inline]
fn resolve_apply_sao_plane_banded() -> SaoPlaneBandedFn {
    *APPLY_SAO_PLANE_BANDED.get_or_init(|| {
        let mut _f: SaoPlaneBandedFn = apply_sao_plane_banded_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::apply_sao_plane_banded_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::apply_sao_plane_banded_sse41;
            }
        }

        _f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn apply_sao_plane(
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
    resolve_apply_sao_plane()(
        dst, src, w, h, x0, y0, x_end, y_end, type_idx, offsets, band_pos, eo_class, bd,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn apply_sao_plane_scalar(
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
    match type_idx {
        1 => apply_sao_band_offset_scalar(dst, src, w, x0, y0, x_end, y_end, offsets, band_pos, bd),
        2 => apply_sao_edge_offset_scalar(
            dst, src, w, h, x0, y0, x_end, y_end, offsets, eo_class, bd,
        ),
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn apply_sao_band_offset_scalar(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    band_pos: u8,
    bd: u8,
) {
    let max_val = ((1u32 << bd) - 1) as i32;
    let shift = bd - 5;

    for y in y0..y_end {
        let row = y * w;
        for x in x0..x_end {
            let s = src[row + x] as i32;
            let band = (s >> shift) as u8;
            let rel = band.wrapping_sub(band_pos);
            if rel < 4 {
                let v = (s + offsets[rel as usize]).clamp(0, max_val);
                dst[row + x] = v as u16;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_sao_edge_offset_scalar(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x_end: usize,
    y_end: usize,
    offsets: &[i32; 4],
    eo_class: u8,
    bd: u8,
) {
    let max_val = ((1u32 << bd) - 1) as i32;

    // Direction vectors for the two neighbors (§8.7.3.2.4).
    let (dx, dy): (i32, i32) = match eo_class {
        0 => (1, 0),  // horizontal
        1 => (0, 1),  // vertical
        2 => (1, 1),  // 135°
        _ => (1, -1), // 45°
    };

    for y in y0..y_end {
        for x in x0..x_end {
            let s = src[y * w + x] as i32;

            let x1 = x as i32 + dx;
            let y1 = y as i32 + dy;
            let x2 = x as i32 - dx;
            let y2 = y as i32 - dy;

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
            let edge_idx = sign1 + sign2 + 2;

            let offset = match edge_idx {
                0 => offsets[0],
                1 => offsets[1],
                3 => offsets[2],
                4 => offsets[3],
                _ => 0,
            };
            if offset != 0 {
                dst[y * w + x] = (s + offset).clamp(0, max_val) as u16;
            }
        }
    }
}

/// Per-CTU SAO parameters, flattened out of the decoder's private `SaoCtb` so
/// this driver has no dependency on `decode.rs` internals. One entry per CTB in
/// raster order (`ry * ctb_cols + rx`). Component order is [luma, Cb, Cr].
#[derive(Clone, Copy, Default)]
pub(crate) struct SaoCtbParams {
    pub type_idx: [u8; 3],
    pub offsets: [[i32; 4]; 3],
    pub band_pos: [u8; 3],
    pub eo_class: [u8; 3],
}

/// Immutable geometry + toggles shared by every SAO band worker.
pub(crate) struct SaoPlanesCtx<'a> {
    pub params: &'a [SaoCtbParams],
    pub ctb_cols: usize,
    pub ctb_rows: usize,
    pub log2_ctb: u32,
    pub w: usize,
    pub h: usize,
    pub cw: usize,
    pub ch: usize,
    pub sub_w: usize,
    pub sub_h: usize,
    pub bd: u8,
    pub bd_c: u8,
    pub sao_luma: bool,
    pub sao_chroma: bool,
}

/// Apply SAO to one CTB-row's worth of output. `luma_dst`/`cb_dst`/`cr_dst` are
/// the *band* slices (rows `[ry*ctb .. ry*ctb+ctb)` of each plane, clipped to
/// the picture); `luma_src`/`cb_src`/`cr_src` are the full, untouched source
/// planes (EO neighbor reads may cross the band's top/bottom edge, so the source
/// must be the whole plane, never the band). `band_y0`/`band_cy0` are the global
/// row of each band's local row 0, so we can translate a CTB's global rectangle
/// into band-local destination coordinates.
#[allow(clippy::too_many_arguments)]
fn apply_sao_ctb_row(
    ctx: &SaoPlanesCtx<'_>,
    ry: usize,
    luma_dst: &mut [u16],
    luma_src: &[u16],
    band_y0: usize,
    cb_dst: &mut [u16],
    cr_dst: &mut [u16],
    cb_src: &[u16],
    cr_src: &[u16],
    band_cy0: usize,
) {
    let ctb = 1usize << ctx.log2_ctb;
    for rx in 0..ctx.ctb_cols {
        let p = &ctx.params[ry * ctx.ctb_cols + rx];
        let x0 = rx * ctb;
        let y0 = ry * ctb;

        if ctx.sao_luma && p.type_idx[0] != 0 {
            let x_end = (x0 + ctb).min(ctx.w);
            let y_end = (y0 + ctb).min(ctx.h);
            // Destination is a band whose local row 0 == global row `band_y0`.
            apply_sao_plane_banded(
                luma_dst,
                luma_src,
                ctx.w,
                ctx.h,
                band_y0,
                x0,
                y0,
                x_end,
                y_end,
                p.type_idx[0],
                &p.offsets[0],
                p.band_pos[0],
                p.eo_class[0],
                ctx.bd,
            );
        }

        if ctx.sao_chroma {
            let cx0 = x0 / ctx.sub_w;
            let cy0 = y0 / ctx.sub_h;
            let cx_end = ((x0 + ctb) / ctx.sub_w).min(ctx.cw);
            let cy_end = ((y0 + ctb) / ctx.sub_h).min(ctx.ch);
            if p.type_idx[1] != 0 {
                apply_sao_plane_banded(
                    cb_dst,
                    cb_src,
                    ctx.cw,
                    ctx.ch,
                    band_cy0,
                    cx0,
                    cy0,
                    cx_end,
                    cy_end,
                    p.type_idx[1],
                    &p.offsets[1],
                    p.band_pos[1],
                    p.eo_class[1],
                    ctx.bd_c,
                );
            }
            if p.type_idx[2] != 0 {
                apply_sao_plane_banded(
                    cr_dst,
                    cr_src,
                    ctx.cw,
                    ctx.ch,
                    band_cy0,
                    cx0,
                    cy0,
                    cx_end,
                    cy_end,
                    p.type_idx[2],
                    &p.offsets[2],
                    p.band_pos[2],
                    p.eo_class[2],
                    ctx.bd_c,
                );
            }
        }
    }
}

/// Banded SAO kernel: reads the *full* source plane at global coordinates
/// (so edge-offset neighbor lookups across the band's top/bottom edge stay
/// correct) and writes to a *band* destination whose local row 0 is global row
/// `band_y0` — i.e. `dst_band[(y - band_y0) * w + x]`. This is the parallel
/// analogue of the serial `apply_sao_plane`, split so disjoint bands can run on
/// separate threads. Bit-exact with the serial path because both read only from
/// the untouched clone and write each output pixel exactly once.
#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_sao_plane_banded(
    dst_band: &mut [u16],
    src_full: &[u16],
    w: usize,
    h: usize,
    band_y0: usize,
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
    resolve_apply_sao_plane_banded()(
        dst_band, src_full, w, h, band_y0, x0, y0, x_end, y_end, type_idx, offsets, band_pos,
        eo_class, bd,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_sao_plane_banded_scalar(
    dst_band: &mut [u16],
    src_full: &[u16],
    w: usize,
    h: usize,
    band_y0: usize,
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
    if w == 0 || x_end <= x0 || y_end <= y0 || y_end <= band_y0 {
        return;
    }

    let Some(max_val) = (1u32)
        .checked_shl(bd as u32)
        .map(|v| v.saturating_sub(1) as i32)
    else {
        return;
    };

    match type_idx {
        // Band offset: purely pointwise, no neighbor reads.
        1 => {
            let shift = bd.saturating_sub(5);
            for y in y0..y_end {
                let src_row = y * w;
                let dst_row = (y - band_y0) * w;
                let src_range = src_row + x0..src_row + x_end;
                let dst_range = dst_row + x0..dst_row + x_end;
                let (Some(src_row), Some(dst_row)) =
                    (src_full.get(src_range), dst_band.get_mut(dst_range))
                else {
                    continue;
                };
                for (s, dst) in src_row.iter().copied().zip(dst_row.iter_mut()) {
                    let s = s as i32;
                    let band = (s >> shift) as u8;
                    let rel = band.wrapping_sub(band_pos);
                    if rel < 4 {
                        *dst = (s + offsets[rel as usize]).clamp(0, max_val) as u16;
                    }
                }
            }
        }
        // Edge offset: reads two neighbors from the full source plane.
        2 => {
            let (dx, dy): (i32, i32) = match eo_class {
                0 => (1, 0),
                1 => (0, 1),
                2 => (1, 1),
                _ => (1, -1),
            };
            let inb = |xx: i32, yy: i32| -> bool {
                xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h
            };
            for y in y0..y_end {
                let src_base = y * w;
                let dst_base = (y - band_y0) * w;
                for x in x0..x_end {
                    let Some(&s0) = src_full.get(src_base + x) else {
                        continue;
                    };
                    let s = s0 as i32;
                    let (x1, y1) = (x as i32 + dx, y as i32 + dy);
                    let (x2, y2) = (x as i32 - dx, y as i32 - dy);
                    let n1 = if inb(x1, y1) {
                        src_full
                            .get(y1 as usize * w + x1 as usize)
                            .copied()
                            .unwrap_or(s0) as i32
                    } else {
                        s
                    };
                    let n2 = if inb(x2, y2) {
                        src_full
                            .get(y2 as usize * w + x2 as usize)
                            .copied()
                            .unwrap_or(s0) as i32
                    } else {
                        s
                    };
                    let sign1 = (s > n1) as i32 - (s < n1) as i32;
                    let sign2 = (s > n2) as i32 - (s < n2) as i32;
                    let offset = match sign1 + sign2 + 2 {
                        0 => offsets[0],
                        1 => offsets[1],
                        3 => offsets[2],
                        4 => offsets[3],
                        _ => 0,
                    };
                    if offset != 0 {
                        if let Some(dst) = dst_band.get_mut(dst_base + x) {
                            *dst = (s + offset).clamp(0, max_val) as u16;
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Parallel SAO over the whole picture. Splits each plane into CTB-row bands
/// (`ctb` output rows each, clipped at the picture edge), hands every band an
/// exclusive `&mut` region via [`DisjointMut`], and applies that CTB-row's SAO
/// on a pool worker. Reads come from full-plane clones, so bands never race.
///
/// `dst_*` are consumed and the filtered planes returned (matching the serial
/// path's in-place semantics from the caller's view). When the pool is
/// single-threaded or there is only one band, it runs serially with no dispatch.
pub(crate) fn apply_sao_parallel(
    pool: &crate::threadpool::ThreadPool,
    ctx: &SaoPlanesCtx<'_>,
    mut y: Vec<u16>,
    mut cb: Vec<u16>,
    mut cr: Vec<u16>,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let ctb = 1usize << ctx.log2_ctb;
    let rows = ctx.ctb_rows;

    // Untouched sources for neighbor/pointwise reads.
    let src_y = y.clone();
    let src_cb = cb.clone();
    let src_cr = cr.clone();

    // Serial fallback: no benefit from dispatch.
    if pool.threads() <= 1 || rows <= 1 {
        for ry in 0..rows {
            let y_lo = ry * ctb * ctx.w;
            let cy0 = (ry * ctb) / ctx.sub_h;
            let c_lo = cy0 * ctx.cw;
            apply_sao_ctb_row(
                ctx,
                ry,
                &mut y[y_lo..],
                &src_y,
                ry * ctb,
                &mut cb[c_lo..],
                &mut cr[c_lo..],
                &src_cb,
                &src_cr,
                cy0,
            );
        }
        return (y, cb, cr);
    }

    let y_dm = crate::threadpool::DisjointMut::new(y);
    let cb_dm = crate::threadpool::DisjointMut::new(cb);
    let cr_dm = crate::threadpool::DisjointMut::new(cr);

    crate::threadpool::parallel_for(pool, rows, |ry| {
        // Luma band: rows [ry*ctb, (ry+1)*ctb) clipped to h.
        let ly0 = ry * ctb;
        let ly1 = ((ry + 1) * ctb).min(ctx.h);
        if ly0 >= ly1 {
            return;
        }
        let y_lo = ly0 * ctx.w;
        let y_hi = ly1 * ctx.w;

        // Chroma band: the chroma rows this CTB-row covers.
        let cy0 = ly0 / ctx.sub_h;
        let cy1 = ((ly0 + ctb) / ctx.sub_h).min(ctx.ch);
        let (c_lo, c_hi) = (cy0 * ctx.cw, cy1.max(cy0) * ctx.cw);

        let mut y_band = y_dm.slice_mut(y_lo..y_hi);
        if ctx.sao_chroma && ctx.cw > 0 && c_lo < c_hi {
            let mut cb_band = cb_dm.slice_mut(c_lo..c_hi);
            let mut cr_band = cr_dm.slice_mut(c_lo..c_hi);
            apply_sao_ctb_row(
                ctx,
                ry,
                &mut y_band,
                &src_y,
                ly0,
                &mut cb_band,
                &mut cr_band,
                &src_cb,
                &src_cr,
                cy0,
            );
        } else {
            apply_sao_ctb_row(
                ctx,
                ry,
                &mut y_band,
                &src_y,
                ly0,
                &mut [],
                &mut [],
                &src_cb,
                &src_cr,
                cy0,
            );
        }
    });

    (y_dm.into_inner(), cb_dm.into_inner(), cr_dm.into_inner())
}
