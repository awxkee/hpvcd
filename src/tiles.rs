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

//! HEVC in-picture tiling (§6.5.1). Resolves the PPS tile spec into per-picture
//! CTB column/row boundaries and the raster↔tile-scan address conversion tables
//! plus the per-CTB TileId map used for CABAC re-init and loop-filter gating.

use crate::config::Pps;

/// Resolved tile geometry for one picture. All addresses are CTB raster (RS) or
/// tile-scan (TS) indices in `0..pic_size_in_ctbs`.
#[derive(Debug, Clone)]
#[allow(dead_code)] // some geometry fields are descriptive / test-only
pub(crate) struct TileGrid {
    pub(crate) cols: usize,
    pub(crate) rows: usize,
    /// Left CTB column x-coordinate of each tile column, length `cols`.
    pub(crate) col_bd: Vec<usize>,
    /// Top CTB row y-coordinate of each tile row, length `rows`.
    pub(crate) row_bd: Vec<usize>,
    /// Width of each tile column in CTBs, length `cols`.
    pub(crate) col_width: Vec<usize>,
    /// Height of each tile row in CTBs, length `rows`.
    pub(crate) row_height: Vec<usize>,
    /// CtbAddrRsToTs[rs] — raster address → tile-scan address (§6.5.1).
    pub(crate) rs_to_ts: Vec<usize>,
    /// CtbAddrTsToRs[ts] — tile-scan address → raster address.
    pub(crate) ts_to_rs: Vec<usize>,
    /// TileId[ts] — tile index for each tile-scan address.
    pub(crate) tile_id: Vec<usize>,
    pub(crate) ctb_cols: usize,
    pub(crate) ctb_rows: usize,
    pub(crate) loop_filter_across_tiles: bool,
}

impl TileGrid {
    /// Resolve the PPS tile spec for a picture of `ctb_cols × ctb_rows` CTBs.
    /// Returns `None` if `tiles_enabled` is false (caller uses plain raster scan).
    pub(crate) fn from_pps(pps: &Pps, ctb_cols: usize, ctb_rows: usize) -> Option<TileGrid> {
        if !pps.tiles_enabled || ctb_cols == 0 || ctb_rows == 0 {
            return None;
        }
        let cols = (pps.num_tile_columns as usize).clamp(1, ctb_cols);
        let rows = (pps.num_tile_rows as usize).clamp(1, ctb_rows);

        // §6.5.1: column widths / row heights in CTBs.
        let col_width = tile_sizes(
            cols,
            ctb_cols,
            pps.tile_uniform_spacing,
            &pps.tile_column_widths,
        );
        let row_height = tile_sizes(
            rows,
            ctb_rows,
            pps.tile_uniform_spacing,
            &pps.tile_row_heights,
        );

        // Boundary (left/top) positions are prefix sums.
        let mut col_bd = Vec::with_capacity(cols);
        let mut acc = 0;
        for &w in &col_width {
            col_bd.push(acc);
            acc += w;
        }
        let mut row_bd = Vec::with_capacity(rows);
        acc = 0;
        for &h in &row_height {
            row_bd.push(acc);
            acc += h;
        }

        let pic = ctb_cols * ctb_rows;
        let (rs_to_ts, ts_to_rs, tile_id) = scan_tables(
            ctb_cols,
            ctb_rows,
            cols,
            rows,
            &col_bd,
            &row_bd,
            &col_width,
            &row_height,
        );

        debug_assert_eq!(rs_to_ts.len(), pic);
        Some(TileGrid {
            cols,
            rows,
            col_bd,
            row_bd,
            col_width,
            row_height,
            rs_to_ts,
            ts_to_rs,
            tile_id,
            ctb_cols,
            ctb_rows,
            loop_filter_across_tiles: pps.loop_filter_across_tiles,
        })
    }

    /// Tile-scan address of the raster address, i.e. `CtbAddrRsToTs[rs]`.
    #[inline]
    pub(crate) fn rs_to_ts(&self, rs: usize) -> usize {
        self.rs_to_ts.get(rs).copied().unwrap_or(rs)
    }

    /// Raster address of the tile-scan address, i.e. `CtbAddrTsToRs[ts]`.
    #[inline]
    pub(crate) fn ts_to_rs(&self, ts: usize) -> usize {
        self.ts_to_rs.get(ts).copied().unwrap_or(ts)
    }

    /// True when the raster CTB at `rs` is the first CTB (in tile-scan order) of
    /// its tile — i.e. CABAC must be (re-)initialized here (§9.3.1).
    #[inline]
    pub(crate) fn is_tile_start_rs(&self, rs: usize) -> bool {
        let ts = self.rs_to_ts(rs);
        if ts == 0 {
            return true;
        }
        self.tile_id.get(ts) != self.tile_id.get(ts - 1)
    }

    /// Tile column index containing CTB x-coordinate `cx`.
    #[inline]
    pub(crate) fn col_of(&self, cx: usize) -> usize {
        // col_bd is ascending; find last boundary ≤ cx.
        let mut c = 0;
        for (i, &b) in self.col_bd.iter().enumerate() {
            if b <= cx {
                c = i;
            } else {
                break;
            }
        }
        c
    }

    /// Tile row index containing CTB y-coordinate `cy`.
    #[inline]
    pub(crate) fn row_of(&self, cy: usize) -> usize {
        let mut r = 0;
        for (i, &b) in self.row_bd.iter().enumerate() {
            if b <= cy {
                r = i;
            } else {
                break;
            }
        }
        r
    }

    /// TileId for a CTB at grid position `(cx, cy)`.
    #[inline]
    pub(crate) fn tile_id_at(&self, cx: usize, cy: usize) -> usize {
        self.row_of(cy) * self.cols + self.col_of(cx)
    }

    /// Left CTB-column boundary of the tile containing CTB column `cx`.
    #[inline]
    pub(crate) fn tile_col_start(&self, cx: usize) -> usize {
        self.col_bd[self.col_of(cx)]
    }

    /// Top CTB-row boundary of the tile containing CTB row `cy`.
    #[inline]
    pub(crate) fn tile_row_start(&self, cy: usize) -> usize {
        self.row_bd[self.row_of(cy)]
    }
}

/// §6.5.1 column-width / row-height derivation for one dimension.
fn tile_sizes(n: usize, total_ctbs: usize, uniform: bool, explicit: &[u32]) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    if uniform {
        // colWidth[i] = ((i+1)*total)/n - (i*total)/n
        for i in 0..n {
            let a = ((i + 1) * total_ctbs) / n;
            let b = (i * total_ctbs) / n;
            out.push(a - b);
        }
    } else {
        // Explicit sizes for the first n-1; last is the remainder.
        let mut used = 0usize;
        for i in 0..n.saturating_sub(1) {
            let w = explicit.get(i).copied().unwrap_or(1).max(1) as usize;
            let w = w.min(total_ctbs.saturating_sub(used).saturating_sub(n - 1 - i));
            let w = w.max(1);
            out.push(w);
            used += w;
        }
        out.push(total_ctbs.saturating_sub(used).max(1));
    }
    out
}

/// Build CtbAddrRsToTs, CtbAddrTsToRs and TileId per §6.5.1.
#[allow(clippy::too_many_arguments)]
fn scan_tables(
    ctb_cols: usize,
    ctb_rows: usize,
    cols: usize,
    rows: usize,
    col_bd: &[usize],
    row_bd: &[usize],
    col_width: &[usize],
    row_height: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let pic = ctb_cols * ctb_rows;
    let mut rs_to_ts = vec![0usize; pic];
    // §6.5.1: for each raster address, count CTBs preceding it in tile-scan order.
    for (rs, dst) in rs_to_ts[..pic].iter_mut().enumerate() {
        let tbx = rs % ctb_cols;
        let tby = rs / ctb_cols;
        let tile_x = col_index(col_bd, col_width, cols, tbx);
        let tile_y = row_index(row_bd, row_height, rows, tby);
        let mut ts = 0usize;
        // Whole tiles before this tile in tile-scan order.
        for &rh in row_height[..tile_y].iter() {
            ts += rh * ctb_cols;
        }
        let rwhy = row_height[tile_y];
        for &col_w in col_width[..tile_x].iter() {
            ts += rwhy * col_w;
        }
        // Within-tile raster offset.
        ts += (tby - row_bd[tile_y]) * col_width[tile_x] + (tbx - col_bd[tile_x]);
        *dst = ts;
    }
    let mut ts_to_rs = vec![0usize; pic];
    for (rs, &ts) in rs_to_ts.iter().enumerate() {
        ts_to_rs[ts] = rs;
    }
    // TileId per tile-scan address.
    let mut tile_id = vec![0usize; pic];
    let mut ts = 0usize;
    for ty in 0..rows {
        for tx in 0..cols {
            let id = ty * cols + tx;
            let start_x = col_bd[tx];
            let start_y = row_bd[ty];
            for y in start_y..start_y + row_height[ty] {
                for x in start_x..start_x + col_width[tx] {
                    let rs = y * ctb_cols + x;
                    tile_id[rs_to_ts[rs]] = id;
                    ts += 1;
                }
            }
        }
    }
    debug_assert_eq!(ts, pic);
    (rs_to_ts, ts_to_rs, tile_id)
}

#[inline]
fn col_index(col_bd: &[usize], col_width: &[usize], cols: usize, cx: usize) -> usize {
    let mut idx = 0;
    for i in 0..cols {
        if cx >= col_bd[i] && cx < col_bd[i] + col_width[i] {
            idx = i;
            break;
        }
    }
    idx
}

#[inline]
fn row_index(row_bd: &[usize], row_height: &[usize], rows: usize, cy: usize) -> usize {
    let mut idx = 0;
    for i in 0..rows {
        if cy >= row_bd[i] && cy < row_bd[i] + row_height[i] {
            idx = i;
            break;
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pps_tiles(cols: u32, rows: u32, uniform: bool, cw: Vec<u32>, rh: Vec<u32>) -> Pps {
        // Minimal PPS with only the fields TileGrid reads.
        let mut p = crate::config::Pps::test_default();
        p.tiles_enabled = true;
        p.num_tile_columns = cols;
        p.num_tile_rows = rows;
        p.tile_uniform_spacing = uniform;
        p.tile_column_widths = cw;
        p.tile_row_heights = rh;
        p.loop_filter_across_tiles = false;
        p
    }

    #[test]
    fn no_tiles_returns_none() {
        let p = Pps::test_default();
        assert!(TileGrid::from_pps(&p, 4, 4).is_none());
    }

    #[test]
    fn uniform_2x2_scan_order() {
        // 4×4 CTBs, 2×2 uniform tiles → each tile 2×2.
        let p = pps_tiles(2, 2, true, vec![], vec![]);
        let g = TileGrid::from_pps(&p, 4, 4).unwrap();
        assert_eq!(g.col_width, vec![2, 2]);
        assert_eq!(g.row_height, vec![2, 2]);
        // Tile-scan order: top-left tile first (rs 0,1,4,5), then top-right
        // (2,3,6,7), then bottom-left (8,9,12,13), then bottom-right.
        assert_eq!(g.rs_to_ts(0), 0);
        assert_eq!(g.rs_to_ts(1), 1);
        assert_eq!(g.rs_to_ts(4), 2);
        assert_eq!(g.rs_to_ts(5), 3);
        assert_eq!(g.rs_to_ts(2), 4);
        assert_eq!(g.rs_to_ts(8), 8);
        // Round-trips.
        for rs in 0..16 {
            assert_eq!(g.ts_to_rs(g.rs_to_ts(rs)), rs);
        }
        // Tile boundaries: rs 2 (top-right start) is a tile start; rs 1 is not.
        assert!(g.is_tile_start_rs(2));
        assert!(!g.is_tile_start_rs(1));
        // tile membership via grid position (ctb_cols = 4).
        assert_eq!(g.tile_id_at(0, 0), 0);
        assert_eq!(g.tile_id_at(2, 0), 1);
        assert_eq!(g.tile_id_at(0, 2), 2);
    }

    #[test]
    fn non_uniform_columns() {
        // 5 CTB wide, 2 tile cols, explicit width 3 for first → [3, 2].
        let p = pps_tiles(2, 1, false, vec![3], vec![]);
        let g = TileGrid::from_pps(&p, 5, 2).unwrap();
        assert_eq!(g.col_width, vec![3, 2]);
        assert_eq!(g.col_bd, vec![0, 3]);
    }

    #[test]
    fn uniform_uneven_division() {
        // 5 CTBs, 2 tiles → [2, 3] per the ((i+1)*5)/2 - (i*5)/2 formula.
        let p = pps_tiles(2, 1, true, vec![], vec![]);
        let g = TileGrid::from_pps(&p, 5, 1).unwrap();
        assert_eq!(g.col_width, vec![2, 3]);
    }

    #[test]
    fn asymmetric_3x2_bijection_and_tile_starts() {
        // 7×5 CTBs, 3 tile cols (explicit 2,3 → last=2), 2 tile rows (explicit 2
        // → last=3). Verifies the scan-order tables are a bijection and that
        // exactly one tile-start exists per tile.
        let p = pps_tiles(3, 2, false, vec![2, 3], vec![2]);
        let g = TileGrid::from_pps(&p, 7, 5).unwrap();
        assert_eq!(g.col_width, vec![2, 3, 2]);
        assert_eq!(g.row_height, vec![2, 3]);
        let pic = 7 * 5;
        // Bijection.
        let mut seen = vec![false; pic];
        for rs in 0..pic {
            let ts = g.rs_to_ts(rs);
            assert!(ts < pic);
            assert!(!seen[ts], "ts {ts} produced twice");
            seen[ts] = true;
            assert_eq!(g.ts_to_rs(ts), rs);
        }
        // Exactly 6 tile starts (one per tile), and each is a real tile top-left.
        let starts = (0..pic).filter(|&rs| g.is_tile_start_rs(rs)).count();
        assert_eq!(starts, 6);
        // The CTB at grid (2,0) starts tile col 1; (0,2) starts tile row 1.
        assert!(g.is_tile_start_rs(2)); // rs = 0*7 + 2
        assert!(g.is_tile_start_rs(2 * 7)); // rs = 2*7 + 0
    }
}
