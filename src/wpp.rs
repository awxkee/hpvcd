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

//! WPP (wavefront parallel processing) support for HEIC still images.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cabac::{ContextSet, IntraModeContexts};
use crate::decode::FullDecoder;
use crate::error::DecodeError;
use crate::threadpool::ThreadPool;

/// Byte range `[start, end)` of one CTB-row sub-stream within the unescaped
/// RBSP CABAC payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowSubstream {
    pub start: usize,
    pub end: usize,
}

/// Compute per-row sub-stream byte ranges inside the unescaped RBSP.
///
/// * `src_of` — the NAL→RBSP source-index map from
///   [`crate::bitreader::unescape_rbsp_with_map`] for the whole NAL RBSP.
/// * `cabac_rbsp_off` — RBSP byte offset where CABAC slice data begins
///   (`SliceHeader::cabac_offset`).
/// * `entry_points` — `entry_point_offset_minus1[i] + 1` values (NAL byte
///   lengths of sub-streams `0..n-1`; the last row's length is implied).
/// * `rbsp_len` — total length of the unescaped RBSP.
/// * `ctb_rows` — number of CTB rows (== number of WPP sub-streams).
///
/// Returns `None` when the geometry is inconsistent (e.g. entry-point count does
/// not match `ctb_rows - 1`, or offsets run past the payload), signalling the
/// caller to fall back to the serial decode.
pub(crate) fn row_substreams(
    src_of: &[usize],
    cabac_rbsp_off: usize,
    entry_points: &[u32],
    rbsp_len: usize,
    ctb_rows: usize,
) -> Option<Vec<RowSubstream>> {
    // WPP emits exactly one sub-stream per CTB row, so there must be
    // `ctb_rows - 1` entry points (the final row's end is the payload end).
    if ctb_rows == 0 {
        return None;
    }
    if entry_points.len() + 1 != ctb_rows {
        return None;
    }
    if cabac_rbsp_off > rbsp_len {
        return None;
    }

    // NAL byte offset where CABAC data begins.
    let nal_data_start = if cabac_rbsp_off < src_of.len() {
        src_of[cabac_rbsp_off]
    } else {
        // cabac data begins exactly at RBSP end → empty payload.
        return None;
    };

    let mut rows = Vec::with_capacity(ctb_rows);
    // Cumulative NAL offset of each sub-stream boundary.
    let mut nal_cursor = nal_data_start;
    let mut rbsp_start = cabac_rbsp_off;
    for (i, _) in (0..ctb_rows).enumerate() {
        let rbsp_end = if i + 1 < ctb_rows {
            nal_cursor += entry_points[i] as usize;
            let e = crate::bitreader::nal_to_rbsp_offset(src_of, nal_cursor);
            e.min(rbsp_len)
        } else {
            rbsp_len
        };
        if rbsp_end < rbsp_start {
            return None;
        }
        rows.push(RowSubstream {
            start: rbsp_start,
            end: rbsp_end,
        });
        rbsp_start = rbsp_end;
    }
    Some(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::unescape_rbsp_with_map;

    #[test]
    fn substreams_without_emulation_bytes() {
        // RBSP == NAL (no 0x00 0x00 03). 3 rows, entry points 4 and 5 bytes.
        let nal = vec![0xAAu8; 20];
        let (_rbsp, src_of) = unescape_rbsp_with_map(&nal);
        // cabac starts at offset 2 (after a 2-byte header, say).
        let rows = row_substreams(&src_of, 2, &[4, 5], 20, 3).unwrap();
        assert_eq!(rows[0], RowSubstream { start: 2, end: 6 });
        assert_eq!(rows[1], RowSubstream { start: 6, end: 11 });
        assert_eq!(rows[2], RowSubstream { start: 11, end: 20 });
    }

    #[test]
    fn wrong_entry_point_count_rejected() {
        let nal = vec![0xAAu8; 20];
        let (_r, src_of) = unescape_rbsp_with_map(&nal);
        // 3 rows need 2 entry points; give 1 → None.
        assert!(row_substreams(&src_of, 2, &[4], 20, 3).is_none());
    }

    #[test]
    fn emulation_bytes_shift_rbsp_offsets() {
        // NAL: 00 00 03 00 AA ...  the 03 at index 2 is removed.
        // RBSP indices:        0  1     2  3
        // src_of:              0  1     3  4 ...
        let mut nal = vec![0x00, 0x00, 0x03, 0x00, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        nal.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        let (rbsp, src_of) = unescape_rbsp_with_map(&nal);
        // RBSP is one byte shorter than NAL.
        assert_eq!(rbsp.len(), nal.len() - 1);
        // Single row: whole payload from cabac offset 0.
        let rows = row_substreams(&src_of, 0, &[], rbsp.len(), 1).unwrap();
        assert_eq!(
            rows[0],
            RowSubstream {
                start: 0,
                end: rbsp.len()
            }
        );
        // Two rows, first sub-stream 5 NAL bytes long starting at NAL 0.
        // NAL boundary at 5 → RBSP offset: src_of.partition_point(s<5).
        // src_of = [0,1,3,4,5,6,7,8,9,10,11,12] (index 2 removed).
        let rows2 = row_substreams(&src_of, 0, &[5], rbsp.len(), 2).unwrap();
        // NAL offset 5 maps to first RBSP index whose src >= 5 → that's rbsp idx 4.
        assert_eq!(rows2[0].start, 0);
        assert_eq!(rows2[0].end, 4);
        assert_eq!(rows2[1].start, 4);
        assert_eq!(rows2[1].end, rbsp.len());
    }
}

/// Run the WPP wavefront over `template`'s picture. On success `template`'s
/// planes/grids hold the fully reconstructed (pre-loop-filter) picture.
pub(crate) fn run_wavefront(
    template: &mut FullDecoder,
    rbsp: &[u8],
    rows: &[RowSubstream],
    pool: &ThreadPool,
) -> Result<(), DecodeError> {
    let ctb_rows = rows.len();
    debug_assert_eq!(ctb_rows, template.ctb_rows_pub());

    // Per-row completed-column counters and context-snapshot hand-off slots.
    // These persist across bands so a band's first row can seed from the
    // previous band's last row.
    let progress: Vec<AtomicUsize> = (0..ctb_rows).map(|_| AtomicUsize::new(0)).collect();
    let snapshots: Vec<OnceLock<(ContextSet, IntraModeContexts)>> =
        (0..ctb_rows).map(|_| OnceLock::new()).collect();

    let (init_ctx, init_ictx) = template.init_contexts_pub();

    // One factory: `&mut template` taken exactly here, never during worker runs.
    let factory = template.row_factory();

    let first_err: OnceLock<DecodeError> = OnceLock::new();

    // Band size: at most the worker count, at least 1, capped by rows.
    let band = pool.threads().max(1);

    let mut band_start = 0usize;
    while band_start < ctb_rows {
        // Stop dispatching further bands once an error has occurred.
        if first_err.get().is_some() {
            break;
        }
        let band_end = (band_start + band).min(ctb_rows);

        pool.scope(|scope| {
            for ry in band_start..band_end {
                let sub = rows[ry];
                if sub.start > sub.end || sub.end > rbsp.len() {
                    let _ = first_err.set(DecodeError::Bitstream("wpp substream range".into()));
                    // Publish so the row below (same band) never stalls.
                    let _ = snapshots[ry].set(default_contexts());
                    progress[ry].store(usize::MAX, Ordering::Release);
                    continue;
                }
                let row_cabac = &rbsp[sub.start..sub.end];

                let progress_ref = &progress;
                let snapshots_ref = &snapshots;
                let init_ctx = init_ctx.clone();
                let first_err_ref = &first_err;
                let factory_ref = &factory;

                scope.spawn(move || {
                    // Seed contexts. Row 0 of the whole picture uses I-slice
                    // init; every other row (including a band's first row) waits
                    // for the row above to publish its post-CTB-1 snapshot. For a
                    // band's first row the above row is in the *previous* band,
                    // already finished, so its snapshot is already set.
                    let (ctx, ictx) = if ry == 0 {
                        (init_ctx, init_ictx)
                    } else {
                        loop {
                            if let Some(snap) = snapshots_ref[ry - 1].get() {
                                break snap.clone();
                            }
                            if first_err_ref.get().is_some() {
                                // Predecessor failed; bail without decoding.
                                break default_contexts();
                            }
                            std::hint::spin_loop();
                        }
                    };

                    // SAFETY: disjointness upheld by the 2-CTB lag; buffers live
                    // for the whole run (template outlives every band scope).
                    let mut row = match unsafe { factory_ref.make(row_cabac, ctx, ictx) } {
                        Ok(r) => r,
                        Err(e) => {
                            let _ = first_err_ref.set(e);
                            let _ = snapshots_ref[ry].set(default_contexts());
                            progress_ref[ry].store(usize::MAX, Ordering::Release);
                            return;
                        }
                    };

                    // The gate row is the one directly above. It is in this band
                    // (scheduled) unless ry is the band's first row, in which case
                    // it is in the previous, already-completed band — its
                    // `progress` is `cols+2`, so the gate opens immediately.
                    let above = if ry == 0 {
                        None
                    } else {
                        Some(&progress_ref[ry - 1])
                    };
                    if let Err(e) =
                        row.decode_wavefront_row(ry, &progress_ref[ry], above, &snapshots_ref[ry])
                    {
                        let _ = first_err_ref.set(e);
                    }
                });
            }
        });

        band_start = band_end;
    }

    if let Some(e) = first_err.into_inner() {
        return Err(e);
    }
    Ok(())
}

/// A fresh pair of default entropy contexts, used only to unblock a stalled row
/// below after an error so the wavefront can drain instead of deadlocking.
fn default_contexts() -> (ContextSet, IntraModeContexts) {
    (
        ContextSet::init_islice(26),
        IntraModeContexts::init_islice(26),
    )
}
