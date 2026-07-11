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

pub(crate) const MAX_COMPONENTS: usize = 3;

/// COPY_INDEX_MODE (0) / COPY_ABOVE_MODE (1) (§7.4.9.13).
pub(crate) const COPY_ABOVE_MODE: u8 = 1;

/// CABAC operations palette decoding needs, with the precise context selection
/// the normative process requires (so §9.3.4.2.1 lives in the decoder bridge).
pub(crate) trait PaletteBits {
    /// `palette_run_type_flag` and `copy_above_indices_for_final_run_flag` — both
    /// use the SAME single context variable (§9.3.4.2.1); the bridge must route
    /// them to one `CtxModel`. Returns COPY_INDEX_MODE / COPY_ABOVE_MODE.
    fn run_type_flag(&mut self) -> u8;
    /// `palette_transpose_flag` (its own context).
    fn transpose_flag(&mut self) -> u8;
    /// `palette_run_prefix` bin `bin_idx` for a COPY_ABOVE run, or a non-first
    /// bin of a COPY_INDEX run (§9.3.4.2.1 / Table 9-49); higher bins are bypass.
    fn run_prefix_bin(&mut self, bin_idx: usize, copy_above: bool) -> u8;
    /// First `palette_run_prefix` bin of a COPY_INDEX run: context depends on the
    /// current palette index (§9.3.4.2.1).
    fn run_prefix_index_bin(&mut self, palette_index: u32) -> u8;
    /// One bypass bin.
    fn bypass(&mut self) -> u8;
    /// `n` bypass bins as an integer (MSB first).
    fn bypass_bits(&mut self, n: u32) -> u32;
}

/// Persistent palette predictor (component-major entries), reset per slice/tile/row.
#[derive(Clone, Default)]
pub(crate) struct PalettePredictor {
    pub(crate) entries: [Vec<u16>; MAX_COMPONENTS],
    pub(crate) num_comps: usize,
}

impl PalettePredictor {
    pub(crate) fn size(&self) -> usize {
        self.entries[0].len()
    }

    /// Reset to the SPS/PPS initialiser table (§9.3.2.3).
    pub(crate) fn reset_from(&mut self, init: &[Vec<u16>], num_comps: usize) {
        self.num_comps = num_comps;
        for c in 0..MAX_COMPONENTS {
            self.entries[c].clear();
        }
        for (c, col) in init.iter().enumerate().take(MAX_COMPONENTS) {
            self.entries[c].extend_from_slice(col);
        }
    }

    /// Update after a CU (§8.4.5.3 / "palette predictor update"): the CU palette
    /// moves to the front, then previous predictor entries not reused, truncated
    /// to `max_pred_size`.
    pub(crate) fn update(
        &mut self,
        cu_palette: &[[u16; MAX_COMPONENTS]],
        reused: &[bool],
        max_pred_size: usize,
    ) {
        let num_comps = self.num_comps.max(1);
        let mut new_cols: [Vec<u16>; MAX_COMPONENTS] = Default::default();
        for e in cu_palette {
            for c in 0..num_comps {
                new_cols[c].push(e[c]);
            }
        }
        let prev_len = self.entries[0].len();
        for k in 0..prev_len {
            if !reused.get(k).copied().unwrap_or(false) {
                if new_cols[0].len() >= max_pred_size {
                    break;
                }
                for (c, dst) in new_cols[..num_comps].iter_mut().enumerate() {
                    dst.push(self.entries[c][k]);
                }
            }
        }
        for (entry, new_col) in self.entries[..MAX_COMPONENTS]
            .iter_mut()
            .zip(new_cols[..MAX_COMPONENTS].iter_mut())
        {
            *entry = std::mem::take(new_col);
            entry.truncate(max_pred_size);
        }
    }
}

/// Decode `palette_predictor_run`-coded reuse flags (§7.3.8.13). Each
/// `palette_predictor_run` is EG0 (bypass): value 1 terminates; value 0 reuses
/// the current entry; value >1 skips `run-1` entries then reuses the next.
/// Returns per-predictor reuse flags and the reused count.
pub(crate) fn decode_reuse_flags<B: PaletteBits>(
    bits: &mut B,
    predictor_size: usize,
    palette_max_size: usize,
) -> (Vec<bool>, usize) {
    let mut reused = vec![false; predictor_size];
    let mut num_reused = 0usize;
    let mut idx = 0usize;
    while idx < predictor_size && num_reused < palette_max_size {
        let run = read_eg0(bits);
        if run == 1 {
            break;
        }
        if run > 1 {
            idx += (run - 1) as usize;
        }
        if idx < predictor_size {
            reused[idx] = true;
            num_reused += 1;
            idx += 1;
        }
    }
    (reused, num_reused)
}

/// SCC bypass Exp-Golomb order-0. Unlike the usual UVLC spelling, SCM's
/// `xReadEpExGolomb` uses a run of **ones** terminated by zero, followed by the
/// refinement bits.
pub(crate) fn read_eg0<B: PaletteBits>(bits: &mut B) -> u32 {
    let mut prefix = 0u32;
    while bits.bypass() != 0 {
        prefix += 1;
        if prefix > 31 {
            return 0;
        }
    }
    let suffix = if prefix > 0 {
        bits.bypass_bits(prefix)
    } else {
        0
    };
    ((1u32 << prefix) - 1).wrapping_add(suffix)
}

/// SCC bypass Exp-Golomb order-k, used by predictor runs and escape values.
pub(crate) fn read_egk<B: PaletteBits>(bits: &mut B, k: u32) -> u32 {
    let mut prefix = 0u32;
    while bits.bypass() != 0 {
        prefix += 1;
        if prefix > 31 {
            return 0;
        }
    }
    let total = prefix + k;
    let suffix = if total > 0 {
        bits.bypass_bits(total)
    } else {
        0
    };
    ((1u32 << prefix) - 1)
        .wrapping_mul(1 << k)
        .wrapping_add(suffix)
}

/// Truncated-binary decode of a value in `[0, c_max)` (§9.3.3.x). `c_max <= 1`
/// reads no bins and returns 0.
pub(crate) fn read_truncated_binary<B: PaletteBits>(bits: &mut B, c_max: u32) -> u32 {
    if c_max <= 1 {
        return 0;
    }
    let n = c_max;
    let k = 31 - n.leading_zeros(); // floor(log2 n)
    let u = (1u32 << (k + 1)) - n; // count of shorter codewords
    let prefix = bits.bypass_bits(k);
    if prefix < u {
        prefix
    } else {
        let extra = bits.bypass() as u32;
        (prefix << 1) + extra - u
    }
}

/// Decode `num_palette_indices_minus1` (§9.3.3.14).
///
/// This is *not* the coefficient-remaining code.  The prefix is a truncated
/// Rice code whose unary part is capped at four one-bins.  Only the all-ones
/// prefix has an EGk extension, with order `cRiceParam + 1`.
pub(crate) fn read_num_palette_indices<B: PaletteBits>(
    bits: &mut B,
    max_palette_index: u32,
) -> u32 {
    let rice = 3 + ((max_palette_index + 1) >> 3);
    let mut prefix = 0u32;
    while prefix < 4 {
        if bits.bypass() == 0 {
            let suffix = if rice != 0 { bits.bypass_bits(rice) } else { 0 };
            return (prefix << rice).wrapping_add(suffix);
        }
        prefix += 1;
    }

    (4u32 << rice).wrapping_add(read_egk(bits, rice + 1))
}

/// Decode `palette_run` (§9.3.3.x). `palette_max_run` = remaining samples − 1.
/// The `palette_run_prefix` is context-coded truncated-unary bounded by
/// `Floor(Log2(PaletteMaxRun)) + 1`; when `prefix > 1` a bypass
/// `palette_run_suffix` follows and `run = (1 << (prefix−1)) + suffix`. The first
/// COPY_INDEX prefix bin's context depends on `palette_index` (§9.3.4.2.1).
pub(crate) fn decode_run<B: PaletteBits>(
    bits: &mut B,
    palette_max_run: u32,
    copy_above: bool,
    palette_index: u32,
) -> u32 {
    if palette_max_run == 0 {
        return 0;
    }
    let max_prefix = 32 - palette_max_run.leading_zeros(); // Floor(Log2)+1
    let mut prefix = 0u32;
    while prefix < max_prefix {
        let bin = if !copy_above && prefix == 0 {
            bits.run_prefix_index_bin(palette_index)
        } else {
            bits.run_prefix_bin(prefix as usize, copy_above)
        };
        if bin == 0 {
            break;
        }
        prefix += 1;
    }
    let run = if prefix < 2 {
        prefix
    } else {
        let base = 1u32 << (prefix - 1);
        if palette_max_run == base {
            base
        } else {
            // palette_run_suffix is truncated binary, not a fixed-width field.
            // Its cMax is inclusive in the specification, while
            // `read_truncated_binary` takes the alphabet size.
            let suffix_max = if (base << 1) > palette_max_run {
                palette_max_run - base
            } else {
                base - 1
            };
            base + read_truncated_binary(bits, suffix_max + 1)
        }
    };
    run.min(palette_max_run)
}

/// Palette traverse scan position `i` → `(x, y)` for a `size × size` block.
///
/// Palette syntax always uses `ScanOrder[log2BlockSize][3]`, i.e. the fixed
/// horizontal boustrophedon traverse. `palette_transpose_flag` affects chroma
/// sample placement/reconstruction; it does not rotate the entropy scan.
#[inline]
pub(crate) fn scan_pos(i: usize, size: usize, _transpose: bool) -> (usize, usize) {
    let y = i / size;
    let x_in_row = i % size;
    let x = if y & 1 == 1 {
        size - 1 - x_in_row
    } else {
        x_in_row
    };
    (x, y)
}

/// A fully decoded palette CU, ready to reconstruct.
pub(crate) struct PaletteCu {
    pub(crate) palette: Vec<[u16; MAX_COMPONENTS]>,
    pub(crate) transpose: bool,
    /// Per-scan-position index in the fixed traverse order; `MaxPaletteIndex` marks escape.
    pub(crate) indices: Vec<u32>,
    /// Per-scan-position escape values (only meaningful at escape positions),
    /// component-major triples in traversal order.
    pub(crate) escapes: Vec<[u16; MAX_COMPONENTS]>,
    /// Value that marks an escape sample (== CurrentPaletteSize).
    pub(crate) escape_index: u32,
}

/// Index-value array + final-run flag, read up front per §7.3.8.13 order:
/// `num_palette_indices_minus1`, all `palette_index_idc`, then
/// `copy_above_indices_for_final_run_flag`.
pub(crate) struct PaletteIndexData {
    pub(crate) idc: Vec<u32>,
    pub(crate) final_run_copy_above: bool,
}

/// Read `num_palette_indices_minus1`, the `palette_index_idc` array, and
/// `copy_above_indices_for_final_run_flag`. The first idc uses alphabet
/// `MaxPaletteIndex+1`; subsequent idc use alphabet `MaxPaletteIndex` (the
/// reduction is the adjusted-index mechanism).
pub(crate) fn decode_index_values<B: PaletteBits>(
    bits: &mut B,
    n: usize,
    max_palette_index: u32,
) -> PaletteIndexData {
    let num_indices = (read_num_palette_indices(bits, max_palette_index) as usize + 1).min(n);
    let mut idc = Vec::with_capacity(num_indices);
    for i in 0..num_indices {
        let alphabet = if i == 0 {
            max_palette_index + 1
        } else {
            max_palette_index
        };
        idc.push(read_truncated_binary(bits, alphabet));
    }
    let final_run_copy_above = bits.run_type_flag() == COPY_ABOVE_MODE;
    PaletteIndexData {
        idc,
        final_run_copy_above,
    }
}

/// Walk scan positions decoding `palette_run_type_flag` + `palette_run`, mapping
/// the pre-read `idc` values to actual indices via the adjusted-reference rule
/// (§7.4.9.13 / §8.4.4.2). Once all explicit indices are consumed the tail is a
/// single final run (mode = `final_run_copy_above`) filling to the block end.
pub(crate) fn assign_index_runs<B: PaletteBits>(
    bits: &mut B,
    size: usize,
    _max_palette_index: u32,
    data: &PaletteIndexData,
    _transpose: bool,
) -> Vec<u32> {
    let n = size * size;
    let mut indices = vec![0u32; n];
    let idc = &data.idc;
    let mut idc_pos = 0usize;
    let mut copy_index_runs = idc.len();
    let mut scan = 0usize;
    let mut prev_mode_copy_above = false;

    while scan < n {
        let can_copy_above = scan >= size;

        // SCM infers COPY_INDEX after a COPY_ABOVE run. Otherwise the flag is
        // present only when both run types remain possible; at the last sample
        // it is inferred so that an outstanding COPY_INDEX run can be consumed.
        let copy_above = if can_copy_above && !prev_mode_copy_above {
            if copy_index_runs != 0 && scan + 1 < n {
                bits.run_type_flag() == COPY_ABOVE_MODE
            } else {
                !(scan + 1 == n && copy_index_runs != 0)
            }
        } else {
            false
        };

        // `siCurLevel` is the unadjusted idc symbol and is also the value used
        // to select the COPY_INDEX run-prefix context. The reconstructed palette
        // index excludes the previous reference only from the second run on.
        let (raw_index, index_value) = if copy_above {
            (0, 0)
        } else {
            let raw = idc.get(idc_pos).copied().unwrap_or(0);
            idc_pos = idc_pos.saturating_add(1);
            copy_index_runs = copy_index_runs.saturating_sub(1);

            let actual = if scan == 0 {
                raw
            } else {
                let reference = if prev_mode_copy_above {
                    index_above_actual(&indices, scan, size)
                } else {
                    indices[scan - 1]
                };
                if raw >= reference { raw + 1 } else { raw }
            };
            (raw, actual)
        };

        let last_run = copy_index_runs == 0 && copy_above == data.final_run_copy_above;
        let run = if last_run {
            (n - scan - 1) as u32
        } else {
            // Reserve one sample for every remaining COPY_INDEX run and, when
            // requested, one for the final COPY_ABOVE run.
            let reserved = copy_index_runs + if data.final_run_copy_above { 1 } else { 0 };
            let max_run = n.saturating_sub(scan + 1 + reserved) as u32;
            decode_run(bits, max_run, copy_above, raw_index)
        };

        for _ in 0..=run {
            if scan >= n {
                break;
            }
            let v = if copy_above {
                index_above_actual(&indices, scan, size)
            } else {
                index_value
            };
            indices[scan] = v;
            scan += 1;
        }
        prev_mode_copy_above = copy_above;
    }
    indices
}

/// Test-only convenience: read index values and walk runs in one call (the
/// decoder proper interleaves transpose / delta-QP between the two phases).
#[cfg(test)]
pub(crate) fn decode_index_map<B: PaletteBits>(
    bits: &mut B,
    size: usize,
    current_palette_size: usize,
    escape_present: bool,
    transpose: bool,
) -> Vec<u32> {
    let n = size * size;
    let escape_index = current_palette_size as u32;
    let max_palette_index: i64 = current_palette_size as i64 - 1 + escape_present as i64;
    if max_palette_index <= 0 {
        let mut indices = vec![0u32; n];
        if max_palette_index == 0 && escape_present && current_palette_size == 0 {
            indices.iter_mut().for_each(|v| *v = escape_index);
        }
        return indices;
    }
    let data = decode_index_values(bits, n, max_palette_index as u32);
    assign_index_runs(bits, size, max_palette_index as u32, &data, transpose)
}

#[inline]
fn index_above_actual(indices: &[u32], scan: usize, size: usize) -> u32 {
    let (x, y) = scan_pos(scan, size, false);
    if y == 0 {
        return 0;
    }
    let above_scan = scan_pos_inv(x, y - 1, size, false);
    indices.get(above_scan).copied().unwrap_or(0)
}

/// Inverse of the fixed palette traverse scan.
#[inline]
pub(crate) fn scan_pos_inv(x: usize, y: usize, size: usize, _transpose: bool) -> usize {
    let x_in_row = if y & 1 == 1 { size - 1 - x } else { x };
    y * size + x_in_row
}

/// Reconstruct a decoded palette CU into up to three component planes. Escape
/// samples take their explicit (already-dequantised) values from `cu.escapes`
/// at the matching scan position; palette samples take `cu.palette[index]`.
/// Chroma is written only at sampled geometric positions for 4:2:0 / 4:2:2
/// (`sub[c]` = subsampling; 4:4:4 uses (1,1)).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct(
    cu: &PaletteCu,
    size: usize,
    num_comps: usize,
    planes: [&mut [u16]; MAX_COMPONENTS],
    strides: [usize; MAX_COMPONENTS],
    dims: [(usize, usize); MAX_COMPONENTS],
    origins: [(usize, usize); MAX_COMPONENTS],
    sub: [(usize, usize); MAX_COMPONENTS],
) {
    let [p0, p1, p2] = planes;
    let plist: [&mut [u16]; MAX_COMPONENTS] = [p0, p1, p2];

    // H.265 §8.4.4.2.7 applies palette_transpose_flag at reconstruction, not
    // while parsing the fixed traverse scan. Iterate each component's output
    // grid and map it back to the luma-resolution PaletteIndexMap location.
    for c in 0..num_comps {
        let (sw, sh) = sub[c];
        let out_w = size / sw;
        let out_h = size / sh;
        let (ox, oy) = origins[c];
        let (pw, ph) = dims[c];

        for y in 0..out_h {
            for x in 0..out_w {
                let (x_l, y_l) = if cu.transpose {
                    (y * sh, x * sw)
                } else {
                    (x * sw, y * sh)
                };
                let scan = scan_pos_inv(x_l, y_l, size, false);
                let idx = cu.indices.get(scan).copied().unwrap_or(0);
                let is_escape = idx == cu.escape_index;
                let sample = if is_escape {
                    cu.escapes.get(scan).copied().unwrap_or([0; MAX_COMPONENTS])[c]
                } else {
                    cu.palette
                        .get(idx as usize)
                        .copied()
                        .unwrap_or([0; MAX_COMPONENTS])[c]
                };

                let px = ox + x;
                let py = oy + y;
                if px < pw && py < ph {
                    plist[c][py * strides[c] + px] = sample;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBits {
        bits: Vec<u8>,
        pos: usize,
    }
    impl MockBits {
        fn new(bits: Vec<u8>) -> Self {
            MockBits { bits, pos: 0 }
        }
        fn n(&mut self) -> u8 {
            let b = self.bits.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            b
        }
    }
    impl PaletteBits for MockBits {
        fn run_type_flag(&mut self) -> u8 {
            self.n()
        }
        fn transpose_flag(&mut self) -> u8 {
            self.n()
        }
        fn run_prefix_bin(&mut self, _i: usize, _ca: bool) -> u8 {
            self.n()
        }
        fn run_prefix_index_bin(&mut self, _x: u32) -> u8 {
            self.n()
        }
        fn bypass(&mut self) -> u8 {
            self.n()
        }
        fn bypass_bits(&mut self, k: u32) -> u32 {
            let mut v = 0u32;
            for _ in 0..k {
                v = (v << 1) | self.n() as u32;
            }
            v
        }
    }

    // --- reference encoder mirroring the decoder binarization (for round-trips) ---
    struct Sink {
        b: Vec<u8>,
    }
    impl Sink {
        fn new() -> Self {
            Sink { b: vec![] }
        }
        fn bit(&mut self, x: u32) {
            self.b.push((x & 1) as u8);
        }
        fn bits_n(&mut self, v: u32, n: u32) {
            for i in (0..n).rev() {
                self.bit((v >> i) & 1);
            }
        }
        fn eg_k(&mut self, val: u32, k: u32) {
            let prefix = 31 - ((val >> k) + 1).leading_zeros();
            let base = ((1u32 << prefix) - 1) << k;
            for _ in 0..prefix {
                self.bit(1);
            }
            self.bit(0);
            self.bits_n(val - base, prefix + k);
        }
        fn num_indices(&mut self, count_minus1: u32, max_palette_index: u32) {
            let rice = 3 + ((max_palette_index + 1) >> 3);
            let cmax = 4u32 << rice;
            if count_minus1 < cmax {
                let prefix = count_minus1 >> rice;
                for _ in 0..prefix {
                    self.bit(1);
                }
                self.bit(0);
                self.bits_n(count_minus1 & ((1u32 << rice) - 1), rice);
            } else {
                for _ in 0..4 {
                    self.bit(1);
                }
                self.eg_k(count_minus1 - cmax, rice + 1);
            }
        }
        fn tb(&mut self, val: u32, alphabet: u32) {
            if alphabet <= 1 {
                return;
            }
            let k = 31 - alphabet.leading_zeros();
            let u = (1u32 << (k + 1)) - alphabet;
            if val < u {
                self.bits_n(val, k);
            } else {
                let coded = val + u;
                self.bits_n(coded >> 1, k);
                self.bit(coded & 1);
            }
        }
        fn run(&mut self, run: u32, pmax: u32) {
            if pmax == 0 {
                return;
            }
            let mp = 32 - pmax.leading_zeros();
            let p = (if run < 2 {
                run
            } else {
                32 - run.leading_zeros()
            })
            .min(mp);
            for _ in 0..p {
                self.bit(1);
            }
            if p < mp {
                self.bit(0);
            }
            if p >= 2 {
                let base = 1u32 << (p - 1);
                if pmax != base {
                    let suffix_max = if (base << 1) > pmax {
                        pmax - base
                    } else {
                        base - 1
                    };
                    self.tb(run - base, suffix_max + 1);
                }
            }
        }
    }

    #[test]
    fn truncated_binary_and_eg() {
        assert_eq!(read_truncated_binary(&mut MockBits::new(vec![0, 0]), 5), 0);
        assert_eq!(read_truncated_binary(&mut MockBits::new(vec![1, 0]), 5), 2);
        assert_eq!(
            read_truncated_binary(&mut MockBits::new(vec![1, 1, 1]), 5),
            4
        );
        let mut m = MockBits::new(vec![]);
        assert_eq!(read_truncated_binary(&mut m, 1), 0);
        assert_eq!(m.pos, 0);
        assert_eq!(read_eg0(&mut MockBits::new(vec![0])), 0);
        assert_eq!(read_eg0(&mut MockBits::new(vec![1, 0, 1])), 2);

        for k in 0..=3 {
            for val in 0..128 {
                let mut s = Sink::new();
                s.eg_k(val, k);
                assert_eq!(read_egk(&mut MockBits::new(s.b), k), val);
            }
        }
    }

    #[test]
    fn scan_inverse_roundtrips() {
        let size = 4;
        for i in 0..size * size {
            for tr in [false, true] {
                let (x, y) = scan_pos(i, size, tr);
                assert_eq!(scan_pos_inv(x, y, size, tr), i);
            }
        }
        // Snake geometry: for 4x4, i=4 is (3,1); the sample above (3,0) is scan 3.
        assert_eq!(scan_pos_inv(3, 0, 4, false), 3);
    }

    #[test]
    fn predictor_update_order() {
        let mut p = PalettePredictor::default();
        p.reset_from(&[vec![10, 20, 30], vec![11, 21, 31], vec![12, 22, 32]], 3);
        p.update(&[[5, 5, 5], [6, 6, 6]], &[true, false, true], 64);
        assert_eq!(p.entries[0], vec![5, 6, 20]);
        assert_eq!(p.entries[2], vec![5, 6, 22]);
    }

    #[test]
    fn reuse_flags() {
        let mut s = Sink::new();
        for run in [0, 0, 2, 1] {
            s.eg_k(run, 0);
        }
        let (r, n) = decode_reuse_flags(&mut MockBits::new(s.b), 5, 64);
        assert_eq!(r, vec![true, true, false, true, false]);
        assert_eq!(n, 3);
    }

    #[test]
    fn max_index_zero_reads_nothing() {
        let mut m = MockBits::new(vec![]);
        let idx = decode_index_map(&mut m, 2, 1, false, false);
        assert_eq!(idx, vec![0, 0, 0, 0]);
        assert_eq!(m.pos, 0);
    }

    // Encode COPY_INDEX-only index arrays and round-trip through the decoder.
    fn encode_index_only(indices: &[u32], size: usize, max_idx: u32) -> Vec<u8> {
        let n = size * size;
        let mut runs: Vec<(u32, u32)> = vec![];
        let mut i = 0;
        while i < n {
            let v = indices[i];
            let mut j = i + 1;
            while j < n && indices[j] == v {
                j += 1;
            }
            runs.push((v, (j - i - 1) as u32));
            i = j;
        }
        let mut s = Sink::new();
        s.num_indices(runs.len() as u32 - 1, max_idx);
        for (ri, (idx, _)) in runs.iter().enumerate() {
            let raw = if ri == 0 {
                *idx
            } else if *idx > runs[ri - 1].0 {
                *idx - 1
            } else {
                *idx
            };
            s.tb(raw, if ri == 0 { max_idx + 1 } else { max_idx });
        }
        // copy_above_indices_for_final_run_flag: all runs here are COPY_INDEX.
        s.b.push(0);
        let mut scan = 0;
        for (ri, (_, rl)) in runs.iter().enumerate() {
            let (_, y) = scan_pos(scan, size, false);
            if y > 0 && scan + 1 < n {
                s.b.push(0); // run_type = COPY_INDEX
            }
            let remaining = runs.len() - ri - 1;
            if remaining != 0 {
                s.run(*rl, (n - scan - 1 - remaining) as u32);
            }
            scan += (*rl + 1) as usize;
        }
        s.b
    }

    #[test]
    fn index_map_copy_index_roundtrip() {
        let cases: Vec<(usize, usize, Vec<u32>)> = vec![
            (2, 3, vec![0, 1, 2, 0]),
            (4, 4, vec![0, 1, 1, 2, 3, 3, 0, 0, 1, 2, 2, 2, 0, 0, 1, 3]),
            (2, 2, vec![0, 1, 1, 0]),
        ];
        for (size, psize, arr) in &cases {
            let bits = encode_index_only(arr, *size, *psize as u32 - 1);
            let dec = decode_index_map(&mut MockBits::new(bits), *size, *psize, false, false);
            assert_eq!(&dec, arr, "size={size}");
        }
    }

    #[test]
    fn reconstruct_with_escape_per_scanpos() {
        let cu = PaletteCu {
            palette: vec![[10, 10, 10], [20, 20, 20]],
            transpose: false,
            indices: vec![1, 2, 0, 0], // scan1 == palette_size (2) → escape
            escapes: {
                let mut e = vec![[0u16; 3]; 4];
                e[1] = [99, 88, 77];
                e
            },
            escape_index: 2,
        };
        let mut y = vec![0u16; 4];
        let mut cb = vec![0u16; 4];
        let mut cr = vec![0u16; 4];
        reconstruct(
            &cu,
            2,
            3,
            [&mut y, &mut cb, &mut cr],
            [2, 2, 2],
            [(2, 2), (2, 2), (2, 2)],
            [(0, 0), (0, 0), (0, 0)],
            [(1, 1), (1, 1), (1, 1)],
        );
        // scan: i0→(0,0) idx1=20; i1→(1,0) escape; i2→(1,1) idx0=10; i3→(0,1) idx0=10
        assert_eq!(y, vec![20, 99, 10, 10]);
        assert_eq!(cb, vec![20, 88, 10, 10]);
        assert_eq!(cr, vec![20, 77, 10, 10]);
    }
}
