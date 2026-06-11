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

/// intraPredAngle for modes 2..=34 (Table 8-5), indexed by mode-2.
pub(crate) static INTRA_PRED_ANGLE: [i32; 33] = [
    32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9,
    -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];

/// invAngle for modes 11..=25 (Table 8-6), indexed by mode-11.
pub(crate) static INV_ANGLE: [i32; 15] = [
    -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096,
];

pub(crate) const PLANAR: u8 = 0;
pub(crate) const DC: u8 = 1;

// These write into caller-provided slices. FullDecoder holds one `IntraScratch`
// that is reused every TU, avoiding ~4 small heap allocations per block.

/// Pre-allocated scratch for the intra prediction pipeline (reused every TU).
pub(crate) struct IntraScratch {
    pub(crate) sub_s: Vec<u16>, // 4*N+1
    pub(crate) sub_avail: Vec<bool>,
    pub(crate) above: Vec<u16>, // 2*N+1
    pub(crate) left: Vec<u16>,
    pub(crate) fa: Vec<u16>,
    pub(crate) fl: Vec<u16>,
    pub(crate) refs_ang: Vec<i32>,          // 3*N+1
    pub(crate) pred: Vec<u16>,              // N*N
    pub(crate) raw_above: Vec<Option<u16>>, // 2*N, reused for ref gathering (no per-TU alloc)
    pub(crate) raw_left: Vec<Option<u16>>,  // 2*N
}

impl Default for IntraScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl IntraScratch {
    pub(crate) fn new() -> Self {
        let max_n = 32usize;
        IntraScratch {
            sub_s: vec![0u16; 4 * max_n + 1],
            sub_avail: vec![false; 4 * max_n + 1],
            above: vec![0u16; 2 * max_n + 1],
            left: vec![0u16; 2 * max_n + 1],
            fa: vec![0u16; 2 * max_n + 1],
            fl: vec![0u16; 2 * max_n + 1],
            refs_ang: vec![0i32; 3 * max_n + 1],
            pred: vec![0u16; max_n * max_n],
            raw_above: vec![None; 2 * max_n],
            raw_left: vec![None; 2 * max_n],
        }
    }
}

/// Like `substitute_refs` but writes into caller-provided slices (length ≥ 2N+1).
#[allow(clippy::too_many_arguments)]
pub(crate) fn substitute_refs_into(
    raw_corner: Option<u16>,
    raw_above: &[Option<u16>],
    raw_left: &[Option<u16>],
    n: usize,
    neutral: u16,
    s: &mut [u16],
    avail: &mut [bool],
    above_out: &mut [u16],
    left_out: &mut [u16],
) {
    let total = 4 * n + 1;
    let s = &mut s[..total];
    let avail = &mut avail[..total];

    for v in s.iter_mut() {
        *v = 0;
    }
    for a in avail.iter_mut() {
        *a = false;
    }

    for i in 0..2 * n {
        if let Some(v) = raw_left[2 * n - 1 - i] {
            s[i] = v;
            avail[i] = true;
        }
    }
    if let Some(v) = raw_corner {
        s[2 * n] = v;
        avail[2 * n] = true;
    }
    for i in 0..2 * n {
        if let Some(v) = raw_above[i] {
            s[2 * n + 1 + i] = v;
            avail[2 * n + 1 + i] = true;
        }
    }

    if !avail.iter().any(|&a| a) {
        for v in s.iter_mut() {
            *v = neutral;
        }
    } else {
        let first = avail.iter().position(|&a| a).unwrap();
        for i in 0..first {
            s[i] = s[first];
        }
        for i in 1..total {
            if !avail[i] {
                s[i] = s[i - 1];
            }
        }
    }

    let above_out = &mut above_out[..2 * n + 1];
    let left_out = &mut left_out[..2 * n + 1];
    above_out[0] = s[2 * n];
    left_out[0] = s[2 * n];
    for i in 0..2 * n {
        above_out[1 + i] = s[2 * n + 1 + i];
        left_out[1 + i] = s[2 * n - 1 - i];
    }
}

/// Like `filter_refs` but writes into `fa_out` / `fl_out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_refs_into(
    above: &[u16],
    left: &[u16],
    n: usize,
    mode: u8,
    is_luma: bool,
    strong_intra_smoothing: bool,
    bit_depth: u8,
    fa_out: &mut [u16],
    fl_out: &mut [u16],
) {
    let len = 2 * n + 1;
    fa_out[..len].copy_from_slice(&above[..len]);
    fl_out[..len].copy_from_slice(&left[..len]);

    if !is_luma || n == 4 || mode == DC {
        return;
    }

    let min_dist = (mode as i32 - 26).abs().min((mode as i32 - 10).abs());
    let thresh = match n {
        8 => 7,
        16 => 1,
        32 => 0,
        _ => 8,
    };
    if !(mode == PLANAR || min_dist > thresh) {
        return;
    }

    if strong_intra_smoothing && n == 32 {
        let thr = 1 << (bit_depth - 5);
        let bl = left[2 * n] as i32;
        let tl = above[0] as i32;
        let tr = above[2 * n] as i32;
        let cl = left[n] as i32;
        let ct = above[n] as i32;
        if (bl + tl - 2 * cl).abs() < thr && (tr + tl - 2 * ct).abs() < thr {
            for i in 1..2 * n {
                fa_out[i] = ((((2 * n - i) as i32) * tl + (i as i32) * tr + n as i32) >> 6) as u16;
                fl_out[i] = ((((2 * n - i) as i32) * tl + (i as i32) * bl + n as i32) >> 6) as u16;
            }
            fa_out[0] = above[0];
            fl_out[0] = left[0];
            fa_out[2 * n] = above[2 * n];
            fl_out[2 * n] = left[2 * n];
            return;
        }
    }

    let corner = above[0] as i32;
    fa_out[0] = ((left[1] as i32 + 2 * corner + above[1] as i32 + 2) >> 2) as u16;
    fl_out[0] = fa_out[0];
    for i in 1..2 * n {
        fa_out[i] =
            ((above[i - 1] as i32 + 2 * above[i] as i32 + above[i + 1] as i32 + 2) >> 2) as u16;
        fl_out[i] =
            ((left[i - 1] as i32 + 2 * left[i] as i32 + left[i + 1] as i32 + 2) >> 2) as u16;
    }
    fa_out[2 * n] = above[2 * n];
    fl_out[2 * n] = left[2 * n];
}

/// Like `predict` but writes into `out[..n*n]`; `refs_ang` is work space (≥ 3N+1).
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_into(
    mode: u8,
    above: &[u16],
    left: &[u16],
    n: usize,
    is_luma: bool,
    bit_depth: u8,
    out: &mut [u16],
    refs_ang: &mut [i32],
) {
    let out = &mut out[..n * n];
    match mode {
        PLANAR => {
            let tr = above[n + 1] as i32;
            let bl = left[n + 1] as i32;
            for y in 0..n {
                let out = &mut out[y * n..y * n + n];
                for (x, dst) in out.iter_mut().enumerate() {
                    let h = (n - 1 - x) as i32 * left[y + 1] as i32 + (x + 1) as i32 * tr;
                    let v = (n - 1 - y) as i32 * above[x + 1] as i32 + (y + 1) as i32 * bl;
                    *dst = ((h + v + n as i32) >> (n.trailing_zeros() as i32 + 1)) as u16;
                }
            }
        }
        DC => {
            // Mirror predict_dc exactly: N samples from above + N from left.
            let mut sum = 0i32;
            for i in 0..n {
                sum += above[i + 1] as i32 + left[i + 1] as i32;
            }
            let dc = (sum + n as i32) >> ((n as u32).trailing_zeros() + 1);
            for v in out.iter_mut() {
                *v = dc as u16;
            }
            // Edge filter: luma only, and only for blocks smaller than 32×32
            if is_luma && n < 32 {
                out[0] = ((left[1] as i32 + 2 * dc + above[1] as i32 + 2) >> 2) as u16;
                for (dst, &above) in out[1..n].iter_mut().zip(above[2..n + 1].iter()) {
                    *dst = ((above as i32 + 3 * dc + 2) >> 2) as u16;
                }
                for (y, &left) in (1..n).zip(left[2..n + 1].iter()) {
                    out[y * n] = ((left as i32 + 3 * dc + 2) >> 2) as u16;
                }
            }
        }
        _ => {
            let angle = INTRA_PRED_ANGLE[mode as usize - 2];
            let vertical = mode >= 18;
            let main = if vertical { above } else { left };
            let side = if vertical { left } else { above };
            let base = n as i32;
            let refs = &mut refs_ang[..3 * n + 1];
            for (dst, &main) in refs[n..=3 * n].iter_mut().zip(main[..=2 * n].iter()) {
                *dst = main as i32;
            }
            if angle < 0 {
                let inv = INV_ANGLE[mode as usize - 11];
                let lim = (n as i32 * angle) >> 5;
                let mut k = -1i32;
                while k >= lim {
                    let idx = ((k * inv + 128) >> 8).min(2 * n as i32);
                    refs[(k + base) as usize] = side[idx.max(0) as usize] as i32;
                    k -= 1;
                }
            }
            let max_v = ((1i32 << bit_depth) - 1).max(0);
            for row in 0..n as i32 {
                let pos = (row + 1) * angle;
                let i_idx = pos >> 5;
                let frac = pos & 31;
                for col in 0..n as i32 {
                    let r0 = refs[(col + i_idx + 1 + base) as usize];
                    let v = if frac == 0 {
                        r0
                    } else {
                        let r1 = refs[(col + i_idx + 2 + base) as usize];
                        ((32 - frac) * r0 + frac * r1 + 16) >> 5
                    };
                    if vertical {
                        out[row as usize * n + col as usize] = v.clamp(0, max_v) as u16;
                    } else {
                        out[col as usize * n + row as usize] = v.clamp(0, max_v) as u16;
                    }
                }
            }
            if is_luma && n < 32 {
                let max = ((1i32 << bit_depth) - 1).max(0);
                if mode == 26 {
                    for (y, &left) in left[1..n + 1].iter().enumerate() {
                        let v = above[1] as i32 + ((left as i32 - above[0] as i32) >> 1);
                        out[y * n] = v.clamp(0, max) as u16;
                    }
                } else if mode == 10 {
                    for (dst, &above_p1) in out.iter_mut().zip(above[1..n + 1].iter()) {
                        let v = left[1] as i32 + ((above_p1 as i32 - above[0] as i32) >> 1);
                        *dst = v.clamp(0, max) as u16;
                    }
                }
            }
        }
    }
}
