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

pub(crate) type ReconstructFn = fn(&mut [u16], usize, &[u16], &[i32], usize, usize, usize, u8);
pub(crate) type ReconstructFn16 = fn(&mut [u16], usize, &[u16], &[i16], usize, usize, usize, u8);

static RECONSTRUCT_ADD_CLIP: std::sync::OnceLock<ReconstructFn> = std::sync::OnceLock::new();
static RECONSTRUCT_ADD_CLIP16: std::sync::OnceLock<ReconstructFn16> = std::sync::OnceLock::new();

#[inline]
pub(crate) fn resolve_reconstruct_add_clip() -> ReconstructFn {
    *RECONSTRUCT_ADD_CLIP.get_or_init(|| {
        let mut _f: ReconstructFn = add_residual_into_scalar;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::add_residual_into_neon;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::add_residual_into_sse41;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn resolve_reconstruct_add_clip16() -> ReconstructFn16 {
    *RECONSTRUCT_ADD_CLIP16.get_or_init(|| {
        let mut _f: ReconstructFn16 = add_residual_into_scalar16;

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            _f = crate::neon::add_residual_into_neon16;
        }

        #[cfg(all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::add_residual_into_sse41_16;
            }
        }

        _f
    })
}

#[inline]
pub(crate) fn sample_max(bit_depth: u8) -> i32 {
    if bit_depth == 0 {
        0
    } else if bit_depth <= 15 {
        (1i32 << bit_depth) - 1
    } else {
        u16::MAX as i32
    }
}

#[inline]
pub(crate) fn has_full_dst(dst: &[u16], stride: usize, n: usize) -> bool {
    if n == 0 || stride < n {
        return false;
    }
    let Some(last_row) = n.checked_sub(1).and_then(|y| y.checked_mul(stride)) else {
        return false;
    };
    let Some(end) = last_row.checked_add(n) else {
        return false;
    };
    dst.len() >= end
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn can_reconstruct_full_block<R>(
    dst: &[u16],
    stride: usize,
    pred: &[u16],
    res: &[R],
    n: usize,
    valid_w: usize,
    valid_h: usize,
    bit_depth: u8,
) -> bool {
    let Some(n2) = n.checked_mul(n) else {
        return false;
    };
    valid_w == n
        && valid_h == n
        && bit_depth != 0
        && has_full_dst(dst, stride, n)
        && pred.len() >= n2
        && res.len() >= n2
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_residual_into_scalar(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i32],
    n: usize,
    valid_w: usize,
    valid_h: usize,
    bit_depth: u8,
) {
    add_residual_generic(dst, stride, pred, res, n, valid_w, valid_h, bit_depth);
}

/// i16-residual scalar reconstruct (8-bit depth path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_residual_into_scalar16(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i16],
    n: usize,
    valid_w: usize,
    valid_h: usize,
    bit_depth: u8,
) {
    add_residual_generic(dst, stride, pred, res, n, valid_w, valid_h, bit_depth);
}

/// Residual element: `i16` for 8-bit depth, `i32` for 10/12-bit.
pub(crate) trait Res: Copy {
    fn widen(self) -> i32;
}
impl Res for i32 {
    #[inline(always)]
    fn widen(self) -> i32 {
        self
    }
}
impl Res for i16 {
    #[inline(always)]
    fn widen(self) -> i32 {
        self as i32
    }
}

#[allow(clippy::too_many_arguments)]
fn add_residual_generic<R: Res>(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[R],
    n: usize,
    valid_w: usize,
    valid_h: usize,
    bit_depth: u8,
) {
    if n == 0 || stride == 0 {
        return;
    }

    let Some(n2) = n.checked_mul(n) else {
        return;
    };
    if pred.len() < n2 || res.len() < n2 {
        return;
    }

    let valid_w = valid_w.min(n).min(stride);
    let valid_h = valid_h.min(n);
    if valid_w == 0 || valid_h == 0 {
        return;
    }

    let max = sample_max(bit_depth);
    let pred = &pred[..n2];
    let res = &res[..n2];

    for y in 0..valid_h {
        let dst_off = y.saturating_mul(stride);
        if dst_off >= dst.len() {
            break;
        }
        let cols = valid_w.min(dst.len() - dst_off);
        if cols == 0 {
            break;
        }
        let row_off = y * n;
        let dst_row = &mut dst[dst_off..dst_off + cols];
        let pred_row = &pred[row_off..row_off + cols];
        let res_row = &res[row_off..row_off + cols];
        for ((dst, &pred), &res) in dst_row.iter_mut().zip(pred_row.iter()).zip(res_row.iter()) {
            *dst = (pred as i32 + res.widen()).clamp(0, max) as u16;
        }
    }
}
