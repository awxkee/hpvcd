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

type ReconstructFn = fn(&mut [u16], usize, &[u16], &[i32], usize, u8);

static RECONSTRUCT_ADD_CLIP: std::sync::OnceLock<ReconstructFn> = std::sync::OnceLock::new();

#[inline]
fn resolve_reconstruct_add_clip() -> ReconstructFn {
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

/// Adds inverse-transform residuals to the predicted block, clips to the active
/// bit depth, and stores into a potentially-strided destination plane.
#[inline]
pub(crate) fn add_residual_into(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i32],
    n: usize,
    bit_depth: u8,
) {
    resolve_reconstruct_add_clip()(dst, stride, pred, res, n, bit_depth)
}

#[inline]
pub(crate) fn add_residual_into_scalar(
    dst: &mut [u16],
    stride: usize,
    pred: &[u16],
    res: &[i32],
    n: usize,
    bit_depth: u8,
) {
    let max = (1i32 << bit_depth) - 1;
    let pred = &pred[..n * n];
    let res = &res[..n * n];

    for ((dst_row, pred_row), res_row) in dst
        .chunks_mut(stride)
        .take(n)
        .zip(pred.chunks_exact(n))
        .zip(res.chunks_exact(n))
    {
        let dst_row = &mut dst_row[..n];
        for ((dst, &pred), &res) in dst_row.iter_mut().zip(pred_row.iter()).zip(res_row.iter()) {
            *dst = (pred as i32 + res).clamp(0, max) as u16;
        }
    }
}
