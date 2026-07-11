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

//! Adaptive Color Transform (SCC §8.6.7). When `cu_residual_act_flag` is set,
//! the three co-located 4:4:4 residual blocks are jointly de-correlated with the
//! reversible YCgCo-R transform before being added to the prediction. Component
//! order follows HM: plane 0 = Y, plane 1 = Cb→Cg, plane 2 = Cr→Co.
//!
//! Inverse (residual → RGB-proxy), matching the forward
//!   Co = R − B; t = B + (Co>>1); Cg = G − t; Y = t + (Cg>>1):
//!   t = Y − (Cg>>1); G = Cg + t; B = t − (Co>>1); R = B + Co.
//! The transform is exactly reversible in integer arithmetic (lossless path).

/// Inverse ACT on one triple of co-located residual samples `(y, cg, co)`,
/// returning `(r0, r1, r2)` for the (luma, Cb, Cr) planes respectively.
#[inline]
pub(crate) fn inverse_sample(y: i32, cg: i32, co: i32) -> (i32, i32, i32) {
    let t = y - (cg >> 1);
    let g = cg + t;
    let b = t - (co >> 1);
    let r = b + co;
    (g, b, r)
}

/// Forward ACT (used only to verify reversibility in tests / lossless encode
/// symmetry). `(r0, r1, r2)` = (luma, Cb, Cr) residuals → `(y, cg, co)`.
#[cfg(test)]
#[inline]
pub(crate) fn forward_sample(r0: i32, r1: i32, r2: i32) -> (i32, i32, i32) {
    let co = r2 - r1;
    let t = r1 + (co >> 1);
    let cg = r0 - t;
    let y = t + (cg >> 1);
    (y, cg, co)
}

/// Apply the inverse ACT in place over three residual buffers of `n` samples.
/// For the lossy path a per-component scaling by √2 for the chroma-like Co/Cg
/// terms is folded into the QP offsets (−5/−5/−3) upstream, so this stage is a
/// pure integer lifting on the dequantized residuals.
pub(crate) fn inverse_block(r0: &mut [i32], r1: &mut [i32], r2: &mut [i32], n: usize) {
    let r0 = &mut r0[..n];
    let r1 = &mut r1[..n];
    let r2 = &mut r2[..n];

    for ((r0, r1), r2) in r0.iter_mut().zip(r1.iter_mut()).zip(r2.iter_mut()) {
        let (g, b, r) = inverse_sample(*r0, *r1, *r2);
        *r0 = g;
        *r1 = b;
        *r2 = r;
    }
}

/// i16 residual variant (reserved for an 8-bit ACT fast path).
#[cfg(test)]
pub(crate) fn inverse_block_i16(r0: &mut [i16], r1: &mut [i16], r2: &mut [i16], n: usize) {
    let r0 = &mut r0[..n];
    let r1 = &mut r1[..n];
    let r2 = &mut r2[..n];

    for ((r0, r1), r2) in r0.iter_mut().zip(r1.iter_mut()).zip(r2.iter_mut()) {
        let (g, b, r) = inverse_sample(*r0 as i32, *r1 as i32, *r2 as i32);
        *r0 = g.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        *r1 = b.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        *r2 = r.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ycgco_r_is_reversible() {
        // The lossless guarantee: inverse(forward(x)) == x for all residuals.
        for r0 in (-300..=300).step_by(7) {
            for r1 in (-300..=300).step_by(11) {
                for r2 in (-300..=300).step_by(13) {
                    let (y, cg, co) = forward_sample(r0, r1, r2);
                    assert_eq!(inverse_sample(y, cg, co), (r0, r1, r2));
                }
            }
        }
    }

    #[test]
    fn zero_maps_to_zero() {
        assert_eq!(inverse_sample(0, 0, 0), (0, 0, 0));
        assert_eq!(forward_sample(0, 0, 0), (0, 0, 0));
    }

    #[test]
    fn block_matches_per_sample() {
        let mut a = vec![10i32, -5, 3, 100];
        let mut b = vec![7i32, 2, -8, -100];
        let mut c = vec![1i32, 4, -2, 50];
        let expect: Vec<_> = (0..4).map(|i| inverse_sample(a[i], b[i], c[i])).collect();
        inverse_block(&mut a, &mut b, &mut c, 4);
        for i in 0..4 {
            assert_eq!((a[i], b[i], c[i]), expect[i]);
        }
    }

    #[test]
    fn i16_block_lossless_within_range() {
        // Small residuals round-trip exactly through the i16 path.
        let orig: Vec<(i32, i32, i32)> = (0..8).map(|i| (i - 4, (i * 3) - 12, 10 - i)).collect();
        let mut y = vec![0i16; 8];
        let mut cg = vec![0i16; 8];
        let mut co = vec![0i16; 8];
        for (i, &(r0, r1, r2)) in orig.iter().enumerate() {
            let (a, b, cc) = forward_sample(r0, r1, r2);
            y[i] = a as i16;
            cg[i] = b as i16;
            co[i] = cc as i16;
        }
        inverse_block_i16(&mut y, &mut cg, &mut co, 8);
        for (i, &(r0, r1, r2)) in orig.iter().enumerate() {
            assert_eq!((y[i] as i32, cg[i] as i32, co[i] as i32), (r0, r1, r2));
        }
    }
}
