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

static APPLY_SAO_PLANE: std::sync::OnceLock<SaoPlaneFn> = std::sync::OnceLock::new();

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
