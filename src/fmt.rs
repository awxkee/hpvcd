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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitDepth {
    Eight,
    Ten,
    Twelve,
}

impl BitDepth {
    /// Construct from a raw bit count (8, 10, or 12). Panics on unsupported depths.
    pub fn from_bits(bits: u8) -> Self {
        match bits {
            8 => BitDepth::Eight,
            10 => BitDepth::Ten,
            12 => BitDepth::Twelve,
            other => panic!("unsupported bit depth: {other} (only 8, 10, 12 supported)"),
        }
    }

    /// Bit count (8, 10, or 12).
    pub fn bits(self) -> u8 {
        match self {
            BitDepth::Eight => 8,
            BitDepth::Ten => 10,
            BitDepth::Twelve => 12,
        }
    }

    /// `bit_depth - 8`, the value the SPS/hvcC store as `*_minus8`.
    pub fn minus8(self) -> u8 {
        self.bits() - 8
    }

    /// Maximum representable sample value: `(1 << bits) - 1` (255, 1023, or 4095).
    pub fn max_val(self) -> u16 {
        (1u16 << self.bits()) - 1
    }

    /// Neutral / midpoint sample: `1 << (bits - 1)` (128, 512, or 2048). Used as the
    /// unavailable-reference default in intra prediction.
    pub fn neutral(self) -> u16 {
        1u16 << (self.bits() - 1)
    }

    /// QpBdOffset = `6 * (bit_depth - 8)` (0, 12, or 24). The decoder dequantises at
    /// `SliceQp + QpBdOffset`, so the encoder must use the same effective QP.
    pub fn qp_bd_offset(self) -> u8 {
        6 * self.minus8()
    }
}

/// Chroma subsampling mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaFormat {
    /// 4:0:0 — monochrome (luma only, no chroma). ChromaArrayType = 0.
    Monochrome,
    /// 4:2:0 — chroma is half width, half height. ChromaArrayType = 1.
    Yuv420,
    /// 4:2:2 — chroma is half width, full height. ChromaArrayType = 2.
    Yuv422,
    /// 4:4:4 — chroma is full width, full height. ChromaArrayType = 3.
    Yuv444,
}

impl ChromaFormat {
    /// `chroma_format_idc` value written in the SPS (HEVC Table 6-1).
    pub fn idc(self) -> u32 {
        match self {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::Yuv420 => 1,
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 3,
        }
    }

    /// True when there is no chroma (4:0:0).
    pub fn is_monochrome(self) -> bool {
        matches!(self, ChromaFormat::Monochrome)
    }

    /// Horizontal subsampling factor: luma_width / chroma_width. (1 for monochrome,
    /// which has no chroma; the value is unused but kept well-defined.)
    pub fn sub_w(self) -> usize {
        match self {
            ChromaFormat::Yuv420 | ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 | ChromaFormat::Monochrome => 1,
        }
    }

    /// Vertical subsampling factor: luma_height / chroma_height.
    pub fn sub_h(self) -> usize {
        match self {
            ChromaFormat::Yuv420 => 2,
            ChromaFormat::Yuv422 | ChromaFormat::Yuv444 | ChromaFormat::Monochrome => 1,
        }
    }

    /// Side length of each chroma transform block for an 8×8 luma CU.
    pub fn chroma_tb_size(self) -> usize {
        match self {
            ChromaFormat::Yuv420 | ChromaFormat::Yuv422 => 4,
            ChromaFormat::Yuv444 => 8,
            ChromaFormat::Monochrome => 0,
        }
    }

    /// Number of chroma transform blocks stacked vertically per 8×8 luma CU.
    pub fn chroma_tbs_per_cu(self) -> usize {
        match self {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::Yuv420 => 1,
            ChromaFormat::Yuv422 => 2,
            ChromaFormat::Yuv444 => 1,
        }
    }
}

/// Pixel buffer returned by [`crate::decode_heic`].
///
/// Variants encode both bit depth and channel count:
/// - `Luma8` / `Luma16` — monochrome (4:0:0); one sample per pixel.
/// - `U8` / `U16`       — color (4:2:0/2:2/4:4:4); R₀G₀B₀ R₁G₁B₁ … interleaved.
#[derive(Clone, Debug)]
pub enum ImageBuffer {
    Luma8(Vec<u8>),   // 8-bit grayscale
    Luma16(Vec<u16>), // 10/12-bit grayscale
    Rgb8(Vec<u8>),    // 8-bit RGB interleaved
    Rgb16(Vec<u16>),  // 10/12-bit RGB interleaved
}

impl ImageBuffer {
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::Luma8(v) | Self::Rgb8(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_u16(&self) -> Option<&[u16]> {
        match self {
            Self::Luma16(v) | Self::Rgb16(v) => Some(v),
            _ => None,
        }
    }
    /// Number of channels: 1 for luma-only, 3 for RGB.
    pub fn channels(&self) -> usize {
        match self {
            Self::Luma8(_) | Self::Luma16(_) => 1,
            _ => 3,
        }
    }
    /// Total sample count (= `width × height × channels()`).
    pub fn len(&self) -> usize {
        match self {
            Self::Luma8(v) => v.len(),
            Self::Luma16(v) => v.len(),
            Self::Rgb8(v) => v.len(),
            Self::Rgb16(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn is_luma(&self) -> bool {
        matches!(self, Self::Luma8(_) | Self::Luma16(_))
    }
}

/// A typed single-plane sample buffer — `u8` for 8-bit, `u16` for 10/12-bit.
///
/// Used for the individual Y, Cb, Cr planes of [`crate::DecodedYuv`].
#[derive(Clone, Debug)]
pub enum SampleBuf {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl SampleBuf {
    pub fn as_u8(&self) -> Option<&[u8]> {
        if let Self::U8(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_u16(&self) -> Option<&[u16]> {
        if let Self::U16(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Visible layout of one owned image plane. `stride` and `offset` are measured
/// in samples, not bytes. The backing allocation may include coded padding
/// before, after, or between visible rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub offset: usize,
}

/// An owning, typed, strided image plane.
///
/// Row access exposes only visible samples and hides coded padding. Use
/// [`PlaneBuffer::data`] together with [`PlaneBuffer::stride`] when passing the
/// visible plane to a strided image-processing API. [`PlaneBuffer::storage`]
/// exposes the complete coded allocation, including prefix and trailing padding.
#[derive(Clone, Debug)]
pub struct PlaneBuffer<T> {
    data: Vec<T>,
    layout: PlaneLayout,
}

impl<T> PlaneBuffer<T> {
    pub(crate) fn from_parts(data: Vec<T>, layout: PlaneLayout) -> Self {
        debug_assert!(layout.stride >= layout.width);
        debug_assert!(layout.height == 0 || layout.width == 0 || layout.offset < data.len());
        debug_assert!(
            layout.height == 0
                || (layout.height - 1)
                    .checked_mul(layout.stride)
                    .and_then(|rows| layout.offset.checked_add(rows))
                    .and_then(|v| v.checked_add(layout.width))
                    .is_some_and(|end| end <= data.len())
        );
        Self { data, layout }
    }

    pub(crate) fn tight(data: Vec<T>, width: usize, height: usize) -> Self {
        debug_assert_eq!(data.len(), width.saturating_mul(height));
        Self::from_parts(
            data,
            PlaneLayout {
                width,
                height,
                stride: width,
                offset: 0,
            },
        )
    }

    #[inline]
    pub fn layout(&self) -> PlaneLayout {
        self.layout
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.layout.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.layout.height
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.layout.stride
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset
    }

    /// Distance between rows in bytes.
    #[inline]
    pub fn stride_bytes(&self) -> usize {
        self.layout.stride.saturating_mul(size_of::<T>())
    }

    /// Byte offset of the first visible sample within the allocation.
    #[inline]
    pub fn offset_bytes(&self) -> usize {
        self.layout.offset.saturating_mul(size_of::<T>())
    }

    /// Number of visible samples.
    #[inline]
    pub fn len(&self) -> usize {
        self.layout.width.saturating_mul(self.layout.height)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layout.width == 0 || self.layout.height == 0
    }

    /// Visible width and height in samples.
    #[inline]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.layout.width, self.layout.height)
    }

    /// Whether the visible rows are adjacent with no stride padding.
    ///
    /// A tightly packed plane may still be a subrange of a larger coded
    /// allocation. Use [`PlaneBuffer::as_tight`] to obtain that subrange.
    #[inline]
    pub fn is_tightly_packed(&self) -> bool {
        self.layout.stride == self.layout.width
    }

    /// Plane storage beginning at the visible top-left sample.
    ///
    /// The slice includes any padding between visible rows, but excludes coded
    /// samples before the visible origin and unused storage after the final
    /// visible row. This is the convenient representation for APIs that accept
    /// a plane slice plus a stride measured in samples.
    #[inline]
    pub fn data(&self) -> &[T] {
        let range = self.visible_data_range();
        &self.data[range]
    }

    /// Mutable plane storage beginning at the visible top-left sample.
    ///
    /// As with [`PlaneBuffer::data`], row padding is retained and the caller
    /// must use [`PlaneBuffer::stride`] to advance between rows.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        let range = self.visible_data_range();
        &mut self.data[range]
    }

    /// Complete backing allocation, including coded padding.
    #[inline]
    pub fn storage(&self) -> &[T] {
        &self.data
    }

    /// Consume the plane and return the complete backing allocation.
    #[inline]
    pub fn into_storage(self) -> Vec<T> {
        self.data
    }

    #[inline]
    fn visible_data_range(&self) -> core::ops::Range<usize> {
        if self.is_empty() {
            return self.layout.offset..self.layout.offset;
        }
        let span = (self.layout.height - 1)
            .saturating_mul(self.layout.stride)
            .saturating_add(self.layout.width);
        self.layout.offset..self.layout.offset.saturating_add(span)
    }

    /// One visible row, excluding stride padding.
    pub fn row(&self, y: usize) -> Option<&[T]> {
        if y >= self.layout.height {
            return None;
        }
        let start = y
            .checked_mul(self.layout.stride)
            .and_then(|v| self.layout.offset.checked_add(v))?;
        let end = start.checked_add(self.layout.width)?;
        self.data.get(start..end)
    }

    /// Iterate visible rows while hiding stride and crop offsets.
    #[inline]
    pub fn rows(&self) -> PlaneRows<'_, T> {
        PlaneRows {
            plane: self,
            row: 0,
        }
    }

    /// Contiguous visible samples when rows have no padding. The returned slice
    /// may be a subrange of a larger coded allocation.
    pub fn as_tight(&self) -> Option<&[T]> {
        if !self.is_tightly_packed() {
            return None;
        }
        let len = self.layout.width.checked_mul(self.layout.height)?;
        let end = self.layout.offset.checked_add(len)?;
        self.data.get(self.layout.offset..end)
    }
}

impl<T: Copy + Default> PlaneBuffer<T> {
    /// Consume the plane and return tightly packed visible samples. This is
    /// allocation-free when the visible plane already occupies the full
    /// backing vector; otherwise only the visible rows are copied.
    pub fn into_tight(self) -> Result<Vec<T>, crate::DecodeError> {
        let len = self
            .layout
            .width
            .checked_mul(self.layout.height)
            .ok_or_else(|| {
                crate::DecodeError::Bitstream("plane dimensions overflow usize".into())
            })?;
        if len == 0 {
            return Ok(Vec::new());
        }
        if self.layout.offset == 0
            && self.layout.stride == self.layout.width
            && self.data.len() == len
        {
            return Ok(self.data);
        }

        let mut out = try_vec![T::default(); len, "tightly packed image plane"];
        for (src, dst) in self.rows().zip(out.chunks_exact_mut(self.layout.width)) {
            dst.copy_from_slice(src);
        }
        Ok(out)
    }
}

/// Exact-size iterator over visible rows of a [`PlaneBuffer`].
pub struct PlaneRows<'a, T> {
    plane: &'a PlaneBuffer<T>,
    row: usize,
}

impl<'a, T> Iterator for PlaneRows<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        let row = self.plane.row(self.row)?;
        self.row += 1;
        Some(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.plane.height().saturating_sub(self.row);
        (left, Some(left))
    }
}

impl<T> ExactSizeIterator for PlaneRows<'_, T> {}

/// Three planar components with a single sample type. Chroma planes are absent
/// for monochrome images.
#[derive(Clone, Debug)]
pub struct PlanarImage<T> {
    pub y: PlaneBuffer<T>,
    pub cb: Option<PlaneBuffer<T>>,
    pub cr: Option<PlaneBuffer<T>>,
}

impl<T> PlanarImage<T> {
    #[inline]
    pub fn width(&self) -> usize {
        self.y.width()
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.y.height()
    }
}

/// Typed planar YCbCr storage. Callers match once for the complete image rather
/// than independently matching Y, Cb, and Cr buffers.
#[derive(Clone, Debug)]
pub enum YuvBuffer {
    U8(PlanarImage<u8>),
    U16(PlanarImage<u16>),
}

impl YuvBuffer {
    #[inline]
    pub fn as_u8(&self) -> Option<&PlanarImage<u8>> {
        match self {
            Self::U8(planes) => Some(planes),
            Self::U16(_) => None,
        }
    }

    #[inline]
    pub fn as_u16(&self) -> Option<&PlanarImage<u16>> {
        match self {
            Self::U8(_) => None,
            Self::U16(planes) => Some(planes),
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        match self {
            Self::U8(planes) => planes.width(),
            Self::U16(planes) => planes.width(),
        }
    }

    #[inline]
    pub fn height(&self) -> usize {
        match self {
            Self::U8(planes) => planes.height(),
            Self::U16(planes) => planes.height(),
        }
    }
}

/// One typed plane, used for auxiliary alpha images.
#[derive(Clone, Debug)]
pub enum SamplePlane {
    U8(PlaneBuffer<u8>),
    U16(PlaneBuffer<u16>),
}

impl SamplePlane {
    #[inline]
    pub fn width(&self) -> usize {
        match self {
            Self::U8(plane) => plane.width(),
            Self::U16(plane) => plane.width(),
        }
    }

    #[inline]
    pub fn height(&self) -> usize {
        match self {
            Self::U8(plane) => plane.height(),
            Self::U16(plane) => plane.height(),
        }
    }

    #[inline]
    pub fn as_u8(&self) -> Option<&PlaneBuffer<u8>> {
        match self {
            Self::U8(plane) => Some(plane),
            Self::U16(_) => None,
        }
    }

    #[inline]
    pub fn as_u16(&self) -> Option<&PlaneBuffer<u16>> {
        match self {
            Self::U8(_) => None,
            Self::U16(plane) => Some(plane),
        }
    }
}
//
// /// Full pixel-format description: chroma subsampling and sample bit depth.
// #[derive(Clone, Copy, Debug)]
// pub struct PixelFormat {
//     pub chroma: ChromaFormat,
//     pub bit_depth: BitDepth,
// }
//
// impl PixelFormat {
//     pub fn new(chroma: ChromaFormat, bit_depth: BitDepth) -> Self {
//         PixelFormat { chroma, bit_depth }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitdepth_derived_quantities() {
        assert_eq!(BitDepth::Eight.bits(), 8);
        assert_eq!(BitDepth::Ten.bits(), 10);
        assert_eq!(BitDepth::Twelve.bits(), 12);
        assert_eq!(BitDepth::Eight.minus8(), 0);
        assert_eq!(BitDepth::Ten.minus8(), 2);
        assert_eq!(BitDepth::Twelve.minus8(), 4);
        assert_eq!(BitDepth::Eight.max_val(), 255);
        assert_eq!(BitDepth::Ten.max_val(), 1023);
        assert_eq!(BitDepth::Twelve.max_val(), 4095);
        assert_eq!(BitDepth::Eight.neutral(), 128);
        assert_eq!(BitDepth::Ten.neutral(), 512);
        assert_eq!(BitDepth::Twelve.neutral(), 2048);
        assert_eq!(BitDepth::Eight.qp_bd_offset(), 0);
        assert_eq!(BitDepth::Ten.qp_bd_offset(), 12);
        assert_eq!(BitDepth::Twelve.qp_bd_offset(), 24);
    }

    #[test]
    fn bitdepth_from_bits_roundtrip() {
        assert_eq!(BitDepth::from_bits(8), BitDepth::Eight);
        assert_eq!(BitDepth::from_bits(10), BitDepth::Ten);
        assert_eq!(BitDepth::from_bits(12), BitDepth::Twelve);
    }

    #[test]
    #[should_panic]
    fn bitdepth_rejects_unsupported() {
        let _ = BitDepth::from_bits(16);
    }
}

#[cfg(test)]
mod plane_tests {
    use super::*;

    #[test]
    fn strided_rows_hide_padding_and_offset() {
        let plane = PlaneBuffer::from_parts(
            (0u8..20).collect(),
            PlaneLayout {
                width: 3,
                height: 2,
                stride: 5,
                offset: 6,
            },
        );
        let rows: Vec<&[u8]> = plane.rows().collect();
        assert_eq!(rows, vec![&[6, 7, 8][..], &[11, 12, 13][..]]);
        assert_eq!(plane.stride_bytes(), 5);
        assert_eq!(plane.offset_bytes(), 6);
        assert_eq!(plane.dimensions(), (3, 2));
        assert!(!plane.is_tightly_packed());
        assert_eq!(plane.data(), &[6, 7, 8, 9, 10, 11, 12, 13]);
        assert!(plane.as_tight().is_none());
    }

    #[test]
    fn mutable_data_starts_at_visible_origin() {
        let mut plane = PlaneBuffer::from_parts(
            (0u8..20).collect(),
            PlaneLayout {
                width: 3,
                height: 2,
                stride: 5,
                offset: 6,
            },
        );
        plane.data_mut()[0] = 42;
        plane.data_mut()[5] = 43;
        assert_eq!(plane.row(0), Some(&[42, 7, 8][..]));
        assert_eq!(plane.row(1), Some(&[43, 12, 13][..]));
        assert_eq!(plane.storage()[5], 5);
        assert_eq!(plane.storage()[14], 14);
    }

    #[test]
    fn tight_conversion_copies_only_visible_samples() {
        let plane = PlaneBuffer::from_parts(
            (0u16..20).collect(),
            PlaneLayout {
                width: 3,
                height: 2,
                stride: 5,
                offset: 6,
            },
        );
        assert_eq!(plane.into_tight().unwrap(), vec![6, 7, 8, 11, 12, 13]);
    }

    #[test]
    fn tight_plane_exposes_contiguous_slice() {
        let plane = PlaneBuffer::tight(vec![1u8, 2, 3, 4], 2, 2);
        assert!(plane.is_tightly_packed());
        assert_eq!(plane.data(), &[1, 2, 3, 4]);
        assert_eq!(plane.as_tight(), Some(&[1, 2, 3, 4][..]));
    }
}
