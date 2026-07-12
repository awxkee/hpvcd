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

use crate::color::{Cicp, MatrixCoefficients};
use crate::config::{Pps, Sps, parse_pps, parse_sps};
use crate::decode::{FullDecoder, SliceHeader, parse_slice_header_full};
use crate::demux::{Framing, Nal, detect_framing, for_each_nal, nal};
use crate::dpb::{Dpb, Frame};
use crate::error::DecodeError;
use crate::fmt::{BitDepth, ChromaFormat, ImageBuffer, SampleBuf};
use crate::inter::RefFramePlanes;
use crate::rps::derive_poc;
use crate::yuv::YuvPlanes;

/// Everything needed to turn coded planes into a displayable picture: the
/// conformance-window crop, the chroma format / bit depth, and the colour
/// coefficients. Filled in from the active SPS when a frame is emitted.
#[derive(Clone)]
pub(crate) struct FrameMeta {
    /// Conformance-window crop offsets in luma samples.
    crop_left: usize,
    crop_top: usize,
    /// Display (visible) luma dimensions after cropping.
    disp_w: usize,
    disp_h: usize,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    color: Cicp,
    color_description_present: bool,
    /// VUI frame-rate as time_scale / num_units_in_tick (0 = not signalled).
    time_scale: u32,
    num_units_in_tick: u32,
}

impl Default for FrameMeta {
    fn default() -> Self {
        FrameMeta {
            crop_left: 0,
            crop_top: 0,
            disp_w: 0,
            disp_h: 0,
            chroma: ChromaFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            color: Cicp::unspecified(),
            color_description_present: false,
            time_scale: 0,
            num_units_in_tick: 0,
        }
    }
}

/// One decoded, displayable video frame.
#[derive(Clone)]
pub struct VideoFrame {
    /// Coded planes (padded to the coding grid). Prefer the accessor methods —
    /// this is exposed only for zero-copy advanced use.
    pub planes: YuvPlanes,
    /// Picture order count (presentation order within a coded video sequence).
    pub poc: i32,
    meta: FrameMeta,
}

impl VideoFrame {
    /// Visible picture width in pixels (conformance-window cropped).
    pub fn width(&self) -> usize {
        self.meta.disp_w
    }
    /// Visible picture height in pixels (conformance-window cropped).
    pub fn height(&self) -> usize {
        self.meta.disp_h
    }
    /// A bit depth of the luma/chroma samples (8, 10, or 12).
    pub fn bit_depth(&self) -> u8 {
        self.meta.bit_depth.bits()
    }
    /// Chroma subsampling of the decoded planes.
    pub fn chroma_format(&self) -> ChromaFormat {
        self.meta.chroma
    }

    /// Frame rate in frames per second from the stream's VUI timing info.
    pub fn frame_rate(&self) -> Option<f64> {
        if self.meta.time_scale > 0 && self.meta.num_units_in_tick > 0 {
            Some(self.meta.time_scale as f64 / self.meta.num_units_in_tick as f64)
        } else {
            None
        }
    }

    /// Presentation timestamp in seconds, derived from the picture order count
    /// and the VUI frame rate.
    pub fn timestamp(&self) -> Option<f64> {
        self.frame_rate().map(|fps| self.poc as f64 / fps)
    }

    /// Colour metadata from the active SPS VUI. Unsignalled code points are
    /// returned as `Unspecified`, never fabricated as BT.709.
    pub fn color(&self) -> Cicp {
        self.meta.color
    }

    /// Complete CICP colour description, only when `colour_description_present_flag`
    /// was set in the active SPS VUI.
    pub fn signalled_color(&self) -> Option<Cicp> {
        self.meta
            .color_description_present
            .then_some(self.meta.color)
    }

    /// Convert to tightly-packed 8-bit interleaved RGB at the display size.
    pub fn to_rgb8(&self) -> Vec<u8> {
        self.to_rgb8_with_color(self.meta.color)
    }

    /// Convert to RGB using an explicit colour description instead of the SPS VUI.
    ///
    /// This is needed for elementary/conformance streams that carry GBR component
    /// planes but omit VUI colour signalling. For HEVC identity-matrix coding the
    /// stored component order is G, B, R.
    pub fn to_rgb8_with_color(&self, color: Cicp) -> Vec<u8> {
        let (dw, dh) = (self.meta.disp_w.max(1), self.meta.disp_h.max(1));
        let img = crate::yuv::yuv_to_rgb_window_with_color(
            &self.planes,
            dw,
            dh,
            self.meta.crop_left,
            self.meta.crop_top,
            &color,
        );
        match img {
            ImageBuffer::Rgb8(v) => v,
            ImageBuffer::Rgb16(v) => v.iter().map(|&s| (s >> self.shift()) as u8).collect(),
            ImageBuffer::Luma8(v) => v.iter().flat_map(|&g| [g, g, g]).collect(),
            ImageBuffer::Luma16(v) => v
                .iter()
                .flat_map(|&g| {
                    let g8 = (g >> self.shift()) as u8;
                    [g8, g8, g8]
                })
                .collect(),
        }
    }

    /// Interpret the decoded 4:4:4 component planes as full-range HEVC GBR.
    ///
    /// Use this for untagged HM/RExt RGB conformance streams. Tagged streams with
    /// `matrix_coefficients == 0` are handled automatically by [`Self::to_rgb8`].
    pub fn to_rgb8_gbr(&self) -> Vec<u8> {
        let mut color = self.meta.color;
        color.matrix = MatrixCoefficients::Identity;
        color.full_range = true;
        self.to_rgb8_with_color(color)
    }

    /// Convert to 8-bit interleaved RGBA (opaque alpha) at the display size.
    pub fn to_rgba8(&self) -> Vec<u8> {
        Self::rgb_to_rgba(self.to_rgb8())
    }

    /// Convert to RGBA using an explicit colour description instead of the SPS VUI.
    pub fn to_rgba8_with_color(&self, color: Cicp) -> Vec<u8> {
        Self::rgb_to_rgba(self.to_rgb8_with_color(color))
    }

    /// Interpret the decoded component planes as full-range HEVC GBR and emit RGBA.
    pub fn to_rgba8_gbr(&self) -> Vec<u8> {
        Self::rgb_to_rgba(self.to_rgb8_gbr())
    }

    fn rgb_to_rgba(rgb: Vec<u8>) -> Vec<u8> {
        let mut out = vec![0u8; (rgb.len() / 3) * 4];
        for (dst, px) in out
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(rgb.as_chunks::<3>().0.iter())
        {
            dst[0] = px[0];
            dst[1] = px[1];
            dst[2] = px[2];
            dst[3] = 255;
        }
        out
    }

    /// The visible YCbCr planes, cropped to the display window, with the sample
    /// type matching the bit depth (`U8` for 8-bit, `U16` for 10/12-bit).
    pub fn to_yuv(&self) -> FrameYuv {
        let (cl, ct) = (self.meta.crop_left, self.meta.crop_top);
        let (dw, dh) = (self.meta.disp_w.max(1), self.meta.disp_h.max(1));
        let (sub_w, sub_h) = (self.meta.chroma.sub_w(), self.meta.chroma.sub_h());
        let (coded_w, coded_h) = self.planes.dims();
        let (coded_cw, coded_ch) = self.planes.chroma_dims();
        let (cw, ch) = if self.meta.chroma.is_monochrome() {
            (0, 0)
        } else {
            (dw.div_ceil(sub_w), dh.div_ceil(sub_h))
        };
        let eight = self.meta.bit_depth == BitDepth::Eight;

        let y = crop_plane(&self.planes.y, coded_w, coded_h, cl, ct, dw, dh, eight);
        let ccl = cl / sub_w;
        let cct = ct / sub_h;
        let cb = crop_plane(&self.planes.cb, coded_cw, coded_ch, ccl, cct, cw, ch, eight);
        let cr = crop_plane(&self.planes.cr, coded_cw, coded_ch, ccl, cct, cw, ch, eight);

        FrameYuv {
            y,
            cb,
            cr,
            width: dw,
            height: dh,
            chroma_width: cw,
            chroma_height: ch,
            bit_depth: self.bit_depth(),
            chroma: self.meta.chroma,
        }
    }

    /// Right-shift to map the sample bit depth down to 8 bits.
    fn shift(&self) -> u16 {
        (self.meta.bit_depth.bits() as u16).saturating_sub(8)
    }

    // ---- Low-level accessors (coded/padded planes) --------------------------
    // These expose the raw coding-grid planes (padded up to a multiple of the
    // CTB size) without cropping or colour conversion. Prefer `to_rgb8` /
    // `to_yuv` for display; these exist for zero-copy and testing.

    /// Coded (padded) luma dimensions — usually larger than `width()`/`height()`.
    pub fn coded_dims(&self) -> (usize, usize) {
        self.planes.dims()
    }
    /// Coded chroma dimensions.
    pub fn coded_chroma_dims(&self) -> (usize, usize) {
        self.planes.chroma_dims()
    }
    /// Raw coded luma plane, truncated to 8-bit.
    pub fn y_u8(&self) -> Vec<u8> {
        self.planes.y_u8()
    }
    /// Raw coded Cb plane, truncated to 8-bit.
    pub fn cb_u8(&self) -> Vec<u8> {
        self.planes.cb_u8()
    }
    /// Raw coded Cr plane, truncated to 8-bit.
    pub fn cr_u8(&self) -> Vec<u8> {
        self.planes.cr_u8()
    }
    /// Deprecated alias for `coded_dims`; kept for source compatibility.
    pub fn dims(&self) -> (usize, usize) {
        self.planes.dims()
    }
    /// Deprecated alias for `coded_chroma_dims`.
    pub fn cb_dims(&self) -> (usize, usize) {
        self.planes.chroma_dims()
    }
}

/// Visible YCbCr planes (already cropped to the display window).
pub struct FrameYuv {
    pub y: SampleBuf,
    pub cb: SampleBuf,
    pub cr: SampleBuf,
    /// Luma width/height (pixels).
    pub width: usize,
    pub height: usize,
    /// Chroma plane width/height (pixels).
    pub chroma_width: usize,
    pub chroma_height: usize,
    pub bit_depth: u8,
    pub chroma: ChromaFormat,
}

/// Crop a coded plane to a display window, emitting `U8` or `U16` samples.
#[allow(clippy::too_many_arguments)]
fn crop_plane(
    src: &[u16],
    src_w: usize,
    src_h: usize,
    x0: usize,
    y0: usize,
    dw: usize,
    dh: usize,
    eight: bool,
) -> SampleBuf {
    if dw == 0 || dh == 0 {
        return if eight {
            SampleBuf::U8(Vec::new())
        } else {
            SampleBuf::U16(Vec::new())
        };
    }

    assert!(
        src_w != 0 && src_h != 0 && src.len() >= src_w * src_h,
        "invalid coded plane dimensions"
    );

    let x0 = x0.min(src_w - 1);
    let y0 = y0.min(src_h - 1);

    let copy_w = dw.min(src_w - x0);
    let copy_h = dh.min(src_h - y0);
    let right_pad = dw - copy_w;
    let len = dw
        .checked_mul(dh)
        .expect("cropped plane dimensions overflow");

    if eight {
        let mut out = Vec::with_capacity(len);

        for sy in y0..y0 + copy_h {
            let src_row = &src[sy * src_w + x0..sy * src_w + x0 + copy_w];

            out.extend(src_row.iter().map(|&sample| sample as u8));

            if right_pad != 0 {
                let edge = src_row[copy_w - 1] as u8;
                out.resize(out.len() + right_pad, edge);
            }
        }

        if copy_h < dh {
            let last_row = out.len() - dw;
            for _ in copy_h..dh {
                out.extend_from_within(last_row..last_row + dw);
            }
        }

        SampleBuf::U8(out)
    } else {
        let mut out = Vec::with_capacity(len);

        for sy in y0..y0 + copy_h {
            let src_row = &src[sy * src_w + x0..sy * src_w + x0 + copy_w];

            out.extend_from_slice(src_row);

            if right_pad != 0 {
                let edge = src_row[copy_w - 1];
                out.resize(out.len() + right_pad, edge);
            }
        }

        if copy_h < dh {
            let last_row = out.len() - dw;
            for _ in copy_h..dh {
                out.extend_from_within(last_row..last_row + dw);
            }
        }

        SampleBuf::U16(out)
    }
}

/// Decode a complete HEVC/H.265 elementary stream into displayable frames.
///
/// This is the one-liner entry point. Hand it the raw bytes of an `.hevc` /
/// `.265` file (Annex-B start codes) or a length-prefixed `hvcC` payload — the
/// framing is auto-detected — and get back the frames in presentation order.
/// Each [`VideoFrame`] can be turned into pixels with
/// [`VideoFrame::to_rgb8`] or [`VideoFrame::to_yuv`].
///
/// ```no_run
/// let data = std::fs::read("clip.hevc").unwrap();
/// for frame in hpvcd::decode_hevc(&data).unwrap() {
///     let rgb = frame.to_rgb8(); // width()*height()*3 bytes
///     // ... write out a PPM, upload a texture, etc.
/// }
/// ```
pub fn decode_hevc(data: &[u8]) -> Result<Vec<VideoFrame>, DecodeError> {
    VideoDecoder::new().decode_all(data)
}

/// Decode just the frame shown at `seconds`, seeking from the nearest preceding
/// random-access point. Convenience wrapper over [`VideoDecoder::decode_frame_at`]
/// for one-shot use. The frame rate comes from the stream's VUI timing; pass an
/// explicit rate with [`VideoDecoder::decode_frame_at_fps`] if the stream has none.
///
/// ```no_run
/// let data = std::fs::read("clip.hevc").unwrap();
/// if let Some(frame) = hpvcd::decode_hevc_frame_at(&data, 3.5).unwrap() {
///     let rgb = frame.to_rgb8();
/// }
/// ```
pub fn decode_hevc_frame_at(data: &[u8], seconds: f64) -> Result<Option<VideoFrame>, DecodeError> {
    VideoDecoder::new().decode_frame_at(data, seconds)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NalUnitType(u8);

impl NalUnitType {
    #[inline]
    const fn new(value: u8) -> Self {
        Self(value)
    }

    #[inline]
    const fn get(self) -> u8 {
        self.0
    }

    #[inline]
    fn is_vcl(self) -> bool {
        nal::is_vcl(self.0)
    }
}

#[derive(Clone, Debug)]
struct OwnedNalUnit {
    nal_type: NalUnitType,
    bytes: Vec<u8>,
}

impl OwnedNalUnit {
    #[inline]
    fn new(nal_type: u8, bytes: Vec<u8>) -> Self {
        Self {
            nal_type: NalUnitType::new(nal_type),
            bytes,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct AccessUnit {
    nals: Vec<OwnedNalUnit>,
}

impl AccessUnit {
    #[inline]
    fn push(&mut self, nal: OwnedNalUnit) {
        self.nals.push(nal);
    }

    #[inline]
    fn append(&mut self, other: &mut Self) {
        self.nals.append(&mut other.nals);
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.nals.is_empty()
    }

    #[inline]
    fn clear(&mut self) {
        self.nals.clear();
    }

    #[inline]
    fn len(&self) -> usize {
        self.nals.len()
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = &OwnedNalUnit> {
        self.nals.iter()
    }
}

#[derive(Debug)]
struct SliceSegmentNal {
    nal_type: NalUnitType,
    rbsp: Vec<u8>,
    source: Vec<u8>,
}

/// Stateful HEVC video decoder.
///
/// For most uses the free function [`decode_hevc`] (or [`VideoDecoder::decode`])
/// is all you need: it returns every frame, in order, ready to convert to RGB.
/// The struct form lets you reuse one decoder across calls or feed data
/// incrementally.
pub struct VideoDecoder {
    /// Parameter sets keyed by id. A stream may carry several distinct sets and
    /// switch between them per picture via slice_pic_parameter_set_id (§7.4.2.4).
    sps_map: std::collections::HashMap<u32, Sps>,
    pps_map: std::collections::HashMap<u32, Pps>,
    /// Most recently parsed SPS id, used to refresh display metadata.
    last_sps_id: Option<u32>,
    dpb: Dpb,
    /// POC of the previous picture in decode order with TemporalId 0 that is not
    /// a RASL/RADL/sub-layer-non-reference picture — the anchor for POC
    /// derivation (§8.3.1, `prevTid0Pic`).
    prev_tid0_poc: i32,
    /// True until the first coded picture of the stream is seen. The first IRAP
    /// (and any IRAP after an end-of-sequence) has NoRaslOutputFlag = 1.
    first_picture: bool,
    /// Set when the previous access unit was an end-of-sequence NAL, so the next
    /// IRAP is treated as a fresh random-access point.
    after_eos: bool,
    /// Frames ready for output, in decode order (caller sorts/consumes).
    outputs: Vec<VideoFrame>,
    /// First picture seen (POC anchor).
    seen_first: bool,
    /// Display metadata (crop/chroma/depth/color) from the active SPS.
    cur_meta: FrameMeta,
    /// Work-stealing pool for in-picture parallelism (WPP wavefront CABAC +
    /// reconstruction, and parallel deblock/SAO). A single-thread pool takes
    /// the serial path and is byte-identical.
    pool: crate::threadpool::ThreadPool,
    seek_cache: SeekCache,
    /// Accumulator for the incremental [`process_nal`] API: VCL NALs of the
    /// current access unit buffered until the next picture's first slice (or a
    /// flush) arrives, so multi-slice pictures decode as one unit.
    pending_au: AccessUnit,
}

/// Per-stream seek cache (see [`VideoDecoder::seek_cache`]).
#[derive(Default)]
struct SeekCache {
    key: Option<(usize, usize)>,
    /// The stream's access units, split once and reused across calls.
    aus: std::rc::Rc<Vec<AccessUnit>>,
    /// Index into `aus` of the random-access point the cached GOP starts at.
    gop_start: Option<usize>,
    /// Next access unit to decode when resuming the cached GOP forward.
    next_au: usize,
    /// Decoded frames of the cached GOP, keyed by POC.
    gop_frames: std::collections::HashMap<i32, VideoFrame>,
}

impl Default for VideoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoDecoder {
    /// Create a decoder with a pool sized to the machine's available
    /// parallelism. Use [`VideoDecoder::with_threads`] to pin the worker count.
    pub fn new() -> Self {
        VideoDecoder {
            sps_map: std::collections::HashMap::new(),
            pps_map: std::collections::HashMap::new(),
            last_sps_id: None,
            dpb: Dpb::new(16),
            prev_tid0_poc: 0,
            first_picture: true,
            after_eos: false,
            outputs: Vec::new(),
            seen_first: false,
            cur_meta: FrameMeta::default(),
            pool: crate::threadpool::ThreadPool::with_available_parallelism(),
            seek_cache: SeekCache::default(),
            pending_au: AccessUnit::default(),
        }
    }

    /// Create a decoder with a fixed number of worker threads. `1` forces the
    /// serial decode path (no pool dispatch); values are clamped to at least 1.
    /// Output is bit-identical regardless of thread count.
    pub fn with_threads(threads: usize) -> Self {
        VideoDecoder {
            sps_map: std::collections::HashMap::new(),
            pps_map: std::collections::HashMap::new(),
            last_sps_id: None,
            dpb: Dpb::new(16),
            prev_tid0_poc: 0,
            first_picture: true,
            after_eos: false,
            outputs: Vec::new(),
            seen_first: false,
            cur_meta: FrameMeta::default(),
            pool: crate::threadpool::ThreadPool::new(threads.max(1)),
            seek_cache: SeekCache::default(),
            pending_au: AccessUnit::default(),
        }
    }

    /// Number of worker threads in this decoder's pool.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Decode an entire elementary stream, returning frames in presentation
    /// (POC) order.
    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<VideoFrame>, DecodeError> {
        self.decode_all(data)
    }

    /// Decode an entire elementary stream, returning frames in output order.
    pub fn decode_all(&mut self, data: &[u8]) -> Result<Vec<VideoFrame>, DecodeError> {
        let framing = detect_framing(data);
        self.decode_stream(data, framing)?;
        // Flush remaining DPB.
        for f in self.dpb.bump(true) {
            self.outputs.push(VideoFrame {
                planes: f.planes,
                poc: f.poc,
                meta: f.meta.clone().unwrap_or_else(|| self.cur_meta.clone()),
            });
        }
        // Pictures are already emitted in output order: the DPB bumps the
        // lowest-POC pending picture per §C.5.2.2, and each IRAP that starts a
        // new coded video sequence flushes the DPB first, so cross-CVS order is
        // preserved. A global POC sort would be wrong here — POC restarts at
        // each CVS, so it could interleave pictures from different sequences.
        Ok(std::mem::take(&mut self.outputs))
    }

    /// Decode and return the single frame that should be shown at `seconds`.
    ///
    /// This is the random-access ("seek") entry point, like dav1d's picture
    /// pull after a seek: it finds the target picture from the frame rate, then
    /// decodes forward from the nearest preceding random-access point (IDR/IRAP)
    /// so all reference pictures the target depends on are available. The frame
    /// rate is taken from the stream's VUI timing; if the stream carries none,
    /// use [`VideoDecoder::decode_frame_at_fps`] and supply it yourself.
    ///
    /// Returns `Ok(None)` if `seconds` falls past the end of the stream.
    pub fn decode_frame_at(
        &mut self,
        data: &[u8],
        seconds: f64,
    ) -> Result<Option<VideoFrame>, DecodeError> {
        let fps = self.stream_frame_rate(data)?.ok_or_else(|| {
            DecodeError::Bitstream(
                "stream has no VUI frame rate; use decode_frame_at_fps".to_string(),
            )
        })?;
        self.decode_frame_at_fps(data, seconds, fps)
    }

    /// As [`decode_frame_at`], but with a caller-supplied frame rate (fps) for
    /// streams that don't signal timing in-band (the container has it instead).
    pub fn decode_frame_at_fps(
        &mut self,
        data: &[u8],
        seconds: f64,
        fps: f64,
    ) -> Result<Option<VideoFrame>, DecodeError> {
        if fps <= 0.0 || seconds < 0.0 {
            return Ok(None);
        }
        let target_poc = (seconds * fps).floor() as i32;
        self.decode_frame_index(data, target_poc)
    }

    /// Decode and return the frame with the given presentation index (POC),
    /// i.e. the `target_poc`-th picture in display order. Decodes forward from
    /// the nearest preceding random-access point. `Ok(None)` if past the end.
    pub fn decode_frame_index(
        &mut self,
        data: &[u8],
        target_poc: i32,
    ) -> Result<Option<VideoFrame>, DecodeError> {
        let key = (data.as_ptr() as usize, data.len());
        if self.seek_cache.key != Some(key) {
            self.seek_cache = SeekCache {
                key: Some(key),
                ..SeekCache::default()
            };
        }

        // Access-unit split: parse once per buffer, reuse afterward.
        if self.seek_cache.aus.is_empty() {
            let framing = detect_framing(data);
            let aus = self.collect_access_units(data, framing)?;
            self.seek_cache.aus = std::rc::Rc::new(aus);
        }
        let aus = std::rc::Rc::clone(&self.seek_cache.aus);
        if aus.is_empty() {
            return Ok(None);
        }

        let start = self.irap_at_or_before_target(&aus, target_poc);

        // Fast path: the target's GOP is (partly) decoded and the frame is
        // already cached — scrubbing within a GOP costs nothing after the first
        // frame that reaches this far.
        if self.seek_cache.gop_start == Some(start)
            && let Some(f) = self.seek_cache.gop_frames.get(&target_poc)
        {
            return Ok(Some(f.clone()));
        }

        // If we're entering a *different* GOP than the cached one, restart the
        // decode state at its random-access point and drop the old GOP's frames.
        // If it's the *same* GOP, keep the decoder state so we resume forward
        // from where the previous call stopped rather than re-decoding from the
        // IRAP.
        let resuming = self.seek_cache.gop_start == Some(start);
        let first_au = if resuming {
            self.seek_cache.next_au
        } else {
            self.reset_decode_state();
            self.seek_cache.gop_start = Some(start);
            self.seek_cache.gop_frames.clear();
            start
        };

        for (i, au) in aus.iter().enumerate().skip(first_au) {
            self.decode_access_unit(au)?;
            self.seek_cache.next_au = i + 1;
            // Cache every freshly available frame (from `outputs` or the DPB),
            // so later calls into this GOP are served from the map.
            for f in self.outputs.drain(..) {
                self.seek_cache.gop_frames.entry(f.poc).or_insert(f);
            }
            for f in &self.dpb.frames {
                self.seek_cache
                    .gop_frames
                    .entry(f.poc)
                    .or_insert_with(|| VideoFrame {
                        planes: f.planes.clone(),
                        poc: f.poc,
                        meta: f.meta.clone().unwrap_or_else(|| self.cur_meta.clone()),
                    });
            }
            if let Some(f) = self.seek_cache.gop_frames.get(&target_poc) {
                return Ok(Some(f.clone()));
            }
        }

        // Reached the end of the GOP (or stream) without the target: flush the
        // reorder buffer and cache whatever comes out, then look again.
        for f in self.dpb.bump(true) {
            self.seek_cache
                .gop_frames
                .entry(f.poc)
                .or_insert(VideoFrame {
                    planes: f.planes,
                    poc: f.poc,
                    meta: f.meta.clone().unwrap_or_else(|| self.cur_meta.clone()),
                });
        }
        Ok(self.seek_cache.gop_frames.get(&target_poc).cloned())
    }

    /// The stream's VUI frame rate, if it signals one. Parses only far enough to
    /// read the first SPS.
    pub fn stream_frame_rate(&self, data: &[u8]) -> Result<Option<f64>, DecodeError> {
        let framing = detect_framing(data);
        let mut rate = None;
        for_each_nal(data, framing, |n: Nal| {
            if n.nal_type == nal::SPS && rate.is_none() {
                let rbsp = crate::bitreader::unescape_rbsp(&n.bytes[2..]);
                if let Ok(sps) = parse_sps(&rbsp)
                    && sps.vui_time_scale > 0
                    && sps.vui_num_units_in_tick > 0
                {
                    rate = Some(sps.vui_time_scale as f64 / sps.vui_num_units_in_tick as f64);
                }
            }
            Ok(())
        })?;
        Ok(rate)
    }

    /// Reset mutable decode state so a fresh (re)decode can start at an IRAP.
    fn reset_decode_state(&mut self) {
        self.dpb = Dpb::new(16);
        self.prev_tid0_poc = 0;
        self.first_picture = true;
        self.after_eos = false;
        self.outputs.clear();
        self.seen_first = false;
    }

    /// Group the stream into access units, prepending the parameter-set NALs
    /// that configure each AU so a decoded can start at any returned index.
    fn collect_access_units(
        &mut self,
        data: &[u8],
        framing: Framing,
    ) -> Result<Vec<AccessUnit>, DecodeError> {
        let mut nals = Vec::new();
        for_each_nal(data, framing, |n: Nal| {
            nals.push(OwnedNalUnit::new(n.nal_type, n.bytes.to_vec()));
            Ok(())
        })?;
        let mut aus = Vec::new();
        let mut pending_non_vcl = AccessUnit::default();
        let mut cur = AccessUnit::default();
        for nal_unit in nals {
            let nal_type = nal_unit.nal_type.get();
            if nal_unit.nal_type.is_vcl() {
                if self.is_first_slice(nal_type, &nal_unit.bytes) && !cur.is_empty() {
                    aus.push(std::mem::take(&mut cur));
                }
                if cur.is_empty() {
                    cur.append(&mut pending_non_vcl);
                }
                cur.push(nal_unit);
            } else {
                // Apply params now (so slice-header parsing works) and remember
                // them to prepend to the next AU for independent decodability.
                self.process_non_vcl(nal_type, &nal_unit.bytes)?;
                pending_non_vcl.push(nal_unit);
            }
        }
        if !cur.is_empty() {
            aus.push(cur);
        }
        Ok(aus)
    }

    /// Index of the access unit to start decoding from so that the target POC's
    /// references are all available: the last random-access point (IDR/IRAP)
    /// whose POC is <= the target. Derives POC cheaply by parsing slice headers
    /// only (no pixel reconstruction).
    fn irap_at_or_before_target(&self, aus: &[AccessUnit], target_poc: i32) -> usize {
        if self.sps_map.is_empty() || self.pps_map.is_empty() {
            return 0;
        }
        let mut prev_tid0 = 0i32;
        let mut first_pic = true;
        let mut best = 0usize;
        let mut have_best = false;
        for (i, au) in aus.iter().enumerate() {
            // The AU's first VCL NAL carries the picture's POC LSB.
            let Some(nal_unit) = au.iter().find(|nal_unit| nal_unit.nal_type.is_vcl()) else {
                continue;
            };
            let nt = nal_unit.nal_type.get();
            let bytes = &nal_unit.bytes;
            let rbsp = crate::bitreader::unescape_rbsp(bytes);
            // Resolve the parameter-set chain this slice activates.
            let Some((sps, pps)) = crate::decode::peek_slice_pps_id(&rbsp, nt)
                .and_then(|pid| self.pps_map.get(&pid))
                .and_then(|pps| self.sps_map.get(&pps.sps_id).map(|sps| (sps, pps)))
            else {
                continue;
            };
            let max_poc_lsb = 1i32 << sps.log2_max_poc_lsb;
            let tid = bytes
                .get(1)
                .map(|b| (b & 0x07).wrapping_sub(1))
                .unwrap_or(0);
            // NoRaslOutputFlag: IDR/BLA always; CRA only as the first picture.
            let no_rasl = nal::is_idr(nt) || nal::is_bla(nt) || (nal::is_cra(nt) && first_pic);
            let poc = match parse_slice_header_full(&rbsp, sps, pps, nt) {
                Ok(h) => {
                    if nal::is_irap(nt) && no_rasl {
                        if nal::is_idr(nt) { 0 } else { h.poc_lsb }
                    } else {
                        derive_poc(h.poc_lsb, prev_tid0, max_poc_lsb, false)
                    }
                }
                Err(_) => prev_tid0,
            };
            if tid == 0 && !nal::is_rasl(nt) && !nal::is_radl(nt) && !nal::is_sub_layer_non_ref(nt)
            {
                prev_tid0 = poc;
            }
            first_pic = false;
            if nal::is_irap(nt) {
                if poc <= target_poc {
                    best = i;
                    have_best = true;
                } else if have_best {
                    // A later IRAP already sits past the target. POC only rises
                    // within a coded video sequence, so no further IRAP can be a
                    // closer preceding random-access point — stop scanning
                    // instead of parsing every remaining slice header. (An IDR
                    // resets POC to 0, which is <= any non-negative target and so
                    // never reaches this branch.)
                    break;
                }
            }
        }
        best
    }

    fn decode_stream(&mut self, data: &[u8], framing: Framing) -> Result<(), DecodeError> {
        // Collect NALs first (borrow-friendly), then process.
        let mut nals = Vec::new();
        for_each_nal(data, framing, |n: Nal| {
            nals.push(OwnedNalUnit::new(n.nal_type, n.bytes.to_vec()));
            Ok(())
        })?;
        // Group VCL NALs into access units. A new picture starts at a VCL NAL
        // whose slice header has first_slice_segment_in_pic_flag = 1; all later
        // VCL NALs (dependent or independent segments) belong to the same
        // picture until the next such flag. Non-VCL NALs (SPS/PPS/VPS/SEI) are
        // applied immediately and flush any pending access unit first.
        let mut au = AccessUnit::default();
        for nal_unit in nals {
            let nal_type = nal_unit.nal_type.get();
            if nal_unit.nal_type.is_vcl() {
                if self.is_first_slice(nal_type, &nal_unit.bytes) && !au.is_empty() {
                    self.decode_access_unit(&au)?;
                    au.clear();
                }
                au.push(nal_unit);
            } else {
                if !au.is_empty() {
                    self.decode_access_unit(&au)?;
                    au.clear();
                }
                self.process_non_vcl(nal_type, &nal_unit.bytes)?;
            }
        }
        if !au.is_empty() {
            self.decode_access_unit(&au)?;
        }
        Ok(())
    }

    /// Peek whether a VCL NAL begins a new picture (first_slice_in_pic). This is
    /// the first bit after the 2-byte NAL header, so no parameter set is needed.
    fn is_first_slice(&self, _nal_type: u8, bytes: &[u8]) -> bool {
        let rbsp = crate::bitreader::unescape_rbsp(bytes);
        let mut r = crate::bitreader::BitReader::new(&rbsp);
        if r.read_bits(16).is_err() {
            return true;
        }
        r.read_flag().unwrap_or(true)
    }

    /// Resolve the SPS/PPS chain a slice activates. Peeks the slice's
    /// pic_parameter_set_id, finds that PPS, then the SPS it references. Also
    /// refreshes display metadata from the activated SPS. Returns clones so the
    /// picture decode owns stable copies even if new sets arrive mid-stream.
    fn activate_params(&mut self, rbsp: &[u8], nal_type: u8) -> Option<(Sps, Pps)> {
        let pps_id = crate::decode::peek_slice_pps_id(rbsp, nal_type)?;
        let pps = self.pps_map.get(&pps_id)?.clone();
        let sps = self.sps_map.get(&pps.sps_id)?.clone();
        if self.last_sps_id != Some(sps.id) {
            self.cur_meta = frame_meta_from_sps(&sps);
            self.last_sps_id = Some(sps.id);
        }
        Some((sps, pps))
    }

    fn process_non_vcl(&mut self, nal_type: u8, bytes: &[u8]) -> Result<(), DecodeError> {
        match nal_type {
            nal::SPS => {
                let rbsp = crate::bitreader::unescape_rbsp(&bytes[2..]);
                if let Ok(sps) = parse_sps(&rbsp) {
                    self.cur_meta = frame_meta_from_sps(&sps);
                    self.last_sps_id = Some(sps.id);
                    self.sps_map.insert(sps.id, sps);
                }
                Ok(())
            }
            nal::PPS => {
                let rbsp = crate::bitreader::unescape_rbsp(&bytes[2..]);
                if let Ok(pps) = parse_pps(&rbsp, false) {
                    self.pps_map.insert(pps.id, pps);
                }
                Ok(())
            }
            _ => {
                // End-of-sequence (36) / end-of-bitstream (37): the next IRAP
                // begins a fresh random-access period (NoRaslOutputFlag = 1).
                if nal_type == 36 || nal_type == 37 {
                    self.after_eos = true;
                }
                Ok(())
            } // VPS/SEI/AUD/etc.
        }
    }

    pub fn process_nal(&mut self, nal_type: u8, bytes: &[u8]) -> Result<(), DecodeError> {
        if nal::is_vcl(nal_type) {
            // A VCL NAL whose first_slice_segment_in_pic_flag is set begins a new
            // picture: flush the buffered slices of the previous picture first,
            // then start accumulating this one. Later (non-first) slices append
            // to the current picture so multi-slice pictures decode as a unit.
            if self.is_first_slice(nal_type, bytes) && !self.pending_au.is_empty() {
                self.flush_pending_au()?;
            }
            self.pending_au
                .push(OwnedNalUnit::new(nal_type, bytes.to_vec()));
            return Ok(());
        }
        // A non-VCL NAL (parameter set, SEI, EOS/EOB) terminates the current
        // access unit. Flush any buffered picture, then process the NAL so a
        // parameter set that follows a picture is not applied to it.
        if !self.pending_au.is_empty() {
            self.flush_pending_au()?;
        }
        self.process_non_vcl(nal_type, bytes)
    }

    /// Decode and clear the buffered access unit. Callers using the incremental
    /// [`process_nal`] API must call [`flush`](Self::flush) after the last NAL to
    /// emit the final picture.
    fn flush_pending_au(&mut self) -> Result<(), DecodeError> {
        if self.pending_au.is_empty() {
            return Ok(());
        }
        let au = std::mem::take(&mut self.pending_au);
        self.decode_access_unit(&au)
    }

    /// Flush the final buffered picture. Call once after the last
    /// [`process_nal`] when using the incremental API.
    pub fn flush(&mut self) -> Result<(), DecodeError> {
        self.flush_pending_au()
    }

    /// Take the frames decoded so far by the incremental [`process_nal`] API and
    /// (optionally) drain the DPB. Pass `drain_dpb = true` after the final
    /// [`flush`] to emit reordered pictures still held for output; pass `false`
    /// mid-stream to pull only pictures already released by the DPB. Returned
    /// frames are sorted by POC within what has been emitted; the caller is
    /// responsible for not mixing pictures across coded-video-sequence resets.
    pub fn take_frames(&mut self, drain_dpb: bool) -> Vec<VideoFrame> {
        if drain_dpb {
            for f in self.dpb.bump(true) {
                self.outputs.push(VideoFrame {
                    planes: f.planes,
                    poc: f.poc,
                    meta: f.meta.clone().unwrap_or_else(|| self.cur_meta.clone()),
                });
            }
        }
        let mut out = std::mem::take(&mut self.outputs);
        out.sort_by_key(|f| f.poc);
        out
    }

    fn decode_access_unit(&mut self, au: &AccessUnit) -> Result<(), DecodeError> {
        // Apply any parameter-set NALs carried at the head of this access unit
        // (they may precede the first VCL slice, e.g. before an IRAP), and keep
        // only the VCL slice segments for the picture decode. Pre-unescape each
        // slice's RBSP so the borrowed CABAC payloads live for the whole
        // picture decode; the original NAL bytes are kept for WPP/tile entry
        // point mapping.
        let mut segs = Vec::with_capacity(au.len());
        for nal_unit in au.iter() {
            let nal_type = nal_unit.nal_type.get();
            if !nal_unit.nal_type.is_vcl() {
                self.process_non_vcl(nal_type, &nal_unit.bytes)?;
                continue;
            }
            segs.push(SliceSegmentNal {
                nal_type: nal_unit.nal_type,
                rbsp: crate::bitreader::unescape_rbsp(&nal_unit.bytes),
                source: nal_unit.bytes.clone(),
            });
        }
        if segs.is_empty() {
            return Ok(());
        }
        // Activate the parameter-set chain referenced by this picture's first
        // slice: slice_pic_parameter_set_id -> PPS -> its seq_parameter_set_id
        // -> SPS (§7.4.2.4). This lets a stream switch PPS/SPS per picture.
        let (sps, pps) = match self.activate_params(&segs[0].rbsp, segs[0].nal_type.get()) {
            Some(pair) => pair,
            None => return Ok(()), // no matching parameter sets yet
        };
        // Size the output DPB from the active SPS (§C.5.2.2) so reorder latency
        // and buffering follow the stream rather than a fixed capacity.
        self.dpb.configure(
            sps.max_dec_pic_buffering as usize,
            sps.max_num_reorder_pics as usize,
            sps.max_latency_pictures.map(|v| v as usize),
        );

        // Parse the first segment's header to set up picture-level state.
        let first = &segs[0];
        let first_nal = first.nal_type.get();
        let first_rbsp = &first.rbsp;
        let first_bytes = &first.source;
        let hdr0 = parse_slice_header_full(first_rbsp, &sps, &pps, first_nal)?;
        if !hdr0.first_slice_in_pic {
            return Ok(());
        }

        // --- Picture lifecycle (§8.3.1, Annex C) --------------------------------
        let is_irap = nal::is_irap(first_nal);
        let is_idr = nal::is_idr(first_nal);
        let is_bla = nal::is_bla(first_nal);
        let is_cra = nal::is_cra(first_nal);
        // TemporalId of this picture (nuh_temporal_id_plus1 - 1).
        let temporal_id = first_bytes
            .get(1)
            .map(|b| (b & 0x07).wrapping_sub(1))
            .unwrap_or(0);
        // NoRaslOutputFlag: the leading (RASL) pictures of this IRAP cannot be
        // correctly decoded, so they are discarded. It is 1 for IDR/BLA, and for
        // a CRA only when it is the first picture of the stream or follows an
        // end-of-sequence.
        let no_rasl_output = if is_idr || is_bla {
            true
        } else if is_cra {
            self.first_picture || self.after_eos
        } else {
            false
        };

        // Skip RASL pictures attached to an IRAP with NoRaslOutputFlag == 1:
        // their references precede the IRAP and are unavailable (random access).
        if nal::is_rasl(first_nal) && self.dpb.pending_no_rasl_output() {
            return Ok(());
        }

        let max_poc_lsb = 1i32 << sps.log2_max_poc_lsb;
        // POC (§8.3.1): reset to the LSB at an IRAP with NoRaslOutputFlag == 1
        // (IDR forces 0); otherwise derive the MSB from the previous Tid0 anchor.
        let poc = if is_irap && no_rasl_output {
            if is_idr { 0 } else { hdr0.poc_lsb }
        } else {
            derive_poc(hdr0.poc_lsb, self.prev_tid0_poc, max_poc_lsb, false)
        };

        // At an IRAP that starts a new random-access period (IDR/BLA, or a CRA
        // with NoRaslOutputFlag == 1), references from the previous period are no
        // longer usable. The NoRaslOutputFlag is recorded for *every* IRAP so a
        // mid-stream CRA (NoRaslOutputFlag == 0) does not inherit the previous
        // period's suppression and its RASL pictures are still output.
        if is_irap {
            // §C.5.2.2: at an IRAP that starts a new CVS (NoRaslOutputFlag == 1),
            // pictures still pending output are either discarded (when
            // NoOutputOfPriorPicsFlag is 1 — always inferred so for a CRA that
            // begins a random-access decode) or emitted first. For IDR/BLA the
            // parsed flag applies directly.
            if no_rasl_output {
                for f in self.dpb.bump_before_irap() {
                    self.outputs.push(VideoFrame {
                        planes: f.planes,
                        poc: f.poc,
                        meta: f.meta.clone().unwrap_or_else(|| self.cur_meta.clone()),
                    });
                }
                let drop_prior = if is_cra {
                    true
                } else {
                    hdr0.no_output_of_prior_pics
                };
                if drop_prior {
                    self.dpb.discard_pending_output();
                } else {
                    for f in self.dpb.bump(true) {
                        self.outputs.push(VideoFrame {
                            planes: f.planes,
                            poc: f.poc,
                            meta: f.meta.clone().unwrap_or_else(|| self.cur_meta.clone()),
                        });
                    }
                }
            }
            if (is_idr || is_bla) || (is_cra && no_rasl_output) {
                self.dpb.clear_refs();
            }
            self.dpb.set_no_rasl_output(no_rasl_output);
        }
        self.after_eos = false;

        // Apply the RPS reference marking for every non-IDR picture — including
        // I slices (§8.3.2). A non-IDR I picture still carries an RPS that can
        // mark older DPB pictures as unused for reference; skipping it would
        // leave stale references in the DPB. IDR pictures have no RPS (the DPB
        // was already cleared above).
        let rps_pocs = if !is_idr {
            Some(
                self.dpb
                    .apply_rps_lt(poc, &hdr0.cur_rps, &hdr0.lt_refs, max_poc_lsb),
            )
        } else {
            None
        };

        // §C.5.2.2 performs bumping before decoding the current picture. Doing
        // it after insertion can output an IRAP before lower-POC RASL/RADL
        // pictures that follow it in decoding order.
        if !is_irap || !no_rasl_output {
            for f in self.dpb.bump_before_decode() {
                self.outputs.push(VideoFrame {
                    planes: f.planes,
                    poc: f.poc,
                    meta: f.meta.clone().unwrap_or_else(|| self.cur_meta.clone()),
                });
            }
        }

        // Reference lists (built once per picture from the first slice's RPS).
        // With curr_pic_ref (IBC) the current picture is appended to L0 as a
        // reference even on I slice, so those slices also build a list.
        let ibc_active = sps.curr_pic_ref_enabled && pps.curr_pic_ref_enabled;
        let (ref_list0, ref_list1, ref_frames) =
            if hdr0.slice_type != crate::inter::SLICE_I || ibc_active {
                let pocs = rps_pocs.clone().unwrap_or_else(|| {
                    self.dpb
                        .apply_rps_lt(poc, &hdr0.cur_rps, &hdr0.lt_refs, max_poc_lsb)
                });
                let is_b = hdr0.slice_type == crate::inter::SLICE_B;
                let current = ibc_active.then_some(crate::dpb::RefEntry {
                    _dpb_index: usize::MAX,
                    poc,
                    // The current decoded picture is marked long-term while it
                    // is used for current-picture referencing (§8.1.3/C.3.4).
                    long_term: true,
                });
                let (l0, l1) = self.dpb.build_ref_lists(
                    &pocs,
                    hdr0.num_ref_idx_l0,
                    hdr0.num_ref_idx_l1,
                    is_b,
                    current,
                    &hdr0.list_mod_l0,
                    &hdr0.list_mod_l1,
                )?;
                let frames = self.collect_ref_frames(&l0, &l1);
                (l0, l1, frames)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

        // Create the decoder on the first segment and decode it.
        let cabac0 = &first_rbsp[hdr0.cabac_offset.min(first_rbsp.len())..];
        let mut d = FullDecoder::new(cabac0, sps.clone(), pps.clone(), &hdr0)?;
        d.set_inter_state(
            poc,
            ref_list0.clone(),
            ref_list1.clone(),
            ref_frames.clone(),
        );
        let sub0: Vec<usize> = if hdr0.entry_points.is_empty() {
            Vec::new()
        } else {
            let src_of = crate::bitreader::rbsp_src_map(first_bytes);
            crate::wpp::substream_starts_rbsp_rel(
                &src_of,
                hdr0.cabac_offset,
                &hdr0.entry_points,
                first_rbsp.len(),
            )
        };
        let starts0 = if sub0.is_empty() {
            None
        } else {
            Some((cabac0, sub0.as_slice()))
        };
        // Try the parallel WPP wavefront first; it decodes the whole picture
        // when eligible. It threads intra decode; P/B slices decode their CABAC
        // serially but still get parallel deblock/SAO from `finish` below. (The
        // RowFactory carries interstate so inter-WPP can be enabled once its
        // cross-row motion hand-off is validated.)
        let ran_wavefront = if segs.len() == 1 && hdr0.slice_type == crate::inter::SLICE_I {
            d.try_decode_wavefront(first_rbsp, first_bytes, &hdr0, &self.pool)?
        } else {
            false
        };
        if !ran_wavefront {
            d.decode_slice_ctx(hdr0.slice_segment_address, starts0)?;
        }

        // Track the most recent *independent* segment header. A dependent slice
        // segment carries no reconstruction state of its own — it inherits the
        // preceding independent segment's slice type, weighted-prediction table,
        // reference lists, temporal-MVP / collocated settings, chroma QP
        // offsets, SAO and deblock parameters (§7.4.7.1). Merge those in before
        // decoding so a dependent P/B segment is not silently treated as an
        // I segment with no weights.
        let mut last_indep = hdr0.clone();
        // Decode any remaining segments of the same picture into the same
        // decoder (accumulating the reconstructed planes and motion field).
        for segment in segs.iter().skip(1) {
            let seg_rbsp = &segment.rbsp;
            let seg_bytes = &segment.source;
            let mut hdr = parse_slice_header_full(seg_rbsp, &sps, &pps, segment.nal_type.get())?;
            if hdr.first_slice_in_pic {
                // Shouldn't happen (grouping guarantees it), but be safe.
                continue;
            }
            if hdr.dependent_slice_segment {
                merge_dependent_header(&mut hdr, &last_indep);
            } else {
                last_indep = hdr.clone();
            }
            // Rebuild ref lists for independent segments (dependent segments
            // inherit them); reuse the picture's reference frame views.
            if !hdr.dependent_slice_segment
                && (hdr.slice_type != crate::inter::SLICE_I || ibc_active)
            {
                let max_poc_lsb = 1i32 << sps.log2_max_poc_lsb;
                let pocs = self
                    .dpb
                    .apply_rps_lt(poc, &hdr.cur_rps, &hdr.lt_refs, max_poc_lsb);
                let is_b = hdr.slice_type == crate::inter::SLICE_B;
                let current = ibc_active.then_some(crate::dpb::RefEntry {
                    _dpb_index: usize::MAX,
                    poc,
                    long_term: true,
                });
                let (l0, l1) = self.dpb.build_ref_lists(
                    &pocs,
                    hdr.num_ref_idx_l0,
                    hdr.num_ref_idx_l1,
                    is_b,
                    current,
                    &hdr.list_mod_l0,
                    &hdr.list_mod_l1,
                )?;
                d.set_inter_state(
                    poc,
                    l0.clone(),
                    l1.clone(),
                    self.collect_ref_frames(&l0, &l1),
                );
            }
            let sub: Vec<usize> = if hdr.entry_points.is_empty() {
                Vec::new()
            } else {
                let src_of = crate::bitreader::rbsp_src_map(seg_bytes);
                crate::wpp::substream_starts_rbsp_rel(
                    &src_of,
                    hdr.cabac_offset,
                    &hdr.entry_points,
                    seg_rbsp.len(),
                )
            };
            let cabac = &seg_rbsp[hdr.cabac_offset.min(seg_rbsp.len())..];
            d.decode_segment(cabac, &hdr, &sub)?;
        }

        // Deblock/SAO once the whole picture is reconstructed, then store it.
        // Deblock runs serially (its parallel chroma kernel is not yet bit-exact
        // vs the serial reference); SAO runs on the pool.
        let planes = d.finish_with(None, Some(&self.pool))?;
        let (motion, width4, height4) = d.take_motion();

        // §8.3.2: every decoded picture is marked "used for short-term reference"
        // when decoded, regardless of NAL type — a sub-layer non-reference (`_N`)
        // picture may still be referenced by a higher temporal sub-layer. The
        // next picture's RPS performs the subsequent short/long-term marking.
        let frame = Frame {
            planes,
            poc,
            motion,
            width4,
            height4,
            short_term: true,
            long_term: false,
            needed_for_output: hdr0.pic_output_flag,
            latency: 0,
            meta: Some(self.cur_meta.clone()),
        };
        self.dpb.push(frame);
        // Update prevTid0Pic (§8.3.1): only pictures with TemporalId 0 that are
        // not RASL/RADL/sub-layer-non-reference serve as the POC anchor.
        if temporal_id == 0
            && !nal::is_rasl(first_nal)
            && !nal::is_radl(first_nal)
            && !nal::is_sub_layer_non_ref(first_nal)
        {
            self.prev_tid0_poc = poc;
        }
        self.first_picture = false;
        self.seen_first = true;

        Ok(())
    }

    /// Build owned reference-frame views for MC from the DPB entries in the
    /// current slice's reference lists.
    fn collect_ref_frames(
        &self,
        l0: &[crate::dpb::RefEntry],
        l1: &[crate::dpb::RefEntry],
    ) -> Vec<RefFramePlanes> {
        let mut out: Vec<RefFramePlanes> = Vec::new();
        let mut seen = Vec::new();
        for e in l0.iter().chain(l1.iter()) {
            if seen.contains(&e.poc) {
                continue;
            }
            seen.push(e.poc);
            if let Some(f) = self.dpb.frames.iter().find(|f| f.poc == e.poc) {
                let (cw, ch) = chroma_dims(&f.planes);
                out.push(RefFramePlanes {
                    poc: f.poc,
                    y: f.planes.y.clone(),
                    cb: f.planes.cb.clone(),
                    cr: f.planes.cr.clone(),
                    w: f.planes.width,
                    h: f.planes.height,
                    cw,
                    ch,
                    motion: f.motion.clone(),
                    width4: f.width4,
                    height4: f.height4,
                });
            }
        }
        out
    }
}

/// Chroma plane (width, height) for a stored picture, from its chroma format.
/// Fill a dependent slice segment's header with the reconstruction state it
/// inherits from the preceding independent segment (§7.4.7.1). The parser leaves
/// these as placeholders for a dependent segment because they are not coded in
/// its header; without this a dependent P/B segment would decode as an I segment
/// with no weighted prediction and no temporal-MVP / collocated state.
fn merge_dependent_header(dep: &mut SliceHeader, indep: &SliceHeader) {
    dep.slice_type = indep.slice_type;
    dep.slice_qp = indep.slice_qp;
    dep.sao_luma = indep.sao_luma;
    dep.sao_chroma = indep.sao_chroma;
    dep.cb_qp_offset = indep.cb_qp_offset;
    dep.cr_qp_offset = indep.cr_qp_offset;
    dep.cu_chroma_qp_offset_enabled = indep.cu_chroma_qp_offset_enabled;
    dep.act_y_qp_offset = indep.act_y_qp_offset;
    dep.act_cb_qp_offset = indep.act_cb_qp_offset;
    dep.act_cr_qp_offset = indep.act_cr_qp_offset;
    dep.deblocking_disabled = indep.deblocking_disabled;
    dep.beta_offset_div2 = indep.beta_offset_div2;
    dep.tc_offset_div2 = indep.tc_offset_div2;
    dep.cur_rps = indep.cur_rps.clone();
    dep.lt_refs = indep.lt_refs.clone();
    dep.num_ref_idx_l0 = indep.num_ref_idx_l0;
    dep.num_ref_idx_l1 = indep.num_ref_idx_l1;
    dep.list_mod_l0 = indep.list_mod_l0.clone();
    dep.list_mod_l1 = indep.list_mod_l1.clone();
    dep.mvd_l1_zero = indep.mvd_l1_zero;
    dep.cabac_init = indep.cabac_init;
    dep.temporal_mvp = indep.temporal_mvp;
    dep.collocated_from_l0 = indep.collocated_from_l0;
    dep.collocated_ref_idx = indep.collocated_ref_idx;
    dep.max_num_merge_cand = indep.max_num_merge_cand;
    dep.use_integer_mv = indep.use_integer_mv;
    dep.pred_weights = indep.pred_weights.clone();
    dep.slice_loop_filter_across_slices = indep.slice_loop_filter_across_slices;
    dep.pic_output_flag = indep.pic_output_flag;
    // slice_segment_address, cabac_offset, dependent_slice_segment,
    // first_slice_in_pic and entry_points remain the dependent segment's own.
}

fn chroma_dims(planes: &YuvPlanes) -> (usize, usize) {
    use crate::fmt::ChromaFormat::*;
    match planes.chroma {
        Yuv420 => (planes.width.div_ceil(2), planes.height.div_ceil(2)),
        Yuv422 => (planes.width.div_ceil(2), planes.height),
        Yuv444 => (planes.width, planes.height),
        Monochrome => (0, 0),
    }
}
/// Build display metadata (crop, chroma, bit depth, colour) from an SPS.
fn frame_meta_from_sps(sps: &Sps) -> FrameMeta {
    let disp_w = (sps.width as usize)
        .saturating_sub(sps.crop_left as usize)
        .saturating_sub(sps.crop_right as usize);
    let disp_h = (sps.height as usize)
        .saturating_sub(sps.crop_top as usize)
        .saturating_sub(sps.crop_bottom as usize);
    let bit_depth = sps.bit_depth().unwrap_or(BitDepth::Eight);
    let color = Cicp {
        primaries: crate::color::Primaries::from_u8(sps.color_primaries),
        transfer: crate::color::TransferFunction::from_u8(sps.transfer_characteristics),
        matrix: crate::color::MatrixCoefficients::from_u8(sps.matrix_coefficients),
        full_range: sps.video_full_range,
    };
    FrameMeta {
        crop_left: sps.crop_left as usize,
        crop_top: sps.crop_top as usize,
        disp_w,
        disp_h,
        chroma: sps.chroma,
        bit_depth,
        color,
        color_description_present: sps.colour_description_present,
        time_scale: sps.vui_time_scale,
        num_units_in_tick: sps.vui_num_units_in_tick,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stream_yields_nothing() {
        let mut d = VideoDecoder::new();
        let out = d.decode_all(&[0, 0, 0, 1, 0x40, 0x01]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn untagged_gbr_override_uses_hevc_component_order() {
        let frame = VideoFrame {
            planes: YuvPlanes {
                y: vec![20],
                cb: vec![30],
                cr: vec![10],
                width: 1,
                height: 1,
                chroma: ChromaFormat::Yuv444,
                bit_depth: BitDepth::Eight,
            },
            poc: 0,
            meta: FrameMeta {
                crop_left: 0,
                crop_top: 0,
                disp_w: 1,
                disp_h: 1,
                chroma: ChromaFormat::Yuv444,
                bit_depth: BitDepth::Eight,
                color: Cicp::unspecified(),
                color_description_present: false,
                time_scale: 0,
                num_units_in_tick: 0,
            },
        };
        assert_eq!(frame.color(), Cicp::unspecified());
        assert_eq!(frame.signalled_color(), None);
        assert_eq!(frame.to_rgb8_gbr(), vec![10, 20, 30]);
        assert_eq!(frame.to_rgba8_gbr(), vec![10, 20, 30, 255]);
    }

    #[test]
    fn monochrome_yuv_has_no_chroma_planes() {
        let frame = VideoFrame {
            planes: YuvPlanes {
                y: vec![1, 2, 3, 4],
                cb: Vec::new(),
                cr: Vec::new(),
                width: 2,
                height: 2,
                chroma: ChromaFormat::Monochrome,
                bit_depth: BitDepth::Eight,
            },
            poc: 0,
            meta: FrameMeta {
                crop_left: 0,
                crop_top: 0,
                disp_w: 2,
                disp_h: 2,
                chroma: ChromaFormat::Monochrome,
                bit_depth: BitDepth::Eight,
                color: Cicp::unspecified(),
                color_description_present: false,
                time_scale: 0,
                num_units_in_tick: 0,
            },
        };

        let yuv = frame.to_yuv();
        assert_eq!(yuv.y.as_u8(), Some([1, 2, 3, 4].as_slice()));
        assert!(yuv.cb.is_empty());
        assert!(yuv.cr.is_empty());
        assert_eq!((yuv.chroma_width, yuv.chroma_height), (0, 0));
    }
}
