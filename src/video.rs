/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * // BSD-3-Clause OR Apache-2.0
 */

//! Top-level HEVC *video* decode driver. Consumes an elementary stream (Annex-B
//! or length-prefixed), demuxes NAL units, tracks parameter sets, derives POC,
//! maintains the DPB, constructs reference lists, decodes each slice (intra or
//! inter) through [`crate::decode::FullDecoder`], and emits frames in output
//! (POC) order.
//!
//! This is the prototype integration: single-slice-per-picture common path,
//! serial (non-WPP) reconstruction. Multi-segment pictures and the parallel
//! wavefront path for inter are wired incrementally on top of this.

use crate::color::Cicp;
use crate::config::{Pps, Sps, parse_pps, parse_sps};
use crate::decode::{FullDecoder, parse_slice_header_full};
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
struct FrameMeta {
    /// Conformance-window crop offsets in luma samples.
    crop_left: usize,
    crop_top: usize,
    /// Display (visible) luma dimensions after cropping.
    disp_w: usize,
    disp_h: usize,
    chroma: ChromaFormat,
    bit_depth: BitDepth,
    color: Cicp,
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
            color: Cicp::srgb(),
            time_scale: 0,
            num_units_in_tick: 0,
        }
    }
}

/// One decoded, displayable video frame.
///
/// The simplest thing to do with a frame is call [`VideoFrame::to_rgb8`], which
/// always returns tightly-packed 8-bit RGB at the display size — regardless of
/// the stream's bit depth or chroma format. For the raw planes (cropped to the
/// visible picture) use [`VideoFrame::to_yuv`]. Presentation order is given by
/// [`VideoFrame::poc`]; `decode`/`decode_all` already return frames in order.
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
    /// Bit depth of the luma/chroma samples (8, 10, or 12).
    pub fn bit_depth(&self) -> u8 {
        self.meta.bit_depth.bits()
    }
    /// Chroma subsampling of the decoded planes.
    pub fn chroma_format(&self) -> ChromaFormat {
        self.meta.chroma
    }

    /// Frame rate in frames per second from the stream's VUI timing info, if it
    /// signalled any (`time_scale / num_units_in_tick`). Returns `None` when the
    /// elementary stream carries no timing — in that case the container is the
    /// authority and you should supply the rate yourself.
    pub fn frame_rate(&self) -> Option<f64> {
        if self.meta.time_scale > 0 && self.meta.num_units_in_tick > 0 {
            Some(self.meta.time_scale as f64 / self.meta.num_units_in_tick as f64)
        } else {
            None
        }
    }

    /// Presentation timestamp in seconds, derived from the picture order count
    /// and the VUI frame rate. `None` when the stream carries no timing info
    /// (use `poc` plus your own frame rate instead). POC counts pictures in
    /// presentation order, so this is `poc / frame_rate`.
    pub fn timestamp(&self) -> Option<f64> {
        self.frame_rate().map(|fps| self.poc as f64 / fps)
    }

    /// Convert to tightly-packed 8-bit interleaved RGB at the display size.
    pub fn to_rgb8(&self) -> Vec<u8> {
        let (dw, dh) = (self.meta.disp_w.max(1), self.meta.disp_h.max(1));
        let img = crate::yuv::yuv_to_rgb_window_with_color(
            &self.planes,
            dw,
            dh,
            self.meta.crop_left,
            self.meta.crop_top,
            &self.meta.color,
        );
        match img {
            ImageBuffer::Rgb8(v) => v,
            ImageBuffer::Rgb16(v) => v.iter().map(|&s| (s >> (self.shift())) as u8).collect(),
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

    /// Convert to 8-bit interleaved RGBA (opaque alpha) at the display size.
    pub fn to_rgba8(&self) -> Vec<u8> {
        let rgb = self.to_rgb8();
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
        let cw = dw.div_ceil(sub_w);
        let ch = dh.div_ceil(sub_h);
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
    let x0 = x0.min(src_w.saturating_sub(1));
    let y0 = y0.min(src_h.saturating_sub(1));
    if eight {
        let mut out = Vec::with_capacity(dw * dh);
        for row in 0..dh {
            let sy = (y0 + row).min(src_h - 1);
            for col in 0..dw {
                let sx = (x0 + col).min(src_w - 1);
                out.push(src[sy * src_w + sx] as u8);
            }
        }
        SampleBuf::U8(out)
    } else {
        let mut out = Vec::with_capacity(dw * dh);
        for row in 0..dh {
            let sy = (y0 + row).min(src_h - 1);
            for col in 0..dw {
                let sx = (x0 + col).min(src_w - 1);
                out.push(src[sy * src_w + sx]);
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

/// Stateful HEVC video decoder.
///
/// For most uses the free function [`decode_hevc`] (or [`VideoDecoder::decode`])
/// is all you need: it returns every frame, in order, ready to convert to RGB.
/// The struct form lets you reuse one decoder across calls or feed data
/// incrementally.
pub struct VideoDecoder {
    sps: Option<Sps>,
    pps: Option<Pps>,
    dpb: Dpb,
    prev_poc: i32,
    /// Frames ready for output, in decode order (caller sorts/consumes).
    outputs: Vec<VideoFrame>,
    /// First picture seen (POC anchor).
    seen_first: bool,
    /// Display metadata (crop/chroma/depth/colour) from the active SPS.
    cur_meta: FrameMeta,
}

impl Default for VideoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoDecoder {
    pub fn new() -> Self {
        VideoDecoder {
            sps: None,
            pps: None,
            dpb: Dpb::new(16),
            prev_poc: 0,
            outputs: Vec::new(),
            seen_first: false,
            cur_meta: FrameMeta::default(),
        }
    }

    /// Decode an entire elementary stream, returning frames in presentation
    /// (POC) order. Framing (Annex-B vs length-prefixed) is auto-detected. This
    /// is the recommended entry point; [`decode_hevc`] wraps it for one-shot use.
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
                meta: self.cur_meta.clone(),
            });
        }
        let mut out = std::mem::take(&mut self.outputs);
        out.sort_by_key(|f| f.poc);
        Ok(out)
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
        let framing = detect_framing(data);
        let aus = self.collect_access_units(data, framing)?;
        if aus.is_empty() {
            return Ok(None);
        }
        let start = self.irap_at_or_before_target(&aus, target_poc);
        self.reset_decode_state();
        for au in &aus[start..] {
            self.decode_access_unit(au)?;
        }
        // Flush the reorder buffer exactly like decode_all so every decoded
        // picture (including the target, wherever it sits in the DPB) is emitted.
        for f in self.dpb.bump(true) {
            self.outputs.push(VideoFrame {
                planes: f.planes,
                poc: f.poc,
                meta: self.cur_meta.clone(),
            });
        }
        let frame = self.outputs.drain(..).find(|f| f.poc == target_poc);
        Ok(frame)
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
        self.prev_poc = 0;
        self.outputs.clear();
        self.seen_first = false;
    }

    /// Group the stream into access units, prepending the parameter-set NALs
    /// that configure each AU so a decode can start at any returned index.
    fn collect_access_units(
        &mut self,
        data: &[u8],
        framing: Framing,
    ) -> Result<Vec<Vec<(u8, Vec<u8>)>>, DecodeError> {
        let mut nals: Vec<(u8, Vec<u8>)> = Vec::new();
        for_each_nal(data, framing, |n: Nal| {
            nals.push((n.nal_type, n.bytes.to_vec()));
            Ok(())
        })?;
        let mut aus: Vec<Vec<(u8, Vec<u8>)>> = Vec::new();
        let mut pending_non_vcl: Vec<(u8, Vec<u8>)> = Vec::new();
        let mut cur: Vec<(u8, Vec<u8>)> = Vec::new();
        for (nal_type, bytes) in nals {
            if nal::is_vcl(nal_type) {
                if self.is_first_slice(nal_type, &bytes) && !cur.is_empty() {
                    aus.push(std::mem::take(&mut cur));
                }
                if cur.is_empty() {
                    cur.append(&mut pending_non_vcl);
                }
                cur.push((nal_type, bytes));
            } else {
                // Apply params now (so slice-header parsing works) and remember
                // them to prepend to the next AU for independent decodability.
                self.process_non_vcl(nal_type, &bytes)?;
                pending_non_vcl.push((nal_type, bytes));
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
    fn irap_at_or_before_target(&self, aus: &[Vec<(u8, Vec<u8>)>], target_poc: i32) -> usize {
        let (sps, pps) = match (self.sps.as_ref(), self.pps.as_ref()) {
            (Some(s), Some(p)) => (s, p),
            _ => return 0,
        };
        let max_poc_lsb = 1i32 << sps.log2_max_poc_lsb;
        let mut prev_poc = 0i32;
        let mut best = 0usize;
        for (i, au) in aus.iter().enumerate() {
            // The AU's first VCL NAL carries the picture's POC LSB.
            let Some((nal_type, bytes)) = au.iter().find(|(t, _)| nal::is_vcl(*t)) else {
                continue;
            };
            let rbsp = crate::bitreader::unescape_rbsp(bytes);
            let poc = match parse_slice_header_full(&rbsp, sps, pps, *nal_type) {
                Ok(h) => {
                    if nal::is_idr(*nal_type) {
                        0
                    } else {
                        crate::rps::derive_poc(
                            h.poc_lsb,
                            prev_poc,
                            max_poc_lsb,
                            nal::is_irap(*nal_type),
                        )
                    }
                }
                Err(_) => prev_poc,
            };
            prev_poc = poc;
            if nal::is_irap(*nal_type) && poc <= target_poc {
                best = i;
            }
        }
        best
    }

    fn decode_stream(&mut self, data: &[u8], framing: Framing) -> Result<(), DecodeError> {
        // Collect NALs first (borrow-friendly), then process.
        let mut nals: Vec<(u8, Vec<u8>)> = Vec::new();
        for_each_nal(data, framing, |n: Nal| {
            nals.push((n.nal_type, n.bytes.to_vec()));
            Ok(())
        })?;
        // Group VCL NALs into access units. A new picture starts at a VCL NAL
        // whose slice header has first_slice_segment_in_pic_flag = 1; all later
        // VCL NALs (dependent or independent segments) belong to the same
        // picture until the next such flag. Non-VCL NALs (SPS/PPS/VPS/SEI) are
        // applied immediately and flush any pending access unit first.
        let mut au: Vec<(u8, Vec<u8>)> = Vec::new();
        for (nal_type, bytes) in nals {
            if nal::is_vcl(nal_type) {
                if self.is_first_slice(nal_type, &bytes) && !au.is_empty() {
                    self.decode_access_unit(&au)?;
                    au.clear();
                }
                au.push((nal_type, bytes));
            } else {
                if !au.is_empty() {
                    self.decode_access_unit(&au)?;
                    au.clear();
                }
                self.process_non_vcl(nal_type, &bytes)?;
            }
        }
        if !au.is_empty() {
            self.decode_access_unit(&au)?;
        }
        Ok(())
    }

    /// Peek whether a VCL NAL begins a new picture (first_slice_in_pic).
    fn is_first_slice(&self, nal_type: u8, bytes: &[u8]) -> bool {
        let (sps, pps) = match (self.sps.as_ref(), self.pps.as_ref()) {
            (Some(s), Some(p)) => (s, p),
            _ => return true,
        };
        let rbsp = crate::bitreader::unescape_rbsp(bytes);
        match parse_slice_header_full(&rbsp, sps, pps, nal_type) {
            Ok(h) => h.first_slice_in_pic,
            Err(_) => true,
        }
    }

    fn process_non_vcl(&mut self, nal_type: u8, bytes: &[u8]) -> Result<(), DecodeError> {
        match nal_type {
            nal::SPS => {
                let rbsp = crate::bitreader::unescape_rbsp(&bytes[2..]);
                if let Ok(sps) = parse_sps(&rbsp) {
                    self.cur_meta = frame_meta_from_sps(&sps);
                    self.sps = Some(sps);
                }
                Ok(())
            }
            nal::PPS => {
                let rbsp = crate::bitreader::unescape_rbsp(&bytes[2..]);
                let sl = self
                    .sps
                    .as_ref()
                    .map(|s| s.scaling_list_enabled)
                    .unwrap_or(false);
                if let Ok(pps) = parse_pps(&rbsp, sl) {
                    self.pps = Some(pps);
                }
                Ok(())
            }
            _ => Ok(()), // VPS/SEI/AUD/etc.
        }
    }

    pub fn process_nal(&mut self, nal_type: u8, bytes: &[u8]) -> Result<(), DecodeError> {
        match nal_type {
            nal::SPS | nal::PPS => self.process_non_vcl(nal_type, bytes),
            t if nal::is_vcl(t) => self.decode_access_unit(&[(t, bytes.to_vec())]),
            _ => Ok(()),
        }
    }

    fn decode_access_unit(&mut self, au: &[(u8, Vec<u8>)]) -> Result<(), DecodeError> {
        // Apply any parameter-set NALs carried at the head of this access unit
        // (they may precede the first VCL slice, e.g. before an IRAP), and keep
        // only the VCL slice segments for the picture decode. Pre-unescape each
        // slice's RBSP so the borrowed CABAC payloads live for the whole
        // picture decode; the original NAL bytes are kept for WPP/tile entry
        // point mapping.
        let mut segs: Vec<(u8, Vec<u8>, Vec<u8>)> = Vec::with_capacity(au.len());
        for (nal_type, bytes) in au {
            if !nal::is_vcl(*nal_type) {
                self.process_non_vcl(*nal_type, bytes)?;
                continue;
            }
            segs.push((
                *nal_type,
                crate::bitreader::unescape_rbsp(bytes),
                bytes.clone(),
            ));
        }
        if segs.is_empty() {
            return Ok(());
        }
        let (sps, pps) = match (self.sps.clone(), self.pps.clone()) {
            (Some(s), Some(p)) => (s, p),
            _ => return Ok(()), // no parameter sets yet
        };

        // Parse the first segment's header to set up picture-level state.
        let (first_nal, first_rbsp, first_bytes) = (&segs[0].0, &segs[0].1, &segs[0].2);
        let first_nal = *first_nal;
        let hdr0 = match parse_slice_header_full(first_rbsp, &sps, &pps, first_nal) {
            Ok(h) => h,
            Err(_) => return Ok(()),
        };
        if !hdr0.first_slice_in_pic {
            return Ok(());
        }

        // POC derivation (from the first slice of the picture).
        let is_irap = nal::is_irap(first_nal);
        let is_idr = nal::is_idr(first_nal);
        let max_poc_lsb = 1i32 << sps.log2_max_poc_lsb;
        let poc = if is_idr {
            0
        } else {
            derive_poc(hdr0.poc_lsb, self.prev_poc, max_poc_lsb, is_irap)
        };
        if is_idr || nal::is_bla(first_nal) {
            self.dpb.clear_refs();
        }

        // Reference lists (built once per picture from the first slice's RPS).
        let (ref_list0, ref_list1, ref_frames) = if hdr0.slice_type != crate::inter::SLICE_I {
            let max_poc_lsb = 1i32 << sps.log2_max_poc_lsb;
            let pocs = self
                .dpb
                .apply_rps_lt(poc, &hdr0.cur_rps, &hdr0.lt_refs, max_poc_lsb);
            let is_b = hdr0.slice_type == crate::inter::SLICE_B;
            let (l0, l1) = self.dpb.build_ref_lists(
                &pocs,
                hdr0.num_ref_idx_l0,
                hdr0.num_ref_idx_l1,
                is_b,
                &hdr0.list_mod_l0,
                &hdr0.list_mod_l1,
            );
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
        d.decode_slice_ctx(hdr0.slice_segment_address, starts0)?;

        // Decode any remaining segments of the same picture into the same
        // decoder (accumulating the reconstructed planes and motion field).
        for (seg_nal, seg_rbsp, seg_bytes) in segs.iter().skip(1) {
            let hdr = match parse_slice_header_full(seg_rbsp, &sps, &pps, *seg_nal) {
                Ok(h) => h,
                Err(_) => continue,
            };
            if hdr.first_slice_in_pic {
                // Shouldn't happen (grouping guarantees it), but be safe.
                continue;
            }
            // Rebuild ref lists for independent segments (dependent segments
            // inherit them); reuse the picture's reference frame views.
            if !hdr.dependent_slice_segment && hdr.slice_type != crate::inter::SLICE_I {
                let max_poc_lsb = 1i32 << sps.log2_max_poc_lsb;
                let pocs = self
                    .dpb
                    .apply_rps_lt(poc, &hdr.cur_rps, &hdr.lt_refs, max_poc_lsb);
                let is_b = hdr.slice_type == crate::inter::SLICE_B;
                let (l0, l1) = self.dpb.build_ref_lists(
                    &pocs,
                    hdr.num_ref_idx_l0,
                    hdr.num_ref_idx_l1,
                    is_b,
                    &hdr.list_mod_l0,
                    &hdr.list_mod_l1,
                );
                d.set_inter_state(poc, l0, l1, ref_frames.clone());
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
            let _ = d.decode_segment(cabac, &hdr, &sub);
        }

        // Deblock/SAO once the whole picture is reconstructed, then store it.
        let planes = d.finish(None);
        let (motion, width4, height4) = d.take_motion();

        let is_ref = nal::is_reference(first_nal);
        let frame = Frame {
            planes,
            poc,
            motion,
            width4,
            height4,
            short_term: is_ref,
            long_term: false,
            needed_for_output: true,
        };
        self.dpb.push(frame);
        self.prev_poc = poc;
        self.seen_first = true;

        for f in self.dpb.bump(false) {
            self.outputs.push(VideoFrame {
                planes: f.planes,
                poc: f.poc,
                meta: self.cur_meta.clone(),
            });
        }
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
}
