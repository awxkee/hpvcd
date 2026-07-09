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

use crate::config::{Pps, Sps, parse_pps, parse_sps};
use crate::decode::{FullDecoder, SliceHeader, parse_slice_header_full};
use crate::demux::{Framing, Nal, detect_framing, for_each_nal, nal};
use crate::dpb::{Dpb, Frame};
use crate::error::DecodeError;
use crate::inter::{MotionInfo, RefFramePlanes};
use crate::rps::derive_poc;
use crate::yuv::YuvPlanes;

/// A decoded video frame in coded form (planes + POC), ready for color/crop.
pub struct VideoFrame {
    pub planes: YuvPlanes,
    pub poc: i32,
}

impl VideoFrame {
    /// Number of luma samples (coded).
    pub fn luma_len(&self) -> usize {
        self.planes.luma_len()
    }
    /// Coded luma dimensions.
    pub fn dims(&self) -> (usize, usize) {
        self.planes.dims()
    }
    /// Luma plane as 8-bit samples (coded layout).
    pub fn y_u8(&self) -> Vec<u8> {
        self.planes.y_u8()
    }
    /// Cb plane as 8-bit samples (coded layout).
    pub fn cb_u8(&self) -> Vec<u8> {
        self.planes.cb_u8()
    }
    /// Cr plane as 8-bit samples (coded layout).
    pub fn cr_u8(&self) -> Vec<u8> {
        self.planes.cr_u8()
    }
}

/// Stateful video decoder. Feed a whole elementary stream to `decode_all`, or
/// drive incrementally via `push_nal` + `take_outputs`.
pub struct VideoDecoder {
    sps: Option<Sps>,
    pps: Option<Pps>,
    dpb: Dpb,
    prev_poc: i32,
    /// Frames ready for output, in decode order (caller sorts/consumes).
    outputs: Vec<VideoFrame>,
    /// First picture seen (POC anchor).
    seen_first: bool,
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
        }
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
            });
        }
        let mut out = std::mem::take(&mut self.outputs);
        out.sort_by_key(|f| f.poc);
        Ok(out)
    }

    fn decode_stream(&mut self, data: &[u8], framing: Framing) -> Result<(), DecodeError> {
        // Collect NALs first (borrow-friendly), then process.
        let mut nals: Vec<(u8, Vec<u8>)> = Vec::new();
        for_each_nal(data, framing, |n: Nal| {
            nals.push((n.nal_type, n.bytes.to_vec()));
            Ok(())
        })?;
        for (nal_type, bytes) in nals {
            self.process_nal(nal_type, &bytes)?;
        }
        Ok(())
    }

    fn process_nal(&mut self, nal_type: u8, bytes: &[u8]) -> Result<(), DecodeError> {
        match nal_type {
            nal::SPS => {
                let rbsp = crate::bitreader::unescape_rbsp(&bytes[2..]);
                if let Ok(sps) = parse_sps(&rbsp) {
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
            t if nal::is_vcl(t) => self.decode_vcl(t, bytes),
            _ => Ok(()), // VPS/SEI/AUD/etc.
        }
    }

    fn decode_vcl(&mut self, nal_type: u8, bytes: &[u8]) -> Result<(), DecodeError> {
        let (sps, pps) = match (self.sps.clone(), self.pps.clone()) {
            (Some(s), Some(p)) => (s, p),
            _ => return Ok(()), // no parameter sets yet
        };
        let rbsp = crate::bitreader::unescape_rbsp(bytes);
        let hdr = match parse_slice_header_full(&rbsp, &sps, &pps, nal_type) {
            Ok(h) => h,
            Err(_) => return Ok(()),
        };
        // Only handle the first (independent) segment of a picture in the
        // prototype; dependent/extra segments are skipped.
        if !hdr.first_slice_in_pic {
            return Ok(());
        }

        // POC derivation.
        let is_irap = nal::is_irap(nal_type);
        let is_idr = nal::is_idr(nal_type);
        let max_poc_lsb = 1i32 << sps.log2_max_poc_lsb;
        let poc = if is_idr {
            0
        } else {
            derive_poc(hdr.poc_lsb, self.prev_poc, max_poc_lsb, is_irap)
        };

        // IDR/BLA clear the DPB references.
        if is_idr || nal::is_bla(nal_type) {
            self.dpb.clear_refs();
        }

        // Reference marking + list construction (for inter slices).
        let (ref_list0, ref_list1, ref_frames) = if hdr.slice_type != crate::inter::SLICE_I {
            let pocs = self.dpb.apply_rps(poc, &hdr.cur_rps);
            let is_b = hdr.slice_type == crate::inter::SLICE_B;
            let (l0, l1) = self.dpb.build_ref_lists(
                &pocs,
                hdr.num_ref_idx_l0,
                hdr.num_ref_idx_l1,
                is_b,
                &hdr.list_mod_l0,
                &hdr.list_mod_l1,
            );
            let frames = self.collect_ref_frames(&l0, &l1);
            (l0, l1, frames)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // Decode the slice.
        let cabac = &rbsp[hdr.cabac_offset.min(rbsp.len())..];
        let mut d = FullDecoder::new(cabac, sps.clone(), pps.clone(), &hdr)?;
        d.set_inter_state(poc, ref_list0, ref_list1, ref_frames);
        d.decode_slice_ctx(hdr.slice_segment_address, None)?;
        // Deblock/SAO first: the BS finalization consults the motion grid's
        // is_intra flags, so the planes must be finished before motion is moved
        // out for the DPB.
        let planes = d.finish(None);
        let (motion, width4, height4) = d.take_motion();

        // Store the reconstructed picture into the DPB.
        let is_ref = nal::is_reference(nal_type);
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

        // Bump any frames due for output.
        for f in self.dpb.bump(false) {
            self.outputs.push(VideoFrame {
                planes: f.planes,
                poc: f.poc,
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
