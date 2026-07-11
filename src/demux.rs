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

//! NAL unit demultiplexing for HEVC video bitstreams.
//!
//! Two framings are supported and auto-detected:
//! - Annex-B: NAL units separated by 3- or 4-byte start codes (`00 00 01` /
//!   `00 00 00 01`). Used by `.hevc` / `.265` elementary streams.
//! - Length-prefixed: each NAL prefixed by a big-endian length of
//!   `length_size` bytes (1, 2, or 4). Used inside `hvcC` / mp4 samples.

use crate::error::DecodeError;

/// A single NAL unit sliced out of the input, with its parsed header fields.
/// The slice still contains emulation-prevention bytes; callers unescape as
/// needed (`bitreader::unescape_rbsp`).
#[derive(Clone, Copy)]
pub(crate) struct Nal<'a> {
    pub(crate) nal_type: u8,
    /// Full NAL bytes including the 2-byte NAL header. TemporalId, when needed,
    /// is read directly from bytes[1] by the caller.
    pub(crate) bytes: &'a [u8],
}

impl<'a> Nal<'a> {
    #[inline]
    fn parse(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < 2 {
            return None;
        }
        // forbidden_zero_bit(1) type(6) layer_id(6) tid_plus1(3)
        let nal_type = (bytes[0] >> 1) & 0x3f;
        Some(Nal { nal_type, bytes })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Framing {
    AnnexB,
    /// Length-prefixed with the given prefix size in bytes (1/2/4).
    Length(u8),
}

/// Guess the framing of a raw byte stream. Annex-B is detected by a leading
/// start code; otherwise we assume a 4-byte length prefix (the mp4 default).
pub(crate) fn detect_framing(data: &[u8]) -> Framing {
    if starts_with_start_code(data) {
        Framing::AnnexB
    } else {
        Framing::Length(4)
    }
}

#[inline]
fn starts_with_start_code(d: &[u8]) -> bool {
    d.len() >= 3
        && d[0] == 0
        && d[1] == 0
        && (d[2] == 1 || (d.len() >= 4 && d[2] == 0 && d[3] == 1))
}

/// Iterate NAL units for a given framing, invoking `f` for each. Stops early
/// (returning the callback's error) if `f` fails.
pub(crate) fn for_each_nal<'a, F>(
    data: &'a [u8],
    framing: Framing,
    mut f: F,
) -> Result<(), DecodeError>
where
    F: FnMut(Nal<'a>) -> Result<(), DecodeError>,
{
    match framing {
        Framing::AnnexB => for_each_annexb(data, f),
        Framing::Length(sz) => {
            let mut pos = 0usize;
            let sz = sz as usize;
            while pos + sz <= data.len() {
                let mut len = 0usize;
                for k in 0..sz {
                    len = (len << 8) | data[pos + k] as usize;
                }
                pos += sz;
                if len == 0 || pos + len > data.len() {
                    break;
                }
                if let Some(n) = Nal::parse(&data[pos..pos + len]) {
                    f(n)?;
                }
                pos += len;
            }
            Ok(())
        }
    }
}

fn for_each_annexb<'a, F>(data: &'a [u8], mut f: F) -> Result<(), DecodeError>
where
    F: FnMut(Nal<'a>) -> Result<(), DecodeError>,
{
    let mut starts: Vec<(usize, usize)> = Vec::new(); // (payload_start, code_len)
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                starts.push((i + 3, 3));
                i += 3;
                continue;
            } else if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                starts.push((i + 4, 4));
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    for k in 0..starts.len() {
        let (s, _) = starts[k];
        let end = if k + 1 < starts.len() {
            // Trim the trailing start code (and its leading zero) of the next NAL.
            starts[k + 1].0 - starts[k + 1].1
        } else {
            data.len()
        };
        if s < end
            && let Some(n) = Nal::parse(trim_trailing_zeros(&data[s..end]))
        {
            f(n)?;
        }
    }
    Ok(())
}

/// Annex-B NALs may carry trailing `cabac_zero_word`s / padding zeros; trim them
/// so RBSP length is exact. Keeps at least the 2-byte header.
#[inline]
fn trim_trailing_zeros(b: &[u8]) -> &[u8] {
    let mut end = b.len();
    while end > 2 && b[end - 1] == 0 {
        end -= 1;
    }
    &b[..end]
}

/// HEVC NAL unit type constants (Table 7-1).
pub(crate) mod nal {
    pub(crate) const RASL_N: u8 = 8;
    pub(crate) const RASL_R: u8 = 9;
    pub(crate) const RADL_N: u8 = 6;
    pub(crate) const RADL_R: u8 = 7;
    pub(crate) const BLA_W_LP: u8 = 16;
    pub(crate) const BLA_N_LP: u8 = 18;
    pub(crate) const IDR_W_RADL: u8 = 19;
    pub(crate) const IDR_N_LP: u8 = 20;
    pub(crate) const CRA_NUT: u8 = 21;
    pub(crate) const SPS: u8 = 33;
    pub(crate) const PPS: u8 = 34;

    #[inline]
    pub(crate) fn is_vcl(t: u8) -> bool {
        t <= 31
    }
    #[inline]
    pub(crate) fn is_irap(t: u8) -> bool {
        (16..=23).contains(&t)
    }
    #[inline]
    pub(crate) fn is_idr(t: u8) -> bool {
        t == IDR_W_RADL || t == IDR_N_LP
    }
    #[inline]
    pub(crate) fn is_bla(t: u8) -> bool {
        (BLA_W_LP..=BLA_N_LP).contains(&t)
    }
    #[inline]
    pub(crate) fn is_cra(t: u8) -> bool {
        t == CRA_NUT
    }
    /// RASL (random-access-skipped-leading) picture — decoded after an IRAP but
    /// output before it; discarded when the IRAP's NoRaslOutputFlag is 1.
    #[inline]
    pub(crate) fn is_rasl(t: u8) -> bool {
        t == RASL_N || t == RASL_R
    }
    /// RADL (random-access-decodable-leading) picture.
    #[inline]
    pub(crate) fn is_radl(t: u8) -> bool {
        t == RADL_N || t == RADL_R
    }
    /// Sub-layer non-reference (`_N`) VCL types are the even values 0..=14.
    #[inline]
    pub(crate) fn is_sub_layer_non_ref(t: u8) -> bool {
        t <= 14 && (t).is_multiple_of(2)
    }
    /// Reference picture flag: `_R` VCL types and all IRAP are reference.
    #[inline]
    pub(crate) fn is_reference(t: u8) -> bool {
        if t > 31 {
            return false;
        }
        !is_sub_layer_non_ref(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_annexb_4byte() {
        let data = [
            0, 0, 0, 1, 0x40, 0x01, 0xaa, // VPS-ish
            0, 0, 1, 0x42, 0x01, 0xbb, // SPS-ish 3-byte code
        ];
        let mut got = Vec::new();
        for_each_nal(&data, Framing::AnnexB, |n| {
            got.push(n.nal_type);
            Ok(())
        })
        .unwrap();
        assert_eq!(got, vec![32, 33]);
    }

    #[test]
    fn splits_length_prefixed() {
        let data = [
            0, 0, 0, 3, 0x26, 0x01, 0xcc, // len=3
            0, 0, 0, 3, 0x28, 0x01, 0xdd,
        ];
        let mut got = Vec::new();
        for_each_nal(&data, Framing::Length(4), |n| {
            got.push(n.nal_type);
            Ok(())
        })
        .unwrap();
        assert_eq!(got, vec![19, 20]);
    }

    #[test]
    fn detects_framing() {
        assert!(matches!(detect_framing(&[0, 0, 1, 0x40]), Framing::AnnexB));
        assert!(matches!(
            detect_framing(&[0, 0, 0, 5, 0x40]),
            Framing::Length(4)
        ));
    }
}
