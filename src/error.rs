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
#[derive(Debug)]
pub enum DecodeError {
    NotHeif,
    TruncatedBox(usize),
    MissingBox(&'static str),
    UnsupportedItemType(String),
    Bitstream(String),
    CabacDesync,
    UnsupportedChroma(u8),
    UnsupportedBitDepth(u8),
    BadDimensions {
        w: u32,
        h: u32,
    },
    ParamSet(String),
    /// A syntactically valid stream that uses a coding tool this decoder does
    /// not implement (e.g. tiles combined with WPP).
    Unsupported(String),
    /// A configured parse limit was exceeded (e.g. box, item, or image size).
    /// The field names the limit and carries the offending vs. allowed values.
    LimitExceeded {
        what: &'static str,
        value: u64,
        limit: u64,
    },
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotHeif => write!(f, "Not a HEIF/HEIC file (bad ftyp)"),
            Self::TruncatedBox(offset) => write!(f, "Truncated box at offset {offset}"),
            Self::MissingBox(name) => write!(f, "Required box '{name}' not found"),
            Self::UnsupportedItemType(ty) => {
                write!(f, "Unsupported item type '{ty}' — only hvc1 is supported")
            }
            Self::Bitstream(msg) => write!(f, "HEVC bitstream error: {msg}"),
            Self::CabacDesync => write!(f, "CABAC decoder desync"),
            Self::UnsupportedChroma(fmt) => write!(f, "Unsupported chroma format {fmt}"),
            Self::UnsupportedBitDepth(depth) => write!(f, "Unsupported bit depth {depth}"),
            Self::BadDimensions { w, h } => {
                write!(f, "Image dimensions {w}×{h} are zero or exceed limits")
            }
            Self::ParamSet(msg) => write!(f, "SPS/PPS parse error: {msg}"),
            Self::Unsupported(msg) => write!(f, "Unsupported HEVC feature: {msg}"),
            Self::LimitExceeded { what, value, limit } => write!(
                f,
                "parse limit exceeded: {what} = {value} exceeds configured limit {limit}"
            ),
        }
    }
}

impl std::error::Error for DecodeError {}
