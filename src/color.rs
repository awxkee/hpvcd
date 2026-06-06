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
#[repr(u8)]
pub enum Primaries {
    Bt709 = 1, // sRGB, BT.1361, IEC 61966
    Unspecified = 2,
    Bt470M = 4,   // NTSC (historical)
    Bt470Bg = 5,  // BT.601 625-line, PAL/SECAM
    Bt601 = 6,    // BT.601 525-line, NTSC, SMPTE 170M
    Smpte240 = 7, // SMPTE 240M (historical)
    GenericFilm = 8,
    Bt2020 = 9,    // BT.2020, BT.2100
    Xyz = 10,      // SMPTE ST 428-1, CIE 1931 XYZ
    Smpte431 = 11, // DCI-P3 (D65)
    Smpte432 = 12, // Display P3
    Ebu3213 = 22,
}

impl Primaries {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Bt709,
            4 => Self::Bt470M,
            5 => Self::Bt470Bg,
            6 => Self::Bt601,
            7 => Self::Smpte240,
            8 => Self::GenericFilm,
            9 => Self::Bt2020,
            10 => Self::Xyz,
            11 => Self::Smpte431,
            12 => Self::Smpte432,
            22 => Self::Ebu3213,
            _ => Self::Unspecified,
        }
    }
}

/// CICP transfer characteristics (ISO/IEC 23091-2 Table 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TransferFunction {
    Bt709 = 1, // BT.709, BT.601 (functionally equivalent to 6, 14, 15)
    Unspecified = 2,
    Bt470M = 4,   // Gamma 2.2 (historical)
    Bt470Bg = 5,  // Gamma 2.8 (historical)
    Bt601 = 6,    // BT.601, SMPTE 170M (≡ 1, 14, 15)
    Smpte240 = 7, // SMPTE 240M (historical)
    Linear = 8,
    Log100 = 9,
    Log100Sqrt10 = 10,
    Iec61966 = 11,     // IEC 61966-2-4
    Bt1361 = 12,       // BT.1361 extended gamut (historical)
    Srgb = 13,         // IEC 61966-2-1 sRGB / sYCC
    Bt2020_10bit = 14, // BT.2020 10-bit (≡ 1, 6, 15)
    Bt2020_12bit = 15, // BT.2020 12-bit (≡ 1, 6, 14)
    Pq = 16,           // SMPTE ST 2084, BT.2100 PQ
    Smpte428 = 17,     // SMPTE ST 428-1
    Hlg = 18,          // ARIB STD-B67, BT.2100 HLG
}

impl TransferFunction {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Bt709,
            4 => Self::Bt470M,
            5 => Self::Bt470Bg,
            6 => Self::Bt601,
            7 => Self::Smpte240,
            8 => Self::Linear,
            9 => Self::Log100,
            10 => Self::Log100Sqrt10,
            11 => Self::Iec61966,
            12 => Self::Bt1361,
            13 => Self::Srgb,
            14 => Self::Bt2020_10bit,
            15 => Self::Bt2020_12bit,
            16 => Self::Pq,
            17 => Self::Smpte428,
            18 => Self::Hlg,
            _ => Self::Unspecified,
        }
    }
}

/// CICP matrix coefficients for YCbCr ↔ RGB (ISO/IEC 23091-2 Table 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Identity = 0, // GBR (no transform)
    Bt709 = 1,
    Unspecified = 2,
    Fcc = 4,
    Bt470Bg = 5,   // BT.601 625-line
    Smpte170m = 6, // BT.601 525-line (same matrix as 5 functionally)
    Smpte240m = 7,
    YCgCo = 8,
    Bt2020Ncl = 9, // BT.2020 non-constant luminance
    Bt2020Cl = 10, // BT.2020 constant luminance
    Smpte2085 = 11,
    ChromaticityDerivedNcl = 12,
    ChromaticityDerivedCl = 13,
    ICtCp = 14,
    IPtPc2 = 15,
    YCgCoRe = 16,
    YCgCoRo = 17,
}

impl MatrixCoefficients {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Identity,
            1 => Self::Bt709,
            4 => Self::Fcc,
            5 => Self::Bt470Bg,
            6 => Self::Smpte170m,
            7 => Self::Smpte240m,
            8 => Self::YCgCo,
            9 => Self::Bt2020Ncl,
            10 => Self::Bt2020Cl,
            11 => Self::Smpte2085,
            12 => Self::ChromaticityDerivedNcl,
            13 => Self::ChromaticityDerivedCl,
            14 => Self::ICtCp,
            15 => Self::IPtPc2,
            16 => Self::YCgCoRe,
            17 => Self::YCgCoRo,
            _ => Self::Unspecified,
        }
    }
}

/// CICP encoding used for both HEIF `colr` (nclx) boxes and HEVC VUI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorEncoding {
    pub primaries: Primaries,
    pub transfer: TransferFunction,
    pub matrix: MatrixCoefficients,
    pub full_range: bool,
}

impl ColorEncoding {
    pub const fn srgb() -> Self {
        Self {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
        }
    }
    pub const fn bt709() -> Self {
        Self {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Bt709,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
        }
    }
    pub const fn bt2020_pq() -> Self {
        Self {
            primaries: Primaries::Bt2020,
            transfer: TransferFunction::Pq,
            matrix: MatrixCoefficients::Bt2020Ncl,
            full_range: true,
        }
    }

    /// Serialize as an `nclx` `colr` box payload (without the box header).
    pub fn nclx_payload(&self) -> Vec<u8> {
        let mut p = Vec::with_capacity(11);
        p.extend_from_slice(b"nclx");
        p.extend_from_slice(&(self.primaries as u16).to_be_bytes());
        p.extend_from_slice(&(self.transfer as u16).to_be_bytes());
        p.extend_from_slice(&(self.matrix as u16).to_be_bytes());
        p.push(if self.full_range { 0x80 } else { 0x00 });
        p
    }
}

impl Default for ColorEncoding {
    fn default() -> Self {
        Self::srgb()
    }
}

/// Combined colour metadata from a HEIF item.
///
/// Both `cicp` and `icc` can be present simultaneously — Apple HEIC files
/// typically include an `nclx` box for YCbCr parameters and a `prof`/`rICC`
/// box for display colour management.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColorMetadata {
    /// CICP code points from an `nclx` `colr` box.
    pub cicp: Option<ColorEncoding>,
    /// Raw ICC profile bytes from a `prof` or `rICC` `colr` box.
    pub icc: Option<Vec<u8>>,
}

impl ColorMetadata {
    pub fn from_cicp(enc: ColorEncoding) -> Self {
        Self {
            cicp: Some(enc),
            icc: None,
        }
    }
    pub fn from_icc(bytes: Vec<u8>) -> Self {
        Self {
            cicp: None,
            icc: Some(bytes),
        }
    }

    /// CICP encoding for driving the YCbCr→RGB conversion; falls back to sRGB.
    pub fn color_encoding(&self) -> ColorEncoding {
        self.cicp.unwrap_or_else(ColorEncoding::srgb)
    }
    pub fn is_empty(&self) -> bool {
        self.cicp.is_none() && self.icc.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nclx_payload_layout() {
        let p = ColorEncoding::bt709().nclx_payload();
        assert_eq!(&p[0..4], b"nclx");
        assert_eq!(u16::from_be_bytes([p[4], p[5]]), 1);
        assert_eq!(u16::from_be_bytes([p[6], p[7]]), 1);
        assert_eq!(u16::from_be_bytes([p[8], p[9]]), 1);
        assert_eq!(p[10] >> 7, 1);
    }

    #[test]
    fn pq_encoding_values() {
        let e = ColorEncoding::bt2020_pq();
        assert_eq!(e.primaries as u8, 9);
        assert_eq!(e.transfer as u8, 16);
        assert_eq!(e.matrix as u8, 9);
    }

    #[test]
    fn from_u8_roundtrips() {
        assert_eq!(Primaries::from_u8(1), Primaries::Bt709);
        assert_eq!(Primaries::from_u8(12), Primaries::Smpte432);
        assert_eq!(Primaries::from_u8(99), Primaries::Unspecified);
        assert_eq!(
            MatrixCoefficients::from_u8(6),
            MatrixCoefficients::Smpte170m
        );
        assert_eq!(TransferFunction::from_u8(16), TransferFunction::Pq);
    }
}
