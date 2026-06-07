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

use crate::color::{ColorEncoding, ColorMetadata, MatrixCoefficients, Primaries, TransferFunction};
use crate::error::DecodeError;
use crate::metadata::{ContentLightLevel, Orientation};

/// Parsed image item — enough to decode one HEIF image item.
#[derive(Debug, Clone)]
pub(crate) struct HeifItem {
    pub(crate) _item_id: u16,
    /// Absolute file offset of the HEVC sample bytes.
    pub(crate) data_offset: u64,
    /// Length in bytes.
    pub(crate) data_length: u64,
    /// hvcC box payload (SPS/PPS NALUs etc.).
    pub(crate) hvcc: Vec<u8>,
    /// Declared display dimensions from ispe.
    pub(crate) display_w: u32,
    pub(crate) display_h: u32,
    /// Colour metadata.
    pub(crate) color: ColorMetadata,
    pub(crate) orientation: Orientation,
    pub(crate) cll: Option<ContentLightLevel>,
    /// true = this item is the alpha auxiliary image.
    pub(crate) _is_alpha: bool,
}

/// Grid (tiled) image descriptor, present for Apple-style tiled HEIC files.
#[derive(Debug)]
pub(crate) struct GridInfo {
    /// Number of tile rows.
    pub(crate) rows: u32,
    /// Number of tile columns.
    pub(crate) cols: u32,
    /// Final cropped output width (may be smaller than rows×tile_h).
    pub(crate) output_width: u32,
    /// Final cropped output height.
    pub(crate) output_height: u32,
    /// Tile items in row-major order (rows×cols entries).
    pub(crate) tiles: Vec<HeifItem>,
    /// Orientation of the composed image (from the grid item's irot/imir properties).
    pub(crate) orientation: Orientation,
}

/// Top-level parse result.
pub(crate) struct HeifFile {
    /// Primary (color) image.  For grid files this is a placeholder pointing
    /// at the first tile so existing single-image code keeps compiling; use
    /// `grid` for the full picture.
    pub(crate) primary: HeifItem,
    /// Alpha auxiliary item, if present.
    pub(crate) alpha: Option<HeifItem>,
    /// Raw EXIF/TIFF bytes (after the 4-byte header-offset prefix), if present.
    pub(crate) exif: Option<Vec<u8>>,
    /// Grid descriptor — `Some` only when the primary item type is `grid`.
    pub(crate) grid: Option<GridInfo>,
}

#[allow(dead_code)]
fn read_u8(b: &[u8], off: usize) -> Option<u8> {
    b.get(off).copied()
}
fn read_u16(b: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(off..off + 2)?.try_into().ok()?))
}
fn read_u32(b: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(off..off + 4)?.try_into().ok()?))
}
fn read_u64(b: &[u8], off: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(off..off + 8)?.try_into().ok()?))
}

/// Iterator over ISO base-media boxes at one nesting level.
struct Boxes<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Boxes<'a> {
    fn new(data: &'a [u8]) -> Self {
        Boxes { data, pos: 0 }
    }
}

struct BoxEntry<'a> {
    fourcc: [u8; 4],
    payload: &'a [u8], // payload excluding the 8-byte header (size+fourcc)
}

impl<'a> Iterator for Boxes<'a> {
    type Item = BoxEntry<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + 8 > self.data.len() {
            return None;
        }
        let size = read_u32(self.data, self.pos)? as usize;
        if size < 8 || self.pos + size > self.data.len() {
            return None;
        }
        let fourcc: [u8; 4] = self.data[self.pos + 4..self.pos + 8].try_into().ok()?;
        let payload = &self.data[self.pos + 8..self.pos + size];
        self.pos += size;
        Some(BoxEntry { fourcc, payload })
    }
}

fn fullbox_header(payload: &[u8]) -> Option<(u8, u32, &[u8])> {
    if payload.len() < 4 {
        return None;
    }
    let version = payload[0];
    let flags = u32::from_be_bytes([0, payload[1], payload[2], payload[3]]);
    Some((version, flags, &payload[4..]))
}

pub(crate) fn parse(file: &[u8]) -> Result<HeifFile, DecodeError> {
    // Verify ftyp
    let mut has_heic = false;
    for b in Boxes::new(file) {
        if &b.fourcc == b"ftyp" {
            let brands: Vec<_> = b
                .payload
                .chunks(4)
                .map(|c| c.try_into().unwrap_or([0; 4]))
                .collect();
            has_heic = brands
                .iter()
                .any(|br: &[u8; 4]| matches!(br, b"heic" | b"mif1" | b"miaf" | b"heif"));
            break;
        }
    }
    if !has_heic {
        return Err(DecodeError::NotHeif);
    }

    // Find meta box
    let meta_payload = Boxes::new(file)
        .find(|b| &b.fourcc == b"meta")
        .map(|b| b.payload)
        .ok_or(DecodeError::MissingBox("meta"))?;
    // meta is a FullBox
    let meta = fullbox_header(meta_payload)
        .map(|(_, _, rest)| rest)
        .ok_or(DecodeError::MissingBox("meta body"))?;

    // Parse sub-boxes of meta
    let mut pitm: u16 = 1;
    let mut iloc_data: Option<&[u8]> = None;
    let mut iinf_data: Option<&[u8]> = None;
    let mut iref_data: Option<&[u8]> = None;
    let mut iprp_data: Option<&[u8]> = None;

    // Track `idat` so we can resolve construction_method=1 iloc entries.
    let mut idat_file_offset: u64 = 0;

    for b in Boxes::new(meta) {
        match &b.fourcc {
            b"pitm" => {
                if let Some((_, _, rest)) = fullbox_header(b.payload) {
                    pitm = u16::from_be_bytes(rest[..2].try_into().unwrap_or([0; 2]));
                }
            }
            b"iloc" => iloc_data = Some(b.payload),
            b"iinf" => iinf_data = Some(b.payload),
            b"iref" => iref_data = Some(b.payload),
            b"iprp" => iprp_data = Some(b.payload),
            b"idat" => {
                // Record the absolute file offset of the idat *content* so that
                // iloc entries with construction_method=1 can be resolved.
                // b.payload is a &[u8] slice into `file`, so pointer arithmetic
                // gives us the offset.
                let content_start = b.payload.as_ptr() as usize;
                let file_start = file.as_ptr() as usize;
                idat_file_offset = (content_start - file_start) as u64;
            }
            _ => {}
        }
    }

    // Parse iloc: collect (item_id → (absolute_file_offset, length))
    let mut extents: std::collections::HashMap<u16, (u64, u64)> = std::collections::HashMap::new();
    if let Some((version, _, rest)) = iloc_data.and_then(fullbox_header) {
        parse_iloc(rest, version, idat_file_offset, &mut extents);
    }

    // Parse iinf: item types
    let mut item_types: std::collections::HashMap<u16, [u8; 4]> = std::collections::HashMap::new();
    if let Some((_, _, rest)) = iinf_data.and_then(fullbox_header) {
        let count = read_u16(rest, 0).unwrap_or(0) as usize;
        let mut pos = 2usize;
        for _ in 0..count {
            if pos + 8 > rest.len() {
                break;
            }
            let bsz = read_u32(rest, pos).unwrap_or(0) as usize;
            if bsz < 12 || pos + bsz > rest.len() {
                break;
            }
            // infe fullbox: version/flags(4) + item_id(2) + protection(2) + item_type(4)
            let iid = read_u16(rest, pos + 8 + 2).unwrap_or(0); // +8 box header, +2 skip ver/flag word = pos+4+2+2
            // Actually: box header=8, then fullbox ver/flags=4, item_id=2, protection=2, item_type=4
            if bsz >= 20 {
                let itype: [u8; 4] = rest[pos + 16..pos + 20].try_into().unwrap_or([0; 4]);
                let iid2 = read_u16(rest, pos + 12).unwrap_or(0);
                item_types.insert(iid2, itype);
            }
            let _ = iid;
            pos += bsz;
        }
    }

    // Parse iref: find auxl (auxiliary images — alpha/depth/gainmap/etc.), cdsc
    // (EXIF), and dimg (grid tiles) references
    let mut auxl_items: Vec<u16> = Vec::new();
    let mut exif_item: Option<u16> = None;
    // dimg: from=grid_item_id → to=[tile1, tile2, ...]
    let mut dimg_map: std::collections::HashMap<u16, Vec<u16>> = std::collections::HashMap::new();
    if let Some((_, _, rest)) = iref_data.and_then(fullbox_header) {
        let mut pos = 0;
        while pos + 8 <= rest.len() {
            let bsz = read_u32(rest, pos).unwrap_or(0) as usize;
            if bsz < 12 || pos + bsz > rest.len() {
                break;
            }
            let fourcc = &rest[pos + 4..pos + 8];
            let from_id = read_u16(rest, pos + 8).unwrap_or(0);
            if fourcc == b"auxl" {
                auxl_items.push(from_id);
            }
            if fourcc == b"cdsc" {
                exif_item = Some(from_id);
            }
            if fourcc == b"dimg" {
                let n = read_u16(rest, pos + 10).unwrap_or(0) as usize;
                let to: Vec<u16> = (0..n)
                    .filter_map(|i| read_u16(rest, pos + 12 + i * 2))
                    .collect();
                dimg_map.insert(from_id, to);
            }
            pos += bsz;
        }
    }

    // Parse iprp: extract properties per item
    let (props, prop_assoc) = if let Some(iprp) = iprp_data {
        parse_iprp(iprp)
    } else {
        (vec![], std::collections::HashMap::new())
    };

    // Detect if the primary item is a grid
    let primary_type = item_types.get(&pitm).copied().unwrap_or(*b"hvc1");
    let is_grid = &primary_type == b"grid";

    // Build primary and optional grid.
    // For grid files `primary` is the first tile so single-image code still works.
    let (primary, grid) = if is_grid {
        let g = parse_grid_item(pitm, &extents, &dimg_map, &props, &prop_assoc, file);
        let first = g
            .tiles
            .first()
            .cloned()
            .unwrap_or_else(|| build_fallback_item(pitm));
        (first, Some(g))
    } else {
        let p = build_item(pitm, &extents, &props, &prop_assoc, false, file)?;
        (p, None)
    };
    // Among all auxiliary items, keep only the one whose `auxC` type is genuine
    // alpha. iPhone HEICs attach many `auxl` items (HDR gain map, depth, portrait
    // matte, …); treating one of those as an alpha plane would corrupt the output.
    let alpha_item = auxl_items
        .iter()
        .copied()
        .find(|&id| item_is_alpha(id, &props, &prop_assoc));
    let alpha = if let Some(aid) = alpha_item {
        build_item(aid, &extents, &props, &prop_assoc, true, file).ok()
    } else {
        None
    };

    // Extract EXIF bytes
    let exif = if let Some(eid) = exif_item {
        if let Some(&(off, len)) = extents.get(&eid) {
            let sum = off.saturating_add(len);
            let start = off as usize;
            let end = sum as usize;
            if sum <= file.len() as u64 && len > 4 {
                let raw = &file[start + 4..end];
                Some(raw.to_vec())
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    Ok(HeifFile {
        primary,
        alpha,
        exif,
        grid,
    })
}

fn parse_iloc(
    data: &[u8],
    version: u8,
    idat_file_offset: u64,
    out: &mut std::collections::HashMap<u16, (u64, u64)>,
) {
    if data.len() < 4 {
        return;
    }
    let offset_size = ((data[0] >> 4) & 0xF) as usize;
    let length_size = (data[0] & 0xF) as usize;
    let base_offset_size = ((data[1] >> 4) & 0xF) as usize;
    let index_size = if version == 1 || version == 2 {
        (data[1] & 0xF) as usize
    } else {
        0
    };
    let (item_count, mut pos) = if version == 2 {
        (read_u32(data, 2).unwrap_or(0) as usize, 6usize)
    } else {
        (read_u16(data, 2).unwrap_or(0) as usize, 4usize)
    };
    // Cap item_count to avoid O(N) loops on malformed data where pos never
    // advances (e.g. all sizes are 0).
    let item_count = item_count.min(4096);
    for _ in 0..item_count {
        if pos >= data.len() {
            break;
        }
        let pos_before = pos;

        let item_id = if version == 2 {
            let v = read_u32(data, pos).unwrap_or(0) as u16;
            pos += 4;
            v
        } else {
            let v = read_u16(data, pos).unwrap_or(0);
            pos += 2;
            v
        };
        let cm = if version == 1 || version == 2 {
            let m = read_u16(data, pos).unwrap_or(0) & 0xF;
            pos += 2;
            m
        } else {
            0
        };
        let _data_ref_index = read_u16(data, pos).unwrap_or(0);
        pos += 2;
        let base_offset = read_n(data, pos, base_offset_size);
        pos += base_offset_size;
        let extent_count = read_u16(data, pos).unwrap_or(0);
        pos += 2;
        let extent_count = extent_count.min(64); // sanity cap
        for _ in 0..extent_count {
            if pos >= data.len() {
                break;
            }
            if index_size > 0 {
                pos += index_size;
            }
            let ext_offset = read_n(data, pos, offset_size);
            pos += offset_size;
            let ext_length = read_n(data, pos, length_size);
            pos += length_size;
            // Use checked_add to prevent u64 overflow in offset arithmetic.
            let abs_offset = if cm == 1 {
                idat_file_offset
                    .saturating_add(base_offset)
                    .saturating_add(ext_offset)
            } else {
                base_offset.saturating_add(ext_offset)
            };
            out.insert(item_id, (abs_offset, ext_length));
        }
        // Safety: if pos didn't advance at all, bump it to prevent infinite loop.
        if pos == pos_before {
            break;
        }
    }
}

fn read_n(data: &[u8], off: usize, n: usize) -> u64 {
    match n {
        1 => data.get(off).copied().unwrap_or(0) as u64,
        2 => read_u16(data, off).unwrap_or(0) as u64,
        4 => read_u32(data, off).unwrap_or(0) as u64,
        8 => read_u64(data, off).unwrap_or(0),
        _ => 0,
    }
}

#[derive(Clone, Debug, Default)]
struct Prop {
    kind: [u8; 4],
    data: Vec<u8>,
}

fn parse_iprp(iprp: &[u8]) -> (Vec<Prop>, std::collections::HashMap<u16, Vec<u8>>) {
    let mut props: Vec<Prop> = vec![Prop::default()]; // 1-indexed
    let mut assoc: std::collections::HashMap<u16, Vec<u8>> = std::collections::HashMap::new();

    for b in Boxes::new(iprp) {
        if &b.fourcc == b"ipco" {
            for pb in Boxes::new(b.payload) {
                props.push(Prop {
                    kind: pb.fourcc,
                    data: pb.payload.to_vec(),
                });
            }
        }
        if &b.fourcc == b"ipma"
            && let Some((_, _, rest)) = fullbox_header(b.payload)
        {
            let entry_count = read_u32(rest, 0).unwrap_or(0);
            let mut pos = 4usize;
            for _ in 0..entry_count {
                if pos + 3 > rest.len() {
                    break;
                }
                let item_id = read_u16(rest, pos).unwrap_or(0);
                pos += 2;
                let assoc_count = rest[pos] as usize;
                pos += 1;
                let mut indices = Vec::with_capacity(assoc_count);
                for _ in 0..assoc_count {
                    if pos >= rest.len() {
                        break;
                    }
                    let raw = rest[pos];
                    pos += 1;
                    let idx = raw & 0x7F;
                    indices.push(idx);
                }
                assoc.insert(item_id, indices);
            }
        }
    }
    (props, assoc)
}

fn item_is_alpha(
    item_id: u16,
    props: &[Prop],
    prop_assoc: &std::collections::HashMap<u16, Vec<u8>>,
) -> bool {
    let Some(indices) = prop_assoc.get(&item_id) else {
        return false;
    };
    for &pidx in indices {
        let pidx = pidx as usize;
        if pidx == 0 || pidx >= props.len() {
            continue;
        }
        let p = &props[pidx];
        if &p.kind != b"auxC" {
            continue;
        }
        // auxC payload: 1-byte version + 3-byte flags, then NUL-terminated URN.
        if p.data.len() <= 4 {
            continue;
        }
        let urn_bytes = &p.data[4..];
        let end = urn_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(urn_bytes.len());
        let urn = &urn_bytes[..end];
        if urn == b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha"
            || urn == b"urn:mpeg:hevc:2015:auxid:1"
        {
            return true;
        }
    }
    false
}

/// Read orientation (irot/imir) for a given item from ipco/ipma — without
/// requiring the item to have a hvcC property.  Used for grid items.
fn read_item_orientation(
    item_id: u16,
    props: &[Prop],
    prop_assoc: &std::collections::HashMap<u16, Vec<u8>>,
) -> crate::metadata::Orientation {
    use crate::metadata::Orientation;
    let mut orientation = Orientation::Normal;
    if let Some(indices) = prop_assoc.get(&item_id) {
        for &pidx in indices {
            let pidx = pidx as usize;
            if pidx == 0 || pidx >= props.len() {
                continue;
            }
            let p = &props[pidx];
            match &p.kind {
                b"irot" => {
                    if let Some(&steps) = p.data.first() {
                        orientation = irot_to_orientation(steps & 3, orientation);
                    }
                }
                b"imir" => {
                    if let Some(&axis) = p.data.first() {
                        orientation = imir_to_orientation(axis & 1, orientation);
                    }
                }
                _ => {}
            }
        }
    }
    orientation
}

/// Parse the `grid` item and return a fully populated [`GridInfo`].
///
/// The grid item's data is a small binary descriptor (ISO/IEC 23008-12 §6.6.2.3.2):
/// ```text
/// uint8  version (= 0)
/// uint8  flags            bit 0 = 1 → 32-bit dimensions, 0 → 16-bit
/// uint8  rows_minus_one
/// uint8  columns_minus_one
/// uint16/uint32  output_width
/// uint16/uint32  output_height
/// ```
fn parse_grid_item(
    grid_id: u16,
    extents: &std::collections::HashMap<u16, (u64, u64)>,
    dimg_map: &std::collections::HashMap<u16, Vec<u16>>,
    props: &[Prop],
    prop_assoc: &std::collections::HashMap<u16, Vec<u8>>,
    file: &[u8],
) -> GridInfo {
    // Read the grid descriptor blob
    let (rows, cols, ow, oh) = if let Some(&(off, len)) = extents.get(&grid_id) {
        let start = off as usize;
        let end = off.saturating_add(len) as usize;
        if end <= file.len() && len >= 4 {
            let b = &file[start..end];
            let flags = b[1];
            let rows = b[2] as u32 + 1;
            let cols = b[3] as u32 + 1;
            let (ow, oh) = if flags & 1 != 0 {
                // 32-bit dimensions
                let w = if b.len() >= 8 {
                    u32::from_be_bytes(b[4..8].try_into().unwrap_or([0; 4]))
                } else {
                    0
                };
                let h = if b.len() >= 12 {
                    u32::from_be_bytes(b[8..12].try_into().unwrap_or([0; 4]))
                } else {
                    0
                };
                (w, h)
            } else {
                // 16-bit dimensions
                let w = if b.len() >= 6 {
                    u16::from_be_bytes(b[4..6].try_into().unwrap_or([0; 2])) as u32
                } else {
                    0
                };
                let h = if b.len() >= 8 {
                    u16::from_be_bytes(b[6..8].try_into().unwrap_or([0; 2])) as u32
                } else {
                    0
                };
                (w, h)
            };
            (rows, cols, ow, oh)
        } else {
            (1, 1, 0, 0)
        }
    } else {
        (1, 1, 0, 0)
    };

    // Build a HeifItem for each tile (in row-major order from dimg references)
    let tile_ids = dimg_map.get(&grid_id).cloned().unwrap_or_default();
    let tiles: Vec<HeifItem> = tile_ids
        .iter()
        .filter_map(|&tid| build_item(tid, extents, props, prop_assoc, false, file).ok())
        .collect();

    // Read the grid item's orientation directly from its ipma/ipco associations.
    // We can't use build_item() here because grid items have no hvcC property
    // and build_item() returns Err for items without hvcC.
    let orientation = read_item_orientation(grid_id, props, prop_assoc);

    GridInfo {
        rows,
        cols,
        output_width: ow,
        output_height: oh,
        tiles,
        orientation,
    }
}

/// A zeroed-out placeholder HeifItem used when a tile or grid item is missing.
fn build_fallback_item(item_id: u16) -> HeifItem {
    HeifItem {
        _item_id: item_id,
        data_offset: 0,
        data_length: 0,
        hvcc: vec![],
        display_w: 0,
        display_h: 0,
        color: ColorMetadata::default(),
        orientation: crate::metadata::Orientation::Normal,
        cll: None,
        _is_alpha: false,
    }
}

fn build_item(
    item_id: u16,
    extents: &std::collections::HashMap<u16, (u64, u64)>,

    props: &[Prop],
    prop_assoc: &std::collections::HashMap<u16, Vec<u8>>,
    is_alpha: bool,
    _file: &[u8],
) -> Result<HeifItem, DecodeError> {
    let &(offset, length) = extents
        .get(&item_id)
        .ok_or(DecodeError::MissingBox("iloc entry for item"))?;

    let mut hvcc = Vec::new();
    let mut display_w = 0u32;
    let mut display_h = 0u32;
    let mut color = ColorMetadata::default(); // starts empty; filled from colr box(es)
    let mut orientation = Orientation::Normal;
    let mut cll = None;

    if let Some(indices) = prop_assoc.get(&item_id) {
        for &pidx in indices {
            let pidx = pidx as usize;
            if pidx == 0 || pidx >= props.len() {
                continue;
            }
            let p = &props[pidx];
            match &p.kind {
                b"hvcC" => hvcc = p.data.clone(),
                b"ispe" => {
                    if p.data.len() >= 12 {
                        display_w = read_u32(&p.data, 4).unwrap_or(0);
                        display_h = read_u32(&p.data, 8).unwrap_or(0);
                    }
                }
                b"colr" => {
                    color = parse_colr_into(color, &p.data);
                }
                b"irot" => {
                    if let Some(&steps) = p.data.first() {
                        orientation = irot_to_orientation(steps & 3, orientation);
                    }
                }
                b"imir" => {
                    if let Some(&axis) = p.data.first() {
                        orientation = imir_to_orientation(axis & 1, orientation);
                    }
                }
                b"clli" if p.data.len() >= 4 => {
                    let maxcll = read_u16(&p.data, 0).unwrap_or(0);
                    let maxfall = read_u16(&p.data, 2).unwrap_or(0);
                    cll = Some(ContentLightLevel::new(maxcll, maxfall));
                }
                _ => {}
            }
        }
    }

    if hvcc.is_empty() {
        return Err(DecodeError::MissingBox("hvcC property"));
    }

    Ok(HeifItem {
        _item_id: item_id,
        data_offset: offset,
        data_length: length,
        hvcc,
        display_w,
        display_h,
        color,
        orientation,
        cll,
        _is_alpha: is_alpha,
    })
}

/// Update `color` with the data from one `colr` box.  Both `nclx` and
/// `prof`/`rICC` boxes can be present simultaneously; we accumulate them.
fn parse_colr_into(mut color: ColorMetadata, data: &[u8]) -> ColorMetadata {
    if data.len() < 4 {
        return color;
    }
    match &data[..4] {
        b"nclx" if data.len() >= 11 => {
            color.cicp = Some(ColorEncoding {
                primaries: Primaries::from_u8(read_u16(data, 4).unwrap_or(2) as u8),
                transfer: TransferFunction::from_u8(read_u16(data, 6).unwrap_or(2) as u8),
                matrix: MatrixCoefficients::from_u8(read_u16(data, 8).unwrap_or(2) as u8),
                full_range: (data[10] & 0x80) != 0,
            });
        }
        b"prof" | b"rICC" => {
            color.icc = Some(data[4..].to_vec());
        }
        _ => {}
    }
    color
}

fn irot_to_orientation(steps: u8, prev: Orientation) -> Orientation {
    // irot is applied after imir; we decode them independently and combine lazily.
    // For our simple case (single irot, no prior imir), map steps directly.
    match (prev, steps) {
        (Orientation::Normal, 0) => Orientation::Normal,
        (Orientation::Normal, 1) => Orientation::Rotate270,
        (Orientation::Normal, 2) => Orientation::Rotate180,
        (Orientation::Normal, 3) => Orientation::Rotate90,
        _ => prev, // combined transforms: preserve; caller applies
    }
}

fn imir_to_orientation(axis: u8, prev: Orientation) -> Orientation {
    match (prev, axis) {
        (Orientation::Normal, 0) => Orientation::FlipH, // vertical axis
        (Orientation::Normal, 1) => Orientation::FlipV, // horizontal axis
        _ => prev,
    }
}
