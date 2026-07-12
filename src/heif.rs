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

use crate::color::{Cicp, ColorMetadata, MatrixCoefficients, Primaries, TransferFunction};
use crate::error::DecodeError;
use crate::limits::ParseLimits;
use crate::metadata::{CleanAperture, ContentLightLevel, Orientation, PixelAspectRatio};
use std::collections::HashMap;

const APPLE_HDR_GAIN_MAP_URN: &[u8] = b"urn:com:apple:photo:2020:aux:hdrgainmap";

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
    /// color metadata.
    pub(crate) color: ColorMetadata,
    pub(crate) orientation: Orientation,
    pub(crate) cll: Option<ContentLightLevel>,
    /// Clean aperture (`clap` property), if present.
    pub(crate) clap: Option<CleanAperture>,
    /// Pixel aspect ratio (`pasp` property), if present.
    pub(crate) pasp: Option<PixelAspectRatio>,
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

/// Coded image backing an auxiliary item. Apple gain maps may be a single
/// `hvc1` item or a tiled `grid`, just like the primary image.
#[derive(Debug)]
pub(crate) enum HeifImageSource {
    Item(HeifItem),
    Grid(GridInfo),
}

/// Apple HDR gain-map auxiliary image and its opaque, directly associated
/// metadata item. Current Apple files normally store an XMP packet here; the
/// container layer intentionally does not parse that payload.
#[derive(Debug)]
pub(crate) struct HeifGainMap {
    pub(crate) image: HeifImageSource,
    pub(crate) metadata: Option<Vec<u8>>,
}

/// Controls which auxiliary image descriptions are materialized for decode.
/// Auxiliary URNs are still detected when loading is disabled.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HeifParseOptions {
    pub(crate) load_alpha: bool,
    pub(crate) load_gain_map: bool,
}

/// Top-level parse result.
pub(crate) struct HeifFile {
    /// Primary (color) image.  For grid files this is a placeholder pointing
    /// at the first tile so existing single-image code keeps compiling; use
    /// `grid` for the full picture.
    pub(crate) primary: HeifItem,
    /// Whether a recognized alpha auxiliary item is present and has HEVC
    /// configuration. The full item is loaded only when requested.
    pub(crate) has_alpha: bool,
    /// Alpha auxiliary item loaded for decoding, if requested and present.
    pub(crate) alpha: Option<HeifItem>,
    /// Whether an Apple HDR gain-map auxiliary item is present.
    pub(crate) has_gain_map: bool,
    /// Gain-map auxiliary image loaded for decoding, if requested and present.
    pub(crate) gain_map: Option<HeifGainMap>,
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
    /// Reject any box whose declared size exceeds this many bytes. `u64::MAX`
    /// disables the cap (used where a limit is not threaded through).
    max_box_size: u64,
}

impl<'a> Boxes<'a> {
    /// Like [`Boxes::new`] but rejects boxes larger than `max_box_size` bytes.
    fn with_limit(data: &'a [u8], max_box_size: u64) -> Self {
        Boxes {
            data,
            pos: 0,
            max_box_size,
        }
    }
}

struct BoxEntry<'a> {
    fourcc: [u8; 4],
    payload: &'a [u8], // payload excluding the 8-byte header (size+fourcc)
}

#[derive(Clone, Copy, Debug)]
struct FullBox<'a> {
    version: u8,
    #[allow(dead_code)]
    flags: u32,
    body: &'a [u8],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ItemExtent {
    offset: u64,
    length: u64,
}

type ItemExtents = HashMap<u16, ItemExtent>;
type PropertyAssociations = HashMap<u16, Vec<u8>>;

#[derive(Debug)]
struct ItemProperties {
    properties: Vec<Prop>,
    associations: PropertyAssociations,
}

#[derive(Clone, Copy, Debug)]
struct GridDescriptor {
    rows: u32,
    cols: u32,
    output_width: u32,
    output_height: u32,
}

impl Default for GridDescriptor {
    fn default() -> Self {
        Self {
            rows: 1,
            cols: 1,
            output_width: 0,
            output_height: 0,
        }
    }
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
        // Granular box-size limit: a box claiming to be larger than the
        // configured cap is treated as invalid and stops iteration.
        if size as u64 > self.max_box_size {
            return None;
        }
        let fourcc: [u8; 4] = self.data[self.pos + 4..self.pos + 8].try_into().ok()?;
        let payload = &self.data[self.pos + 8..self.pos + size];
        self.pos += size;
        Some(BoxEntry { fourcc, payload })
    }
}

fn fullbox_header(payload: &[u8]) -> Option<FullBox<'_>> {
    if payload.len() < 4 {
        return None;
    }
    Some(FullBox {
        version: payload[0],
        flags: u32::from_be_bytes([0, payload[1], payload[2], payload[3]]),
        body: &payload[4..],
    })
}

pub(crate) fn parse(
    file: &[u8],
    limits: &ParseLimits,
    options: HeifParseOptions,
) -> Result<HeifFile, DecodeError> {
    let mbs = limits.max_box_size;
    // Verify ftyp
    let mut has_heic = false;
    for b in Boxes::with_limit(file, mbs) {
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
    let meta_payload = Boxes::with_limit(file, mbs)
        .find(|b| &b.fourcc == b"meta")
        .map(|b| b.payload)
        .ok_or(DecodeError::MissingBox("meta"))?;
    // meta is a FullBox
    let meta = fullbox_header(meta_payload)
        .map(|header| header.body)
        .ok_or(DecodeError::MissingBox("meta body"))?;

    // Parse sub-boxes of meta
    let mut pitm: u16 = 1;
    let mut iloc_data: Option<&[u8]> = None;
    let mut iinf_data: Option<&[u8]> = None;
    let mut iref_data: Option<&[u8]> = None;
    let mut iprp_data: Option<&[u8]> = None;

    // Track `idat` so we can resolve construction_method=1 iloc entries.
    let mut idat_file_offset: u64 = 0;

    for b in Boxes::with_limit(meta, mbs) {
        match &b.fourcc {
            b"pitm" => {
                if let Some(header) = fullbox_header(b.payload) {
                    pitm = if header.version == 1 {
                        read_u32(header.body, 0).map(|v| v as u16).unwrap_or(pitm)
                    } else {
                        read_u16(header.body, 0).unwrap_or(pitm)
                    };
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

    // Parse iloc: collect each item's absolute file range.
    let mut extents = ItemExtents::new();
    if let Some(header) = iloc_data.and_then(fullbox_header) {
        parse_iloc(
            header.body,
            header.version,
            idat_file_offset,
            &mut extents,
            limits,
        );
    }

    // Parse iinf: item types
    let mut item_types: HashMap<u16, [u8; 4]> = HashMap::new();
    if let Some(header) = iinf_data.and_then(fullbox_header) {
        let rest = header.body;
        let count = (read_u16(rest, 0).unwrap_or(0) as usize).min(limits.max_items);
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
    // (metadata describing an image), and dimg (grid tiles) references.
    let mut auxl_items: Vec<u16> = Vec::new();
    // cdsc: from=metadata item id -> to=[described image ids]
    let mut cdsc_map: HashMap<u16, Vec<u16>> = HashMap::new();
    // dimg: from=grid_item_id → to=[tile1, tile2, ...]
    let mut dimg_map: HashMap<u16, Vec<u16>> = HashMap::new();
    if let Some(header) = iref_data.and_then(fullbox_header) {
        let rest = header.body;
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
                let n = read_u16(rest, pos + 10).unwrap_or(0) as usize;
                let to: Vec<u16> = (0..n)
                    .filter_map(|i| read_u16(rest, pos + 12 + i * 2))
                    .collect();
                cdsc_map.insert(from_id, to);
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
    let ItemProperties {
        properties: props,
        associations: prop_assoc,
    } = if let Some(iprp) = iprp_data {
        parse_iprp(iprp, limits)
    } else {
        ItemProperties {
            properties: Vec::new(),
            associations: PropertyAssociations::new(),
        }
    };

    // Detect if the primary item is a grid
    let primary_type = item_types.get(&pitm).copied().unwrap_or(*b"hvc1");
    let is_grid = &primary_type == b"grid";

    // Build primary and optional grid.
    // For grid files `primary` is the first tile so single-image code still works.
    let (primary, grid) = if is_grid {
        let g = parse_grid_item(pitm, &extents, &dimg_map, &props, &prop_assoc, file, limits)?;
        let first = g
            .tiles
            .first()
            .cloned()
            .unwrap_or_else(|| build_fallback_item(pitm));
        (first, Some(g))
    } else {
        let p = build_item(pitm, &extents, &props, &prop_assoc, false, file, limits)?;
        (p, None)
    };
    // Among all auxiliary items, keep only the one whose `auxC` type is genuine
    // alpha. iPhone HEICs attach many `auxl` items (HDR gain map, depth, portrait
    // matte, …); treating one of those as an alpha plane would corrupt the output.
    let alpha_item = auxl_items
        .iter()
        .copied()
        .find(|&id| item_is_alpha(id, &props, &prop_assoc));
    let has_alpha = alpha_item
        .map(|item_id| item_has_property(item_id, b"hvcC", &props, &prop_assoc))
        .unwrap_or(false);
    let alpha = if options.load_alpha {
        alpha_item
            .and_then(|aid| build_item(aid, &extents, &props, &prop_assoc, true, file, limits).ok())
    } else {
        None
    };

    // Apple HDR gain maps use a dedicated auxC URN. The auxiliary itself may
    // be a regular hvc1 item or a tiled grid (modern iPhone files commonly use
    // the latter), so preserve that distinction for the decode layer.
    let gain_map_item = auxl_items
        .iter()
        .copied()
        .find(|&id| item_is_gain_map(id, &props, &prop_assoc));
    let has_gain_map = gain_map_item.is_some();
    let gain_map = if options.load_gain_map {
        if let Some(gain_id) = gain_map_item {
            let image = if item_types.get(&gain_id) == Some(b"grid") {
                parse_grid_item(
                    gain_id,
                    &extents,
                    &dimg_map,
                    &props,
                    &prop_assoc,
                    file,
                    limits,
                )
                .ok()
                .map(HeifImageSource::Grid)
            } else {
                build_item(gain_id, &extents, &props, &prop_assoc, false, file, limits)
                    .ok()
                    .map(HeifImageSource::Item)
            };
            if let Some(image) = image {
                let metadata =
                    associated_metadata(gain_id, &cdsc_map, &item_types, &extents, file, limits)?;
                Some(HeifGainMap { image, metadata })
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Select an actual Exif item describing the primary image. Previously the
    // parser treated the last cdsc source as Exif, which could expose an XMP
    // packet instead on files carrying gain-map metadata.
    let exif = cdsc_map
        .iter()
        .find_map(|(&metadata_id, targets)| {
            (item_types.get(&metadata_id) == Some(b"Exif") && targets.contains(&pitm))
                .then_some(metadata_id)
        })
        .map(|eid| extract_item_bytes(eid, &extents, file, limits, true))
        .transpose()?
        .flatten();

    Ok(HeifFile {
        primary,
        has_alpha,
        alpha,
        has_gain_map,
        gain_map,
        exif,
        grid,
    })
}

fn extract_item_bytes(
    item_id: u16,
    extents: &ItemExtents,
    file: &[u8],
    limits: &ParseLimits,
    strip_exif_offset: bool,
) -> Result<Option<Vec<u8>>, DecodeError> {
    let Some(extent) = extents.get(&item_id) else {
        return Ok(None);
    };
    let ItemExtent {
        offset: off,
        length: len,
    } = *extent;
    let prefix = if strip_exif_offset { 4 } else { 0 };
    if len < prefix as u64 {
        return Ok(None);
    }
    let payload_len = len - prefix as u64;
    if payload_len > limits.max_exif_size as u64 {
        return Err(DecodeError::LimitExceeded {
            what: if strip_exif_offset {
                "exif payload"
            } else {
                "gain-map metadata"
            },
            value: payload_len,
            limit: limits.max_exif_size as u64,
        });
    }
    let Some(end) = off.checked_add(len) else {
        return Ok(None);
    };
    if end > file.len() as u64 {
        return Ok(None);
    }
    let start = off as usize + prefix;
    Ok(Some(file[start..end as usize].to_vec()))
}

fn associated_metadata(
    image_id: u16,
    cdsc_map: &HashMap<u16, Vec<u16>>,
    item_types: &HashMap<u16, [u8; 4]>,
    extents: &ItemExtents,
    file: &[u8],
    limits: &ParseLimits,
) -> Result<Option<Vec<u8>>, DecodeError> {
    // Prefer MIME/XMP, which is how Apple stores HDRGainMapVersion/headroom,
    // but accept another opaque metadata item if that is all the file has.
    // Sorting keeps the choice deterministic if a file links several items of
    // the same type to the gain map.
    let mut candidates: Vec<u16> = cdsc_map
        .iter()
        .filter_map(|(&metadata_id, targets)| targets.contains(&image_id).then_some(metadata_id))
        .collect();
    candidates.sort_unstable_by_key(|&metadata_id| {
        (item_types.get(&metadata_id) != Some(b"mime"), metadata_id)
    });

    for metadata_id in candidates {
        if let Some(bytes) = extract_item_bytes(metadata_id, extents, file, limits, false)? {
            return Ok(Some(bytes));
        }
    }
    Ok(None)
}

fn parse_iloc(
    data: &[u8],
    version: u8,
    idat_file_offset: u64,
    out: &mut ItemExtents,
    limits: &ParseLimits,
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
    // advances (e.g. all sizes are 0), and to honour the configured limit.
    let item_count = item_count.min(limits.max_items);
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
        let extent_count = (extent_count as usize).min(limits.max_extents_per_item);
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
            out.insert(
                item_id,
                ItemExtent {
                    offset: abs_offset,
                    length: ext_length,
                },
            );
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

fn parse_iprp(iprp: &[u8], limits: &ParseLimits) -> ItemProperties {
    let mut properties: Vec<Prop> = vec![Prop::default()]; // 1-indexed
    let mut associations = PropertyAssociations::new();
    let mbs = limits.max_box_size;

    for b in Boxes::with_limit(iprp, mbs) {
        if &b.fourcc == b"ipco" {
            for pb in Boxes::with_limit(b.payload, mbs) {
                // Bound the number of properties we accumulate; treat max_items
                // as the ceiling since each item can associate several.
                if properties.len() >= limits.max_items.saturating_mul(4).max(64) {
                    break;
                }
                properties.push(Prop {
                    kind: pb.fourcc,
                    data: pb.payload.to_vec(),
                });
            }
        }
        if &b.fourcc == b"ipma"
            && let Some(header) = fullbox_header(b.payload)
        {
            let rest = header.body;
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
                associations.insert(item_id, indices);
            }
        }
    }
    ItemProperties {
        properties,
        associations,
    }
}

fn item_has_property(
    item_id: u16,
    kind: &[u8; 4],
    props: &[Prop],
    prop_assoc: &PropertyAssociations,
) -> bool {
    prop_assoc.get(&item_id).is_some_and(|indices| {
        indices.iter().any(|&property_index| {
            let property_index = property_index as usize;
            property_index != 0
                && props
                    .get(property_index)
                    .is_some_and(|property| &property.kind == kind)
        })
    })
}

fn item_is_alpha(item_id: u16, props: &[Prop], prop_assoc: &PropertyAssociations) -> bool {
    let aux_type = item_aux_type(item_id, props, prop_assoc);
    aux_type == Some(b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha")
        || aux_type == Some(b"urn:mpeg:hevc:2015:auxid:1")
}

fn item_is_gain_map(item_id: u16, props: &[Prop], prop_assoc: &PropertyAssociations) -> bool {
    item_aux_type(item_id, props, prop_assoc) == Some(APPLE_HDR_GAIN_MAP_URN)
}

fn item_aux_type<'a>(
    item_id: u16,
    props: &'a [Prop],
    prop_assoc: &PropertyAssociations,
) -> Option<&'a [u8]> {
    let indices = prop_assoc.get(&item_id)?;
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
        return Some(&urn_bytes[..end]);
    }
    None
}

/// Read orientation (irot/imir) for a given item from ipco/ipma — without
/// requiring the item to have a hvcC property.  Used for grid items.
fn read_item_orientation(
    item_id: u16,
    props: &[Prop],
    prop_assoc: &PropertyAssociations,
) -> Orientation {
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

/// Read the compact `grid` item descriptor.
///
/// The grid item's data follows ISO/IEC 23008-12 §6.6.2.3.2:
/// ```text
/// uint8  version (= 0)
/// uint8  flags            bit 0 = 1 → 32-bit dimensions, 0 → 16-bit
/// uint8  rows_minus_one
/// uint8  columns_minus_one
/// uint16/uint32  output_width
/// uint16/uint32  output_height
/// ```
fn read_grid_descriptor(grid_id: u16, extents: &ItemExtents, file: &[u8]) -> GridDescriptor {
    let Some(extent) = extents.get(&grid_id).copied() else {
        return GridDescriptor::default();
    };
    let Ok(start) = usize::try_from(extent.offset) else {
        return GridDescriptor::default();
    };
    let Some(end_offset) = extent.offset.checked_add(extent.length) else {
        return GridDescriptor::default();
    };
    let Ok(end) = usize::try_from(end_offset) else {
        return GridDescriptor::default();
    };
    let Some(data) = file.get(start..end).filter(|data| data.len() >= 4) else {
        return GridDescriptor::default();
    };

    let rows = data[2] as u32 + 1;
    let cols = data[3] as u32 + 1;
    let (output_width, output_height) = if data[1] & 1 != 0 {
        (
            read_u32(data, 4).unwrap_or(0),
            read_u32(data, 8).unwrap_or(0),
        )
    } else {
        (
            read_u16(data, 4).unwrap_or(0) as u32,
            read_u16(data, 6).unwrap_or(0) as u32,
        )
    };

    GridDescriptor {
        rows,
        cols,
        output_width,
        output_height,
    }
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
    extents: &ItemExtents,
    dimg_map: &HashMap<u16, Vec<u16>>,
    props: &[Prop],
    prop_assoc: &PropertyAssociations,
    file: &[u8],
    limits: &ParseLimits,
) -> Result<GridInfo, DecodeError> {
    let descriptor = read_grid_descriptor(grid_id, extents, file);

    // Reject oversized grid geometry up front (image-size limit).
    limits.check_image(descriptor.output_width, descriptor.output_height)?;

    // Build a HeifItem for each tile (in row-major order from dimg references),
    // capping the number of tiles enumerated.
    let mut tile_ids = dimg_map.get(&grid_id).cloned().unwrap_or_default();
    if tile_ids.len() > limits.max_tiles {
        return Err(DecodeError::LimitExceeded {
            what: "grid tiles",
            value: tile_ids.len() as u64,
            limit: limits.max_tiles as u64,
        });
    }
    // A grid never needs more tiles than rows*cols; drop any stragglers so a
    // bogus dimg list can't inflate work beyond the declared layout.
    let max_needed = (descriptor.rows as usize).saturating_mul(descriptor.cols as usize);
    tile_ids.truncate(max_needed);
    let tiles: Vec<HeifItem> = tile_ids
        .iter()
        .filter_map(|&tid| build_item(tid, extents, props, prop_assoc, false, file, limits).ok())
        .collect();

    // Read the grid item's orientation directly from its ipma/ipco associations.
    // We can't use build_item() here because grid items have no hvcC property
    // and build_item() returns Err for items without hvcC.
    let orientation = read_item_orientation(grid_id, props, prop_assoc);

    Ok(GridInfo {
        rows: descriptor.rows,
        cols: descriptor.cols,
        output_width: descriptor.output_width,
        output_height: descriptor.output_height,
        tiles,
        orientation,
    })
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
        orientation: Orientation::Normal,
        cll: None,
        clap: None,
        pasp: None,
        _is_alpha: false,
    }
}

fn build_item(
    item_id: u16,
    extents: &ItemExtents,
    props: &[Prop],
    prop_assoc: &PropertyAssociations,
    is_alpha: bool,
    _file: &[u8],
    limits: &ParseLimits,
) -> Result<HeifItem, DecodeError> {
    let ItemExtent { offset, length } = extents
        .get(&item_id)
        .copied()
        .ok_or(DecodeError::MissingBox("iloc entry for item"))?;

    // Tag/data-size limit: reject an item whose declared coded size is absurd
    // before anyone tries to slice the file with it.
    if length > limits.max_item_size {
        return Err(DecodeError::LimitExceeded {
            what: "item data size",
            value: length,
            limit: limits.max_item_size,
        });
    }
    let mut hvcc = Vec::new();
    let mut display_w = 0u32;
    let mut display_h = 0u32;
    let mut color = ColorMetadata::default(); // starts empty; filled from colr box(es)
    let mut orientation = Orientation::Normal;
    let mut cll = None;
    let mut clap: Option<CleanAperture> = None;
    let mut pasp: Option<PixelAspectRatio> = None;

    if let Some(indices) = prop_assoc.get(&item_id) {
        for &pidx in indices {
            let pidx = pidx as usize;
            if pidx == 0 || pidx >= props.len() {
                continue;
            }
            let p = &props[pidx];
            match &p.kind {
                b"hvcC" => {
                    if p.data.len() > limits.max_hvcc_size {
                        return Err(DecodeError::LimitExceeded {
                            what: "hvcC size",
                            value: p.data.len() as u64,
                            limit: limits.max_hvcc_size as u64,
                        });
                    }
                    hvcc = p.data.clone();
                }
                b"ispe" => {
                    if p.data.len() >= 12 {
                        display_w = read_u32(&p.data, 4).unwrap_or(0);
                        display_h = read_u32(&p.data, 8).unwrap_or(0);
                        // Reject oversized declared geometry at parse time.
                        limits.check_image(display_w, display_h)?;
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
                b"clap" if p.data.len() >= 32 => {
                    // CleanApertureBox (ISO 14496-12 §12.1.4.2):
                    //   u32 cleanApertureWidthN
                    //   u32 cleanApertureWidthD
                    //   u32 cleanApertureHeightN
                    //   u32 cleanApertureHeightD
                    //   i32 horizOffN   (signed)
                    //   u32 horizOffD
                    //   i32 vertOffN    (signed)
                    //   u32 vertOffD
                    let width_n = read_u32(&p.data, 0).unwrap_or(0);
                    let width_d = read_u32(&p.data, 4).unwrap_or(1);
                    let height_n = read_u32(&p.data, 8).unwrap_or(0);
                    let height_d = read_u32(&p.data, 12).unwrap_or(1);
                    let horiz_off_n =
                        i32::from_be_bytes(p.data[16..20].try_into().unwrap_or([0; 4]));
                    let horiz_off_d = read_u32(&p.data, 20).unwrap_or(1);
                    let vert_off_n =
                        i32::from_be_bytes(p.data[24..28].try_into().unwrap_or([0; 4]));
                    let vert_off_d = read_u32(&p.data, 28).unwrap_or(1);
                    clap = Some(CleanAperture {
                        width_n,
                        width_d,
                        height_n,
                        height_d,
                        horiz_off_n,
                        horiz_off_d,
                        vert_off_n,
                        vert_off_d,
                    });
                }
                b"pasp" if p.data.len() >= 8 => {
                    // PixelAspectRatioBox (ISO 14496-12 §12.1.4.4):
                    //   u32 hSpacing
                    //   u32 vSpacing
                    let h = read_u32(&p.data, 0).unwrap_or(1);
                    let v = read_u32(&p.data, 4).unwrap_or(1);
                    // Clamp zero to 1 so callers never divide by zero.
                    pasp = Some(PixelAspectRatio {
                        h_spacing: h.max(1),
                        v_spacing: v.max(1),
                    });
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
        clap,
        pasp,
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
            color.cicp = Some(Cicp {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one ISOBMFF box: 4-byte big-endian size, 4-byte fourcc, `payload`.
    fn boxed(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = 8 + payload.len();
        let mut v = (size as u32).to_be_bytes().to_vec();
        v.extend_from_slice(fourcc);
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn box_size_limit_stops_iteration() {
        // First box is 12 bytes; with a 10-byte cap it is rejected immediately.
        let data = boxed(b"ftyp", b"heic"); // 8 + 4 = 12 bytes
        let seen: Vec<_> = Boxes::with_limit(&data, 10).collect();
        assert!(seen.is_empty(), "oversized box should be rejected");
        // With a generous cap the same box is accepted.
        let seen2: Vec<_> = Boxes::with_limit(&data, 4096).collect();
        assert_eq!(seen2.len(), 1);
    }

    #[test]
    fn non_heif_input_is_rejected() {
        let limits = ParseLimits::default();
        let junk = boxed(b"ftyp", b"mp42");
        assert!(matches!(
            parse(
                &junk,
                &limits,
                HeifParseOptions {
                    load_alpha: true,
                    load_gain_map: true,
                },
            ),
            Err(DecodeError::NotHeif)
        ));
    }

    #[test]
    fn item_size_limit_rejects_large_item() {
        // build_item should reject an item whose iloc length exceeds the cap.
        let mut limits = ParseLimits::default();
        limits.max_item_size = 100;
        let mut extents = ItemExtents::new();
        extents.insert(
            1u16,
            ItemExtent {
                offset: 0,
                length: 200,
            },
        ); // 200 > 100
        let props: Vec<Prop> = vec![Prop::default()];
        let assoc = PropertyAssociations::new();
        let r = build_item(1, &extents, &props, &assoc, false, &[], &limits);
        assert!(matches!(
            r,
            Err(DecodeError::LimitExceeded {
                what: "item data size",
                ..
            })
        ));
    }

    #[test]
    fn apple_gain_map_aux_type_is_detected() {
        let mut auxc = vec![0, 0, 0, 0];
        auxc.extend_from_slice(APPLE_HDR_GAIN_MAP_URN);
        auxc.push(0);
        let props = vec![
            Prop::default(),
            Prop {
                kind: *b"auxC",
                data: auxc,
            },
        ];
        let assoc = PropertyAssociations::from([(7u16, vec![1u8])]);

        assert!(item_is_gain_map(7, &props, &assoc));
        assert!(!item_is_alpha(7, &props, &assoc));
    }

    #[test]
    fn gain_map_metadata_prefers_linked_mime_item() {
        let file = b"xmp!fallback";
        let cdsc = HashMap::from([(9u16, vec![7u16]), (10u16, vec![7u16])]);
        let item_types = HashMap::from([(9u16, *b"mime"), (10u16, *b"Exif")]);
        let extents = ItemExtents::from([
            (
                9u16,
                ItemExtent {
                    offset: 0,
                    length: 4,
                },
            ),
            (
                10u16,
                ItemExtent {
                    offset: 4,
                    length: 8,
                },
            ),
        ]);

        let metadata = associated_metadata(
            7,
            &cdsc,
            &item_types,
            &extents,
            file,
            &ParseLimits::default(),
        )
        .unwrap();
        assert_eq!(metadata.as_deref(), Some(&b"xmp!"[..]));
    }
}
