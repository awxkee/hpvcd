# hpvcd

A tiny HEVC decoder in Rust.

## Example

```rust
fn main() {
    let decoded = hpvcd::decode_heic(&bytes).unwrap();
    let diff = decoded.bit_depth.bits() - 8;
    let img = DynamicImage::ImageRgb8(
        RgbImage::from_vec(
            decoded.width,
            decoded.height,
            match decoded.pixels {
                ImageBuffer::Luma8(luma) => luma.iter().flat_map(|&x| [x, x, x]).collect(),
                ImageBuffer::Luma16(luma) => luma
                    .iter()
                    .flat_map(|&x| [(x >> diff) as u8, (x >> diff) as u8, (x >> diff) as u8])
                    .collect(),
                ImageBuffer::Rgb8(rgb) => rgb.to_vec(),
                ImageBuffer::Rgb16(rgb16) => rgb16.iter().map(|&x| (x >> diff) as u8).collect(),
            },
        )
            .unwrap(),
    );
    img.save("./out.jpg").unwrap();
}
```

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.