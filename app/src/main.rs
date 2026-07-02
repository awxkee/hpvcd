use hpvcd::ImageBuffer;
use image::{DynamicImage, RgbImage};
use std::fs;
use std::time::Instant;

fn main() {
    let bytes = fs::read("./assets/old-safe-wall.heic").unwrap();
    let mut durations = Vec::with_capacity(20);
    for i in 0..20 {
        let instant = Instant::now();
        let decoded = hpvcd::decode_heic(&bytes).unwrap();
        let elapsed = instant.elapsed();
        println!("Iteration {i}: {:?}", elapsed);
        durations.push(elapsed);
    }

    let total: std::time::Duration = durations.iter().sum();
    let avg = total / durations.len() as u32;
    println!("Average: {:?}", avg);
    let instant = Instant::now();
    let decoded = hpvcd::decode_heic(&bytes).unwrap();
    println!("Decoded: {:?}", instant.elapsed());
    let decoded_yuv = hpvcd::decode_heic_yuv(&bytes).unwrap();
    println!("Decoded WxH {:?}x{:?}", decoded.width, decoded.height);
    println!(
        "Decoded YUV WxH {:?}x{:?}",
        decoded_yuv.width, decoded_yuv.height
    );
    println!("Decoded: {:?}", instant.elapsed());
    println!("Decoded: {:?}", decoded_yuv.bit_depth);
    println!("Decoded: {:?}", decoded_yuv.orientation);
    println!(
        "bit depth {}, orient {:?}",
        decoded.bit_depth as u8, decoded.orientation
    );
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
