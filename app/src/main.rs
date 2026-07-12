mod gainmap;

use hpvcd::{ImageBuffer, VideoDecoder};
use image::{DynamicImage, RgbImage};
use std::fs;
use std::time::Instant;

fn main() {
    // let bytes = fs::read("./assets/file_example_MP4_480_1_5MG.265").unwrap();
    // let mut video_decoder = VideoDecoder::new();
    // let instant = Instant::now();
    // let decoded_frame = video_decoder
    //     .decode_frame_at_fps(&bytes, 1., 24.)
    //     .unwrap()
    //     .unwrap();
    // let elapsed = instant.elapsed();
    // println!("Video {:?}", elapsed);
    // let rgb_image = decoded_frame.to_rgb8();
    // let iamge = RgbImage::from_vec(
    //     decoded_frame.width() as u32,
    //     decoded_frame.height() as u32,
    //     rgb_image.to_vec(),
    // )
    // .unwrap();
    // iamge.save("./out_v.jpg").unwrap();

    let bytes = fs::read("./assets/IMG_0073.HEIC").unwrap();
    let mut durations = Vec::with_capacity(20);
    for i in 0..20 {
        let instant = Instant::now();
        let decoded = hpvcd::decode_heic(&bytes).unwrap();
        let elapsed = instant.elapsed();
        if let Some(metadata) = decoded
            .gain_map
            .and_then(|x| x.metadata)
            .and_then(|x| String::from_utf8(x).ok())
        {
            println!("{:?}", metadata);
        }
        println!("Iteration {i}: {:?} w:{}", elapsed, decoded.width);
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

    let bytes1 = fs::read("./assets/ADJUST_IPRED_ANGLE_A_RExt_Mitsubishi_2.bit").unwrap();
    let decoded = hpvcd::decode_hevc(&bytes1).unwrap();
    let rgb_image = decoded[0].to_rgb8();
    let iamge = RgbImage::from_vec(
        decoded[0].width() as u32,
        decoded[0].height() as u32,
        rgb_image.to_vec(),
    )
    .unwrap();
    iamge.save("./out_v.jpg").unwrap();
}
