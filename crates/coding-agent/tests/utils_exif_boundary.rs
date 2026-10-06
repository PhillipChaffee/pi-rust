//! Boundary tests for the EXIF belt: the byte parsers' branches the
//! image-processing suite does not reach, synthesized from the TIFF/JPEG/
//! WebP layouts upstream's helpers walk.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use image::{Rgba, RgbaImage};
use pi_coding_agent::utils::exif_orientation::{apply_exif_orientation, get_exif_orientation};

const fn red() -> Rgba<u8> {
    Rgba([255, 0, 0, 255])
}

const fn blue() -> Rgba<u8> {
    Rgba([0, 0, 255, 255])
}

fn le_tiff(orientation: u16) -> Vec<u8> {
    // "II" byte order, IFD at offset 8, one entry: 0x0112 with the value.
    let mut bytes = vec![0x49, 0x49, 0x2a, 0x00, 8, 0, 0, 0, 1, 0];
    // Entry: tag, type, count, value/offset (inline SHORT).
    bytes.extend_from_slice(&0x0112u16.to_le_bytes());
    bytes.extend_from_slice(&3u16.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&orientation.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes
}

fn be_tiff(orientation: u16) -> Vec<u8> {
    // "MM" byte order, same IFD shape in big-endian fields.
    let mut bytes = vec![0x4d, 0x4d, 0x00, 0x2a, 0, 0, 0, 8, 0, 1];
    bytes.extend_from_slice(&0x0112u16.to_be_bytes());
    bytes.extend_from_slice(&3u16.to_be_bytes());
    bytes.extend_from_slice(&1u32.to_be_bytes());
    bytes.extend_from_slice(&orientation.to_be_bytes());
    bytes.extend_from_slice(&[0, 0]);
    bytes
}

fn jpeg_with_exif(tiff: &[u8]) -> Vec<u8> {
    // SOI, an APP1 segment carrying "Exif\0\0" + the TIFF, then an EOI.
    let mut jpeg = vec![0xff, 0xd8];
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(tiff);
    jpeg.extend_from_slice(&[0xff, 0xe1]);
    let length = u16::try_from(payload.len() + 2).expect("fixture length");
    jpeg.extend_from_slice(&length.to_be_bytes());
    jpeg.extend(payload);
    jpeg.push(0xff);
    jpeg.push(0xd9);
    jpeg
}

fn webp_with_exif(tiff: &[u8], prefixed: bool) -> Vec<u8> {
    // RIFF header, WEBP tag, then an EXIF chunk whose size is even-padded.
    let mut payload = b"Exif\0\0".to_vec();
    if !prefixed {
        payload.clear();
    }
    payload.extend_from_slice(tiff);
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&[0, 0, 0, 0]); // size placeholder
    bytes.extend_from_slice(b"WEBP");
    bytes.extend_from_slice(b"EXIF");
    let chunk_size = u32::try_from(payload.len()).expect("fixture length");
    bytes.extend_from_slice(&chunk_size.to_le_bytes());
    bytes.extend(payload);
    if chunk_size % 2 == 1 {
        bytes.push(0);
    }
    let riff_size = u32::try_from(bytes.len() - 8).expect("fixture length");
    bytes[4..8].copy_from_slice(&riff_size.to_le_bytes());
    bytes
}

#[test]
fn jpeg_orientation_reads_both_byte_orders() {
    assert_eq!(get_exif_orientation(&jpeg_with_exif(&le_tiff(6))), 6);
    assert_eq!(get_exif_orientation(&jpeg_with_exif(&be_tiff(8))), 8);
}

#[test]
fn jpeg_orientation_clamps_and_defaults() {
    // An orientation value outside 1..=8 reads as 1.
    assert_eq!(get_exif_orientation(&jpeg_with_exif(&le_tiff(9))), 1);
    assert_eq!(get_exif_orientation(&jpeg_with_exif(&le_tiff(0))), 1);
    // No EXIF segment: the marker walk ends at EOI.
    assert_eq!(get_exif_orientation(&[0xff, 0xd8, 0xff, 0xd9]), 1);
    // A non-EXIF APP1 (XMP) is skipped and the walk continues.
    let mut jpeg = vec![0xff, 0xd8];
    let xmp = b"http://ns.adobe.com/xap/1.0/\0<meta/>";
    jpeg.extend_from_slice(&[0xff, 0xe1]);
    let xmp_length = u16::try_from(xmp.len() + 2).expect("fixture length");
    jpeg.extend_from_slice(&xmp_length.to_be_bytes());
    jpeg.extend(xmp);
    jpeg.extend_from_slice(&jpeg_with_exif(&le_tiff(3))[2..]);
    assert_eq!(get_exif_orientation(&jpeg), 3);
    // A marker whose segment length overruns the bytes answers 1.
    let mut truncated = vec![0xff, 0xd8, 0xff, 0xe1, 0x7f, 0xff];
    truncated.extend_from_slice(b"Exif\0\0");
    assert_eq!(get_exif_orientation(&truncated), 1);
    // The walk stops at a non-marker byte.
    assert_eq!(get_exif_orientation(&[0xff, 0xd8, 0x00, 0x01]), 1);
}

#[test]
fn webp_orientation_reads_both_prefix_shapes() {
    assert_eq!(get_exif_orientation(&webp_with_exif(&le_tiff(5), true)), 5);
    assert_eq!(get_exif_orientation(&webp_with_exif(&be_tiff(7), false)), 7);
    // An EXIF chunk whose declared size overruns the bytes answers 1.
    let mut overrun = webp_with_exif(&le_tiff(6), false);
    let end = overrun.len();
    overrun.truncate(end - 4);
    assert_eq!(get_exif_orientation(&overrun), 1);
    // A RIFF body without an EXIF chunk answers 1.
    let mut plain = b"RIFF".to_vec();
    plain.extend_from_slice(&4u32.to_le_bytes());
    plain.extend_from_slice(b"WEBP");
    plain.extend_from_slice(b"VP8 ");
    plain.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(get_exif_orientation(&plain), 1);
    // Unknown chunks are skipped with even-size padding.
    assert_eq!(get_exif_orientation(&webp_with_exif(&le_tiff(2), true)), 2);
}

#[test]
fn tiff_bounds_and_jpeg_marker_edges_answer_one() {
    // An IFD offset past the bytes reads 1.
    let mut far_ifd = le_tiff(6);
    far_ifd[4..8].copy_from_slice(&9999u32.to_le_bytes());
    assert_eq!(get_exif_orientation(&jpeg_with_exif(&far_ifd)), 1);
    // An IFD entry count that overruns the bytes reads 1 — the tiff must
    // carry no orientation entry first, since a found tag answers before
    // the overrun is reached.
    let mut plain = vec![0x49, 0x49, 0x2a, 0x00, 8, 0, 0, 0, 1, 0];
    plain.extend_from_slice(&0x0100u16.to_le_bytes());
    plain.extend_from_slice(&3u16.to_le_bytes());
    plain.extend_from_slice(&1u32.to_le_bytes());
    plain.extend_from_slice(&1u16.to_le_bytes());
    plain.extend_from_slice(&0u16.to_le_bytes());
    let mut many = plain.clone();
    many[8..10].copy_from_slice(&9999u16.to_le_bytes());
    assert_eq!(get_exif_orientation(&jpeg_with_exif(&many)), 1);
    // The same shape with a real entry still answers its orientation.
    assert_eq!(get_exif_orientation(&jpeg_with_exif(&le_tiff(6))), 6);
    // Fill bytes between the marker and the segment corrupt the length
    // field, so the walk runs past the chain and answers 1.
    let mut filled = jpeg_with_exif(&le_tiff(6));
    filled.splice(4..4, [0xff, 0xff, 0xff]);
    assert_eq!(get_exif_orientation(&filled), 1);
    // A walk that runs off the end of a marker chain reads 1.
    assert_eq!(get_exif_orientation(&[0xff, 0xd8, 0xff]), 1);
}

#[test]
fn other_formats_answer_one() {
    assert_eq!(get_exif_orientation(b"plain bytes"), 1);
    assert_eq!(get_exif_orientation(&[]), 1);
}

#[test]
fn orientation_two_flips_horizontally() {
    let image = RgbaImage::from_fn(2, 1, |x, _| if x == 0 { red() } else { blue() });
    let flipped = apply_exif_orientation(image, &jpeg_with_exif(&le_tiff(2)));
    // Red moved to the right.
    assert_eq!(flipped.get_pixel(1, 0).0, [255, 0, 0, 255]);
}

#[test]
fn orientation_three_flips_both() {
    let image = RgbaImage::from_fn(2, 1, |x, _| if x == 0 { red() } else { blue() });
    let flipped = apply_exif_orientation(image, &jpeg_with_exif(&be_tiff(3)));
    assert_eq!(flipped.get_pixel(1, 0).0, [255, 0, 0, 255]);
}

#[test]
fn orientation_four_flips_vertically() {
    let image = RgbaImage::from_fn(1, 2, |_, y| if y == 0 { red() } else { blue() });
    let flipped = apply_exif_orientation(image, &jpeg_with_exif(&le_tiff(4)));
    assert_eq!(flipped.get_pixel(0, 1).0, [255, 0, 0, 255]);
}

#[test]
fn orientation_six_rotates_clockwise() {
    // 1x2: red on the top pixel; CW rotation makes it the right pixel of a
    // 2x1 image.
    let image = RgbaImage::from_fn(1, 2, |_, y| if y == 0 { red() } else { blue() });
    let rotated = apply_exif_orientation(image, &jpeg_with_exif(&le_tiff(6)));
    assert_eq!(rotated.dimensions(), (2, 1));
    assert_eq!(rotated.get_pixel(1, 0).0, [255, 0, 0, 255]);
}

#[test]
fn orientation_five_rotates_then_flips() {
    let image = RgbaImage::from_fn(1, 2, |_, y| if y == 0 { red() } else { blue() });
    let rotated = apply_exif_orientation(image, &jpeg_with_exif(&be_tiff(5)));
    assert_eq!(rotated.dimensions(), (2, 1));
    assert_eq!(rotated.get_pixel(0, 0).0, [255, 0, 0, 255]);
}

#[test]
fn orientation_eight_rotates_counterclockwise() {
    let image = RgbaImage::from_fn(1, 2, |_, y| if y == 0 { red() } else { blue() });
    let rotated = apply_exif_orientation(image, &jpeg_with_exif(&le_tiff(8)));
    assert_eq!(rotated.dimensions(), (2, 1));
    assert_eq!(rotated.get_pixel(0, 0).0, [255, 0, 0, 255]);
}

#[test]
fn orientation_seven_rotates_then_flips() {
    let image = RgbaImage::from_fn(1, 2, |_, y| if y == 0 { red() } else { blue() });
    let rotated = apply_exif_orientation(image, &jpeg_with_exif(&be_tiff(7)));
    assert_eq!(rotated.dimensions(), (2, 1));
    assert_eq!(rotated.get_pixel(1, 0).0, [255, 0, 0, 255]);
}

#[test]
fn orientation_one_is_the_identity() {
    let image = RgbaImage::new(2, 2);
    let same = apply_exif_orientation(image, b"no exif");
    assert_eq!(same.dimensions(), (2, 2));
}

#[test]
fn jpeg_marker_walk_edges_answer_one() {
    // Filler FF bytes continue the marker walk, upstream's filler loop.
    assert_eq!(get_exif_orientation(&[0xff, 0xd8, 0xff, 0xff, 0x00]), 1);
    // An APP1 marker with no segment-length bytes ends the walk.
    assert_eq!(get_exif_orientation(&[0xff, 0xd8, 0xff, 0xe1]), 1);
    // An APP1 marker whose EXIF preamble does not fit ends the walk.
    assert_eq!(
        get_exif_orientation(&[0xff, 0xd8, 0xff, 0xe1, 0x00, 0x10, 0x45]),
        1
    );
    // A segment length past the buffer walks off the chain's end.
    assert_eq!(
        get_exif_orientation(&[0xff, 0xd8, 0xff, 0x00, 0x00, 0x10]),
        1
    );
}

#[test]
fn tiff_ifd_without_the_orientation_tag_answers_one() {
    // An IFD whose single entry is not the orientation tag completes the
    // walk and falls through to the default, upstream's post-loop answer.
    let mut plain = vec![0x49, 0x49, 0x2a, 0x00, 8, 0, 0, 0, 1, 0];
    plain.extend_from_slice(&0x0100u16.to_le_bytes());
    plain.extend_from_slice(&3u16.to_le_bytes());
    plain.extend_from_slice(&1u32.to_le_bytes());
    plain.extend_from_slice(&1u16.to_le_bytes());
    plain.extend_from_slice(&0u16.to_le_bytes());
    assert_eq!(get_exif_orientation(&jpeg_with_exif(&plain)), 1);
}
