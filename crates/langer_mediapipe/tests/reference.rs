//! Parity with MediaPipe's own `LanguageDetector`.
//!
//! Ignored by default because it needs external files:
//!
//! - `LANGER_MP_TEXTS`: input texts, one per line;
//! - `LANGER_MP_REFERENCE`: MediaPipe's top-3 result per line, as `code probability` pairs
//!   (e.g. from the Python `mediapipe` package with `max_results=3`).
//!
//! Run with `cargo test --release -p langer_mediapipe --test reference -- --ignored --nocapture`.

use std::fs;

use langer_mediapipe::Detector;

/// Largest probability difference expected from float rounding (e.g. XNNPACK's exp
/// approximation in softmax).
const TOLERANCE: f32 = 1e-5;

/// Share of texts allowed to exceed [`TOLERANCE`]. When lowercasing shortens a text by two
/// or more bytes (e.g. two Turkish `İ`), MediaPipe tokenizes past the end of its lowercased
/// buffer and reads uninitialized memory, which cannot be reproduced. The top-1 language
/// still has to match for those texts.
const MAX_OVER_TOLERANCE: f64 = 1e-4;

#[test]
#[ignore = "needs MediaPipe reference predictions"]
fn matches_mediapipe() {
    let path = |key: &str| std::env::var(key).unwrap_or_else(|_| panic!("{key} is not set"));
    let texts = fs::read_to_string(path("LANGER_MP_TEXTS")).expect("texts are readable");
    let reference = fs::read_to_string(path("LANGER_MP_REFERENCE")).expect("reference is readable");
    let detector = Detector::new();

    let (mut lines, mut top1, mut ranking, mut max_diff, mut over) = (0_u32, 0_u32, 0_u32, 0.0_f32, 0_u32);
    let mut examples = Vec::new();
    let mut worst: Vec<(f32, String)> = Vec::new();
    for (text, expected) in texts.lines().zip(reference.lines()) {
        lines += 1;
        let expected: Vec<(&str, f32)> = expected
            .split(' ')
            .collect::<Vec<_>>()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[code, p]| (code, p.parse().expect("probability is a number")))
            .collect();
        let got = detector.predict(text, expected.len());
        let same_labels = got.iter().map(|p| p.language).eq(expected.iter().map(|e| e.0));
        if got.first().map(|p| p.language) != expected.first().map(|e| e.0) {
            top1 += 1;
            if examples.len() < 10 {
                examples.push(format!("{text:?}: got {got:?}, expected {expected:?}"));
            }
        } else if !same_labels {
            ranking += 1;
        }
        let diff = got
            .iter()
            .zip(&expected)
            .filter(|(p, e)| p.language == e.0)
            .map(|(p, e)| (p.probability - e.1).abs())
            .fold(0.0_f32, f32::max);
        max_diff = max_diff.max(diff);
        if diff > TOLERANCE {
            over += 1;
            worst.push((diff, format!("{text:?}: got {got:?}, expected {expected:?}")));
        }
    }
    println!(
        "{lines} texts: {top1} top-1 mismatches, {ranking} top-3 order mismatches, \
         {over} over tolerance, max probability diff {max_diff:e}"
    );
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    for e in examples.iter().chain(worst.iter().take(10).map(|(_, e)| e)) {
        println!("  {e}");
    }
    assert!(lines > 0, "no reference lines");
    assert_eq!(top1, 0, "top-1 language differs from MediaPipe");
    let over_share = f64::from(over) / f64::from(lines);
    assert!(
        over_share <= MAX_OVER_TOLERANCE,
        "{over} texts differ by more than {TOLERANCE} (up to {max_diff})"
    );
}
