//! Accuracy of both methods on a labelled dataset.
//!
//! Ignored by default because it needs external files:
//!
//! - `LANGER_EVAL_TSV`: lines of `expected_code<TAB>text`, with ISO 639-1 codes.
//!
//! Run with `cargo test --release -p langer --test accuracy -- --ignored --nocapture`.

use std::fs;
use std::time::Instant;

use langer::{Detectors, Method};
use langer_mediapipe::Detector;
use lingua::LanguageDetectorBuilder;

#[test]
#[ignore = "needs a real model and a labelled dataset"]
fn report_accuracy() {
    let path = |key: &str| std::env::var(key).unwrap_or_else(|_| panic!("{key} is not set"));
    let detectors = Detectors::new(Detector::new(), LanguageDetectorBuilder::from_all_languages().build());
    let data = fs::read_to_string(path("LANGER_EVAL_TSV")).expect("dataset is readable");
    let rows: Vec<(&str, &str)> = data.lines().filter_map(|l| l.split_once('\t')).collect();
    assert!(!rows.is_empty(), "dataset is empty");

    for method in [Method::Mediapipe, Method::Lingua] {
        let started = Instant::now();
        let correct = rows
            .iter()
            .filter(|(expected, text)| {
                let detection = detectors.detect(text, method);
                let language = iso_639_1(&detection.language);
                // Lingua reports Norwegian as Bokmål or Nynorsk.
                language == *expected || (matches!(language, "nb" | "nn") && *expected == "no")
            })
            .count();
        #[expect(clippy::cast_precision_loss, reason = "row counts are small")]
        let accuracy = correct as f64 / rows.len() as f64 * 100.0;
        println!(
            "{method:?}: {accuracy:.2}% of {} rows in {:?}",
            rows.len(),
            started.elapsed()
        );
    }
}

/// ISO 639-1 code of a reported language: MediaPipe uses `iw`, `fil` and `xx-Latn` codes.
fn iso_639_1(code: &str) -> &str {
    match code.split('-').next().unwrap_or(code) {
        "iw" => "he",
        "fil" => "tl",
        code => code,
    }
}
