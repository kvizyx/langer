//! Accuracy of both methods on Lingua's test data (Wortschatz Leipzig): 1000 single words,
//! word pairs and sentences per language, for the languages both methods detect. The score of a
//! text type is the mean of the per-language accuracies, as in Lingua's own reports.
//!
//! Ignored by default because it needs the data: `LANGER_EVAL_DATA` is the `language-models`
//! directory of a lingua-rs checkout, with `<code>/testdata/<type>.txt` files (see README).

#![expect(clippy::expect_used, reason = "test helpers report failures by panicking")]

use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};
use std::{env, fs};

use langer::{Detectors, Method};
use langer_mediapipe::Detector;
use lingua::{Language, LanguageDetectorBuilder};

const TEXT_TYPES: [&str; 3] = ["single-words", "word-pairs", "sentences"];

#[test]
#[ignore = "needs Lingua's test data"]
fn report_accuracy() {
    let data = env::var("LANGER_EVAL_DATA").expect("LANGER_EVAL_DATA is set");
    let mediapipe = Detector::new().expect("the bundled model loads");
    let languages = shared_languages(&mediapipe);
    let detectors = Detectors::new(mediapipe, LanguageDetectorBuilder::from_all_languages().build());

    println!(
        "{} languages: {}",
        languages.len(),
        languages.iter().cloned().collect::<Vec<_>>().join(" ")
    );

    for text_type in TEXT_TYPES {
        let samples: Vec<(&str, Vec<String>)> = languages
            .iter()
            .map(|code| {
                (
                    code.as_str(),
                    read_lines(
                        &Path::new(&data)
                            .join(code)
                            .join("testdata")
                            .join(format!("{text_type}.txt")),
                    ),
                )
            })
            .collect();
        let count: usize = samples.iter().map(|(_, lines)| lines.len()).sum();
        assert!(count > 0, "no {text_type} in {data}");

        println!("\n{text_type} ({count} texts):");
        for method in [Method::MediaPipe, Method::Lingua] {
            let mut elapsed = Duration::ZERO;
            let accuracies: Vec<f64> = samples
                .iter()
                .map(|(code, lines)| {
                    let started = Instant::now();
                    let correct = lines
                        .iter()
                        .filter(|text| {
                            let detection = detectors.detect(text, method).expect("detection succeeds");
                            iso_639_1(&detection.language) == *code
                        })
                        .count();
                    elapsed += started.elapsed();

                    ratio(correct, lines.len())
                })
                .collect();

            let mean = accuracies.iter().sum::<f64>() / ratio(accuracies.len(), 1) * 100.0;
            let per_text = elapsed.as_secs_f64() * 1e6 / ratio(count, 1);
            let name = method.to_string();
            println!("  {name:<10} {mean:5.1}%   {per_text:7.1} µs per text");
        }
    }
}

/// ISO 639-1 codes of the languages both methods detect.
fn shared_languages(mediapipe: &Detector) -> BTreeSet<String> {
    let mediapipe: BTreeSet<&str> = mediapipe.labels().map(iso_639_1).collect();

    Language::all()
        .into_iter()
        .map(|language| language.iso_code_639_1().to_string())
        .filter(|code| mediapipe.contains(code.as_str()))
        .collect()
}

/// ISO 639-1 code of a reported language: MediaPipe uses `iw`, `fil` and `xx-Latn` codes.
fn iso_639_1(code: &str) -> &str {
    match code.split('-').next().unwrap_or(code) {
        "iw" => "he",
        "fil" => "tl",
        code => code,
    }
}

fn read_lines(path: &Path) -> Vec<String> {
    let text = fs::read_to_string(path).expect("test data is readable");

    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_owned)
        .collect()
}

#[expect(clippy::cast_precision_loss, reason = "sample counts are far below 2^52")]
fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}
