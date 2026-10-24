//! `Detectors` on arbitrary input.

use langer::{Detectors, Method};
use langer_mediapipe::Detector;
use lingua::Language::{English, French, Russian};
use lingua::LanguageDetectorBuilder;
use proptest::prelude::*;
use rstest::rstest;

/// Lingua sums probabilities in parallel, so repeated computations differ in the last digits.
const LINGUA_TOLERANCE: f64 = 1e-12;

/// Whether two confidences are equal up to [`LINGUA_TOLERANCE`].
fn same_confidence(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= LINGUA_TOLERANCE,
        (a, b) => a.is_none() && b.is_none(),
    }
}

/// Lingua's documented way to get the language and its confidence, which `detect` computes in
/// one pass.
fn lingua_reference(lingua: &lingua::LanguageDetector, text: &str) -> (String, Option<f64>) {
    match lingua.detect_language_of(text) {
        Some(language) => (
            language.iso_code_639_1().to_string(),
            Some(lingua.compute_language_confidence(text, language)),
        ),
        None => ("unknown".to_owned(), None),
    }
}

#[rstest]
#[case::russian("Привет, как дела?")]
#[case::english("hello there, how are you")]
#[case::ambiguous_word("test")]
#[case::belarusian_looking("удаляй")]
#[case::german("Guten Morgen zusammen")]
#[case::japanese("これはテストです")]
#[case::chat("gg wp")]
#[case::no_letters("😀 !!! 42")]
#[case::mixed("Hello, как дела? Wie geht's?")]
fn detect_with_lingua_matches_lingua_reference(#[case] text: &str) {
    let lingua = LanguageDetectorBuilder::from_all_languages().build();
    let expected = lingua_reference(&lingua, text);
    let detectors = Detectors::new(Detector::new().expect("the bundled model loads"), lingua);
    let detection = detectors.detect(text, Method::Lingua).expect("detection succeeds");
    assert_eq!(detection.language, expected.0);
    assert!(
        same_confidence(detection.confidence, expected.1),
        "{:?} vs {:?}",
        detection.confidence,
        expected.1
    );
}

proptest! {
    #[test]
    fn detect_returns_a_language_for_any_text(text in any::<String>()) {
        let lingua = LanguageDetectorBuilder::from_languages(&[English, French, Russian]).build();
        let detectors = Detectors::new(Detector::new().expect("the bundled model loads"), lingua);

        let mediapipe = detectors.detect(&text, Method::MediaPipe).expect("detection succeeds");
        prop_assert_eq!(mediapipe.method, Method::MediaPipe);
        prop_assert!(!mediapipe.language.is_empty());
        prop_assert!(mediapipe.confidence.is_some_and(|c| (0.0..=1.0).contains(&c)));

        // Lingua decides or reports `unknown` without a confidence, as its reference API does.
        let reference = lingua_reference(&LanguageDetectorBuilder::from_languages(&[English, French, Russian]).build(), &text);
        let lingua = detectors.detect(&text, Method::Lingua).expect("detection succeeds");
        prop_assert_eq!(&lingua.language, &reference.0);
        prop_assert!(same_confidence(lingua.confidence, reference.1));
        prop_assert_eq!(lingua.method, Method::Lingua);
        prop_assert_eq!(lingua.language == "unknown", lingua.confidence.is_none());
        prop_assert!(lingua.confidence.is_none_or(|c| (0.0..=1.0).contains(&c)));
    }
}

/// Letters of the languages of the test detector, and spaces.
const LETTERS: &str = "abcdefghijklmnopqrstuvwxyzéèàçâêëïôœабвгдеёжзийклмнопрстуфхцчшщъыьэюя    ";

proptest! {
    #[test]
    fn detect_with_lingua_matches_lingua_reference_on_words(
        letters in proptest::collection::vec(proptest::sample::select(LETTERS.chars().collect::<Vec<_>>()), 1..40),
    ) {
        let text: String = letters.into_iter().collect();
        let lingua = LanguageDetectorBuilder::from_languages(&[English, French, Russian]).build();
        let expected = lingua_reference(&lingua, &text);
        let detectors = Detectors::new(Detector::new().expect("the bundled model loads"), lingua);
        let detection = detectors.detect(&text, Method::Lingua).expect("detection succeeds");
        prop_assert_eq!(&detection.language, &expected.0);
        prop_assert!(same_confidence(detection.confidence, expected.1));
    }
}
