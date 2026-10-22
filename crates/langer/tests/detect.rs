//! `Detectors` on arbitrary input.

use langer::{Detectors, Method};
use langer_mediapipe::Detector;
use lingua::Language::{English, French, Russian};
use lingua::LanguageDetectorBuilder;
use proptest::prelude::*;

proptest! {
    #[test]
    fn detect_returns_a_language_for_any_text(text in any::<String>()) {
        let lingua = LanguageDetectorBuilder::from_languages(&[English, French, Russian]).build();
        let detectors = Detectors::new(Detector::new().expect("the bundled model loads"), lingua);

        let mediapipe = detectors.detect(&text, Method::MediaPipe).expect("detection succeeds");
        prop_assert_eq!(mediapipe.method, Method::MediaPipe);
        prop_assert!(!mediapipe.language.is_empty());
        prop_assert!(mediapipe.confidence.is_some_and(|c| (0.0..=1.0).contains(&c)));

        // Lingua decides or reports `unknown` without a confidence.
        let lingua = detectors.detect(&text, Method::Lingua).expect("detection succeeds");
        prop_assert_eq!(lingua.method, Method::Lingua);
        prop_assert_eq!(lingua.language == "unknown", lingua.confidence.is_none());
        prop_assert!(lingua.confidence.is_none_or(|c| (0.0..=1.0).contains(&c)));
    }
}
