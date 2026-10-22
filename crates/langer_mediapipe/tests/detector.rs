//! Behavior of `Detector` with the bundled model.

use langer_mediapipe::Detector;

fn top(detector: &Detector, text: &str) -> String {
    detector.predict(text, 1)[0].language.to_owned()
}

#[test]
fn bundled_model_knows_110_languages() {
    let detector = Detector::new();
    let labels: Vec<&str> = detector.labels().collect();
    assert_eq!(labels.len(), 111);
    assert_eq!(labels[0], "unknown");
    for code in ["en", "ru", "uk", "de", "ja", "zh", "zh-Latn", "iw", "fil"] {
        assert!(labels.contains(&code), "{code}");
    }
}

#[test]
fn detects_common_languages() {
    let detector = Detector::new();
    let cases = [
        ("Hello, how are you doing today?", "en"),
        ("Привет, как у тебя дела сегодня?", "ru"),
        ("Привіт, як у тебе справи сьогодні?", "uk"),
        ("Hallo, wie geht es dir heute?", "de"),
        ("今日はとても良い天気ですね", "ja"),
        ("Bonjour, comment ça va aujourd'hui ?", "fr"),
    ];
    for (text, expected) in cases {
        assert_eq!(top(&detector, text), expected, "{text}");
    }
}

#[test]
fn predictions_are_a_sorted_distribution() {
    let detector = Detector::new();
    let all = detector.predict("¿Dónde está la biblioteca?", usize::MAX);
    assert_eq!(all.len(), 111);
    assert!(all.windows(2).all(|p| p[0].probability >= p[1].probability));
    let sum: f32 = all.iter().map(|p| p.probability).sum();
    assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
    assert_eq!(all[0].language, "es");
}

#[test]
fn case_does_not_matter() {
    let detector = Detector::new();
    assert_eq!(detector.predict("ПРИВЕТ МИР", 3), detector.predict("привет мир", 3));
}

#[test]
fn empty_input_or_zero_k_has_no_predictions() {
    let detector = Detector::new();
    assert!(detector.predict("", 1).is_empty());
    assert!(detector.predict("hello", 0).is_empty());
    // Whitespace is valid input, as in MediaPipe.
    assert_eq!(detector.predict(" ", 1).len(), 1);
}

#[test]
fn rejects_data_that_is_not_the_model() {
    let err = Detector::from_bytes(b"definitely not a tflite file").unwrap_err();
    assert!(err.to_string().starts_with("invalid language detector model"), "{err}");
}
