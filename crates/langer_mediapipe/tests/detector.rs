//! Behavior of `Detector` with the bundled model.

use langer_mediapipe::Detector;
use proptest::prelude::*;
use rstest::rstest;

/// Languages of the bundled model and `unknown`.
const LABELS: usize = 111;

#[test]
fn labels_are_unknown_and_110_languages() {
    let detector = Detector::new().expect("the bundled model loads");
    let labels: Vec<&str> = detector.labels().collect();
    assert_eq!(labels.len(), LABELS);
    assert_eq!(labels[0], "unknown");
}

#[test]
fn heap_size_is_close_to_the_model_size() {
    // Weights are kept quantized, as in the ~300 KiB file.
    let size = Detector::new().expect("the bundled model loads").heap_size();
    assert!((200 << 10..400 << 10).contains(&size), "{size} bytes");
}

#[rstest]
#[case::english("Hello, how are you doing today?", "en")]
#[case::russian("Привет, как у тебя дела сегодня?", "ru")]
#[case::ukrainian("Привіт, як у тебе справи сьогодні?", "uk")]
#[case::german("Hallo, wie geht es dir heute?", "de")]
#[case::french("Bonjour, comment ça va aujourd'hui ?", "fr")]
#[case::japanese("今日はとても良い天気ですね", "ja")]
// MediaPipe's own codes rather than ISO 639-1.
#[case::hebrew("שלום, מה שלומך היום?", "iw")]
#[case::romanized_japanese("Kyou wa ii tenki desu ne", "ja-Latn")]
fn predict_detects_the_language(#[case] text: &str, #[case] expected: &str) {
    assert_eq!(
        Detector::new()
            .expect("the bundled model loads")
            .predict(text, 1)
            .expect("prediction succeeds")[0]
            .language,
        expected
    );
}

#[test]
fn predict_ignores_letter_case() {
    let detector = Detector::new().expect("the bundled model loads");
    assert_eq!(
        detector.predict("ПРИВЕТ МИР", 3).expect("prediction succeeds"),
        detector.predict("привет мир", 3).expect("prediction succeeds")
    );
}

#[rstest]
#[case::empty("")]
#[case::whitespace(" ")]
#[case::emoji("😀")]
fn predict_classifies_texts_without_letters(#[case] text: &str) {
    // As in MediaPipe: every text gets a prediction, even without a single letter.
    assert_eq!(
        Detector::new()
            .expect("the bundled model loads")
            .predict(text, 1)
            .expect("prediction succeeds")
            .len(),
        1
    );
}

#[test]
fn predict_with_zero_k_returns_nothing() {
    assert!(
        Detector::new()
            .expect("the bundled model loads")
            .predict("hello", 0)
            .expect("prediction succeeds")
            .is_empty()
    );
}

#[test]
fn from_bytes_rejects_data_that_is_not_a_model() {
    let err = Detector::from_bytes(b"definitely not a tflite file").unwrap_err();
    assert!(err.to_string().starts_with("invalid language detector model"), "{err}");
}

proptest! {
    #[test]
    fn predict_returns_a_sorted_distribution_for_any_text(text in any::<String>(), k in 1_usize..200) {
        let detector = Detector::new().expect("the bundled model loads");
        let all = detector.predict(&text, usize::MAX).expect("prediction succeeds");
        prop_assert_eq!(all.len(), LABELS);
        prop_assert!(all.windows(2).all(|p| p[0].probability >= p[1].probability));
        let sum: f32 = all.iter().map(|p| p.probability).sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "sum = {}", sum);
        // The top k are a prefix of the full ranking.
        let top = detector.predict(&text, k).expect("prediction succeeds");
        prop_assert_eq!(&*top, &all[..k.min(LABELS)]);
    }
}

/// The model `Detector::new` embeds.
const MODEL: &[u8] = include_bytes!("../vendor/model/language_detector.tflite");

proptest! {
    #[test]
    fn from_bytes_rejects_corrupted_models_without_panicking(
        flips in proptest::collection::vec((0..MODEL.len(), 1..=u8::MAX), 1..8),
    ) {
        let mut model = MODEL.to_vec();
        for (offset, mask) in flips {
            model[offset] ^= mask;
        }
        // Flipped bytes in weights still make a valid model: it must then predict or report an error.
        if let Ok(detector) = Detector::from_bytes(&model)
            && let Ok(top) = detector.predict("hello", 1)
        {
            prop_assert_eq!(top.len(), 1);
        }
    }

    #[test]
    fn from_bytes_rejects_truncated_models(len in 0..MODEL.len()) {
        prop_assert!(Detector::from_bytes(&MODEL[..len]).is_err());
    }
}

#[test]
fn from_bytes_rejects_oversized_labels() {
    // The model is a FlatBuffer followed by a zip archive with the labels: replace the archive.
    let fb_len = zip::ZipArchive::new(std::io::Cursor::new(MODEL)).unwrap().offset();
    let fb = &MODEL[..usize::try_from(fb_len).unwrap()];

    let mut archive = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    archive.start_file("labels.txt", options).unwrap();
    std::io::Write::write_all(&mut archive, "xx\n".repeat(40_000).as_bytes()).unwrap();
    let labels = archive.finish().unwrap().into_inner();

    let model = [fb, &labels].concat();

    let err = Detector::from_bytes(&model).unwrap_err();
    assert!(err.to_string().contains("labels.txt is larger than"), "{err}");
}
