//! Parity with MediaPipe: this crate's predictions against MediaPipe's own, for the texts of
//! `data/parity.jsonl` (see `data/README.md`).

#![expect(clippy::expect_used, reason = "test helpers report failures by panicking")]

use std::collections::BTreeSet;

use langer_mediapipe::{Detector, Prediction};
use serde::Deserialize;

/// One JSON object per line: `{"text": ..., "top": [[label, probability], ...]}`.
const DATA: &str = include_str!("data/parity.jsonl");

/// Largest difference allowed between two probabilities: the computation is the same, but float
/// sums can differ in their last bits.
const TOLERANCE: f32 = 1e-5;

/// Mismatches shown when the test fails.
const SHOWN: usize = 20;

/// A text and MediaPipe's most probable labels for it, most probable first.
#[derive(Deserialize)]
struct Case {
    text: String,
    top: Vec<(String, f32)>,
}

impl Case {
    /// How our prediction differs from MediaPipe's, if it does.
    fn mismatch(&self, ours: &[Prediction<'_>]) -> Option<String> {
        self.probability_mismatch(ours).or_else(|| self.top_mismatch(ours))
    }

    /// The first label whose probability differs by more than [`TOLERANCE`].
    fn probability_mismatch(&self, ours: &[Prediction<'_>]) -> Option<String> {
        self.top.iter().find_map(|(label, expected)| {
            let actual = ours.iter().find(|p| p.language == label).map(|p| p.probability);
            let close = actual.is_some_and(|actual| (actual - expected).abs() <= TOLERANCE);

            (!close).then(|| format!("{label} is {actual:?}, MediaPipe {expected}"))
        })
    }

    /// A different most probable label, unless MediaPipe's first two are too close to rank.
    fn top_mismatch(&self, ours: &[Prediction<'_>]) -> Option<String> {
        if let [(_, first), (_, second), ..] = self.top.as_slice()
            && first - second <= 2.0 * TOLERANCE
        {
            return None;
        }

        let expected = self.top.first().map(|(label, _)| label.as_str());
        let actual = ours.first().map(|p| p.language);

        (actual != expected).then(|| format!("top is {actual:?}, MediaPipe {expected:?}"))
    }
}

fn cases() -> Vec<Case> {
    DATA.lines()
        .map(|line| serde_json::from_str(line).expect("every line is a case"))
        .collect()
}

/// Every label is among MediaPipe's most probable for some text. Not necessarily the most
/// probable: MediaPipe takes Bosnian for Croatian or Serbian.
#[test]
fn parity_data_covers_every_label() {
    let detector = Detector::new().expect("the bundled model loads");
    let labels: BTreeSet<&str> = detector.labels().collect();

    let cases = cases();
    let predicted: BTreeSet<&str> = cases
        .iter()
        .flat_map(|case| case.top.iter().map(|(label, _)| label.as_str()))
        .collect();

    let missing: Vec<&&str> = labels.difference(&predicted).collect();
    assert!(
        missing.is_empty(),
        "no text has {missing:?} among its most probable labels"
    );
}

#[test]
fn predictions_match_mediapipe() {
    let detector = Detector::new().expect("the bundled model loads");
    let cases = cases();

    let mismatches: Vec<String> = cases
        .iter()
        .filter_map(|case| {
            let ours = detector.predict(&case.text, usize::MAX).expect("prediction succeeds");
            case.mismatch(&ours)
                .map(|mismatch| format!("{:?}: {mismatch}", case.text))
        })
        .collect();

    assert!(
        mismatches.is_empty(),
        "{} of {} texts differ from MediaPipe, first {}:\n{}",
        mismatches.len(),
        cases.len(),
        SHOWN.min(mismatches.len()),
        mismatches.iter().take(SHOWN).cloned().collect::<Vec<_>>().join("\n")
    );
}
