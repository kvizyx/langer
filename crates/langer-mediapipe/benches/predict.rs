//! Loading and prediction latency of the bundled model.

#![expect(clippy::expect_used, reason = "a benchmark cannot run without the model")]

use divan::Bencher;
use langer_mediapipe::Detector;

/// Texts by kind, named so that the report stays readable.
const TEXTS: [&str; 5] = ["word", "chat", "sentence", "mixed_scripts", "long"];

fn main() {
    divan::main();
}

fn text(kind: &str) -> String {
    match kind {
        "word" => "kekw".to_owned(),
        "chat" => "привет как дела".to_owned(),
        "sentence" => "hello there, how are you doing today? I hope everything is fine".to_owned(),
        "mixed_scripts" => "Hello, как дела? Wie geht's? 你好".to_owned(),
        // A 64 KiB text: only its first characters are read, so it costs as much as a short one.
        _ => "Это обычное сообщение на русском языке. ".repeat(900),
    }
}

/// Loading and checking the whole model, once per process in the service.
#[divan::bench]
fn load() -> Detector {
    Detector::new().expect("the bundled model loads")
}

/// The most probable language of a text, as the service asks for it.
#[divan::bench(args = TEXTS)]
fn predict(bencher: Bencher<'_, '_>, kind: &str) {
    let detector = Detector::new().expect("the bundled model loads");
    let text = text(kind);

    bencher.bench_local(|| detector.predict(divan::black_box(&text), 1));
}

/// All languages ranked, for comparison with the top one.
#[divan::bench]
fn predict_all(bencher: Bencher<'_, '_>) {
    let detector = Detector::new().expect("the bundled model loads");
    let text = text("chat");

    bencher.bench_local(|| detector.predict(divan::black_box(&text), usize::MAX));
}
