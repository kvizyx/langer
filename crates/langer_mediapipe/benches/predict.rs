//! Prediction latency of the bundled model.

use langer_mediapipe::Detector;

const TEXTS: [&str; 4] = [
    "привет как дела",
    "hello there how are you doing today",
    "kekw",
    "bonjour à tous",
];

fn main() {
    divan::main();
}

#[divan::bench]
fn predict(bencher: divan::Bencher<'_, '_>) {
    let detector = Detector::new();
    bencher.bench(|| {
        for text in TEXTS {
            divan::black_box(detector.predict(divan::black_box(text), 1));
        }
    });
}
