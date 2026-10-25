//! Detection latency of both methods, and the cost of the HTTP layer around it.

#![expect(clippy::expect_used, reason = "a benchmark cannot run without its setup")]

use std::sync::OnceLock;

use axum::Router;
use axum::body::Body;
use axum::http::Request;
use divan::Bencher;
use http_body_util::BodyExt;
use langer::{Detectors, Limits, LinguaLimits, Method, Service};
use langer_mediapipe::Detector;
use lingua::LanguageDetectorBuilder;
use tower::ServiceExt;

/// Texts by kind, named so that the report stays readable. Lingua's time depends on the script:
/// Cyrillic leaves it few candidate languages, Latin several dozen.
const TEXTS: [&str; 5] = ["word", "cyrillic", "latin", "mixed_scripts", "long"];

fn main() {
    divan::main();
}

fn text(kind: &str) -> String {
    match kind {
        "word" => "kekw".to_owned(),
        "cyrillic" => "Привет, как дела? Это обычное сообщение.".to_owned(),
        "latin" => "hello there, how are you doing today".to_owned(),
        "mixed_scripts" => "Hello, как дела? Wie geht's? 你好".to_owned(),
        // Close to the default body limit of 64 KiB.
        _ => "Это обычное сообщение на русском языке. ".repeat(900),
    }
}

/// Detectors of the service, with Lingua's models paged in by a first detection of each text.
fn detectors() -> &'static Detectors {
    static DETECTORS: OnceLock<Detectors> = OnceLock::new();

    DETECTORS.get_or_init(|| {
        let mediapipe = Detector::new().expect("the bundled model loads");
        let detectors = Detectors::new(mediapipe, LanguageDetectorBuilder::from_all_languages().build());
        for kind in TEXTS {
            detectors
                .detect(&text(kind), Method::Lingua)
                .expect("detection succeeds");
        }

        detectors
    })
}

#[divan::bench(args = TEXTS)]
fn mediapipe(bencher: Bencher<'_, '_>, kind: &str) {
    let detectors = detectors();
    let text = text(kind);

    bencher.bench_local(|| detectors.detect(divan::black_box(&text), Method::MediaPipe));
}

#[divan::bench(args = TEXTS, sample_count = 50)]
fn lingua(bencher: Bencher<'_, '_>, kind: &str) {
    let detectors = detectors();
    let text = text(kind);

    bencher.bench_local(|| detectors.detect(divan::black_box(&text), Method::Lingua));
}

/// `POST /detect` with `mediapipe` through the whole router: body reading, JSON, the service,
/// metrics and the other middleware. The difference to [`mediapipe`] is the HTTP layer's cost.
#[divan::bench]
fn http_detect(bencher: Bencher<'_, '_>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("the runtime starts");
    let app = runtime.block_on(async { app() });
    let body = r#"{"text": "Привет, как дела? Это обычное сообщение."}"#;

    bencher.bench_local(|| {
        runtime.block_on(async {
            let request = Request::post("/detect")
                .body(Body::from(body))
                .expect("request is valid");
            let response = app.clone().oneshot(request).await.expect("router is infallible");

            response.into_body().collect().await.expect("body is readable")
        })
    });
}

fn app() -> Router {
    let service = Service::new(detectors().clone(), LinguaLimits::default());
    let metrics = langer::install_metrics().expect("the recorder installs once");

    langer::router(service, Limits::default(), metrics)
}
