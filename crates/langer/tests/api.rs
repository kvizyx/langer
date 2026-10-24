//! HTTP API behavior, with the bundled MediaPipe model and a small Lingua detector.

#![expect(clippy::expect_used, reason = "test helpers report failures by panicking")]

use std::num::NonZeroUsize;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use langer::{Detectors, Limits};
use langer_mediapipe::Detector;
use lingua::Language::{English, French, Russian};
use lingua::LanguageDetectorBuilder;
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;

/// Body limit of the test app: small, so that tests stay fast.
const BODY_LIMIT: usize = 1024;

fn bytes(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("limit is positive")
}

fn app() -> Router {
    app_with(Limits {
        body_bytes: bytes(BODY_LIMIT),
        ..Limits::default()
    })
}

fn app_with(limits: Limits) -> Router {
    let lingua = LanguageDetectorBuilder::from_languages(&[English, French, Russian]).build();
    langer::router(
        Detectors::new(Detector::new().expect("the bundled model loads"), lingua),
        limits,
    )
}

async fn send(request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app().oneshot(request).await.expect("router is infallible");
    let status = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body is readable")
        .to_bytes();
    (status, body.to_vec())
}

async fn post_detect(body: &str) -> (StatusCode, Value) {
    let request = Request::post("/detect")
        .body(Body::from(body.to_owned()))
        .expect("request is valid");
    let (status, body) = send(request).await;
    (status, serde_json::from_slice(&body).expect("response is JSON"))
}

async fn get(uri: &str) -> (StatusCode, Vec<u8>) {
    send(Request::get(uri).body(Body::empty()).expect("request is valid")).await
}

#[tokio::test]
async fn detect_without_method_uses_mediapipe() {
    let (status, body) = post_detect(r#"{"text": "Привет!!! 123 😀"}"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["language"], "ru");
    assert_eq!(body["method"], "mediapipe");
    let confidence = body["confidence"].as_f64().expect("confidence is a number");
    assert!((0.9..=1.0).contains(&confidence), "{confidence}");
}

#[tokio::test]
async fn detect_with_lingua_method_uses_lingua() {
    let (status, body) = post_detect(r#"{"text": "Это обычное сообщение на русском языке", "method": "lingua"}"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["language"], "ru");
    assert_eq!(body["method"], "lingua");
    assert!(body["confidence"].as_f64().is_some_and(|c| c > 0.5), "{body}");
}

#[tokio::test]
async fn detect_with_lingua_returns_unknown_without_letters() {
    let (_, body) = post_detect(r#"{"text": "😀 !!! 42", "method": "lingua"}"#).await;
    assert_eq!(
        body,
        json!({"language": "unknown", "confidence": null, "method": "lingua"})
    );
}

#[tokio::test]
async fn detect_with_mediapipe_returns_mediapipe_codes() {
    let (_, body) = post_detect(r#"{"text": "שלום, מה שלומך היום?"}"#).await;
    assert_eq!(body["language"], "iw");
}

#[rstest]
#[case::empty("")]
#[case::spaces("   ")]
#[case::newlines_and_tabs("\n\t")]
#[tokio::test]
async fn detect_with_blank_text_returns_400(#[case] text: &str) {
    let (status, body) = post_detect(&json!({ "text": text }).to_string()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body, json!({"status_code": 400, "detail": "Text cannot be empty"}));
}

#[rstest]
#[case::not_json("not_json", "not json")]
#[case::missing_text("missing_text", r#"{"method": "lingua"}"#)]
#[case::text_not_a_string("text_not_a_string", r#"{"text": 42}"#)]
#[case::unknown_method("unknown_method", r#"{"text": "hi", "method": "fasttext"}"#)]
#[case::unknown_field("unknown_field", r#"{"text": "hi", "methd": "lingua"}"#)]
#[tokio::test]
async fn detect_with_invalid_body_returns_400(#[case] name: &str, #[case] body: &str) {
    let (status, response) = post_detect(body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    insta::assert_json_snapshot!(format!("invalid_body_{name}"), response);
}

#[tokio::test]
async fn languages_returns_the_language_list() {
    let (status, body) = get("/detect/languages").await;
    assert_eq!(status, StatusCode::OK);
    let body: Value = serde_json::from_slice(&body).expect("response is JSON");
    let expected: Value = serde_json::from_str(include_str!("../data/languages.json")).expect("list is JSON");
    assert_eq!(body, expected);
}

#[tokio::test]
async fn health_returns_ok() {
    let (status, body) = get("/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        serde_json::from_slice::<Value>(&body).expect("JSON"),
        json!({"status": "ok"})
    );
}

#[rstest]
#[tokio::test]
async fn docs_aliases_redirect_to_schema(#[values("/", "/docs")] uri: &str) {
    let response = app()
        .oneshot(Request::get(uri).body(Body::empty()).expect("request is valid"))
        .await
        .expect("router is infallible");
    assert_eq!(response.status(), StatusCode::FOUND);
    assert_eq!(response.headers()[header::LOCATION], "/schema");
}

#[tokio::test]
async fn schema_serves_pinned_scalar_with_the_spec() {
    let (status, html) = get("/schema").await;
    assert_eq!(status, StatusCode::OK);
    let html = String::from_utf8_lossy(&html);
    assert!(html.contains("@scalar/api-reference@1.75.1/"), "Scalar is pinned");
    assert!(html.contains(r#""theme":"deepSpace""#));
    assert!(html.contains(r#""paths":"#), "the spec is embedded");
}

#[tokio::test]
async fn openapi_document_is_unchanged() {
    let (status, spec) = get("/schema/openapi.json").await;
    assert_eq!(status, StatusCode::OK);
    let spec: Value = serde_json::from_slice(&spec).expect("spec is JSON");
    insta::assert_json_snapshot!("openapi", spec);
}

#[tokio::test]
async fn unknown_path_returns_404() {
    let (status, body) = get("/nowhere").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_slice(&body).expect("response is JSON");
    assert_eq!(body, json!({"status_code": 404, "detail": "Not found"}));
}

#[tokio::test]
async fn wrong_method_returns_405_with_allowed_methods() {
    let response = app()
        .oneshot(Request::get("/detect").body(Body::empty()).expect("request is valid"))
        .await
        .expect("router is infallible");
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(response.headers()[header::ALLOW], "POST");
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body is readable")
        .to_bytes();
    let body: Value = serde_json::from_slice(&body).expect("response is JSON");
    assert_eq!(body, json!({"status_code": 405, "detail": "Method not allowed"}));
}

#[tokio::test]
async fn detect_with_body_over_the_limit_returns_413() {
    let (status, body) = post_detect(&json!({ "text": "a".repeat(BODY_LIMIT) }).to_string()).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(body["status_code"], 413);
    assert_eq!(body["detail"], "Cannot read the request body");
}

#[tokio::test]
async fn detect_with_body_at_the_limit_succeeds() {
    // `{"text":"…"}` takes 11 bytes besides the text.
    let (status, _) = post_detect(&json!({ "text": "a".repeat(BODY_LIMIT - 11) }).to_string()).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn detect_with_lingua_over_the_in_flight_limit_returns_503() {
    let app = app_with(Limits {
        body_bytes: bytes(1 << 20),
        lingua_in_flight: NonZeroUsize::MIN,
    });
    let request = || {
        let body = json!({ "text": "Это обычное сообщение на русском языке. ".repeat(500), "method": "lingua" });
        Request::post("/detect")
            .body(Body::from(body.to_string()))
            .expect("request is valid")
    };
    // The first request takes the only slot before the second one is polled.
    let (first, second) = tokio::join!(app.clone().oneshot(request()), app.oneshot(request()));
    assert_eq!(first.expect("router is infallible").status(), StatusCode::OK);
    let second = second.expect("router is infallible");
    assert_eq!(second.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(second.headers()[header::RETRY_AFTER], "1");
}
