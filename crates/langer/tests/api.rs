//! HTTP API behavior, with the bundled MediaPipe model and a small Lingua detector.

#![expect(clippy::expect_used, reason = "test helpers report failures by panicking")]

use std::num::NonZeroUsize;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use langer::{Detectors, Service};
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

fn service(lingua_in_flight: NonZeroUsize) -> Service {
    let lingua = LanguageDetectorBuilder::from_languages(&[English, French, Russian]).build();
    let detectors = Detectors::new(Detector::new().expect("the bundled model loads"), lingua);

    Service::new(detectors, lingua_in_flight)
}

fn app() -> Router {
    langer::router(service(langer::default_lingua_in_flight()), bytes(BODY_LIMIT))
}

/// Status and body of `request` sent to `app`.
async fn send_to(app: Router, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app.oneshot(request).await.expect("router is infallible");
    let status = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body is readable")
        .to_bytes();

    (status, body.to_vec())
}

/// Status and JSON body of `request` sent to `app`.
async fn call(app: Router, request: Request<Body>) -> (StatusCode, Value) {
    let (status, body) = send_to(app, request).await;

    (status, serde_json::from_slice(&body).expect("response is JSON"))
}

async fn post_detect(body: &str) -> (StatusCode, Value) {
    let request = Request::post("/detect")
        .body(Body::from(body.to_owned()))
        .expect("request is valid");

    call(app(), request).await
}

async fn get(uri: &str) -> (StatusCode, Vec<u8>) {
    send_to(app(), Request::get(uri).body(Body::empty()).expect("request is valid")).await
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
async fn detect_with_blank_text_returns_422(#[case] text: &str) {
    let (status, body) = post_detect(&json!({ "text": text }).to_string()).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body, json!({"status_code": 422, "detail": "Text cannot be empty"}));
}

#[rstest]
#[case::not_json("not_json", "not json")]
#[case::truncated_json("truncated_json", r#"{"text": "hi""#)]
#[tokio::test]
async fn detect_with_malformed_json_returns_400(#[case] name: &str, #[case] body: &str) {
    let (status, response) = post_detect(body).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    insta::assert_json_snapshot!(format!("invalid_body_{name}"), response);
}

#[rstest]
#[case::missing_text("missing_text", r#"{"method": "lingua"}"#)]
#[case::text_not_a_string("text_not_a_string", r#"{"text": 42}"#)]
#[case::unknown_method("unknown_method", r#"{"text": "hi", "method": "fasttext"}"#)]
#[case::unknown_field("unknown_field", r#"{"text": "hi", "methd": "lingua"}"#)]
#[tokio::test]
async fn detect_with_invalid_fields_returns_422(#[case] name: &str, #[case] body: &str) {
    let (status, response) = post_detect(body).await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
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

#[tokio::test]
async fn ready_returns_ok_while_serving() {
    let (status, body) = get("/health/ready").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        serde_json::from_slice::<Value>(&body).expect("JSON"),
        json!({"status": "ok"})
    );
}

#[tokio::test]
async fn ready_returns_503_while_shutting_down_but_detection_still_works() {
    let service = service(langer::default_lingua_in_flight());
    let app = langer::router(service.clone(), langer::DEFAULT_BODY_LIMIT);

    service.shut_down();

    let ready = Request::get("/health/ready")
        .body(Body::empty())
        .expect("request is valid");
    let (status, body) = call(app.clone(), ready).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body, json!({"status_code": 503, "detail": "Shutting down"}));

    let detect = Request::post("/detect")
        .body(Body::from(r#"{"text": "Привет"}"#))
        .expect("request is valid");
    let (status, _) = call(app, detect).await;
    assert_eq!(status, StatusCode::OK);
}

#[rstest]
#[tokio::test]
async fn docs_aliases_redirect_to_schema(#[values("/", "/docs")] uri: &str) {
    let response = app()
        .oneshot(Request::get(uri).body(Body::empty()).expect("request is valid"))
        .await
        .expect("router is infallible");
    assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
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
    let app = langer::router(service(NonZeroUsize::MIN), bytes(1 << 20));
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
    assert_eq!(second.headers()[header::CACHE_CONTROL], "no-store");
}
