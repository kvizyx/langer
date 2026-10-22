//! HTTP API behavior, with the bundled MediaPipe model and a small Lingua detector.

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use langer::Detectors;
use langer_mediapipe::Detector;
use lingua::Language::{English, French, Russian};
use lingua::LanguageDetectorBuilder;
use serde_json::{Value, json};
use tower::ServiceExt;

fn app() -> Router {
    let lingua = LanguageDetectorBuilder::from_languages(&[English, French, Russian]).build();
    langer::router(Detectors::new(Detector::new(), lingua))
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
async fn detects_with_mediapipe_by_default() {
    let (status, body) = post_detect(r#"{"text": "Привет!!! 123 😀"}"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["language"], "ru");
    assert_eq!(body["method"], "mediapipe");
    let confidence = body["confidence"].as_f64().expect("confidence is a number");
    assert!((0.9..=1.0).contains(&confidence), "{confidence}");
}

#[tokio::test]
async fn detects_with_lingua_on_request() {
    let (status, body) = post_detect(r#"{"text": "Это обычное сообщение на русском языке", "method": "lingua"}"#).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["language"], "ru");
    assert_eq!(body["method"], "lingua");
    assert!(body["confidence"].as_f64().is_some_and(|c| c > 0.5), "{body}");
}

#[tokio::test]
async fn lingua_reports_unknown_without_words() {
    let (_, lingua) = post_detect(r#"{"text": "😀 !!! 42", "method": "lingua"}"#).await;
    assert_eq!(
        lingua,
        json!({"language": "unknown", "confidence": null, "method": "lingua"})
    );
}

#[tokio::test]
async fn mediapipe_reports_its_own_codes() {
    let (_, body) = post_detect(r#"{"text": "שלום, מה שלומך היום?"}"#).await;
    assert_eq!(body["language"], "iw");
}

#[tokio::test]
async fn rejects_empty_text() {
    let (status, body) = post_detect(r#"{"text": "   "}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body, json!({"status_code": 400, "detail": "Text cannot be empty"}));
}

#[tokio::test]
async fn rejects_invalid_bodies() {
    for body in [
        "not json",
        r#"{"method": "lingua"}"#,
        r#"{"text": "hi", "method": "cld3"}"#,
        r#"{"text": "hi", "method": "fasttext"}"#,
    ] {
        let (status, response) = post_detect(body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(response["detail"], "Validation failed for POST /detect", "{body}");
        assert!(response["extra"][0]["message"].is_string(), "{body}");
    }
}

#[tokio::test]
async fn lists_languages_sorted_with_methods() {
    let (status, body) = get("/detect/languages").await;
    assert_eq!(status, StatusCode::OK);
    let languages: Vec<Value> = serde_json::from_slice(&body).expect("response is a JSON array");
    let codes: Vec<&str> = languages
        .iter()
        .map(|l| l["iso_639_1"].as_str().expect("code is a string"))
        .collect();
    assert!(codes.is_sorted());
    let english = languages
        .iter()
        .find(|l| l["iso_639_1"] == "en")
        .expect("English is listed");
    assert_eq!(english["name"], "English");
    assert_eq!(english["methods"], json!(["mediapipe", "lingua"]));
}

#[tokio::test]
async fn reports_health() {
    let (status, body) = get("/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        serde_json::from_slice::<Value>(&body).expect("JSON"),
        json!({"status": "ok"})
    );
}

#[tokio::test]
async fn serves_api_reference() {
    for uri in ["/", "/docs"] {
        let response = app()
            .oneshot(Request::get(uri).body(Body::empty()).expect("valid"))
            .await
            .expect("infallible");
        assert_eq!(response.status(), StatusCode::FOUND);
        assert_eq!(response.headers()[header::LOCATION], "/schema");
    }

    let (status, html) = get("/schema").await;
    assert_eq!(status, StatusCode::OK);
    assert!(String::from_utf8_lossy(&html).contains("@scalar/api-reference"));

    let (status, spec) = get("/schema/openapi.json").await;
    assert_eq!(status, StatusCode::OK);
    let spec: Value = serde_json::from_slice(&spec).expect("spec is JSON");
    for path in ["/detect", "/detect/languages", "/health"] {
        assert!(spec["paths"][path].is_object(), "{path} is documented");
    }
}
