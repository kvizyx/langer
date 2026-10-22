//! HTTP routes, request handling and the `OpenAPI` document.

use std::time::Instant;

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_scalar::{Scalar, Servable};

use crate::detect::{Detection, Detectors, Method};
use crate::languages::{self, Language};

/// Where the API reference is served; `/` and `/docs` redirect here.
const DOCS_PATH: &str = "/schema";

/// Page of the API reference: Scalar with the `fastify` theme, pinned to one version.
const SCALAR_TEMPLATE: &str = include_str!("../assets/scalar.html");

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Language Detection API",
        description = "Detects the language of a text with MediaPipe (`mediapipe` method) or Lingua (`lingua` method)."
    ),
    tags((name = "detect", description = "Language detection"))
)]
struct ApiDoc;

/// Builds the application router.
///
/// # Panics
///
/// Never in practice: the generated `OpenAPI` document always serializes to JSON.
///
/// # Panics
///
/// Never in practice: the generated `OpenAPI` document always serializes to JSON.
pub fn router(detectors: Detectors) -> Router {
    let (api, openapi) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .routes(routes!(detect))
        .routes(routes!(languages))
        .routes(routes!(health))
        .with_state(detectors)
        .split_for_parts();

    let openapi_json = serde_json::to_string(&openapi).expect("OpenAPI document serializes");

    api.merge(
        Scalar::with_url(DOCS_PATH, openapi)
            .title("Language Detection API")
            .custom_html(SCALAR_TEMPLATE),
    )
    .route(
        "/schema/openapi.json",
        get(|| async move { ([(header::CONTENT_TYPE, "application/json")], openapi_json) }),
    )
    .route("/", get(redirect_to_docs))
    .route("/docs", get(redirect_to_docs))
    .layer(CorsLayer::permissive())
}

async fn redirect_to_docs() -> Response {
    (StatusCode::FOUND, [(header::LOCATION, DOCS_PATH)]).into_response()
}

/// Body of `POST /detect`.
#[derive(Debug, Deserialize, ToSchema)]
struct DetectRequest {
    /// Text to detect the language of.
    #[schema(example = "Привет, как дела?")]
    text: String,
    /// Detection method.
    #[serde(default)]
    method: Method,
}

/// Error body, same shape as the Python service (Litestar).
#[derive(Debug, Serialize, ToSchema)]
struct ErrorBody {
    status_code: u16,
    detail: String,
    /// Validation problems, if any.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    extra: Vec<ErrorExtra>,
}

#[derive(Debug, Serialize, ToSchema)]
struct ErrorExtra {
    message: String,
}

fn error(status: StatusCode, detail: impl Into<String>, extra: Vec<ErrorExtra>) -> Response {
    let body = ErrorBody {
        status_code: status.as_u16(),
        detail: detail.into(),
        extra,
    };
    (status, Json(body)).into_response()
}

/// Detect the language of a text.
///
/// `mediapipe` (default) answers in tens of microseconds; `lingua` is more accurate on one or two
/// words but takes about a millisecond.
#[utoipa::path(
    post,
    path = "/detect",
    tag = "detect",
    request_body = DetectRequest,
    responses(
        (status = 200, description = "Detected language", body = Detection),
        (status = 400, description = "Invalid body or empty text", body = ErrorBody),
    )
)]
async fn detect(State(detectors): State<Detectors>, body: Bytes) -> Response {
    // Parsed by hand to accept any content type and answer with Litestar-style errors.
    let request: DetectRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => {
            let extra = vec![ErrorExtra { message: e.to_string() }];
            return error(StatusCode::BAD_REQUEST, "Validation failed for POST /detect", extra);
        }
    };
    if request.text.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "Text cannot be empty", Vec::new());
    }

    let started = Instant::now();
    let text_len = request.text.chars().count();
    let detection = match request.method {
        Method::Mediapipe => detectors.detect(&request.text, Method::Mediapipe),
        Method::Lingua => {
            let task = tokio::task::spawn_blocking(move || detectors.detect(&request.text, Method::Lingua));
            match task.await {
                Ok(detection) => detection,
                Err(e) => {
                    tracing::event!(
                        name: "detect.lingua.failure",
                        tracing::Level::ERROR,
                        error.message = %e,
                        "lingua detection failed",
                    );
                    return error(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error", Vec::new());
                }
            }
        }
    };
    tracing::event!(
        name: "detect.request.success",
        tracing::Level::DEBUG,
        detect.method = %detection.method,
        detect.language = %detection.language,
        detect.confidence = detection.confidence,
        text.length = text_len,
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "detected language",
    );
    (StatusCode::OK, Json(detection)).into_response()
}

/// List supported languages.
///
/// Every language with the methods that can detect it, sorted by code.
#[utoipa::path(
    get,
    path = "/detect/languages",
    tag = "detect",
    responses((status = 200, description = "Supported languages", body = [Language]))
)]
async fn languages() -> Response {
    ([(header::CONTENT_TYPE, "application/json")], languages::JSON).into_response()
}

/// Health of the service.
#[derive(Debug, Serialize, ToSchema)]
struct Health {
    #[schema(example = "ok")]
    status: &'static str,
}

/// Health check.
#[utoipa::path(get, path = "/health", responses((status = 200, description = "Service is up", body = Health)))]
async fn health() -> Json<Health> {
    Json(Health { status: "ok" })
}
