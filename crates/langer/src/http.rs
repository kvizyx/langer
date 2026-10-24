//! HTTP routes, request handling and the `OpenAPI` document.

use std::any::Any;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::CorsLayer;
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_scalar::{Scalar, Servable};

use crate::detect::{Detection, Detectors, Method};
use crate::languages::{self, Language};

/// Where the API reference is served; `/` and `/docs` redirect here.
const DOCS_PATH: &str = "/schema";

/// Page of the API reference: Scalar with the `deepSpace` theme, pinned to one version.
const SCALAR_TEMPLATE: &str = include_str!("../assets/scalar.html");

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Langer API",
        description = "Self-host language detection HTTP service, based on Lingua and MediaPipe."
    ),
    tags((name = "Detect", description = "Language detection and supported languages"))
)]
struct ApiDoc;

/// Limits that protect the service from oversized or excessive requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Largest accepted request body, in bytes; larger ones get `413`.
    ///
    /// Detection time grows with the text: about 0.1 s for 64 KiB with `lingua`.
    pub body_bytes: usize,
    /// Most `lingua` detections running or waiting for a thread at once; more get `503`.
    ///
    /// Lingua runs on blocking threads, so this bounds how long a request can wait for one.
    pub lingua_in_flight: NonZeroUsize,
}

/// `lingua` detections allowed in flight per CPU core: one running, three waiting.
const LINGUA_SLOTS_PER_CORE: NonZeroUsize = match NonZeroUsize::new(4) {
    Some(slots) => slots,
    None => NonZeroUsize::MIN,
};

impl Default for Limits {
    /// 64 KiB bodies and [`LINGUA_SLOTS_PER_CORE`] `lingua` detections per CPU core.
    fn default() -> Self {
        let cores = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
        Self {
            body_bytes: 64 << 10,
            lingua_in_flight: cores.saturating_mul(LINGUA_SLOTS_PER_CORE),
        }
    }
}

/// State shared by the request handlers.
#[derive(Clone, Debug)]
struct AppState {
    detectors: Detectors,
    /// One permit per `lingua` detection allowed in flight.
    lingua_slots: Arc<Semaphore>,
}

/// Builds the application router.
pub fn router(detectors: Detectors, limits: Limits) -> Router {
    let (api, openapi) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .routes(routes!(detect))
        .routes(routes!(languages))
        .routes(routes!(health))
        .with_state(AppState {
            detectors,
            lingua_slots: Arc::new(Semaphore::new(limits.lingua_in_flight.get())),
        })
        .split_for_parts();

    let spec = openapi.clone();

    api.merge(
        Scalar::with_url(DOCS_PATH, openapi)
            .title("Langer API")
            .custom_html(SCALAR_TEMPLATE),
    )
    .route(
        "/schema/openapi.json",
        get(move || std::future::ready(Json(spec.clone()))),
    )
    .route("/", get(redirect_to_docs))
    .route("/docs", get(redirect_to_docs))
    .fallback(not_found)
    .method_not_allowed_fallback(method_not_allowed)
    .layer(DefaultBodyLimit::max(limits.body_bytes))
    .layer(CorsLayer::permissive())
    .layer(CatchPanicLayer::custom(panic_response))
}

async fn redirect_to_docs() -> Response {
    (StatusCode::FOUND, [(header::LOCATION, DOCS_PATH)]).into_response()
}

async fn not_found() -> Response {
    error(StatusCode::NOT_FOUND, "Not found", Vec::new())
}

async fn method_not_allowed() -> Response {
    error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed", Vec::new())
}

/// Body of `POST /detect`.
#[derive(Debug, Deserialize, ToSchema)]
#[schema(description = "Body of `POST /detect`.")]
struct DetectRequest {
    /// Text to detect the language of.
    #[schema(example = "Hi, how are you?")]
    text: String,
    /// Detection method.
    #[serde(default)]
    method: Method,
}

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

/// Answers a request whose handler panicked. Handlers return errors instead of panicking; this
/// keeps the service up should one panic anyway.
#[expect(
    clippy::needless_pass_by_value,
    reason = "CatchPanicLayer passes the panic payload by value"
)]
fn panic_response(panic: Box<dyn Any + Send + 'static>) -> Response {
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic");
    tracing::event!(
        name: "request.panic",
        tracing::Level::ERROR,
        error.message = message,
        "request handler panicked",
    );
    error(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error", Vec::new())
}

/// `503` for a `lingua` request beyond [`Limits::lingua_in_flight`].
fn overloaded() -> Response {
    tracing::event!(
        name: "detect.lingua.overloaded",
        tracing::Level::WARN,
        "too many lingua detections at once",
    );
    let mut response = error(
        StatusCode::SERVICE_UNAVAILABLE,
        "Too many lingua detections at once, retry later",
        Vec::new(),
    );
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    response
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
#[utoipa::path(
    post,
    path = "/detect",
    tag = "Detect",
    summary = "Detect the language of a text",
    description = "`mediapipe` (default) answers in tens of microseconds, `lingua` is more accurate on one or two words but takes about a millisecond.",
    request_body = DetectRequest,
    responses(
        (status = 200, description = "Detected language", body = Detection),
        (status = 400, description = "Invalid body or empty text", body = ErrorBody),
        (status = 413, description = "Body larger than the limit", body = ErrorBody),
        (status = 500, description = "Detection failed", body = ErrorBody),
        (status = 503, description = "Too many `lingua` detections at once; retry later", body = ErrorBody),
    )
)]
async fn detect(State(state): State<AppState>, body: Result<Bytes, BytesRejection>) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(rejection) => {
            let extra = vec![ErrorExtra {
                message: rejection.body_text(),
            }];
            return error(rejection.status(), "Cannot read the request body", extra);
        }
    };
    let request: DetectRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(e) => {
            let extra = vec![ErrorExtra { message: e.to_string() }];
            return error(StatusCode::BAD_REQUEST, "Body validation failed", extra);
        }
    };
    if request.text.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "Text cannot be empty", Vec::new());
    }

    let started = Instant::now();
    let text_len = request.text.chars().count();
    let method = request.method;
    let detection = match method {
        Method::MediaPipe => state.detectors.detect(&request.text, method).map_err(|e| e.to_string()),
        Method::Lingua => {
            let Ok(slot) = Arc::clone(&state.lingua_slots).try_acquire_owned() else {
                return overloaded();
            };
            let detectors = state.detectors;
            tokio::task::spawn_blocking(move || {
                // Released when the detection ends, even if the client is gone by then.
                let _slot = slot;
                detectors.detect(&request.text, method)
            })
            .await
            .map_err(|e| e.to_string())
            .and_then(|result| result.map_err(|e| e.to_string()))
        }
    };
    let detection = match detection {
        Ok(detection) => detection,
        Err(message) => {
            tracing::event!(
                name: "detect.request.failure",
                tracing::Level::ERROR,
                detect.method = %method,
                error.message = message,
                "language detection failed",
            );
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Language detection failed",
                Vec::new(),
            );
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

/// List supported languages for detection.
#[utoipa::path(
    get,
    path = "/detect/languages",
    tag = "Detect",
    summary = "List supported languages for detection",
    description = "Every language with the methods that can detect it, sorted by code.",
    responses((status = 200, description = "Supported languages", body = [Language]))
)]
async fn languages() -> Response {
    ([(header::CONTENT_TYPE, "application/json")], languages::JSON).into_response()
}

/// Health of the service.
#[derive(Debug, Serialize, ToSchema)]
#[schema(description = "Health of the service.")]
struct Health {
    #[schema(example = "ok")]
    status: &'static str,
}

/// Health check.
#[utoipa::path(
    get,
    path = "/health",
    summary = "Health check",
    responses((status = 200, description = "Service is up", body = Health))
)]
async fn health() -> Json<Health> {
    Json(Health { status: "ok" })
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;

    async fn boom() -> &'static str {
        panic!("boom")
    }

    #[tokio::test]
    async fn panicking_handler_returns_500() {
        let app = Router::new()
            .route("/panic", get(boom))
            .layer(CatchPanicLayer::custom(panic_response));
        let response = app
            .oneshot(Request::get("/panic").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body, json!({"status_code": 500, "detail": "Internal server error"}));
    }
}
