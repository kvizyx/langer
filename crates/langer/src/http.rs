//! HTTP routes, request handling and the `OpenAPI` document.

use std::any::Any;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use serde::{Deserialize, Serialize};
use serde_json::error::Category;
use tokio::sync::Semaphore;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::CorsLayer;
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_scalar::Scalar;

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
    tags(
        (name = "Detect", description = "Language detection and supported languages"),
        (name = "Health", description = "Whether the service is running and able to serve detections"),
    )
)]
struct ApiDoc;

/// Limits that protect the service from oversized or excessive requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Largest accepted request body, in bytes; larger ones get `413`.
    ///
    /// Detection time grows with the text: about 0.1 s for 64 KiB with `lingua`.
    pub body_bytes: NonZeroUsize,
    /// Most `lingua` detections running or waiting for a thread at once; more get `503`.
    ///
    /// Lingua runs on blocking threads, so this bounds how long a request can wait for one.
    pub lingua_in_flight: NonZeroUsize,
}

/// Default body limit: far more than any chat message.
const DEFAULT_BODY_BYTES: NonZeroUsize = match NonZeroUsize::new(64 << 10) {
    Some(bytes) => bytes,
    None => NonZeroUsize::MIN,
};

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
            body_bytes: DEFAULT_BODY_BYTES,
            lingua_in_flight: cores.saturating_mul(LINGUA_SLOTS_PER_CORE),
        }
    }
}

/// Whether the service takes new work, shared between the router and whoever stops it.
///
/// Cloning gives a handle to the same state.
#[derive(Clone, Debug, Default)]
pub struct Readiness {
    shutting_down: Arc<AtomicBool>,
}

impl Readiness {
    /// Marks the service as shutting down: `GET /health/ready` answers `503` from now on, while
    /// requests are still served until the server stops.
    pub fn shut_down(&self) {
        self.shutting_down.store(true, Ordering::Relaxed);
    }

    fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Relaxed)
    }
}

/// State shared by the request handlers.
#[derive(Clone, Debug)]
struct AppState {
    detectors: Detectors,
    /// One permit per `lingua` detection allowed in flight.
    lingua_slots: Arc<Semaphore>,
    readiness: Readiness,
}

/// Builds the application router.
pub fn router(detectors: Detectors, limits: Limits, readiness: Readiness) -> Router {
    let (api, openapi) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .routes(routes!(detect))
        .routes(routes!(languages))
        .routes(routes!(health))
        .routes(routes!(ready))
        .with_state(AppState {
            detectors,
            lingua_slots: Arc::new(Semaphore::new(limits.lingua_in_flight.get())),
            readiness,
        })
        .split_for_parts();

    // axum clones route handlers for every connection, so the documents are rendered once into
    // `Bytes`, whose clones share the data: a captured `OpenApi` took ~100 KiB per connection.
    let spec = match serde_json::to_vec(&openapi) {
        Ok(json) => Some(Bytes::from(json)),
        Err(e) => {
            tracing::event!(
                name: "openapi.serialize.failure",
                tracing::Level::ERROR,
                error.message = %e,
                "cannot serialize the OpenAPI document",
            );
            None
        }
    };
    let page = Bytes::from(
        Scalar::new(openapi)
            .title("Langer API")
            .custom_html(SCALAR_TEMPLATE)
            .to_html(),
    );

    let app = api
        .route(DOCS_PATH, get(move || std::future::ready(Html(page.clone()))))
        .route(
            "/schema/openapi.json",
            get(move || std::future::ready(openapi_response(spec.clone()))),
        )
        .route("/", get(redirect_to_docs))
        .route("/docs", get(redirect_to_docs))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed);

    with_middleware(app, limits)
}

/// Adds the layers every response goes through. The last one added runs first, so CORS wraps
/// everything: its headers are set on every response, including the `500` of a panic.
fn with_middleware(app: Router, limits: Limits) -> Router {
    app.layer(DefaultBodyLimit::max(limits.body_bytes.get()))
        .layer(CatchPanicLayer::custom(panic_response))
        .layer(CorsLayer::permissive())
}

/// The `OpenAPI` document, or `500` if it could not be serialized at startup.
fn openapi_response(spec: Option<Bytes>) -> Response {
    match spec {
        Some(json) => ([(header::CONTENT_TYPE, "application/json")], json).into_response(),
        None => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "OpenAPI document unavailable",
            Vec::new(),
        ),
    }
}

async fn redirect_to_docs() -> Response {
    // Permanent: these paths are fixed aliases of the API reference.
    (StatusCode::PERMANENT_REDIRECT, [(header::LOCATION, DOCS_PATH)]).into_response()
}

async fn not_found() -> Response {
    error(StatusCode::NOT_FOUND, "Not found", Vec::new())
}

async fn method_not_allowed() -> Response {
    error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed", Vec::new())
}

/// Body of `POST /detect`. Unknown fields are rejected, so that a misspelt `method` is not
/// silently ignored.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
#[schema(description = "Body of `POST /detect`.")]
struct DetectRequest {
    /// Text to detect the language of.
    #[schema(example = "Hi, how are you?")]
    text: String,
    /// Detection method.
    #[serde(default)]
    method: Method,
}

/// Body of every error response.
#[derive(Debug, Serialize, ToSchema)]
#[schema(description = "Body of every error response.")]
struct ErrorBody {
    /// HTTP status code of the response.
    #[schema(example = 422)]
    status_code: u16,
    /// What went wrong.
    #[schema(example = "Text cannot be empty")]
    detail: String,
    /// Details of the problem, if any, e.g. why the body is invalid.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    extra: Vec<ErrorExtra>,
}

/// One detail of an error.
#[derive(Debug, Serialize, ToSchema)]
#[schema(description = "One detail of an error.")]
struct ErrorExtra {
    /// Description of the problem.
    #[schema(example = "missing field `text` at line 1 column 2")]
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
    // A temporary condition: the client should retry soon, and nothing should cache this answer.
    let headers = response.headers_mut();
    headers.insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));

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
        (status = 400, description = "Body is not valid JSON", body = ErrorBody),
        (status = 422, description = "Valid JSON with invalid fields, or empty text", body = ErrorBody),
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
            // Malformed JSON is a bad request; well-formed JSON with wrong or missing fields is
            // understood but cannot be processed.
            let (status, detail) = match e.classify() {
                Category::Data => (StatusCode::UNPROCESSABLE_ENTITY, "Body validation failed"),
                Category::Syntax | Category::Eof | Category::Io => (StatusCode::BAD_REQUEST, "Body is not valid JSON"),
            };
            let extra = vec![ErrorExtra { message: e.to_string() }];

            return error(status, detail, extra);
        }
    };

    if request.text.trim().is_empty() {
        return error(StatusCode::UNPROCESSABLE_ENTITY, "Text cannot be empty", Vec::new());
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

/// Body of a successful health check.
#[derive(Debug, Serialize, ToSchema)]
#[schema(description = "Body of a successful health check.")]
struct Health {
    /// Always `ok`.
    #[schema(example = "ok")]
    status: &'static str,
}

/// Text and expected language of the readiness self-check.
const SELF_CHECK: (&str, &str) = ("Hello, how are you today?", "en");

/// Liveness: whether the process is up.
#[utoipa::path(
    get,
    path = "/health",
    tag = "Health",
    summary = "Check that the service is running",
    description = "Answers `200` as long as the process runs and handles requests. It checks nothing else on purpose: a busy or shutting down service is still alive, and restarting it would not help. Whether it can take requests is `GET /health/ready`.",
    responses((status = 200, description = "The service is running", body = Health))
)]
async fn health() -> Json<Health> {
    Json(Health { status: "ok" })
}

/// Readiness: whether the service can take requests.
#[utoipa::path(
    get,
    path = "/health/ready",
    tag = "Health",
    summary = "Check that the service can take requests",
    description = "Answers `200` when the service is not shutting down and `mediapipe` detects the language of a fixed English text correctly (about 20 µs). Answers `503` once the service starts shutting down, a few seconds before it stops accepting connections (`SHUTDOWN_DELAY_SECS`), so that traffic can move to other instances; requests are still served meanwhile. Load does not affect it: excess `lingua` requests get their own `503`.",
    responses(
        (status = 200, description = "The service can take requests", body = Health),
        (status = 503, description = "The service is shutting down or failed its self-check", body = ErrorBody),
    )
)]
async fn ready(State(state): State<AppState>) -> Response {
    if state.readiness.is_shutting_down() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "Shutting down", Vec::new());
    }

    let (text, expected) = SELF_CHECK;
    let problem = match state.detectors.detect(text, Method::MediaPipe) {
        Ok(detection) if detection.language == expected => return Json(Health { status: "ok" }).into_response(),
        Ok(detection) => format!("detected {:?} instead of {expected:?}", detection.language),
        Err(e) => e.to_string(),
    };

    tracing::event!(
        name: "health.self_check.failure",
        tracing::Level::ERROR,
        error.message = problem,
        "readiness self-check failed",
    );

    let extra = vec![ErrorExtra { message: problem }];
    error(StatusCode::SERVICE_UNAVAILABLE, "Self-check failed", extra)
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
    async fn panicking_handler_returns_500_with_cors_headers() {
        let app = with_middleware(Router::new().route("/panic", get(boom)), Limits::default());

        let request = Request::get("/panic")
            .header(header::ORIGIN, "https://example.com")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN], "*");
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body, json!({"status_code": 500, "detail": "Internal server error"}));
    }
}
