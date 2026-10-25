//! HTTP routes, request handling and the `OpenAPI` document.

use std::any::Any;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use metrics::{counter, histogram};
use metrics_exporter_prometheus::PrometheusHandle;
use serde::de::IgnoredAny;
use serde::{Deserialize, Serialize};
use serde_json::error::Category;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::CorsLayer;
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_scalar::Scalar;

use crate::detect::{Detection, Method};
use crate::languages::{self, Language};
use crate::service::{Service, ServiceError};
use crate::telemetry::{HTTP_REQUEST_DURATION, HTTP_REQUESTS};

/// Where the API reference is served; `/` and `/docs` redirect here.
const DOCS_PATH: &str = "/schema";

/// Page of the API reference: Scalar with the `deepSpace` theme, pinned to one version.
const SCALAR_TEMPLATE: &str = include_str!("../assets/scalar.html");

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Langer API",
        description = "Self-hosted language detection HTTP service, based on Lingua and MediaPipe."
    ),
    tags(
        (name = "Detect", description = "Language detection and supported languages"),
        (name = "Health", description = "Whether the service is running and able to serve detections"),
    )
)]
struct ApiDoc;

/// Default limit of request bodies: far more than any chat message. Detection time grows with
/// the text, about 0.1 s for 64 KiB with `lingua`.
pub const DEFAULT_BODY_LIMIT: NonZeroUsize = NonZeroUsize::new(64 << 10).expect("64 KiB is not zero");

/// Default time limit of a request, from its headers to the response: far more than a
/// detection takes, so that only clients that stall sending the body hit it.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Limits every request is held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Largest accepted body; larger ones get `413`.
    pub body: NonZeroUsize,
    /// Longest time to read the body and answer; slower requests get `408`.
    pub request_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            body: DEFAULT_BODY_LIMIT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }
}

/// Builds the HTTP API over `service`, holding requests to `limits`; `GET /metrics` renders
/// `metrics` (see [`install_metrics`](crate::install_metrics)).
pub fn router(service: Service, limits: Limits, metrics: PrometheusHandle) -> Router {
    let (api, openapi) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .routes(routes!(detect))
        .routes(routes!(languages))
        .routes(routes!(health))
        .routes(routes!(ready))
        .with_state(service)
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
        // Only matched routes, so that unknown paths cannot add metric series; `/metrics` comes
        // after, so that scrapes are not counted.
        .route_layer(middleware::from_fn(track_metrics))
        .route("/metrics", get(move || std::future::ready(metrics_response(&metrics))))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed);

    with_middleware(app, limits)
}

/// Adds the layers every response goes through. The last one added runs first, so CORS wraps
/// everything: its headers are set on every response, including the `500` of a panic and the
/// `408` of a timeout.
fn with_middleware(app: Router, limits: Limits) -> Router {
    app.layer(DefaultBodyLimit::max(limits.body.get()))
        .layer(middleware::from_fn_with_state(limits.request_timeout, time_out))
        .layer(CatchPanicLayer::custom(panic_response))
        .layer(CorsLayer::permissive())
}

/// Answers `408` and closes the connection when a request takes longer than `timeout`, e.g.
/// because its client stalls sending the body.
async fn time_out(State(timeout): State<Duration>, request: Request, next: Next) -> Response {
    let Ok(response) = tokio::time::timeout(timeout, next.run(request)).await else {
        let mut response = error(StatusCode::REQUEST_TIMEOUT, "Request timed out", Vec::new());
        response
            .headers_mut()
            .insert(header::CONNECTION, HeaderValue::from_static("close"));

        return response;
    };

    response
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

/// Metrics in the Prometheus text format.
fn metrics_response(metrics: &PrometheusHandle) -> Response {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        metrics.render(),
    )
        .into_response()
}

/// Counts and times a request to a known route, labelled with the route's pattern.
async fn track_metrics(request: Request, next: Next) -> Response {
    let started = Instant::now();
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(String::new, |path| path.as_str().to_owned());
    // Only the methods the routes accept reach this layer, so the label has few values.
    let method = request.method().to_string();

    let response = next.run(request).await;

    let status = response.status().as_u16().to_string();
    counter!(HTTP_REQUESTS, "method" => method.clone(), "route" => route.clone(), "status" => status).increment(1);
    histogram!(HTTP_REQUEST_DURATION, "method" => method, "route" => route).record(started.elapsed().as_secs_f64());

    response
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

/// The response to a [`ServiceError`]: `422` for invalid input, `503` while unavailable (not to
/// be cached, with `Retry-After` when known), `500` otherwise.
fn service_error(e: &ServiceError) -> Response {
    let status = if e.is_invalid_input() {
        StatusCode::UNPROCESSABLE_ENTITY
    } else if e.is_unavailable() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    let extra = e
        .client_details()
        .map(|message| ErrorExtra {
            message: message.to_owned(),
        })
        .into_iter()
        .collect();

    let mut response = error(status, e.summary(), extra);

    let headers = response.headers_mut();
    if status == StatusCode::SERVICE_UNAVAILABLE {
        headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    if let Some(retry_after) = e.retry_after()
        && let Ok(seconds) = HeaderValue::from_str(&retry_after.as_secs().to_string())
    {
        headers.insert(header::RETRY_AFTER, seconds);
    }

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
        (status = 408, description = "Body not received in time", body = ErrorBody),
        (status = 413, description = "Body larger than the limit", body = ErrorBody),
        (status = 500, description = "Detection failed", body = ErrorBody),
        (status = 503, description = "Too many `lingua` detections at once; retry later", body = ErrorBody),
    )
)]
async fn detect(State(service): State<Service>, body: Result<Bytes, BytesRejection>) -> Response {
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
            // understood but cannot be processed. serde_json reports some wrong types as syntax
            // errors (`"method": null`), so the body is parsed again without a schema to tell.
            let valid_json = e.classify() == Category::Data || serde_json::from_slice::<IgnoredAny>(&body).is_ok();
            let (status, detail) = if valid_json {
                (StatusCode::UNPROCESSABLE_ENTITY, "Body validation failed")
            } else {
                (StatusCode::BAD_REQUEST, "Body is not valid JSON")
            };
            let extra = vec![ErrorExtra { message: e.to_string() }];

            return error(status, detail, extra);
        }
    };

    match service.detect(request.text, request.method).await {
        Ok(detection) => (StatusCode::OK, Json(detection)).into_response(),
        Err(e) => service_error(&e),
    }
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
    description = "Answers `200` when the service is not shutting down and `mediapipe` detects the language of a fixed English text correctly (about 20 µs). Answers `503` once the service starts shutting down, a few seconds before it stops accepting connections (`SHUTDOWN_DELAY_SECONDS`), so that traffic can move to other instances; requests are still served meanwhile. Load does not affect it: excess `lingua` requests get their own `503`.",
    responses(
        (status = 200, description = "The service can take requests", body = Health),
        (status = 503, description = "The service is shutting down or failed its self-check", body = ErrorBody),
    )
)]
async fn ready(State(service): State<Service>) -> Response {
    match service.check_ready() {
        Ok(()) => Json(Health { status: "ok" }).into_response(),
        Err(e) => service_error(&e),
    }
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

    async fn stall() -> &'static str {
        tokio::time::sleep(Duration::from_secs(60)).await;
        "too late"
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

    #[tokio::test]
    async fn slow_request_returns_408_and_closes_the_connection() {
        let limits = Limits {
            request_timeout: Duration::from_millis(10),
            ..Limits::default()
        };
        let app = with_middleware(Router::new().route("/stall", get(stall)), limits);

        let request = Request::get("/stall")
            .header(header::ORIGIN, "https://example.com")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(response.headers()[header::CONNECTION], "close");
        assert_eq!(response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN], "*");
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body, json!({"status_code": 408, "detail": "Request timed out"}));
    }
}
