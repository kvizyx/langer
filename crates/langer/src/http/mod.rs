//! HTTP routes, request handling and the `OpenAPI` document.

mod body;
mod error;

use std::any::Any;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use metrics::{counter, histogram};
use metrics_exporter_prometheus::PrometheusHandle;
use serde::{Deserialize, Serialize};
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::CorsLayer;
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_scalar::Scalar;

use crate::detect::{Detection, Method};
use crate::languages::{self, Language};
use crate::service::Service;
use crate::telemetry::{HTTP_REQUEST_DURATION, HTTP_REQUESTS};

use body::RequestBody;
use error::{ApiError, ErrorBody};

/// Where the API reference is served; `/` and `/docs` redirect here.
const DOCS_PATH: &str = "/schema";

/// Content type of the Prometheus text format.
const PROMETHEUS_TEXT: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Page of the API reference: Scalar with the `deepSpace` theme, pinned to one version.
const SCALAR_TEMPLATE: &str = include_str!("../../assets/scalar.html");

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

    let docs = Router::new()
        .route(DOCS_PATH, get(docs_page))
        .route("/schema/openapi.json", get(openapi_document))
        .with_state(Docs::new(openapi));
    let metrics = Router::new().route("/metrics", get(render_metrics)).with_state(metrics);

    let app = api
        .merge(docs)
        .route("/", get(redirect_to_docs))
        .route("/docs", get(redirect_to_docs))
        // Only matched routes, so that unknown paths cannot add metric series; `/metrics` comes
        // after, so that scrapes are not counted.
        .route_layer(middleware::from_fn(track_metrics))
        .merge(metrics)
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
    tokio::time::timeout(timeout, next.run(request))
        .await
        .unwrap_or_else(|_| {
            ApiError::new(StatusCode::REQUEST_TIMEOUT, "Request timed out")
                .closing()
                .into_response()
        })
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

    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error").into_response()
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

/// Metrics in the Prometheus text format.
async fn render_metrics(State(metrics): State<PrometheusHandle>) -> impl IntoResponse {
    ([(header::CONTENT_TYPE, PROMETHEUS_TEXT)], metrics.render())
}

/// The API reference page and the `OpenAPI` document, rendered once.
///
/// axum clones the state for every connection, and `Bytes` clones share the data: a captured
/// `OpenApi` took ~100 KiB per connection.
#[derive(Clone)]
struct Docs {
    page: Bytes,
    /// `None` if the document could not be serialized.
    spec: Option<Bytes>,
}

impl Docs {
    fn new(openapi: utoipa::openapi::OpenApi) -> Self {
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

        let page = Scalar::new(openapi)
            .title("Langer API")
            .custom_html(SCALAR_TEMPLATE)
            .to_html();

        Self {
            page: Bytes::from(page),
            spec,
        }
    }
}

async fn docs_page(State(docs): State<Docs>) -> Html<Bytes> {
    Html(docs.page)
}

/// The `OpenAPI` document, or `500` if it could not be serialized at startup.
async fn openapi_document(State(docs): State<Docs>) -> Result<impl IntoResponse, ApiError> {
    let spec = docs.spec.ok_or(ApiError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "OpenAPI document unavailable",
    ))?;

    Ok(([(header::CONTENT_TYPE, "application/json")], spec))
}

async fn redirect_to_docs() -> Response {
    // Permanent: these paths are fixed aliases of the API reference.
    (StatusCode::PERMANENT_REDIRECT, [(header::LOCATION, DOCS_PATH)]).into_response()
}

async fn not_found() -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "Not found")
}

async fn method_not_allowed() -> ApiError {
    ApiError::new(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed")
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
        (status = 503, description = "No `lingua` slot freed up in time; retry later", body = ErrorBody),
    )
)]
async fn detect(
    State(service): State<Service>,
    RequestBody(request): RequestBody<DetectRequest>,
) -> Result<Json<Detection>, ApiError> {
    let detection = service.detect(request.text, request.method).await?;

    Ok(Json(detection))
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
async fn ready(State(service): State<Service>) -> Result<Json<Health>, ApiError> {
    service.check_ready()?;

    Ok(Json(Health { status: "ok" }))
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
