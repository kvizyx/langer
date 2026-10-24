//! Metrics: their names, recorded through the `metrics` facade, and the Prometheus exporter that
//! renders them for `GET /metrics`.

use metrics::{Unit, describe_counter, describe_gauge, describe_histogram};
use metrics_exporter_prometheus::{BuildError, Matcher, PrometheusBuilder, PrometheusHandle};

/// Detection requests, by `method` and `outcome` (`ok`, `invalid_input`, `overloaded`, `failed`).
pub(crate) const DETECTIONS: &str = "langer_detections_total";
/// Time of successful detections, by `method`.
pub(crate) const DETECTION_DURATION: &str = "langer_detection_duration_seconds";
/// Most `lingua` detections allowed in flight.
pub(crate) const LINGUA_SLOTS: &str = "langer_lingua_slots";
/// `lingua` detections in flight.
pub(crate) const LINGUA_SLOTS_IN_USE: &str = "langer_lingua_slots_in_use";
/// HTTP requests to known routes, by `method`, `route` and `status`.
pub(crate) const HTTP_REQUESTS: &str = "langer_http_requests_total";
/// Time to answer HTTP requests to known routes, by `method` and `route`.
pub(crate) const HTTP_REQUEST_DURATION: &str = "langer_http_request_duration_seconds";

/// Bucket bounds of the duration histograms, in seconds: from `mediapipe`'s tens of
/// microseconds to `lingua` on long texts.
const DURATION_BUCKETS: [f64; 17] = [
    0.000_01, 0.000_025, 0.000_05, 0.000_1, 0.000_25, 0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
    0.5, 1.0, 2.5,
];

/// Installs the Prometheus recorder for the whole process and returns the handle that renders
/// the metrics.
///
/// Call it once, before creating the [`Service`](crate::Service), so that its metrics are
/// recorded. The handle needs [`PrometheusHandle::run_upkeep`] every few seconds, which keeps
/// histogram data from growing between scrapes.
///
/// # Errors
///
/// Returns an error if a recorder is already installed.
pub fn install_metrics() -> Result<PrometheusHandle, BuildError> {
    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(Matcher::Suffix("_duration_seconds".to_owned()), &DURATION_BUCKETS)?
        .install_recorder()?;

    describe_counter!(DETECTIONS, "Detection requests, by method and outcome.");
    describe_histogram!(
        DETECTION_DURATION,
        Unit::Seconds,
        "Time of successful detections, by method."
    );
    describe_gauge!(LINGUA_SLOTS, "Most lingua detections allowed in flight.");
    describe_gauge!(LINGUA_SLOTS_IN_USE, "Lingua detections in flight.");
    describe_counter!(
        HTTP_REQUESTS,
        "HTTP requests to known routes, by method, route and status."
    );
    describe_histogram!(
        HTTP_REQUEST_DURATION,
        Unit::Seconds,
        "Time to answer HTTP requests to known routes, by method and route."
    );

    Ok(handle)
}
