//! HTTP service that detects the language of short texts such as chat messages.
//!
//! Two detectors are available through `POST /detect`:
//!
//! - `mediapipe` (default): MediaPipe's language detector via [`langer_mediapipe`], a
//!   ~300 KiB model answering in tens of microseconds;
//! - `lingua`: [Lingua](https://github.com/pemistahl/lingua-rs) in high-accuracy mode, the
//!   most accurate on one or two words, but around a millisecond per text. Its models are
//!   embedded in the executable and paged in on first use: up to ~250 MiB of reclaimable,
//!   file-backed memory rather than heap.
//!
//! `GET /detect/languages` lists the supported languages, `GET /health` and
//! `GET /health/ready` report the service's state, `GET /metrics` serves Prometheus metrics,
//! `GET /schema` the API reference (Scalar) and `GET /schema/openapi.json` the `OpenAPI` document.
//!
//! The crate has three layers:
//!
//! - [`Detectors`]: the two detectors, synchronous;
//! - [`Service`]: everything a request needs regardless of the transport: input checks, the
//!   limit of `lingua` detections in flight, running Lingua off the async threads, readiness, and
//!   [`ServiceError`]s that say what went wrong without choosing a status code;
//! - [`router`]: the HTTP API over a [`Service`]: request decoding, body limits, status codes and
//!   headers, the `OpenAPI` document, and `GET /metrics`.
//!
//! Both [`Service`] and [`router`] record metrics through the `metrics` facade;
//! [`install_metrics`] installs the Prometheus exporter that renders them. The binary adds
//! configuration, logging, metrics upkeep and graceful shutdown.

mod detect;
mod http;
mod languages;
mod service;
mod telemetry;

#[doc(inline)]
pub use detect::{DetectError, Detection, Detectors, Method};
#[doc(inline)]
pub use http::{DEFAULT_BODY_LIMIT, router};
#[doc(inline)]
pub use service::{Service, ServiceError, default_lingua_in_flight};
#[doc(inline)]
pub use telemetry::install_metrics;
