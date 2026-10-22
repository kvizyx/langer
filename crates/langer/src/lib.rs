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
//! `GET /detect/languages` lists the supported languages, `GET /schema` serves the API
//! reference (Scalar) and `GET /schema/openapi.json` the `OpenAPI` document.
//!
//! [`router`] builds the service from [`Detectors`]; the binary adds configuration,
//! logging and graceful shutdown.

mod detect;
mod http;
mod languages;

#[doc(inline)]
pub use detect::{Detection, Detectors, Method};
#[doc(inline)]
pub use http::router;
