//! Language detection independent of the transport: input checks, the `lingua` slots, running
//! Lingua off the async threads, readiness, and errors that a transport maps to its own codes.

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use tokio::sync::Semaphore;

use crate::detect::{Detection, Detectors, Method};

/// `lingua` detections allowed in flight per CPU core: one running, three waiting.
const LINGUA_SLOTS_PER_CORE: NonZeroUsize = match NonZeroUsize::new(4) {
    Some(slots) => slots,
    None => NonZeroUsize::MIN,
};

/// How long an overloaded client should wait before retrying.
const RETRY_AFTER: Duration = Duration::from_secs(1);

/// Text and expected language of the readiness self-check.
const SELF_CHECK: (&str, &str) = ("Hello, how are you today?", "en");

/// Default limit of `lingua` detections in flight: four per CPU core, one running and three
/// waiting.
#[must_use]
pub fn default_lingua_in_flight() -> NonZeroUsize {
    let cores = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    cores.saturating_mul(LINGUA_SLOTS_PER_CORE)
}

/// The language detection service, shared by every request. Cloning gives a handle to the same
/// service.
#[derive(Clone)]
pub struct Service {
    inner: Arc<Inner>,
}

struct Inner {
    detectors: Detectors,
    /// One permit per `lingua` detection allowed in flight.
    lingua_slots: Arc<Semaphore>,
    shutting_down: AtomicBool,
}

impl Service {
    /// Creates the service. At most `lingua_in_flight` `lingua` detections run or wait for a
    /// thread at once; more are refused as overloaded rather than queued without bound.
    #[must_use]
    pub fn new(detectors: Detectors, lingua_in_flight: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(Inner {
                detectors,
                lingua_slots: Arc::new(Semaphore::new(lingua_in_flight.get())),
                shutting_down: AtomicBool::new(false),
            }),
        }
    }

    /// Detects the language of `text` with `method`.
    ///
    /// `mediapipe` runs on the calling task: it takes tens of microseconds. `lingua` runs on a
    /// blocking thread, holding one of the `lingua` slots until it ends, even if the caller is
    /// gone by then.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is empty or whitespace only
    /// ([`ServiceError::is_invalid_input`]), if every `lingua` slot is taken
    /// ([`ServiceError::is_unavailable`]), or if the detection itself fails.
    pub async fn detect(&self, text: String, method: Method) -> Result<Detection, ServiceError> {
        if text.trim().is_empty() {
            return Err(ServiceError::new(ErrorKind::EmptyText));
        }

        let started = Instant::now();
        let text_len = text.chars().count();

        let detection = match method {
            Method::MediaPipe => self.inner.detectors.detect(&text, method).map_err(|e| e.to_string()),
            Method::Lingua => {
                let Ok(slot) = Arc::clone(&self.inner.lingua_slots).try_acquire_owned() else {
                    tracing::event!(
                        name: "detect.lingua.overloaded",
                        tracing::Level::WARN,
                        "too many lingua detections at once",
                    );
                    return Err(ServiceError::new(ErrorKind::Overloaded));
                };

                let service = self.clone();
                tokio::task::spawn_blocking(move || {
                    let _slot = slot;
                    service.inner.detectors.detect(&text, method)
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

                return Err(ServiceError::with_details(ErrorKind::DetectionFailed, message));
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

        Ok(detection)
    }

    /// Checks that the service can take requests: it is not shutting down, and `mediapipe`
    /// detects a fixed English text correctly (about 20 µs). Load does not affect the answer.
    ///
    /// # Errors
    ///
    /// Returns an [unavailable](ServiceError::is_unavailable) error while shutting down or if the
    /// self-check fails.
    pub fn check_ready(&self) -> Result<(), ServiceError> {
        if self.inner.shutting_down.load(Ordering::Relaxed) {
            return Err(ServiceError::new(ErrorKind::ShuttingDown));
        }

        let (text, expected) = SELF_CHECK;
        let problem = match self.inner.detectors.detect(text, Method::MediaPipe) {
            Ok(detection) if detection.language == expected => return Ok(()),
            Ok(detection) => format!("detected {:?} instead of {expected:?}", detection.language),
            Err(e) => e.to_string(),
        };

        tracing::event!(
            name: "health.self_check.failure",
            tracing::Level::ERROR,
            error.message = problem,
            "readiness self-check failed",
        );

        Err(ServiceError::with_details(ErrorKind::SelfCheckFailed, problem))
    }

    /// Marks the service as shutting down: [`Service::check_ready`] fails from now on, while
    /// detections are still served.
    pub fn shut_down(&self) {
        self.inner.shutting_down.store(true, Ordering::Relaxed);
    }
}

impl fmt::Debug for Service {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Service")
            .field("detectors", &self.inner.detectors)
            .field("lingua_slots_free", &self.inner.lingua_slots.available_permits())
            .field("shutting_down", &self.inner.shutting_down.load(Ordering::Relaxed))
            .finish()
    }
}

/// Why the service could not answer a request.
#[derive(Debug)]
pub struct ServiceError {
    kind: ErrorKind,
    /// Explanation of the problem, e.g. the self-check result or the detector's error.
    details: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErrorKind {
    EmptyText,
    Overloaded,
    ShuttingDown,
    SelfCheckFailed,
    DetectionFailed,
}

impl ServiceError {
    fn new(kind: ErrorKind) -> Self {
        Self { kind, details: None }
    }

    fn with_details(kind: ErrorKind, details: String) -> Self {
        Self {
            kind,
            details: Some(details),
        }
    }

    /// The request itself is invalid, e.g. its text is empty; retrying it cannot help.
    #[must_use]
    pub fn is_invalid_input(&self) -> bool {
        self.kind == ErrorKind::EmptyText
    }

    /// The service cannot take the request now (overloaded, shutting down or failing its
    /// self-check); the same request may succeed later or on another instance.
    #[must_use]
    pub fn is_unavailable(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::Overloaded | ErrorKind::ShuttingDown | ErrorKind::SelfCheckFailed
        )
    }

    /// How long to wait before retrying, when that is known.
    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        (self.kind == ErrorKind::Overloaded).then_some(RETRY_AFTER)
    }

    /// Short description of the problem, fit to show to clients.
    #[must_use]
    pub fn summary(&self) -> &'static str {
        match self.kind {
            ErrorKind::EmptyText => "Text cannot be empty",
            ErrorKind::Overloaded => "Too many lingua detections at once, retry later",
            ErrorKind::ShuttingDown => "Shutting down",
            ErrorKind::SelfCheckFailed => "Self-check failed",
            ErrorKind::DetectionFailed => "Language detection failed",
        }
    }

    /// Details fit to show to clients, if any. Internal failures keep theirs out of responses:
    /// they are logged instead.
    #[must_use]
    pub fn client_details(&self) -> Option<&str> {
        match self.kind {
            ErrorKind::SelfCheckFailed => self.details.as_deref(),
            _ => None,
        }
    }
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.summary())?;
        if let Some(details) = &self.details {
            write!(f, ": {details}")?;
        }

        Ok(())
    }
}

impl Error for ServiceError {}

#[cfg(test)]
mod tests {
    use langer_mediapipe::Detector;
    use lingua::Language::{English, French, Russian};
    use lingua::LanguageDetectorBuilder;
    use rstest::rstest;

    use super::*;

    fn service(lingua_in_flight: usize) -> Service {
        let lingua = LanguageDetectorBuilder::from_languages(&[English, French, Russian]).build();
        let detectors = Detectors::new(Detector::new().unwrap(), lingua);

        Service::new(detectors, NonZeroUsize::new(lingua_in_flight).unwrap())
    }

    #[rstest]
    #[case::empty("")]
    #[case::whitespace(" \n\t")]
    #[tokio::test]
    async fn detect_rejects_blank_text_as_invalid_input(#[case] text: &str) {
        let err = service(1).detect(text.to_owned(), Method::MediaPipe).await.unwrap_err();

        assert!(err.is_invalid_input());
        assert!(!err.is_unavailable());
        assert_eq!(err.summary(), "Text cannot be empty");
    }

    #[rstest]
    #[tokio::test]
    async fn detect_answers_with_either_method(#[values(Method::MediaPipe, Method::Lingua)] method: Method) {
        let detection = service(1).detect("Привет, как дела?".to_owned(), method).await.unwrap();

        assert_eq!(detection.language, "ru");
        assert_eq!(detection.method, method);
    }

    #[tokio::test]
    async fn detect_with_lingua_beyond_the_slots_is_unavailable() {
        let service = service(1);
        let text = "Это обычное сообщение на русском языке. ".repeat(500);

        // The first detection takes the only slot before the second one is polled.
        let (first, second) = tokio::join!(
            service.detect(text.clone(), Method::Lingua),
            service.detect(text, Method::Lingua),
        );

        first.unwrap();
        let err = second.unwrap_err();
        assert!(err.is_unavailable());
        assert_eq!(err.retry_after(), Some(RETRY_AFTER));
    }

    #[tokio::test]
    async fn detect_with_lingua_frees_its_slot_when_done() {
        let service = service(1);

        for _ in 0..3 {
            service.detect("bonjour".to_owned(), Method::Lingua).await.unwrap();
        }
    }

    #[test]
    fn check_ready_passes_until_shut_down() {
        let service = service(1);
        service.check_ready().unwrap();

        service.shut_down();

        let err = service.check_ready().unwrap_err();
        assert!(err.is_unavailable());
        assert_eq!(err.retry_after(), None);
        assert_eq!(err.summary(), "Shutting down");
    }
}
