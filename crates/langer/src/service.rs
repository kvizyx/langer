//! Language detection independent of the transport: input checks, the `lingua` slots, running
//! Lingua off the async threads, readiness, and errors that transport maps to its own codes.

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use metrics::{counter, gauge, histogram};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use crate::detect::{Detection, Detectors, Method};
use crate::telemetry::{DETECTION_DURATION, DETECTIONS, LINGUA_SLOT_WAIT, LINGUA_SLOTS, LINGUA_SLOTS_IN_USE};

/// `lingua` detections allowed in flight per CPU core: one running, three waiting.
const LINGUA_SLOTS_PER_CORE: NonZeroUsize = NonZeroUsize::new(4).expect("4 is not zero");

/// How long an overloaded client should wait before retrying: the shortest `Retry-After`, as it
/// counts whole seconds. Requests wait for a free slot first, so a refused one meets a load
/// that a few milliseconds did not clear.
const RETRY_AFTER: Duration = Duration::from_secs(1);

/// Default longest wait for a free `lingua` slot: enough for a burst to clear, as a detection
/// takes a few milliseconds, and short against the client's own timeouts.
pub const DEFAULT_LINGUA_QUEUE_TIMEOUT: Duration = Duration::from_millis(100);

/// Text and expected language of the readiness self-check.
const SELF_CHECK: (&str, &str) = ("Hello, how are you today?", "en");

/// Default limit of `lingua` detections in flight: four per CPU core, one running and three
/// waiting.
#[must_use]
pub fn default_lingua_in_flight() -> NonZeroUsize {
    let cores = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    cores.saturating_mul(LINGUA_SLOTS_PER_CORE)
}

/// Limits of `lingua` detections, which take a thread each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinguaLimits {
    /// Most detections running or waiting for a thread at once, the `lingua` slots.
    pub in_flight: NonZeroUsize,
    /// Longest wait for a free slot; a request still without one is refused as overloaded.
    pub queue_timeout: Duration,
}

impl Default for LinguaLimits {
    fn default() -> Self {
        Self {
            in_flight: default_lingua_in_flight(),
            queue_timeout: DEFAULT_LINGUA_QUEUE_TIMEOUT,
        }
    }
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
    lingua_queue_timeout: Duration,
    shutting_down: AtomicBool,
}

impl Service {
    /// Creates the service. At most `lingua.in_flight` `lingua` detections run or wait for a
    /// thread at once; more wait up to `lingua.queue_timeout` for a slot, in order of arrival,
    /// and are refused as overloaded if none frees up.
    #[must_use]
    pub fn new(detectors: Detectors, lingua: LinguaLimits) -> Self {
        #[expect(clippy::cast_precision_loss, reason = "a slot count is far below 2^52")]
        gauge!(LINGUA_SLOTS).set(lingua.in_flight.get() as f64);

        Self {
            inner: Arc::new(Inner {
                detectors,
                lingua_slots: Arc::new(Semaphore::new(lingua.in_flight.get())),
                lingua_queue_timeout: lingua.queue_timeout,
                shutting_down: AtomicBool::new(false),
            }),
        }
    }

    /// Detects the language of `text` with `method`.
    ///
    /// `mediapipe` runs on the calling task: it takes tens of microseconds. `lingua` waits for
    /// one of the `lingua` slots, then runs on a blocking thread, holding the slot until it ends,
    /// even if the caller is gone by then. A caller that is gone while waiting leaves the queue.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is empty or whitespace only
    /// ([`ServiceError::is_invalid_input`]), if no `lingua` slot frees up in time
    /// ([`ServiceError::is_unavailable`]), or if the detection itself fails.
    pub async fn detect(&self, text: String, method: Method) -> Result<Detection, ServiceError> {
        if text.trim().is_empty() {
            count(method, "invalid_input");
            return Err(ServiceError::new(ErrorKind::EmptyText));
        }

        let started = Instant::now();
        let text_len = text.chars().count();

        let detection: Result<Detection, BoxError> = match method {
            Method::MediaPipe => self.inner.detectors.detect(&text, method).map_err(Into::into),
            Method::Lingua => {
                let Some(slot) = LinguaSlot::take(&self.inner.lingua_slots, self.inner.lingua_queue_timeout).await
                else {
                    tracing::event!(
                        name: "detect.lingua.overloaded",
                        tracing::Level::WARN,
                        "no lingua slot freed up in time",
                    );
                    count(method, "overloaded");
                    return Err(ServiceError::new(ErrorKind::Overloaded));
                };

                let service = self.clone();
                tokio::task::spawn_blocking(move || {
                    let _slot = slot;
                    service.inner.detectors.detect(&text, method)
                })
                .await
                .map_err(Into::into)
                .and_then(|result| result.map_err(Into::into))
            }
        };

        let detection = match detection {
            Ok(detection) => detection,
            Err(source) => {
                tracing::event!(
                    name: "detect.request.failure",
                    tracing::Level::ERROR,
                    detect.method = %method,
                    error.message = %source,
                    "language detection failed",
                );
                count(method, "failed");

                return Err(ServiceError::caused_by(ErrorKind::DetectionFailed, source));
            }
        };

        let elapsed = started.elapsed();
        count(method, "ok");
        histogram!(DETECTION_DURATION, "method" => method.as_str()).record(elapsed.as_secs_f64());

        tracing::event!(
            name: "detect.request.success",
            tracing::Level::DEBUG,
            detect.method = %detection.method,
            detect.language = %detection.language,
            detect.confidence = detection.confidence,
            text.length = text_len,
            duration_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
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

/// Counts a detection request by method and outcome.
fn count(method: Method, outcome: &'static str) {
    counter!(DETECTIONS, "method" => method.as_str(), "outcome" => outcome).increment(1);
}

/// A taken `lingua` slot, counted in [`LINGUA_SLOTS_IN_USE`] until it is dropped.
struct LinguaSlot {
    _permit: OwnedSemaphorePermit,
}

impl LinguaSlot {
    /// Takes a free slot, waiting up to `timeout` for one, and records the wait.
    async fn take(slots: &Arc<Semaphore>, timeout: Duration) -> Option<Self> {
        let started = Instant::now();

        // Without waiting first, so that the common case needs no timer.
        let permit = match Arc::clone(slots).try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::NoPermits) => tokio::time::timeout(timeout, Arc::clone(slots).acquire_owned())
                .await
                .ok()?
                .ok()?,
            Err(TryAcquireError::Closed) => return None,
        };

        histogram!(LINGUA_SLOT_WAIT).record(started.elapsed().as_secs_f64());
        gauge!(LINGUA_SLOTS_IN_USE).increment(1.0);

        Some(Self { _permit: permit })
    }
}

impl Drop for LinguaSlot {
    fn drop(&mut self) {
        gauge!(LINGUA_SLOTS_IN_USE).decrement(1.0);
    }
}

impl fmt::Debug for Service {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Service")
            .field("detectors", &self.inner.detectors)
            .field("lingua_slots_free", &self.inner.lingua_slots.available_permits())
            .field("lingua_queue_timeout", &self.inner.lingua_queue_timeout)
            .field("shutting_down", &self.inner.shutting_down.load(Ordering::Relaxed))
            .finish()
    }
}

/// Error of a detector or of the thread it ran on.
type BoxError = Box<dyn Error + Send + Sync>;

/// Why the service could not answer a request.
#[derive(Debug)]
pub struct ServiceError {
    kind: ErrorKind,
    /// Explanation of the problem, e.g. the self-check result.
    details: Option<String>,
    /// The error that caused this one, e.g. the detector's.
    source: Option<BoxError>,
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
        Self {
            kind,
            details: None,
            source: None,
        }
    }

    fn with_details(kind: ErrorKind, details: String) -> Self {
        Self {
            details: Some(details),
            ..Self::new(kind)
        }
    }

    fn caused_by(kind: ErrorKind, source: BoxError) -> Self {
        Self {
            source: Some(source),
            ..Self::new(kind)
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

impl Error for ServiceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|e| e as _)
    }
}

#[cfg(test)]
mod tests {
    use langer_mediapipe::Detector;
    use lingua::Language::{English, French, Russian};
    use lingua::LanguageDetectorBuilder;
    use rstest::rstest;

    use super::*;

    /// A service with `lingua_in_flight` slots that refuses `lingua` requests at once when they
    /// are taken.
    fn service(lingua_in_flight: usize) -> Service {
        service_waiting(lingua_in_flight, Duration::ZERO)
    }

    fn service_waiting(lingua_in_flight: usize, queue_timeout: Duration) -> Service {
        let lingua = LanguageDetectorBuilder::from_languages(&[English, French, Russian]).build();
        let detectors = Detectors::new(Detector::new().unwrap(), lingua);
        let limits = LinguaLimits {
            in_flight: NonZeroUsize::new(lingua_in_flight).unwrap(),
            queue_timeout,
        };

        Service::new(detectors, limits)
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
    async fn detect_with_lingua_waits_for_a_slot_within_the_queue_timeout() {
        let service = service_waiting(1, Duration::from_secs(30));
        let text = "Это обычное сообщение на русском языке. ".repeat(500);

        // The second detection waits for the slot of the first one.
        let (first, second) = tokio::join!(
            service.detect(text.clone(), Method::Lingua),
            service.detect(text, Method::Lingua),
        );

        assert_eq!(first.unwrap().language, "ru");
        assert_eq!(second.unwrap().language, "ru");
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
