//! Language detectors behind `POST /detect`.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use langer_mediapipe::{Detector, PredictError};
use lingua::LanguageDetector;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Language code reported when no language could be detected.
const UNKNOWN: &str = "unknown";

/// Detection method requested by a client.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
#[schema(description = "Detection method requested by a client.")]
pub enum Method {
    /// MediaPipe's language detector (110 languages): small and fast.
    #[default]
    MediaPipe,
    /// Lingua in high-accuracy mode (75 languages): most accurate on one or two words.
    Lingua,
}

impl fmt::Display for Method {
    /// Writes the method as it appears in the API, e.g. `mediapipe`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MediaPipe => "mediapipe",
            Self::Lingua => "lingua",
        })
    }
}

/// Result of a detection.
#[derive(Clone, Debug, PartialEq, Serialize, ToSchema)]
#[schema(description = "Result of a detection.")]
pub struct Detection {
    /// Language code. MediaPipe reports its own codes (e.g. `iw`, `fil`, `zh-Latn`) and
    /// `unknown`; Lingua reports ISO 639-1 codes, or `unknown` when it cannot decide.
    #[schema(example = "en")]
    pub language: String,
    /// Confidence in `0.0..=1.0`; `null` when Lingua cannot decide.
    #[schema(example = 0.97)]
    pub confidence: Option<f64>,
    /// Method that produced the result.
    pub method: Method,
}

/// Shared, cheaply cloneable set of detectors.
#[derive(Clone)]
pub struct Detectors {
    inner: Arc<Inner>,
}

struct Inner {
    mediapipe: Detector,
    lingua: LanguageDetector,
}

impl Detectors {
    /// Creates detectors from a MediaPipe detector and a Lingua detector.
    #[must_use]
    pub fn new(mediapipe: Detector, lingua: LanguageDetector) -> Self {
        Self {
            inner: Arc::new(Inner { mediapipe, lingua }),
        }
    }

    /// Detects the language of `text` with `method`.
    ///
    /// MediaPipe answers in tens of microseconds. Lingua takes up to a few milliseconds per
    /// text (more while its models are paged in for the first time), so async callers
    /// should run it on a blocking thread.
    ///
    /// # Errors
    ///
    /// Returns an error only if MediaPipe fails, which it does not with a valid model.
    pub fn detect(&self, text: &str, method: Method) -> Result<Detection, DetectError> {
        match method {
            Method::MediaPipe => self.detect_mediapipe(text),
            Method::Lingua => Ok(self.detect_lingua(text)),
        }
    }

    fn detect_mediapipe(&self, text: &str) -> Result<Detection, DetectError> {
        let top = self.inner.mediapipe.predict(text, 1).map_err(|e| DetectError {
            message: "MediaPipe failed",
            source: Some(e),
        })?;
        let top = top.first().ok_or(DetectError {
            message: "MediaPipe returned no prediction",
            source: None,
        })?;
        Ok(Detection {
            language: top.language.to_owned(),
            confidence: Some(f64::from(top.probability)),
            method: Method::MediaPipe,
        })
    }

    /// Lingua's `detect_language_of`, then the confidence of the detected language.
    fn detect_lingua(&self, text: &str) -> Detection {
        let lingua = &self.inner.lingua;
        let (language, confidence) = match lingua.detect_language_of(text) {
            Some(language) => (
                language.iso_code_639_1().to_string(),
                Some(lingua.compute_language_confidence(text, language)),
            ),
            None => (UNKNOWN.to_owned(), None),
        };
        Detection {
            language,
            confidence,
            method: Method::Lingua,
        }
    }
}

/// A detector failed to produce a result.
#[derive(Debug)]
pub struct DetectError {
    message: &'static str,
    source: Option<PredictError>,
}

impl fmt::Display for DetectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "language detection failed: {}", self.message)?;
        if let Some(source) = &self.source {
            write!(f, ": {source}")?;
        }
        Ok(())
    }
}

impl Error for DetectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|e| e as _)
    }
}

impl fmt::Debug for Detectors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Detectors")
            .field("mediapipe", &self.inner.mediapipe)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::Method;

    #[rstest]
    fn method_display_matches_api_name(#[values(Method::MediaPipe, Method::Lingua)] method: Method) {
        assert_eq!(serde_json::to_string(&method).unwrap(), format!("\"{method}\""));
    }
}
