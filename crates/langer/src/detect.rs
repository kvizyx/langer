//! Language detectors behind `POST /detect`.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use langer_mediapipe::{Detector, PredictError};
use lingua::{Language, LanguageDetector};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// BCP 47 tag reported when no language could be detected.
const UNDETERMINED: &str = "und";

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

impl Method {
    /// The method as it appears in the API, e.g. `mediapipe`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MediaPipe => "mediapipe",
            Self::Lingua => "lingua",
        }
    }
}

impl fmt::Display for Method {
    /// Writes the method as it appears in the API, e.g. `mediapipe`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result of a detection.
#[derive(Clone, Debug, PartialEq, Serialize, ToSchema)]
#[schema(description = "Result of a detection.")]
pub struct Detection {
    /// BCP 47 tag, the same for both methods and listed by `GET /detect/languages`: an
    /// ISO 639-1 code where one exists (`en`), else ISO 639-3 (`ceb`), with a script subtag for
    /// romanized text (`ru-Latn`, `mediapipe` only); `und` when no language is detected.
    #[schema(example = "en")]
    pub language: String,
    /// Confidence in `0.0..=1.0`; `null` with `und`.
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

    /// Detects the language of `text` with `method`: `und` if it has no letters, as only emoji,
    /// digits or punctuation do.
    ///
    /// MediaPipe answers in tens of microseconds. Lingua takes up to a few milliseconds per
    /// text (more while its models are paged in for the first time), so async callers
    /// should run it on a blocking thread.
    ///
    /// # Errors
    ///
    /// Returns an error only if MediaPipe fails, which it does not with a valid model.
    pub fn detect(&self, text: &str, method: Method) -> Result<Detection, DetectError> {
        // MediaPipe guesses a language for any text, Lingua only with letters of its languages.
        if !text.chars().any(char::is_alphabetic) {
            return Ok(Detection {
                language: UNDETERMINED.to_owned(),
                confidence: None,
                method,
            });
        }

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

        // The model's own `unknown` label is no language, so it has no confidence either.
        let language = tag(top.language);
        let confidence = (language != UNDETERMINED).then(|| widen(top.probability));

        Ok(Detection {
            language: language.to_owned(),
            confidence,
            method: Method::MediaPipe,
        })
    }

    /// Lingua's `detect_language_of` and the confidence of the detected language, from one
    /// computation of the confidence values instead of two: `compute_language_confidence`
    /// computes all of them again.
    fn detect_lingua(&self, text: &str) -> Detection {
        let values = self.inner.lingua.compute_language_confidence_values(text);
        let (language, confidence) = match most_likely(&values) {
            Some((language, confidence)) => (tag(&language.iso_code_639_1().to_string()).to_owned(), Some(confidence)),
            None => (UNDETERMINED.to_owned(), None),
        };

        Detection {
            language,
            confidence,
            method: Method::Lingua,
        }
    }
}

/// The BCP 47 tag of a code reported by either detector, so that both name a language the same
/// way: MediaPipe's deprecated `iw` and its `fil` and `unknown`, and Lingua's Norwegian written
/// standards `nb` and `nn`, which MediaPipe does not tell apart.
pub(crate) fn tag(code: &str) -> &str {
    match code {
        "iw" => "he",
        "fil" => "tl",
        "nb" | "nn" => "no",
        "unknown" => UNDETERMINED,
        code => code,
    }
}

/// `value` as the `f64` closest to its shortest decimal form, e.g. 0.99982196 rather than
/// 0.9998219609260559: `f64::from` keeps the binary error of the `f32`, which then shows up
/// in the JSON response as noise digits.
fn widen(value: f32) -> f64 {
    value.to_string().parse().unwrap_or_else(|_| f64::from(value))
}

/// The language `detect_language_of` reports for these confidence values, sorted as Lingua
/// returns them, with its confidence: none if the values are empty, all zero or tied for the top.
///
/// Mirrors Lingua 1.8's rules; `tests/detect.rs` checks it against `detect_language_of`.
fn most_likely(values: &[(Language, f64)]) -> Option<(Language, f64)> {
    match *values {
        [] => None,
        [(language, confidence)] => (confidence > 0.0).then_some((language, confidence)),
        [(language, confidence), (_, second), ..] => {
            ((confidence - second).abs() >= f64::EPSILON).then_some((language, confidence))
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

    use super::{Method, tag, widen};

    #[rstest]
    #[case::probability(0.999_821_96, 0.999_821_96)]
    #[case::tenth(0.1, 0.1)]
    #[case::one(1.0, 1.0)]
    #[case::tiny(1e-30, 1e-30)]
    fn widen_keeps_the_decimal_form(#[case] value: f32, #[case] expected: f64) {
        assert_eq!(widen(value).to_string(), expected.to_string());
    }

    #[rstest]
    #[case::deprecated("iw", "he")]
    #[case::filipino("fil", "tl")]
    #[case::bokmal("nb", "no")]
    #[case::nynorsk("nn", "no")]
    #[case::unknown("unknown", "und")]
    #[case::romanized("ru-Latn", "ru-Latn")]
    #[case::plain("en", "en")]
    fn tag_unifies_codes(#[case] code: &str, #[case] expected: &str) {
        assert_eq!(tag(code), expected);
    }

    #[rstest]
    fn method_display_matches_api_name(#[values(Method::MediaPipe, Method::Lingua)] method: Method) {
        assert_eq!(serde_json::to_string(&method).unwrap(), format!("\"{method}\""));
    }
}
