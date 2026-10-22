use std::backtrace::{Backtrace, BacktraceStatus};
use std::borrow::Cow;
use std::fmt;

/// Error returned when a model cannot be loaded.
///
/// Loading only fails for data that is not a supported MediaPipe language detector model,
/// so there is a single failure kind; [`source`](std::error::Error::source) carries the
/// underlying flatbuffer, flexbuffer or zip error where there is one.
#[derive(Debug)]
pub struct LoadError {
    message: Cow<'static, str>,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
    backtrace: Backtrace,
}

impl LoadError {
    pub(crate) fn new(message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            message: message.into(),
            source: None,
            backtrace: Backtrace::capture(),
        }
    }

    pub(crate) fn with_source(
        message: impl Into<Cow<'static, str>>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            source: Some(Box::new(source)),
            ..Self::new(message)
        }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid language detector model: {}", self.message)?;
        if let Some(source) = &self.source {
            write!(f, ": {source}")?;
        }
        if self.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n{}", self.backtrace)?;
        }
        Ok(())
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|e| e as _)
    }
}

/// Error returned when a prediction cannot be computed.
///
/// Only an inconsistency inside a loaded model can cause it, and loading checks the model
/// for those; it is returned rather than panicking in case a check is missing.
#[derive(Debug)]
pub struct PredictError {
    message: &'static str,
    backtrace: Backtrace,
}

impl PredictError {
    pub(crate) fn new(message: &'static str) -> Self {
        Self {
            message,
            backtrace: Backtrace::capture(),
        }
    }
}

impl fmt::Display for PredictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "language prediction failed: {}", self.message)?;
        if self.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n{}", self.backtrace)?;
        }
        Ok(())
    }
}

impl std::error::Error for PredictError {}
