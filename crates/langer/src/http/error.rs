//! Error responses: every error the API answers with has the same JSON body.

use std::time::Duration;

use axum::Json;
use axum::extract::rejection::BytesRejection;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use utoipa::ToSchema;

use crate::service::ServiceError;

/// An error response: a status, an [`ErrorBody`], and the headers that go with the status.
#[derive(Debug)]
pub(super) struct ApiError {
    status: StatusCode,
    detail: &'static str,
    extra: Vec<ErrorExtra>,
    retry_after: Option<Duration>,
    close: bool,
}

impl ApiError {
    pub(super) const fn new(status: StatusCode, detail: &'static str) -> Self {
        Self {
            status,
            detail,
            extra: Vec::new(),
            retry_after: None,
            close: false,
        }
    }

    /// Adds a detail of the problem, e.g. why the body is invalid.
    pub(super) fn with_extra(mut self, message: impl Into<String>) -> Self {
        self.extra.push(ErrorExtra {
            message: message.into(),
        });
        self
    }

    /// Asks the client to close the connection after this response.
    pub(super) const fn closing(mut self) -> Self {
        self.close = true;
        self
    }
}

/// `422` for invalid input, `503` while unavailable, `500` otherwise.
impl From<ServiceError> for ApiError {
    fn from(e: ServiceError) -> Self {
        let status = if e.is_invalid_input() {
            StatusCode::UNPROCESSABLE_ENTITY
        } else if e.is_unavailable() {
            StatusCode::SERVICE_UNAVAILABLE
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };

        let mut error = Self::new(status, e.summary());
        if let Some(details) = e.client_details() {
            error = error.with_extra(details);
        }
        error.retry_after = e.retry_after();

        error
    }
}

/// A body that could not be read, e.g. because it is larger than the limit (`413`).
impl From<BytesRejection> for ApiError {
    fn from(rejection: BytesRejection) -> Self {
        Self::new(rejection.status(), "Cannot read the request body").with_extra(rejection.body_text())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            status_code: self.status.as_u16(),
            detail: self.detail,
            extra: self.extra,
        };
        let mut response = (self.status, Json(body)).into_response();

        let headers = response.headers_mut();
        // An unavailable instance says nothing about the next answer, possibly from another one.
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        }
        if let Some(retry_after) = self.retry_after {
            headers.insert(header::RETRY_AFTER, HeaderValue::from(retry_after.as_secs()));
        }
        if self.close {
            headers.insert(header::CONNECTION, HeaderValue::from_static("close"));
        }

        response
    }
}

/// Body of every error response.
#[derive(Debug, Serialize, ToSchema)]
#[schema(description = "Body of every error response.")]
pub(super) struct ErrorBody {
    /// HTTP status code of the response.
    #[schema(example = 422)]
    status_code: u16,
    /// What went wrong.
    #[schema(example = "Text cannot be empty")]
    detail: &'static str,
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
