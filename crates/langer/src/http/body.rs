//! The JSON request body extractor.

use axum::body::Bytes;
use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use serde::de::{DeserializeOwned, IgnoredAny};
use serde_json::error::Category;

use super::error::ApiError;

/// A JSON request body: `400` if it is not JSON, `422` if it is JSON of the wrong shape.
///
/// Unlike axum's `Json`, it needs no `Content-Type` header, and tells the two cases apart even
/// where `serde_json` reports a wrong type as a syntax error (`"method": null`).
#[derive(Debug)]
pub(super) struct RequestBody<T>(pub(super) T);

impl<T: DeserializeOwned, S: Send + Sync> FromRequest<S> for RequestBody<T> {
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let body = Bytes::from_request(req, state).await?;

        serde_json::from_slice(&body).map(Self).map_err(|e| {
            // Parsing again without a schema runs only for rejected bodies.
            let valid_json = e.classify() == Category::Data || serde_json::from_slice::<IgnoredAny>(&body).is_ok();
            let error = if valid_json {
                ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "Body validation failed")
            } else {
                ApiError::new(StatusCode::BAD_REQUEST, "Body is not valid JSON")
            };

            error.with_extra(e.to_string())
        })
    }
}
