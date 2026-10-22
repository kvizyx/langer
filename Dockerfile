FROM rust:1.98-slim-bookworm AS builder
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p langer \
    && cp target/release/langer /usr/local/bin/langer

FROM gcr.io/distroless/cc-debian12:nonroot
LABEL org.opencontainers.image.title="Language Processor"
LABEL org.opencontainers.image.description="HTTP service for language detection (MediaPipe and Lingua)"
LABEL org.opencontainers.image.licenses="MIT"

# Both models are embedded in the executable.
COPY --from=builder /usr/local/bin/langer /usr/local/bin/langer

ENV PORT=3000
EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/langer"]
