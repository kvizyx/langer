# Dependencies are built in their own layer with cargo-chef, keyed by Cargo.lock and the crates
# manifests, so the layer is reused (also from a registry cache in CI) until dependencies change.
# Base images are pinned by digest (of the multi-platform index) for reproducible builds; the tag
# is kept for readers and update tools. To update, change the tag and take the new digest from
# `docker buildx imagetools inspect <image:tag>`.
FROM lukemathwalker/cargo-chef:0.1.78-rust-1.98-slim-bookworm@sha256:9a2089c084b6f8b85b7cbc6dac81f38afcdcd87708fe5a5a1697da254d21c147 AS chef
WORKDIR /src
# The toolchain in rust-toolchain.toml builds everything, also when it differs from the image's.
COPY rust-toolchain.toml .
RUN rustup toolchain install

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
# CPU level the executable requires, see .cargo/config.toml. RUSTFLAGS replaces the flags set
# there; dependencies and the service must be built with the same flags, so it is set before cook.
ARG TARGET_CPU=x86-64-v2
ENV RUSTFLAGS="-C target-cpu=${TARGET_CPU}"
COPY --from=planner /src/recipe.json recipe.json
RUN cargo chef cook --release --locked -p langer --recipe-path recipe.json
COPY . .
RUN cargo build --release --locked -p langer

FROM gcr.io/distroless/cc-debian12:nonroot@sha256:9dac0a79194e45a7da0158a9c6da57b217585af0786db3845d1f0ec1a0dd182f

LABEL org.opencontainers.image.title="Langer"
LABEL org.opencontainers.image.description="Self-host language detection HTTP service, based on Lingua and MediaPipe."
LABEL org.opencontainers.image.licenses="MIT"

# Both models are embedded in the executable.
COPY --from=builder /src/target/release/langer /usr/local/bin/langer

ENV PORT=3000
EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/langer"]
