# Langer

[![CI](https://img.shields.io/github/actions/workflow/status/kvizyx/langer/ci.yaml?branch=main&label=CI&logo=githubactions&logoColor=white)](https://github.com/kvizyx/langer/actions/workflows/ci.yaml)
[![Image](https://img.shields.io/badge/Image-ghcr.io%2Fkvizyx%2Flanger-2496ED?logo=docker&logoColor=white)](https://github.com/kvizyx/langer/pkgs/container/langer)
[![Rust](https://img.shields.io/badge/Rust-1.88%2B-CE422B?logo=rust)](Cargo.toml)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

A tiny self-hosted language detection server, supports [MediaPipe's](https://ai.google.dev/edge/mediapipe/solutions/text/language_detector) language
detector and Lingua ([lingua-rs](https://github.com/pemistahl/lingua-rs)) detection methods. For now, only HTTP transport is supported.

## Quick start

After starting the server, you can open the [interactive API reference](http://localhost:3000/schema) to browse the OpenAPI
schema and send requests.

### Docker

```bash
docker run --rm -p 3000:3000 ghcr.io/kvizyx/langer:latest
```

### From source

```bash
git clone git@github.com:kvizyx/langer.git && cd langer

cargo run --release -p langer
```

## Detection method

Each request can pick its detection method. Which one fits depends on your use case:
how short the texts are, how much accuracy matters, and how much load you expect.

- **`mediapipe`** (default): very cheap, 110 languages, less accurate on 1-2 words.
- **`lingua`**: the most accurate on short texts, but more expensive and 75 languages.

A common pattern is `mediapipe` for everything and `lingua` for 1-2 words or when `mediapipe`'s
confidence is low.

## Accuracy

On Lingua's test data (1000 texts of each kind for each language, see `crates/langer/tests/accuracy.rs`):

| Method      | 1 word | 2 words | Sentences |
|-------------|-------:|--------:|----------:|
| `mediapipe` |  66.2% |   78.4% |     94.2% |
| `lingua`    |  75.4% |   89.6% |     96.3% |

Other Rust detectors are less accurate on short texts (e.g. [whatlang](https://github.com/greyblake/whatlang-rs)) or slower (Lingua's low accuracy
mode), so this two methods is the most optimal choices.

## Load handling

`lingua` runs on one blocking thread per core, behind `LINGUA_MAX_IN_FLIGHT` slots. A request
without a free slot waits for one up to `LINGUA_QUEUE_TIMEOUT_MILLISECONDS`, then gets `503` with
`Retry-After: 1`. `mediapipe` and the health checks never wait for it.

Every request must finish within `REQUEST_TIMEOUT_SECONDS` (else `408`), and bodies are limited to
`MAX_BODY_BYTES` (else `413`).

## Metrics

[Prometheus](https://prometheus.io) metrics are served on the same port through `GET /metrics` endpoint.

## Docker

CI publishes Docker images to `ghcr.io/kvizyx/langer` in two variants. Docker pulls the
architecture of the host:

| Variant | Tag suffix | Architectures                                                                                                     |
|---------|------------|-------------------------------------------------------------------------------------------------------------------|
| Default | None       | `linux/amd64` with SSE4.2 and POPCNT (Intel Nehalem, AMD Bulldozer and newer), `linux/arm64` (any 64-bit ARM CPU) |
| AVX2    | `-avx2`    | `linux/amd64` with AVX2 and FMA, detection is 25–40% faster                                                       |

Every variant is tagged the same way, plus its suffix:

| Tag                                          | Image                                                  |
|----------------------------------------------|--------------------------------------------------------|
| `latest`                                     | The latest commit on `main`                            |
| `<major>.<minor>.<patch>`, `<major>.<minor>` | A release, from the git tag `v<major>.<minor>.<patch>` |
| `sha-<commit>`                               | A commit, by its short hash                            |

For example, `ghcr.io/kvizyx/langer:latest-avx2` is the AVX2 variant of the latest commit on
`main`.

To release, run [cargo-release](https://github.com/crate-ci/cargo-release) on `main`: it bumps the
version, commits, tags `v<version>` and pushes, and CI publishes the images.

```bash
cargo release 1.0.0 --execute
```

The Dockerfile picks the CPU level by architecture, `TARGET_CPU` overrides it (e.g. `docker build --build-arg TARGET_CPU=x86-64-v3 -t langer .`)

## License

This project is licensed under the [MIT](LICENSE) license.

Third-party vendor in `crates/langer-mediapipe` keep its own license, see the [`langer-mediapipe` README](crates/langer-mediapipe/README.md).
