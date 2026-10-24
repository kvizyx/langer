# Langer

HTTP service that detects the language of short texts such as chat messages.

| Method                | Engine                                                                                                                                | Memory                                       | Speed              | Use case                          |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|--------------------|-----------------------------------|
| `mediapipe` (default) | [MediaPipe's language detector](https://ai.google.dev/edge/mediapipe/solutions/text/language_detector), ported to Rust, 110 languages | ~300 KiB model                               | ~15–20 µs per text | The bulk of messages              |
| `lingua`              | [Lingua](https://github.com/pemistahl/lingua-rs), high accuracy, 75 languages                                                         | Up to ~250 MiB of page cache, no heap growth | ~0.3–2 ms per text | Best accuracy on one or two words |

## API

The interactive API reference (Scalar) is served at `/schema`, the OpenAPI document at
`/schema/openapi.json`, `/` and `/docs` redirect to `/schema`.

The page is [`crates/langer/assets/scalar.html`](crates/langer/assets/scalar.html), embedded in the
executable; utoipa-scalar replaces `$title` and `$spec` in it. The theme is set in
`data-configuration` ([available themes](https://github.com/scalar/scalar/blob/main/documentation/themes.md)).
Scalar is loaded from jsDelivr, pinned to one version and checked by subresource integrity. To
update it, change the version in the URL and replace the `integrity` hash with the output of:

```bash
curl -sL <url> | openssl dgst -sha384 -binary | openssl base64 -A
```

### `POST /detect`

```json
{"text": "Привет, как дела?", "method": "mediapipe"}
```

`method` is optional and defaults to `mediapipe`; the other value is `lingua`.

```json
{"language": "ru", "confidence": 0.9997, "method": "mediapipe"}
```

- `language` is a [BCP 47](https://www.rfc-editor.org/info/bcp47) tag, the same for both
  methods and listed by `GET /detect/languages`: the ISO 639-1 code where one exists (`ru`), else
  the ISO 639-3 one (`ceb`). `mediapipe` also reports romanized text with a script subtag
  (`ru-Latn`). Norwegian is `no` and Filipino is `tl` for both methods. `und` means that no
  language was detected.
- `confidence` is a number in `0..=1`; for `lingua` it is `null` when Lingua cannot decide.
- Body that is not valid JSON: `400` with `"detail": "Body is not valid JSON"`.
- Valid JSON with a missing `text`, a field of the wrong type, an unknown `method` or unknown
  fields: `422` with `"detail": "Body validation failed"`. Both give the reason in `extra`.
- Empty or whitespace-only text: `422 {"status_code":422,"detail":"Text cannot be empty"}`.
- Body larger than `MAX_BODY_BYTES`: `413`.
- Body not received within `REQUEST_TIMEOUT_SECS`: `408`, and the connection is closed.
- More `lingua` requests than `LINGUA_MAX_IN_FLIGHT` at once: `503` with `Retry-After` and
  `Cache-Control: no-store`.

Every error, including unknown paths (`404`) and methods (`405`), has the same
`{"status_code", "detail", "extra"?}` shape.

### `GET /detect/languages`

All languages `POST /detect` can report, with their tags, sorted by tag:

```json
[{"code": "af", "name": "Afrikaans", "methods": ["mediapipe", "lingua"]}, ...]
```

The list lives in [`crates/langer/data/languages.json`](crates/langer/data/languages.json);
`build.rs` validates it and compiles it into the response body. Tests check it against the
MediaPipe model's labels and the `lingua` crate.

### `GET /health` and `GET /health/ready`

Both answer `{"status": "ok"}` with `200` when all is well.

- `/health` (liveness) answers as long as the process runs, and checks nothing else: a busy or
  shutting down service is still alive, and restarting it would not help.
- `/health/ready` (readiness) answers `200` when the service can take requests: it is not shutting
  down, and `mediapipe` detects a fixed English text correctly (about 20 µs). It answers `503`
  with the usual error body otherwise. On a stop signal it turns `503` at once, and the service
  keeps serving for `SHUTDOWN_DELAY_SECS` before it stops accepting connections, so that traffic
  moves to other instances first. Load does not affect it: excess `lingua` requests get their
  own `503`.

### `GET /metrics`

Metrics in the Prometheus text format, on the same port:

| metric | type | labels | meaning |
|---|---|---|---|
| `langer_detections_total` | counter | `method`, `outcome` | detection requests; `outcome` is `ok`, `invalid_input`, `overloaded` or `failed` |
| `langer_detection_duration_seconds` | histogram | `method` | time of successful detections |
| `langer_lingua_slots` | gauge | | `REQUEST_TIMEOUT_SECS` | `10` | longest time to read a request's body and answer it; slower requests get `408` and their connection is closed. After the shutdown delay, connections still open this long are closed |
| `LINGUA_MAX_IN_FLIGHT` |
| `langer_lingua_slots_in_use` | gauge | | `lingua` detections running or waiting for a thread |
| `langer_http_requests_total` | counter | `method`, `route`, `status` | requests to known routes |
| `langer_http_request_duration_seconds` | histogram | `method`, `route` | time to answer them |

`route` is the route's pattern (`/detect`), so that unknown paths and scrapes of `/metrics` add
no series. Histogram buckets go from 10 µs (`mediapipe`) to 2.5 s (`lingua` on long texts).
`langer_lingua_slots_in_use` close to `langer_lingua_slots`, or a growing count of `overloaded`,
means `lingua` needs more CPU or a higher `LINGUA_MAX_IN_FLIGHT`.

## Configuration

| env | default | description |
|---|---|---|
| `PORT` | `3000` | HTTP port |
| `LOG_LEVEL` | `info` | `tracing` log filter (`warn`, `info,langer=debug`, ...); an invalid one stops the start |
| `MAX_BODY_BYTES` | `65536` | largest accepted request body; detection time grows with the text, about 0.1 s for 64 KiB with `lingua` |
| `REQUEST_TIMEOUT_SECS` | `10` | longest time to read a request's body and answer it; slower requests get `408` and their connection is closed. After the shutdown delay, connections still open this long are closed |
| `LINGUA_MAX_IN_FLIGHT` | 4 per CPU core | most `lingua` detections running or queued at once; more get `503` with `Retry-After: 1` |
| `SHUTDOWN_DELAY_SECS` | `5` | seconds between a stop signal and closing the listener, during which `/health/ready` answers `503` and requests are still served; `0` for local runs |

Both models are embedded in the executable; there is nothing else to configure.

## Accuracy

Lingua's test data (Wortschatz Leipzig): 1000 single words, word pairs and sentences for each of
the 70 languages both methods detect; the score is the mean of the per-language accuracies, as in
Lingua's own reports. One core, release build. The data comes from the same source as Lingua's
training data, which favors Lingua.

| method | 1 word | 2 words | sentences | time per text |
|---|---:|---:|---:|---:|
| `mediapipe` | 66.2% | 78.4% | 94.2% | 23–46 µs |
| `lingua` | 75.4% | 89.6% | 96.3% | 0.3–2.1 ms |

To reproduce (see `crates/langer/tests/accuracy.rs`), fetch the test data only and run:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/pemistahl/lingua-rs.git
```

```bash
git -C lingua-rs sparse-checkout set --no-cone '/language-models/*/testdata/'
```

```bash
LANGER_EVAL_DATA=$PWD/lingua-rs/language-models cargo test --release -p langer --test accuracy -- --ignored --nocapture
```

### Alternatives

Rust detectors compared on the same data, for the 56 languages all of them support (measured
once, not part of the repository's tests):

| detector | 1 word | 2 words | sentences | time per text |
|---|---:|---:|---:|---:|
| `mediapipe` (this repository) | 70.0% | 82.4% | 96.0% | 21–46 µs |
| [whatlang](https://crates.io/crates/whatlang) 0.18 | 58.0% | 72.4% | 95.3% | 27–46 µs |
| Lingua, low accuracy mode | 67.7% | 84.1% | 96.1% | 0.25–1.3 ms |
| Lingua, high accuracy mode (`lingua`) | 77.3% | 91.2% | 98.1% | 0.3–1.9 ms |

whatlang is as fast as `mediapipe` but much less accurate on short texts; Lingua's low accuracy
mode is as accurate but ten times slower. Other detectors are C++ bindings (CLD2, CLD3,
fastText) or cover few languages (whichlang, 16).

### Parity with MediaPipe

`mediapipe` returns the same top language as MediaPipe 1.1.1 on all 97 439 texts tested; details
in the [`langer_mediapipe` README](crates/langer_mediapipe/README.md#parity).

### Lingua versions

The `lingua` method uses lingua-rs 1.8.0, the engine of the Python package
`lingua-language-detector` 2.2.0. A side-by-side run against the Python service built with
`lingua-language-detector` 2.1.1 agreed on 2998 of 3010 texts; the 12 differences are texts where
both versions are unsure (confidence 0.08–0.4). Results are identical only against the same
Lingua version.

## Memory and performance

One server core (`taskset`), load generated by `oha` on other cores. Throughput varies by ~30%
between runs on the test machine (WSL).

| scenario | text | RSS | requests/s | p99 |
|---|---|---:|---:|---:|
| idle | | 6 MiB | — | — |
| `mediapipe`, 50 connections | `Привет, как дела?` | 10 MiB | 16k–17k | 6 ms |
| `lingua`, 8 connections | `Привет, как дела? Это обычное сообщение.` | 25 MiB | 4.3k–4.8k | 3–4 ms |
| `lingua`, 8 connections | `Hello, how are you?` | 94 MiB | ~2.1k | 9 ms |
| `lingua`, 8 connections | `hello there, how are you doing today` | 110 MiB | ~950 | 15 ms |
| `lingua`, texts in 34 languages | | ~250 MiB, almost all page cache | — | — |

Lingua's time depends on the script: Cyrillic leaves it a few candidate languages, Latin several
dozen.

Lingua 1.8 embeds its models in the executable as FSTs and reads them in place, so its "memory" is
executable pages in the page cache. The kernel can evict them under memory pressure instead of
OOM-killing the process; the heap stays below 10 MiB.

### Allocator

The service uses the system allocator (glibc). mimalloc was measured as an alternative: it gave no
throughput gain, because the hot path barely allocates, and it kept ~10 MiB more RSS unless built
without transparent huge pages. If the CPU limit is raised to several cores, re-measure; glibc's
per-thread arenas can be capped with `MALLOC_ARENA_MAX=2` if fragmentation shows up.

## Layout

```text
crates/
  langer/            service: detectors, a transport-independent `Service` layer, and the HTTP
                     API over it (axum, OpenAPI with utoipa, Scalar)
  langer_mediapipe/  MediaPipe language detector in pure Rust
    vendor/          pinned, unmodified third-party files (vendir): the model, MediaPipe's
                     character tables and the TFLite schema (Apache 2.0)
```

`langer_mediapipe` reads the `.tflite` file with bindings generated by `flatc` and reimplements
the model's operators: MediaPipe's `NGramHash` (Plan 9 Unicode tables, MurmurHash64 variant) and
`KmeansEmbeddingLookup` custom ops, and the fully connected layers as run by TFLite's XNNPACK
delegate (dynamic `i8` input quantization, fused multiply-add rescaling).

Vendored files are managed with [vendir](https://carvel.dev/vendir/):
[`crates/langer_mediapipe/vendir.yml`](crates/langer_mediapipe/vendir.yml) pins each one to an
exact upstream URL and sha256. Building and testing need no vendir. The crate's
[README](crates/langer_mediapipe/README.md) explains how it works, each vendored file, and how to
update and check them.

## Development

The Rust version is pinned in [`rust-toolchain.toml`](rust-toolchain.toml) for local builds, CI and
the Docker image; rustup installs it on first use.

Git hooks run the CI checks locally with [lefthook](https://lefthook.dev): formatting and clippy
on commit, tests on push. Enable them once:

```bash
lefthook install
```

```bash
cargo test --workspace
```

```bash
cargo clippy --workspace --all-targets --all-features
```

The tests include:

- property tests ([proptest](https://proptest-rs.github.io/proptest/)) on random input. A case
  proptest finds goes into an explicit, named test next to the property; the seeds it saves to
  `proptest-regressions/` and `*.proptest-regressions` stay local (ignored by git);
- snapshots ([insta](https://insta.rs)) of error bodies and the OpenAPI document in
  `crates/langer/tests/snapshots/`. After an intended change, review and accept them with
  [`cargo-insta`](https://insta.rs/docs/cli/):

```bash
cargo insta test --review
```

Tests that need external data are ignored by default: the accuracy report (see
[Accuracy](#accuracy)).

Inference benchmark:

```bash
cargo bench -p langer_mediapipe
```

Builds target `x86-64-v2`, see [`.cargo/config.toml`](.cargo/config.toml). Raise it to `x86-64-v3`
only if every node supports AVX2 and FMA; inference gets ~25% faster. Otherwise the executable
dies with `SIGILL` on start. To see which levels a machine supports:

```bash
/lib64/ld-linux-x86-64.so.2 --help | grep x86-64-v
```

For one build, `RUSTFLAGS` replaces the flags from `.cargo/config.toml`:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release -p langer
```

```bash
docker build --build-arg TARGET_CPU=x86-64-v3 -t langer .
```

## Dependencies

[cargo-deny](https://embarkstudios.github.io/cargo-deny/) checks dependencies in CI against
RustSec advisories, allowed licenses and crate sources ([`deny.toml`](deny.toml)); locally:

```bash
cargo deny check
```

[Dependabot](.github/dependabot.yml) opens weekly pull requests updating crates, base image
digests and actions.

## Docker images

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) checks every push and pull request and
publishes two `linux/amd64` images to `ghcr.io/kvizyx/langer` on pushes to `main` and `v*` tags:

| tag | CPU level | runs on |
|---|---|---|
| `latest`, `1.2.3`, `1.2`, `sha-abc1234` | `x86-64-v2` | any x86-64 server CPU since ~2009 |
| `latest-avx2`, `1.2.3-avx2`, `1.2-avx2`, `sha-abc1234-avx2` | `x86-64-v3` | AVX2 and FMA: Intel Haswell, AMD Excavator and newer; ~25% faster |

```bash
docker run --rm -p 3000:3000 ghcr.io/kvizyx/langer:latest
```

## License

[MIT](LICENSE). Vendored third-party files in `crates/langer_mediapipe/vendor/` keep their own
license (Apache 2.0), see the [`langer_mediapipe` README](crates/langer_mediapipe/README.md#vendored-files).
