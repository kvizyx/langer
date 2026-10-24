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

- `language`: MediaPipe reports its own codes, as the Python service did: mostly ISO 639-1, plus
  `iw` (Hebrew), `fil` (Filipino), romanized variants such as `zh-Latn`, and `unknown`. Lingua
  reports ISO 639-1 codes, or `unknown` when it cannot decide.
- `confidence` is a number in `0..=1`; for `lingua` it is `null` when Lingua cannot decide.
- Empty or whitespace-only text: `400 {"status_code":400,"detail":"Text cannot be empty"}`.
- Invalid JSON, missing `text` or unknown `method`: `400` with
  `"detail": "Body validation failed"` and the reason in `extra`.
- Body larger than `MAX_BODY_BYTES`: `413`. Every error, including unknown paths (`404`) and
  methods (`405`), has the same `{"status_code", "detail", "extra"?}` shape.

### `GET /detect/languages`

All supported languages with ISO 639-1 codes, sorted by code, identical to the Python service:

```json
[{"iso_639_1": "af", "name": "Afrikaans", "methods": ["mediapipe", "lingua"]}, ...]
```

The list lives in [`crates/langer/data/languages.json`](crates/langer/data/languages.json);
`build.rs` validates it and compiles it into the response body. Tests check it against the
MediaPipe model's labels and the `lingua` crate.

### `GET /health`

```json
{"status": "ok"}
```

## Configuration

| env | default | description |
|---|---|---|
| `PORT` | `3000` | HTTP port |
| `LOG_LEVEL` | `info` | `tracing` log filter (`warn`, `langer=debug`, ...) |
| `MAX_BODY_BYTES` | `65536` | largest accepted request body; detection time grows with the text, about 0.1 s for 64 KiB with `lingua` |

Both models are embedded in the executable; there is nothing else to configure.

## Accuracy

Datasets:

- Lingua's test data (Wortschatz Leipzig), 300 samples per language and text type;
- [papluca/language-identification](https://huggingface.co/datasets/papluca/language-identification),
  20 languages, 10 000 texts.

Lingua's test data comes from the same source as its training data, which favors Lingua.

| method | languages | sentences | 2 words | 1 word |
|---|---|---:|---:|---:|
| `mediapipe` | 34 common | 95.78% | 84.55% | 69.53% |
| `lingua` | 34 common | 98.11% | 91.45% | 76.91% |
| `mediapipe` | all 75 Lingua languages | 89.39% | 73.94% | 62.23% |
| `lingua` | all 75 Lingua languages | 96.27% | 89.29% | 74.23% |

On papluca both are on par: 98.74% (`mediapipe`) and 98.80% (`lingua`).

The 34 common languages: en, ru, uk, de, fr, es, pt, it, pl, tr, ja, ko, zh, kk, be, nl, sv, cs, fi,
ar, he, hu, ro, bg, sr, hr, lt, et, vi, id, th, el, da, no.

### Parity with MediaPipe

`mediapipe` returns the same top language as MediaPipe 1.1.1 on all 97 439 texts tested (the
datasets above plus edge cases); details in the
[`langer_mediapipe` README](crates/langer_mediapipe/README.md#parity).

### Lingua versions

The `lingua` method uses lingua-rs 1.8.0, the engine of the Python package
`lingua-language-detector` 2.2.0. A side-by-side run against the Python service built with
`lingua-language-detector` 2.1.1 agreed on 2998 of 3010 texts; the 12 differences are texts where
both versions are unsure (confidence 0.08–0.4). Results are identical only against the same
Lingua version.

## Memory and performance

One server core (`taskset`), load generated by `oha` on other cores. Throughput varies by ~30%
between runs on the test machine (WSL).

| scenario | RSS | requests/s | p99 |
|---|---:|---:|---:|
| idle | 6 MiB | — | — |
| `mediapipe`, 50 connections | 7 MiB | ~16k | 8–9 ms |
| `lingua`, Cyrillic text, 8 connections | 28 MiB | 2.5k–3.2k | 5–12 ms |
| `lingua`, Latin text, 8 connections | 81 MiB | 1.9k–2.3k | 6–10 ms |
| `lingua`, texts in 34 languages | ~250 MiB, almost all page cache | — | — |

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
  langer/            HTTP service: axum, OpenAPI (utoipa) + Scalar, detectors
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

```bash
cargo test --workspace
```

```bash
cargo clippy --workspace --all-targets --all-features
```

Tests that need external data are ignored by default.

Parity with MediaPipe: see the [`langer_mediapipe` README](crates/langer_mediapipe/README.md#parity).

Accuracy of both methods on a labelled dataset (`code<TAB>text` per line):

```bash
LANGER_EVAL_TSV=eval.tsv cargo test --release -p langer --test accuracy -- --ignored --nocapture
```

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
