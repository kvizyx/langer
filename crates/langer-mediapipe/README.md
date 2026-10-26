# langer-mediapipe

This crate implements [MediaPipe's](https://ai.google.dev/edge/mediapipe/solutions/text/language_detector) language detector,
with no TFLite runtime and build-time downloads (model is embedded in the crate).

## Usage

Here is how to detect the most probable language of a text.

```rust
// Loads the embedded model, do it once.
let detector = Detector::new()?;

// The `k` most probable languages, most probable first.
let top = detector.predict("Hello, how are you?", 1)?;
```

The model always picks a language, even for a text with no letters (e.g.`"😀"` is
Telugu with a probability of 0.40). A caller that doesn’t want this behavior must check for letters itself.

```rust
/// The most probable language of `text`, or `None` if it has no letters.
fn detect_only_letters<'a>(detector: &'a Detector, text: &str) -> Result<Option<&'a str>, PredictError> {
    if !text.chars().any(char::is_alphabetic) {
        return Ok(None);
    }

    let top = detector.predict(text, 1)?;
    Ok(top.first().map(|best| best.language))
}
```

Detector is also immutable, so one detector can serve any number
of threads at once, without a lock or a copy per thread. You should load it once and share a reference, or
an `Arc` where it must outlive the current scope (e.g. in a server's state).

```rust
let detector = Detector::new()?;
let texts = ["Bonjour à tous", "Hallo zusammen", "Hola a todos"];

// Every thread borrows the same detector.
let languages: Vec<&str> = thread::scope(|scope| {
    let workers: Vec<_> = texts
        .iter()
        .map(|text| scope.spawn(|| detector.predict(text, 1)))
        .collect();

    workers
        .into_iter()
        .filter_map(|worker| worker.join().ok()?.ok()?.first().map(|best| best.language))
        .collect()
});
```

## Vendor

Third-party files in `vendor/` are managed with [vendir](https://carvel.dev/vendir/).

Building and testing need no vendir, only updating vendored files does.

### Updating

Follow this steps if you need to update vendor:

1. Change the file's `url` and `sha256` in `vendir.yaml`.
2. Run vendir in this directory.

   ```bash
   vendir sync -f vendir.yaml --lock-file vendir.lock.yaml
   ```

3. For the schema, regenerate the bindings with the `flatc` matching the `flatbuffers` crate:

   ```bash
   flatc --rust -o src vendor/tflite/schema.fbs
   ```

4. Run the tests. A change to the model or the character tables must be checked against
   MediaPipe itself.

   ```bash
   cargo test -p langer-mediapipe
   ```

### Checking

To check that `vendor/` matches `vendir.yaml` (e.g. in CI), sync and look for changes:

```bash
vendir sync -f vendir.yaml --lock-file vendir.lock.yaml && git diff --exit-code vendor/
```

## License

This crate is licensed under the [MIT](../../LICENSE) license.

The vendor keep its own license (Apache License 2.0).
