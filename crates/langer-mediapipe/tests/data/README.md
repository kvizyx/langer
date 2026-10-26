# Data (parity)

`parity.jsonl` holds MediaPipe's own predictions, which `tests/parity.rs` checks this crate
against: for each text, its three most probable labels with their probabilities.

`generate_parity.py` writes it, running MediaPipe 1.1.1 on the bundled model. MediaPipe needs
OpenGL libraries even for text, so it runs in a container built from `parity.Dockerfile`. From
the repository root:

```bash
docker build -f crates/langer-mediapipe/tests/data/parity.Dockerfile --output crates/langer-mediapipe/tests/data crates/langer-mediapipe
```

Regenerate it after a change to the model.

## Texts

- Sentences from [Tatoeba](https://tatoeba.org) in every language of the model, and short texts
  made of their first words. Tatoeba's sentences are licensed under
  [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/).
- Phrases for the romanized labels (`ru-Latn`, `ja-Latn`, ...), sentences of several languages
  joined in one text, and edge cases: empty and whitespace-only text, emoji, digits, long texts.

Texts MediaPipe cannot predict reproducibly are left out: those with a NUL, which its Python
bindings cut, and those with a character whose lowercase is shorter in UTF-8 (`İ`), for which it
reads uninitialized memory.
