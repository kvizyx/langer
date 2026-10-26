"""Writes parity.jsonl: MediaPipe's predictions for the texts that tests/parity.rs checks.

The texts are sentences from Tatoeba in every language of the model, short texts taken from
them, phrases for the romanized labels, texts mixing several scripts, and edge cases. Run it
through parity.Dockerfile, which provides MediaPipe (see README.md).
"""

import bz2
import json
import random
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import text

DATA = Path(__file__).parent
MODEL = DATA.parents[1] / "vendor" / "model" / "language_detector.tflite"
OUTPUT = DATA / "parity.jsonl"

TATOEBA = "https://downloads.tatoeba.org/exports/per_language/{code}/{code}_sentences.tsv.bz2"

# Labels recorded per text, most probable first.
TOP = 3
# Per language: sentences, then short texts made of the first words of further sentences.
SENTENCES = 15
SHORT_TEXTS = 5
MIXED_TEXTS = 200
SEED = 10911

# Tatoeba's language code for every label of the model but the romanized ones and `unknown`.
TATOEBA_CODES = {
    "af": "afr", "am": "amh", "ar": "ara", "az": "aze", "be": "bel", "bg": "bul", "bn": "ben",
    "bs": "bos", "ca": "cat", "ceb": "ceb", "co": "cos", "cs": "ces", "cy": "cym", "da": "dan",
    "de": "deu", "el": "ell", "en": "eng", "eo": "epo", "es": "spa", "et": "est", "eu": "eus",
    "fa": "pes", "fi": "fin", "fil": "tgl", "fr": "fra", "fy": "fry", "ga": "gle", "gd": "gla",
    "gl": "glg", "gu": "guj", "ha": "hau", "haw": "haw", "hi": "hin", "hmn": "mww", "hr": "hrv",
    "ht": "hat", "hu": "hun", "hy": "hye", "id": "ind", "ig": "ibo", "is": "isl", "it": "ita",
    "iw": "heb", "ja": "jpn", "jv": "jav", "ka": "kat", "kk": "kaz", "km": "khm", "kn": "kan",
    "ko": "kor", "ku": "kmr", "ky": "kir", "la": "lat", "lb": "ltz", "lo": "lao", "lt": "lit",
    "lv": "lvs", "mg": "mlg", "mi": "mri", "mk": "mkd", "ml": "mal", "mn": "mon", "mr": "mar",
    "ms": "zsm", "mt": "mlt", "my": "mya", "ne": "npi", "nl": "nld", "no": "nob", "ny": "nya",
    "pa": "pan", "pl": "pol", "ps": "pus", "pt": "por", "ro": "ron", "ru": "rus", "sd": "snd",
    "si": "sin", "sk": "slk", "sl": "slv", "sm": "smo", "sn": "sna", "so": "som", "sq": "sqi",
    "sr": "srp", "st": "sot", "su": "sun", "sv": "swe", "sw": "swh", "ta": "tam", "te": "tel",
    "tg": "tgk", "th": "tha", "tr": "tur", "uk": "ukr", "ur": "urd", "uz": "uzb", "vi": "vie",
    "xh": "xho", "yi": "yid", "yo": "yor", "zh": "cmn", "zu": "zul",
}

# For the romanized labels, which Tatoeba has no sentences for.
ROMANIZED = [
    "privet kak dela", "spasibo bolshoe, do vstrechi zavtra",
    "zdravei kak si", "blagodarya, vsichko e nared",
    "kalimera ti kaneis", "efcharisto poli, ta leme avrio",
    "namaste aap kaise hain", "mujhe bhi achha laga, phir milenge",
    "marhaban kayf halak", "shukran jazilan, ila al liqa",
    "Kyou wa ii tenki desu ne", "arigatou gozaimasu, mata ashita",
    "ni hao ni zenme yang", "xie xie, mingtian jian",
]

EDGE_CASES = [
    "",
    " ",
    "\n\t",
    "😀",
    "😀 123",
    "!!!",
    "12345",
    "e\u0301",
    "a" * 1000,
    "\u023a" * 300,
    " ".join(["word"] * 200),
]


def reproducible(t: str) -> bool:
    """Whether MediaPipe predicts `t` deterministically and as a whole.

    MediaPipe tokenizes the lowercased text with the original byte length, so a character whose
    lowercase is shorter in UTF-8 makes it read past its buffer, into uninitialized memory. Its
    Python bindings pass the text as a C string, which ends at the first NUL.
    """
    shortened = any(len(c.lower().encode()) < len(c.encode()) for c in t)
    return not shortened and "\u0130" not in t and "\0" not in t


def tatoeba(code: str, count: int) -> Iterator[str]:
    """The first `count` usable sentences of a language on Tatoeba, oldest first.

    Exports are sorted by sentence id, so only their beginning is downloaded.
    """
    with urllib.request.urlopen(TATOEBA.format(code=code), timeout=60) as response:
        lines = bz2.open(response, "rt", encoding="utf-8")
        found = 0
        for line in lines:
            sentence = line.rstrip("\n").split("\t")[2]
            if reproducible(sentence):
                yield sentence
                found += 1
                if found == count:
                    return


def short(sentence: str, words: int) -> str:
    """The first `words` words of `sentence`, or its first characters if it has no spaces."""
    split = sentence.split()
    return " ".join(split[:words]) if len(split) > 1 else sentence[: 2 * words]


def texts() -> list[str]:
    sentences = {}
    shorts = []
    for label, code in TATOEBA_CODES.items():
        found = list(tatoeba(code, SENTENCES + SHORT_TEXTS))
        sentences[label] = found[:SENTENCES]
        shorts += [short(s, 1 + i % 2) for i, s in enumerate(found[SENTENCES:])]

    # Two or three sentences in different languages, often different scripts, in one text.
    rng = random.Random(SEED)
    pool = [s for found in sentences.values() for s in found]
    mixed = [" ".join(rng.sample(pool, rng.randint(2, 3))) for _ in range(MIXED_TEXTS)]

    every = [s for found in sentences.values() for s in found] + shorts + ROMANIZED + mixed + EDGE_CASES
    return [t for t in every if reproducible(t)]


def main() -> None:
    options = text.LanguageDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        max_results=TOP,
    )
    detector = text.LanguageDetector.create_from_options(options)

    lines = []
    for t in texts():
        top = [[d.language_code, d.probability] for d in detector.detect(t).detections]
        lines.append(json.dumps({"text": t, "top": top}, ensure_ascii=False))

    # MediaPipe fails to close it later, while the interpreter shuts down.
    detector.close()

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} texts written to {OUTPUT.name}")


if __name__ == "__main__":
    main()
