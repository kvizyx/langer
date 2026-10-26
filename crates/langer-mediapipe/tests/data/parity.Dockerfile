# Generates parity.jsonl with MediaPipe itself, see README.md for the command.
FROM python:3.12-slim AS generate

# MediaPipe loads OpenGL libraries even for text tasks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgles2 libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --root-user-action=ignore mediapipe==1.1.1

WORKDIR /crate
COPY vendor/model/language_detector.tflite vendor/model/
COPY tests/data/generate_parity.py tests/data/
RUN python tests/data/generate_parity.py

# Only the output is exported, to the directory given to `docker build --output`.
FROM scratch
COPY --from=generate /crate/tests/data/parity.jsonl /
