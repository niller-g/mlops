# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY data/processed data/processed/
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir --default-timeout=100 --verbose .

ENTRYPOINT ["python", "-u", "src/mlops/train.py"]
