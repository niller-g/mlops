FROM python:3.11-slim AS base

EXPOSE $PORT

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

ENV DOCKER_BUILDKIT=1
# Add environment variable to force offline mode
ENV TRANSFORMERS_OFFLINE=1 

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir --default-timeout=100 --verbose -r requirements.txt

# Fix the model path copying
COPY src/ /src/
COPY models/ /models/

WORKDIR /src

# Use shell to expand environment variable
CMD ["sh", "-c", "uvicorn mlops.api:app --port ${PORT} --host 0.0.0.0 --workers 1"]
