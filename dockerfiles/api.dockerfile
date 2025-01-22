FROM python:3.11-slim AS base

EXPOSE $PORT

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 --verbose -r requirements.txt

COPY src src/


COPY secrets/api_key.json default.json
COPY .dvc/config .dvc/config
RUN dvc init --no-scm
COPY models.dvc models.dvc
RUN dvc config core.no_scm true
RUN dvc pull

WORKDIR /src/mlops

CMD ["uvicorn", "api:app", "--port", "$PORT", "--host", "0.0.0.0", "--workers", "1"]
