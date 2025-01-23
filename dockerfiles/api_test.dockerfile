FROM python:3.11-slim AS base

EXPOSE $PORT

ENV GOOGLE_APPLICATION_CREDENTIALS=default.json

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
RUN pip install --no-cache-dir --default-timeout=100 --verbose -r requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 --verbose -r requirements_dev.txt

COPY src src/
COPY integrationtests integrationtests/
COPY configs configs/
COPY secrets/api_key.json default.json

RUN dvc init --no-scm
RUN dvc remote add -d gcs_remote gs://mlops-grp-43-2025/
COPY ./*.dvc .dvc/
RUN dvc config core.no_scm true
RUN dvc pull #.dvc/models.dvc --force

CMD ["pytest", "--cov=src", "--cov-report=term-missing", "integrationtests/"]
