FROM python:3.11-slim AS base

EXPOSE $PORT

ENV GOOGLE_APPLICATION_CREDENTIALS=default.json

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
#RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir --default-timeout=100 --verbose -r requirements.txt

RUN pip install --no-cache-dir --default-timeout=100 --verbose -r requirements.txt


COPY src src/


COPY secrets/api_key.json default.json
RUN dvc init --no-scm
#COPY .dvc/config .dvc/config
RUN dvc remote add -d gcs_remote gs://mlops-grp-43-2025/
COPY ./*.dvc .dvc/
RUN dvc config core.no_scm true
#RUN dvc remote modify gcs_remote --local gdrive_service_account_json_file_path default.json
#COPY models.dvc models.dvc
#RUN dvc gc --all-branches --all-tags -f
RUN dvc pull #.dvc/models.dvc --force

WORKDIR /src/mlops

CMD ["uvicorn", "api:app", "--port", "$PORT", "--host", "0.0.0.0", "--workers", "1"]
