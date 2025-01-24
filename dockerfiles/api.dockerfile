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
COPY configs configs/


COPY secrets/api_key.json default.json
RUN dvc init --no-scm
#COPY .dvc/config .dvc/config
#COPY *.dvc .dvc/
COPY models.dvc models.dvc
COPY data.dvc data.dvc
RUN dvc remote add -d gcs_remote gs://mlops-grp-43-2025/
RUN dvc config core.no_scm true
#RUN dvc remote modify gcs_remote --local gdrive_service_account_json_file_path default.json
#COPY models.dvc models.dvc
RUN dvc gc --workspace -f
#RUN dvc checkout
RUN dvc config cache.type symlink
RUN dvc config cache.protected true
RUN dvc fetch  --remote gcs_remote --no-run-cache
RUN dvc pull  --remote gcs_remote --no-run-cache

#RUN mv /models /src/models

WORKDIR /src

CMD ["sh", "-c", "uvicorn mlops.api:app --port $PORT --host 0.0.0.0 --workers 1"]
#CMD ["uvicorn", "api:app", "--port", "${PORT}", "--host", "0.0.0.0", "--workers", "1"]
