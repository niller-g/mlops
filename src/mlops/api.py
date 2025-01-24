from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from fastapi import FastAPI
from http import HTTPStatus
from transformers import AutoTokenizer
from prometheus_client import make_asgi_app
import mlops.model as model
import mlops.predict as predict
from mlops.monitoring import MLOpsMetrics


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    model_path = "../models/distilgpt2-finetuned-final"

    global m, tokenizer, gen_kwargs, metrics
    print(f"Loading model from {model_path}...")
    m = model.DistilGPT2Model.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    gen_kwargs = {"max_length": 16}
    
    # Initialize metrics
    metrics = MLOpsMetrics()

    yield

    print("Cleaning up...")
    del m, tokenizer, gen_kwargs


app = FastAPI(lifespan=lifespan)

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
def root():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/infer")
def infer(prompt: str, max_length: Optional[int] = None):
    # Use the metrics context manager to time the inference
    with metrics.time_inference():
        generated_text = predict.generate_text(
            prompt, 
            m, 
            tokenizer, 
            max_length if max_length else gen_kwargs["max_length"]
        )

    return generated_text
