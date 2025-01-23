from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from http import HTTPStatus
from transformers import AutoTokenizer
import mlops.model as model
import mlops.predict as predict


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    model_path = "models/distilgpt2-finetuned-final"

    global m, tokenizer, gen_kwargs
    print("Loading model...")
    m = model.DistilGPT2Model.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    gen_kwargs = {"max_length": 16}

    yield

    print("Cleaning up...")
    del m, tokenizer, gen_kwargs


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/infer/{prompt}")
def infer(prompt: str, max_length: None | int):
    generated_text = predict.generate_text(prompt, m, tokenizer, max_length if max_length else gen_kwargs["max_length"])

    return generated_text
