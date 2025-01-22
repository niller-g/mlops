from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from http import HTTPStatus
import torch
from transformers import AutoTokenizer
from model import DistilGPT2Model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    model_path = "models/distilgpt2-finetuned-final"

    global model, tokenizer, gen_kwargs
    print("Loading model...")
    model = DistilGPT2Model.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    gen_kwargs = {"max_length": 16}

    yield

    print("Cleaning up...")
    del model, tokenizer, gen_kwargs


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict/{prompt}")
def predict(prompt: str, max_length: None | int):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        output_sequences = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length if max_length else gen_kwargs["max_length"],
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text
