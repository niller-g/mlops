from transformers import AutoTokenizer
from src.mlops.model import DistilGPT2Model


def test_model_forward():
    model = DistilGPT2Model("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    inputs = tokenizer("Hello, this is a test.", return_tensors="pt")
    outputs = model(**inputs)
    assert outputs.logits is not None, "Model output logits is None."
