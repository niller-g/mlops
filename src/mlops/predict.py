import torch
from transformers import AutoTokenizer
from .model import DistilGPT2Model


def generate_text(prompt: str, model_path: str, max_length: int = 50) -> str:
    """Generate text from a prompt using our fine-tuned model."""
    # Load model and tokenizer
    model = DistilGPT2Model.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate
    with torch.no_grad():
        output_sequences = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and return the generated text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    # Test the model with a medical question
    model_path = "models/distilgpt2-finetuned-final"
    test_prompt = "What are the symptoms of "

    print(f"Prompt: {test_prompt}")
    generated = generate_text(test_prompt, model_path)
    print(f"Generated: {generated}")
