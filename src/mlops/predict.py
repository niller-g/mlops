import argparse
import torch
from transformers import AutoTokenizer
import mlops.model as model


def load_and_generate_text(prompt: str, model_path: str, max_length: int = 50) -> str:
    m = model.DistilGPT2Model.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return generate_text(prompt, m, tokenizer, max_length)


def generate_text(prompt: str, model, tokenizer, max_length: int = 50) -> str:
    """Generate text from a prompt using our fine-tuned model."""
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output_sequences = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a fine-tuned DistilGPT2 model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="What are the symptoms of ",
        help="Prompt to pass to the model.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/distilgpt2-finetuned-final",
        help="Path to the fine-tuned model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum length of generated sequence.",
    )
    args = parser.parse_args()

    generated = load_and_generate_text(args.prompt, args.model_path, max_length=args.max_length)
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated}")
