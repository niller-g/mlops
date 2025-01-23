from torch import nn
from transformers import AutoModelForCausalLM


class DistilGPT2Model(nn.Module):
    def __init__(self, model_name: str = "models/distilgpt2-finetuned-final",
                 local_files_only: bool = True):
        """
        model_name should point to a local directory containing:
          - config.json
          - pytorch_model.bin
          - tokenizer.json / merges.txt / vocab.json (if needed)
        Pass local_files_only=True so it never tries to fetch from HF hub.
        """
        super().__init__()
        # Load the underlying GPT-2 model from disk only
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        # Ensure the lm_head weight is not shared
        self.model.lm_head.weight = nn.Parameter(self.model.lm_head.weight.clone())

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def save_pretrained(self, path: str):
        """Custom save method to handle weight saving properly."""
        self.model.save_pretrained(path, safe_serialization=False)

    @classmethod
    def from_pretrained(cls, path: str, local_files_only: bool = True):
        """
        Custom load method that uses local_files_only=True
        so it does not attempt to fetch anything online.
        """
        # Just call our constructor with the requested path and local_files_only
        return cls(model_name=path, local_files_only=local_files_only)