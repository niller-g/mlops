from torch import nn
from transformers import AutoModelForCausalLM


class DistilGPT2Model(nn.Module):
    def __init__(self, model_name="models/distilgpt2-finetuned-final"):
        """
        model_name should point to a local directory containing:
          - config.json
          - pytorch_model.bin
          - tokenizer.json / merges.txt / vocab.json (if needed)
        Pass local_files_only=True so it never tries to fetch from HF hub.
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,  # Only load from local disk
        )
        # Ensure the lm_head weight is not shared
        self.model.lm_head.weight = nn.Parameter(self.model.lm_head.weight.clone())

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def save_pretrained(self, path):
        """Custom save method to handle weight saving properly."""
        self.model.save_pretrained(path, safe_serialization=False)

    @classmethod
    def from_pretrained(cls, path):
        """
        Custom load method that also uses local_files_only=True
        so that it does not attempt to fetch anything online.
        """
        instance = cls(model_name=path)  # Initialize with dummy name
        instance.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
        return instance
