from torch import nn
from transformers import AutoModelForCausalLM


class DistilGPT2Model(nn.Module):
    def __init__(self, model_name="distilgpt2"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Ensure the lm_head weight is not shared
        self.model.lm_head.weight = nn.Parameter(self.model.lm_head.weight.clone())

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def save_pretrained(self, path):
        """Custom save method to handle weight saving properly"""
        self.model.save_pretrained(
            path, safe_serialization=False
        )  # Disable safetensors

    @classmethod
    def from_pretrained(cls, path):
        """Custom load method"""
        instance = cls()
        instance.model = AutoModelForCausalLM.from_pretrained(path)
        return instance
