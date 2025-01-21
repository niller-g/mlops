import os
import logging
import hydra
from omegaconf import DictConfig

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
import wandb

from .model import DistilGPT2Model

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    logging.info("Starting training with Hydra config...")
    logging.info(f"Working directory: {os.getcwd()}")

    # Access config params
    batch_size = cfg.train.batch_size
    lr = cfg.train.lr
    max_epochs = cfg.train.max_epochs
    eval_steps = cfg.train.eval_steps
    max_samples = cfg.train.max_samples  # Number of examples to use for quick testing

    # 1) Load processed data
    processed_path = os.path.join(
        cfg.paths.data_dir, "processed/medical_questions_processed"
    )
    logging.info(f"Loading dataset from {processed_path}")
    ds = load_from_disk(processed_path)

    # Take a small subset for quick testing
    ds = ds.shuffle(seed=42).select(range(max_samples))
    logging.info(f"Using {len(ds)} examples for quick testing")

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(ex):
        return tokenizer(
            ex["clean_text"], truncation=True, padding="max_length", max_length=128
        )

    # Prepare dataset
    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(
        [col for col in ds.column_names if col not in ["input_ids", "attention_mask"]]
    )
    ds = ds.train_test_split(
        test_size=0.2, seed=42
    )  # Use 20% for test since we have few examples
    train_ds = ds["train"]
    val_ds = ds["test"]

    # 3) Prepare model
    model = DistilGPT2Model("distilgpt2")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4) Setup W&B (M14)
    if cfg.train.wandb.project:
        wandb.init(project=cfg.train.wandb.project, entity=cfg.train.wandb.entity)

    # Create output directory
    os.makedirs(cfg.paths.models_dir, exist_ok=True)

    # 5) HF Trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.paths.models_dir, "distilgpt2-finetuned"),
        overwrite_output_dir=True,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        save_strategy="no",  # We'll handle saving ourselves
        learning_rate=lr,
        report_to="wandb" if cfg.train.wandb.project else None,
        # Add some parameters to speed up training
        no_cuda=True,  # Use CPU for this quick test
        fp16=False,
        dataloader_num_workers=0,
        # Save total number of steps
        max_steps=4,  # Just do a few steps for testing
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    logging.info("Beginning training...")
    trainer.train()
    logging.info("Training complete.")

    # 6) Save final model using our custom save method
    final_model_path = os.path.join(cfg.paths.models_dir, "distilgpt2-finetuned-final")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)  # Use our custom save method
    tokenizer.save_pretrained(final_model_path)
    logging.info(f"Model saved to {final_model_path}")

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    train()
