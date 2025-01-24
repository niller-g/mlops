import os
import logging
import hydra
from omegaconf import DictConfig

import wandb
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_from_disk

from monitoring import MLOpsMetrics
from data_validation import DataValidator
from model import DistilGPT2Model

from google.cloud import secretmanager

logging.basicConfig(level=logging.INFO)

def get_secret(secret_id: str, project_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


class MetricsCallback(TrainerCallback):
    def __init__(self, metrics: MLOpsMetrics):
        self.metrics = metrics

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.metrics.record_training_step(latest_log['loss'])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.metrics.record_validation_loss(metrics['eval_loss'])

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None:
            progress = state.epoch - int(state.epoch)
            self.metrics.record_epoch_progress(progress)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    if cfg.get("sweep", False):
        wandb.init()
        cfg.train.batch_size = wandb.config.batch_size
        cfg.train.lr = wandb.config.learning_rate
        cfg.train.max_epochs = wandb.config.num_epochs
        cfg.train.warmup_steps = wandb.config.warmup_steps
        cfg.train.weight_decay = wandb.config.weight_decay
        cfg.train.gradient_accumulation_steps = wandb.config.gradient_accumulation_steps

        if 'max_samples' in wandb.config:
            cfg.train.max_samples = wandb.config.max_samples

    elif cfg.train.wandb.project:
        # Regular non-sweep W&B init
        gcp_project_id = cfg.train.wandb.gcp_project_id
        secret_id = cfg.train.wandb.secret_id
        wandb_key = get_secret(secret_id, gcp_project_id)
        wandb.login(key=wandb_key)
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
        )

    metrics = MLOpsMetrics()

    logging.info("Starting training with configuration...")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Batch size: {cfg.train.batch_size}")
    logging.info(f"Learning rate: {cfg.train.lr}")
    logging.info(f"Max epochs: {cfg.train.max_epochs}")

    batch_size = cfg.train.batch_size
    lr = cfg.train.lr
    max_epochs = cfg.train.max_epochs
    eval_steps = cfg.train.eval_steps
    max_samples = cfg.train.max_samples 

    processed_path = os.path.join(cfg.paths.data_dir, "processed/medical_questions_processed")
    logging.info(f"Loading dataset from {processed_path}")
    ds = load_from_disk(processed_path)

    total_samples = len(ds)
    logging.info(f"Dataset loaded with {total_samples} total samples")

    if max_samples and max_samples < total_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
        logging.info(f"Using {max_samples} examples for quick testing")
    else:
        logging.info(f"Using all {total_samples} examples for training")

    validator = DataValidator()
    is_test_mode = max_samples and max_samples < 100
    validation_results = validator.validate_dataset(ds, is_test_mode=is_test_mode)
    if not validation_results["success"]:
        logging.error("Data validation failed!")
        raise ValueError("Dataset failed validation checks")
    logging.info("Data validation passed successfully!")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(example["clean_text"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns([c for c in ds.column_names if c not in ["input_ids", "attention_mask"]])
    ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = ds["train"], ds["test"]

    model = DistilGPT2Model("distilgpt2")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.paths.models_dir, "distilgpt2-finetuned"),
        overwrite_output_dir=True,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        save_strategy="no",
        learning_rate=lr,
        warmup_steps=cfg.train.warmup_steps,
        weight_decay=cfg.train.weight_decay,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        report_to="wandb",
        no_cuda=True, 
        fp16=False,
        bf16=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[MetricsCallback(metrics)],
    )

    logging.info("Beginning training...")
    trainer.train()
    logging.info("Training complete.")

    final_model_path = os.path.join(cfg.paths.models_dir, "distilgpt2-finetuned-final")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logging.info(f"Model saved to {final_model_path}")

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    train()