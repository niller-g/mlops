import os
import logging
import hydra
from omegaconf import DictConfig

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_from_disk
import wandb

# 1) Import Google Secret Manager client
from google.cloud import secretmanager
from data_validation import DataValidator
from model import DistilGPT2Model
from monitoring import MLOpsMetrics

logging.basicConfig(level=logging.INFO)


def get_secret(secret_id: str, project_id: str) -> str:
    """
    Retrieve a secret's value from Google Cloud Secret Manager.
    Requires 'google-cloud-secret-manager' and proper GCP credentials.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


class MetricsCallback(TrainerCallback):
    def __init__(self, metrics: MLOpsMetrics):
        self.metrics = metrics
    
    def on_step_end(self, args, state, control, **kwargs):
        """Record metrics after each training step"""
        if state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log:
                self.metrics.record_training_step(latest_log['loss'])
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Record validation metrics"""
        if metrics and 'eval_loss' in metrics:
            self.metrics.record_validation_loss(metrics['eval_loss'])
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Record epoch progress"""
        if state.epoch is not None:
            # Calculate progress within current epoch
            progress = state.epoch - int(state.epoch)
            self.metrics.record_epoch_progress(progress)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    # Initialize metrics (use different port than validation)
    metrics = MLOpsMetrics()
    
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
    
    total_samples = len(ds)
    logging.info(f"Dataset loaded with {total_samples} total samples")

    # Take a subset for quick testing if max_samples is specified and less than total
    if max_samples and max_samples < total_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
        logging.info(f"Using {max_samples} examples for quick testing")
    else:
        logging.info(f"Using all {total_samples} examples for training")

    # Validate dataset using Great Expectations
    validator = DataValidator()
    # Use test mode if we're using a small sample
    is_test_mode = max_samples and max_samples < 100
    validation_results = validator.validate_dataset(ds, is_test_mode=is_test_mode)
    
    if not validation_results["success"]:
        logging.error("Data validation failed! Check the logs above for details.")
        raise ValueError("Dataset failed validation checks")

    logging.info("Data validation passed successfully!")

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(
            example["clean_text"], truncation=True, padding="max_length", max_length=128
        )

    # Prepare dataset
    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(
        [col for col in ds.column_names if col not in ["input_ids", "attention_mask"]]
    )
    ds = ds.train_test_split(test_size=0.2, seed=42)  # Use 20% for test
    train_ds = ds["train"]
    val_ds = ds["test"]

    # 3) Prepare model
    model = DistilGPT2Model("distilgpt2")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4) Setup W&B
    if cfg.train.wandb.project:
        # Dynamically fetch the W&B API key from Secret Manager
        gcp_project_id = cfg.train.wandb.gcp_project_id
        secret_id = cfg.train.wandb.secret_id
        wandb_key = get_secret(secret_id, gcp_project_id)

        # Log into W&B with the retrieved key
        wandb.login(key=wandb_key)

        # Initialize the W&B run
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
        )

    # Create output directory
    os.makedirs(cfg.paths.models_dir, exist_ok=True)

    # 5) HF Trainer
    # Detect hardware
    import torch
    use_gpu = torch.cuda.is_available()
    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
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
        no_cuda=not use_gpu,  # Only use GPU if available
        fp16=use_gpu,  # Only use fp16 if CUDA GPU is available
        dataloader_num_workers=4,  # Use multiple workers for data loading
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[MetricsCallback(metrics)],  # Add metrics callback
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

    # Finish W&B session
    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    train()
