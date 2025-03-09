# train.py
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import get_last_checkpoint

# Configure training parameters
MODEL_NAME = "t5-small"  # Use t5-small for faster training, t5-base for better results
OUTPUT_DIR = "./t5_summarization_model"
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
print(f"Loading {MODEL_NAME} model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)

# Load dataset
print(f"Loading {DATASET_NAME} dataset...")
dataset = load_dataset(DATASET_NAME, DATASET_VERSION)

# Preprocess function
def preprocess_function(examples):
    # Prepare inputs
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    
    # Prepare targets
    targets = tokenizer(
        examples["highlights"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )
    
    # Create labels (targets)
    model_inputs["labels"] = targets["input_ids"]
    
    return model_inputs

# Preprocess datasets
print("Preprocessing datasets...")
train_dataset = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

validation_dataset = dataset["validation"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["validation"].column_names,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# Resume from checkpoint if available
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")

# Configure training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),  # Use mixed precision training if possible
    report_to="none",  # Disable wandb, tensorboard, etc.
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
print("Starting training...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# Save final model
print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")