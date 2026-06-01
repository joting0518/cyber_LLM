import pandas as pd

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# Load
train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")

# only for testing
train_df = train_df.head(1000)
valid_df = valid_df.head(200)


# Build Text
train_df["text"] = (
    "[SUBJECT] "
    + train_df["subject"].fillna("")
    + " [BODY] "
    + train_df["body"].fillna("")
)

valid_df["text"] = (
    "[SUBJECT] "
    + valid_df["subject"].fillna("")
    + " [BODY] "
    + valid_df["body"].fillna("")
)


# Dataset
train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]]
)

valid_dataset = Dataset.from_pandas(
    valid_df[["text", "label"]]
)


# Tokenizer
MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-base",
    use_fast=False
)

def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

valid_dataset = valid_dataset.map(
    tokenize,
    batched=True
)


# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# Trainer
training_args = TrainingArguments(
    output_dir="./checkpoints",

    num_train_epochs=1,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    logging_steps=20,

    eval_strategy="epoch",

    save_strategy="epoch",

    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)


# Train
trainer.train()


# Eval
result = trainer.evaluate()

print(result)