import argparse
import csv
import json
import math
import os
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


csv.field_size_limit(sys.maxsize)


SYSTEM_PROMPT = (
    "You are a cybersecurity email classifier. "
    "Decide whether the email is phishing or legitimate. "
    "Return exactly one word only: phishing or legitimate."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_to_text(label: str) -> str:
    return "phishing" if str(label).strip() == "1" else "legitimate"


def compact_email(subject: str, body: str, max_chars: int) -> str:
    subject = subject or ""
    body = body or ""
    text = f"Subject: {subject}\nBody: {body}"
    if len(text) <= max_chars:
        return text

    # Preserve the opening and ending because phishing links often appear near either.
    head = text[: int(max_chars * 0.65)]
    tail = text[-int(max_chars * 0.25) :]
    return head + "\n\n[...truncated...]\n\n" + tail


def load_rows(zip_path: Path, member: str, max_samples: int | None, seed: int) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        reader = csv.DictReader(
            (line.decode("utf-8-sig") for line in f),
        )
        rows = list(reader)
    if max_samples is not None and max_samples < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)
    return rows


def load_rows_from_csv(csv_path: Path, max_samples: int | None, seed: int) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if max_samples is not None and max_samples < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)
    return rows


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


class EmailInstructionDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_length: int,
        max_chars: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_chars = max_chars

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        email = compact_email(row.get("subject", ""), row.get("body", ""), self.max_chars)
        answer = label_to_text(row["label"])

        messages_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email},
        ]
        messages_full = messages_prompt + [{"role": "assistant", "content": answer}]

        prompt_text = self.tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        tokenized = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        # If truncation removed the answer entirely, keep the final token trainable.
        if all(value == -100 for value in labels):
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class DataCollator:
    tokenizer: object

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            batch["labels"].append(item["labels"] + [-100] * pad_len)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def evaluate_loss(model, loader, device, max_batches: int | None) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            if max_batches is not None and step >= max_batches:
                break
    model.train()
    return sum(losses) / max(1, len(losses))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset-dir", default=str(default_data_dir()))
    parser.add_argument("--dataset-zip", default=None)
    parser.add_argument("--output-dir", default="outputs/qwen25_1_5b_phishing_qlora")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "training_args.json", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.dataset_zip:
        train_rows = load_rows(Path(args.dataset_zip), "train.csv", args.max_train_samples, args.seed)
        valid_rows = load_rows(Path(args.dataset_zip), "valid.csv", args.max_valid_samples, args.seed)
    else:
        data_dir = Path(args.dataset_dir)
        train_rows = load_rows_from_csv(data_dir / "train.csv", args.max_train_samples, args.seed)
        valid_rows = load_rows_from_csv(data_dir / "valid.csv", args.max_valid_samples, args.seed)

    train_dataset = EmailInstructionDataset(train_rows, tokenizer, args.max_length, args.max_chars)
    valid_dataset = EmailInstructionDataset(valid_rows, tokenizer, args.max_length, args.max_chars)
    collator = DataCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = next(model.parameters()).device
    total_update_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    print(f"train_rows={len(train_rows)} valid_rows={len(valid_rows)}")
    print(f"total_update_steps={total_update_steps} device={device}")

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,update_step,train_loss,valid_loss,lr,elapsed_sec\n", encoding="utf-8")

    best_valid = float("inf")
    global_update = 0
    start_time = time.time()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += float(outputs.loss.detach().cpu())

            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

                avg_loss = running_loss / max(1, args.grad_accum)
                running_loss = 0.0
                progress.set_postfix(loss=f"{avg_loss:.4f}", update=global_update)

                if global_update % args.log_every == 0:
                    elapsed = time.time() - start_time
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"{epoch},{step},{global_update},{avg_loss:.6f},,{args.learning_rate},{elapsed:.1f}\n")

        valid_loss = evaluate_loss(model, valid_loader, device, args.eval_batches)
        elapsed = time.time() - start_time
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{len(train_loader)},{global_update},,{valid_loss:.6f},{args.learning_rate},{elapsed:.1f}\n")
        print(f"epoch={epoch} valid_loss={valid_loss:.6f} elapsed_sec={elapsed:.1f}")

        epoch_dir = out_dir / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_dir = out_dir / "best_adapter"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            write_json(out_dir / "best_metrics.json", {"epoch": epoch, "valid_loss": valid_loss})

    write_json(out_dir / "final_metrics.json", {"best_valid_loss": best_valid, "elapsed_sec": time.time() - start_time})
    print(f"training_complete output_dir={out_dir}")


if __name__ == "__main__":
    main()
