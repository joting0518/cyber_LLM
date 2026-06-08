import argparse
import csv
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a cybersecurity email classifier. "
    "Decide whether the email is phishing or legitimate. "
    "Do not copy the email. Do not explain. "
    "Output only one label: phishing or legitimate."
)


def default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_prediction(text: str) -> str:
    text = (text or "").strip().lower()
    if "phishing" in text:
        return "1"
    if "legitimate" in text:
        return "0"
    return "invalid"


def label_text(label: str) -> str:
    return "phishing" if str(label).strip() == "1" else "legitimate"


def build_prompt(tokenizer, subject: str, body: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Subject: {subject}\nBody: {body}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def read_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "subject",
        "label",
        "expected",
        "raw_prediction",
        "predicted_label",
        "correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["predicted_label"] in {"0", "1"}]
    tp = sum(row["label"] == "1" and row["predicted_label"] == "1" for row in valid)
    tn = sum(row["label"] == "0" and row["predicted_label"] == "0" for row in valid)
    fp = sum(row["label"] == "0" and row["predicted_label"] == "1" for row in valid)
    fn = sum(row["label"] == "1" and row["predicted_label"] == "0" for row in valid)
    total = len(rows)
    valid_total = len(valid)
    invalid_total = total - valid_total
    accuracy = (tp + tn) / valid_total if valid_total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    conservative_accuracy = (tp + tn) / total if total else 0.0
    return {
        "total": total,
        "valid_predictions": valid_total,
        "invalid_predictions": invalid_total,
        "accuracy_valid_only": accuracy,
        "conservative_accuracy_invalid_as_wrong": conservative_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    root = default_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", default=str(root / "model" / "best_adapter"))
    parser.add_argument("--test-csv", default=str(root / "data" / "test.csv"))
    parser.add_argument("--output-csv", default=str(root / "evaluation" / "finetuned" / "reproduced_test_predictions.csv"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir, local_files_only=True)
    model.eval()

    rows = read_rows(Path(args.test_csv), args.max_samples)
    output_rows = []
    for idx, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row.get("subject", ""), row.get("body", ""))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predicted = normalize_prediction(raw)
        label = str(row.get("label", "")).strip()
        output_rows.append(
            {
                "index": idx,
                "subject": row.get("subject", ""),
                "label": label,
                "expected": label_text(label),
                "raw_prediction": raw,
                "predicted_label": predicted,
                "correct": predicted == label,
            }
        )

    write_rows(Path(args.output_csv), output_rows)
    for key, value in compute_metrics(output_rows).items():
        print(f"{key}={value}")
    print(f"predictions_written={args.output_csv}")


if __name__ == "__main__":
    main()
