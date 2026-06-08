# Group 8 Week 16 Final Submission

Project: Defending against Conventional and AI-generated Phishing Using an LLM-based System

## Team

- 楊大明 111701043 國立政治大學
- 林佩璇 111302055 國立政治大學
- 陳若庭 111306011 國立政治大學
- 李宜恩 111208001 國立政治大學

## Main Files

- `Group8_Final_Report.pdf`: final integrated report.
- `poc/app/BEC.ipynb`: main proof-of-concept notebook.
- `poc/app/demo.png`: demo interface screenshot.
- `data/train.csv`, `data/valid.csv`, `data/test.csv`: dataset splits.
- `model/best_adapter/`: fine-tuned LoRA adapter and tokenizer files.

## Online Demo Links

- Colab notebook: https://colab.research.google.com/drive/1Fv-GlxEPfdZPR_q3EnJTPMpCgeYqYr9G?usp=sharing
- Demo video: https://drive.google.com/file/d/1Bn9_QrXNJQ1dEVGxulVnasttunsR-Yqd/view?usp=drivesdk

## Project Summary

This project builds an LLM-based phishing and Business Email Compromise email detection workflow. The system classifies email subject/body pairs as phishing or legitimate and documents a practical proof-of-concept pipeline for cybersecurity email defense.

## Dataset

The dataset combines legitimate email and phishing/BEC sources:

| Dataset | Purpose |
|---|---|
| Enron Email Dataset | Legitimate emails |
| Nazario Phishing Corpus | Phishing emails |
| Adversarial BEC Email Dataset | BEC and adversarial phishing emails |

Dataset split:

| File | Rows | Purpose |
|---|---:|---|
| `data/train.csv` | 10,060 | Fine-tuning |
| `data/valid.csv` | 1,258 | Validation |
| `data/test.csv` | 1,258 | Final evaluation |

Schema:

| Column | Meaning |
|---|---|
| `subject` | Email subject |
| `body` | Email body |
| `label` | `1 = phishing`, `0 = legitimate` |
| `dataset` | Source dataset ID |

## Model and Fine-tuning

Base model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Fine-tuning:

```text
LoRA BF16 supervised fine-tuning on NVIDIA A100
```

Key settings:

| Setting | Value |
|---|---|
| Epochs | 3 |
| Max sequence length | 2048 |
| Batch size per device | 8 |
| Gradient accumulation steps | 2 |
| Learning rate | 1e-4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |

## Reproducing the PoC

Install dependencies from the submission root:

```bash
cd w16
pip install -r poc/requirements.txt
```

Open the main notebook:

```text
poc/app/BEC.ipynb
```

The submitted package includes the fine-tuned LoRA adapter in:

```text
model/best_adapter/
```

The full base model weights are not included because they are large and can be retrieved by model name. To reproduce inference, the runtime must be able to load:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Then attach the submitted adapter from `model/best_adapter/`. The dataset used by the notebook and evaluation scripts is in:

```text
data/test.csv
```

The prompt used by the PoC is stored in:

```text
poc/prompts/email_classifier_prompt.txt
```

## Reproducing Evaluation

Run a small smoke test:

```bash
python poc/scripts/evaluate_predictions.py --max-samples 10
```

Run the full test split:

```bash
python poc/scripts/evaluate_predictions.py
```

The script writes reproduced predictions to:

```text
evaluation/finetuned/reproduced_test_predictions.csv
```

Existing fine-tuned evaluation artifacts are included in:

```text
evaluation/finetuned/
```

Provided evaluation CSV files:

- `evaluation/finetuned/test_predictions.csv`
- `evaluation/finetuned/test_metrics_overall.csv`
- `evaluation/finetuned/test_metrics_by_dataset.csv`
- `evaluation/finetuned/before_after_comparison.csv`

## Reproducing Training

The training script now uses relative data paths by default:

```bash
python poc/scripts/train_qwen25_qlora.py
```

For a faster smoke run:

```bash
python poc/scripts/train_qwen25_qlora.py --max-train-samples 32 --max-valid-samples 16 --epochs 1
```

The script reads:

```text
data/train.csv
data/valid.csv
```

## Results

| Model | Accuracy | Precision | Recall | F1-score | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Before fine-tuning | 0.5684 | 0.5416 | 0.8903 | 0.6735 | 560 | 155 | 474 | 69 |
| After fine-tuning, valid outputs only | 0.9960 | 0.9984 | 0.9936 | 0.9960 | 625 | 606 | 1 | 4 |

Additional evaluation details:

```text
test_total = 1258
valid_predictions = 1236
invalid_predictions = 22
conservative_accuracy_invalid_as_wrong = 0.978537
```

## Notes

- The full base model weights are not included in this zip. The adapter is included under `model/best_adapter/`.
- Private credentials are not included.
