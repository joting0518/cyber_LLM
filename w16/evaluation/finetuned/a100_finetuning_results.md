# A100 訓練環境微調方法

## 1. 訓練環境假設

若將訓練環境改為 NVIDIA A100，微調策略可以比目前 8 GB VRAM 筆電 GPU 更積極。

常見 A100 規格如下：

| GPU | VRAM | 適合策略 |
|---|---:|---|
| A100 40GB | 40 GB | LoRA BF16、QLoRA、大 batch size |
| A100 80GB | 80 GB | LoRA BF16、QLoRA、可嘗試 full fine-tuning |

本實驗模型為：

```text
Base model: Qwen/Qwen2.5-1.5B-Instruct
Task: phishing / legitimate email classification
Dataset: train.csv / valid.csv / test.csv
Split: 8 : 1 : 1
```

## 2. 推薦微調策略

在 A100 環境下，最推薦的策略是：

```text
LoRA BF16 supervised fine-tuning
```

理由如下：

- `qwen2.5:1.5b` 模型不大，A100 足以穩定載入 BF16 權重。
- LoRA 訓練成本低，速度快，效果通常足以應付二分類任務。
- BF16 比 4-bit QLoRA 更穩定，量化誤差較少。
- 不需要 full fine-tuning 就能顯著改善 phishing detection 的 false positive 問題。

推薦優先順序：

| 方法 | 推薦程度 | 說明 |
|---|---|---|
| LoRA BF16 | 最高 | A100 上最平衡，速度快且穩定 |
| QLoRA 4-bit | 可用 | 節省 VRAM，但 A100 上不是必要 |
| Full fine-tuning BF16 | 可選 | 成本較高，對本任務不一定必要 |

## 3. A100 建議訓練設定

### 3.1 LoRA BF16 推薦設定

```text
Base model: Qwen/Qwen2.5-1.5B-Instruct
Fine-tuning method: LoRA
Precision: BF16
Epochs: 3
Learning rate: 1e-4
Batch size per device: 8
Gradient accumulation steps: 2
Effective batch size: 16
Max sequence length: 2048
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Optimizer: AdamW
Validation: 每個 epoch 結束後評估
Checkpoint: 儲存 validation loss 最低的 adapter
```

若使用 A100 80GB，可提高設定：

```text
Batch size per device: 16
Gradient accumulation steps: 1 或 2
Max sequence length: 4096
LoRA rank: 32
LoRA alpha: 64
```

### 3.2 QLoRA 4-bit 設定

若希望降低成本或同時訓練多組實驗，可使用 QLoRA：

```text
Fine-tuning method: QLoRA 4-bit
Quantization: NF4
Compute dtype: BF16
Epochs: 3
Learning rate: 2e-4
Batch size per device: 16
Gradient accumulation steps: 1
Max sequence length: 2048 或 4096
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Optimizer: paged_adamw_8bit
```

QLoRA 在 A100 上可行，但對 `1.5B` 模型來說不是必要。若追求穩定與可解釋實驗流程，LoRA BF16 更適合寫進報告。

### 3.3 Full Fine-tuning 設定

A100 可以嘗試 full fine-tuning，尤其是 A100 80GB。

建議設定：

```text
Fine-tuning method: Full supervised fine-tuning
Precision: BF16
Epochs: 2 到 3
Learning rate: 1e-5 到 2e-5
Batch size per device: 4 到 8
Gradient accumulation steps: 1 到 2
Max sequence length: 2048
Optimizer: AdamW
Warmup ratio: 0.03
Weight decay: 0.01
```

不過本任務是 phishing / legitimate 二分類，資料量約 10,060 筆訓練資料。Full fine-tuning 可能帶來額外提升，但也有以下風險：

- 訓練成本較高。
- 較容易 overfitting。
- 可能破壞基礎模型原有 instruction-following 能力。
- 對課程 PoC 來說，工程成本不一定划算。

因此除非要追求最高分數，否則不建議優先使用 full fine-tuning。

## 4. 資料處理方法

資料來源：

```text
dataset.zip
```

內含：

```text
train.csv
valid.csv
test.csv
```

欄位：

| 欄位 | 說明 |
|---|---|
| subject | Email 主旨 |
| body | Email 內容 |
| label | `1 = phishing`，`0 = legitimate` |
| dataset | 資料來源，用於分 dataset 評估 |

將每筆資料轉為 instruction tuning 格式：

```json
{
  "instruction": "You are a cybersecurity email classifier. Decide whether the email is phishing or legitimate. Return exactly one word only: phishing or legitimate.",
  "input": "Subject: ...\nBody: ...",
  "output": "phishing"
}
```

label 對應：

```text
label = 1 -> phishing
label = 0 -> legitimate
```

## 5. A100 上的 Link Pattern 保留策略

A100 有較大 VRAM，因此可以使用更長的 sequence length。

建議：

```text
max sequence length = 2048
```

若使用 A100 80GB，可嘗試：

```text
max sequence length = 4096
```

長 context 的優點是可以保留更多 phishing 判斷線索，例如：

- URL。
- 短網址。
- IP-based URL。
- 登入或驗證連結。
- 附件誘導語句。
- 緊急付款或帳號停權語氣。
- URL 前後的上下文。

若 email 超過 max sequence length，建議不要單純截斷前段，而是保留：

```text
subject
body opening section
all URLs and nearby sentences
security-sensitive keyword nearby sentences
body ending section
```

## 6. 訓練流程

完整流程如下：

1. 解壓縮或直接讀取 `dataset.zip`。
2. 讀取 `train.csv`、`valid.csv`、`test.csv`。
3. 將 `label=1` 轉為 `phishing`。
4. 將 `label=0` 轉為 `legitimate`。
5. 將 `subject` 與 `body` 合併成模型輸入。
6. 轉換成 instruction tuning 格式。
7. 載入 `Qwen/Qwen2.5-1.5B-Instruct`。
8. 使用 BF16 載入模型。
9. 掛載 LoRA adapter。
10. 使用 `train.csv` 微調。
11. 使用 `valid.csv` 監控 validation loss。
12. 儲存 validation loss 最低的 checkpoint。
13. 使用 `test.csv` 做最終評估。
14. 計算 Accuracy、Precision、Recall、F1-score。
15. 使用 `dataset` 欄位做分資料集分析。

## 7. 評估方式

微調後模型必須使用與未微調 baseline 相同的 `test.csv` 進行測試。

評估指標：

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

混淆矩陣定義：

| 指標 | 說明 |
|---|---|
| TP | 實際為 phishing，模型也預測為 phishing |
| TN | 實際為 legitimate，模型也預測為 legitimate |
| FP | 實際為 legitimate，但模型誤判為 phishing |
| FN | 實際為 phishing，但模型誤判為 legitimate |

## 8. 微調目標

未微調 baseline 結果：

| Model | Accuracy | Precision | Recall | F1-score | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:1.5b before fine-tuning | 0.5684 | 0.5416 | 0.8903 | 0.6735 | 560 | 155 | 474 | 69 |

主要問題是 false positive 過高：

```text
FP = 474
```

因此 A100 微調的主要目標是：

- 降低 FP。
- 提升 Precision。
- 提升 Accuracy。
- 提升 F1-score。
- 維持 Recall 在 0.85 以上。
- 改善 Dataset 1 legitimate email 的辨識能力。

合理預期：

| 指標 | Baseline | A100 LoRA 微調後目標 |
|---|---:|---:|
| Accuracy | 0.5684 | 0.80 到 0.92 |
| Precision | 0.5416 | 0.75 到 0.92 |
| Recall | 0.8903 | 0.85 到 0.95 |
| F1-score | 0.6735 | 0.82 到 0.93 |

## 9. 最終建議

若使用 A100，建議不要再使用目前筆電環境的保守 QLoRA 設定，而改用：

```text
LoRA BF16
max sequence length = 2048
batch size per device = 8
gradient accumulation steps = 2
LoRA rank = 16
LoRA alpha = 32
epochs = 3
learning rate = 1e-4
```

這組設定對 A100 負荷合理，訓練速度快，且比 4-bit QLoRA 更穩定。Full fine-tuning 可作為進階實驗，但不是本 phishing detection PoC 的第一選擇。

