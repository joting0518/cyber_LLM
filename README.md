## Dataset Preparation

| Dataset          | Label          | Final Size |
| ---------------- | -------------- | ---------: |
| Enron            | Legitimate (0) |    228,221 |
| Nazario          | Phishing (1)   |        950 |
| BEC              | Phishing (1)   |      5,338 |
| Training Dataset | Balanced       |     12,576 |

### Source Datasets

This project uses three publicly available email datasets.
| Dataset                       | Purpose                                | Link                                                                                                                                           |
| ----------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Enron Email Dataset           | Legitimate Emails                      | [https://www.kaggle.com/datasets/wcukierski/enron-email-dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)               |
| Nazario Phishing Corpus       | Phishing Emails                        | [https://monkey.org/~jose/phishing/phishing3.mbox](https://monkey.org/~jose/phishing/phishing3.box)                                                                       |
| Adversarial BEC Email Dataset | Phishing & Adversarial Phishing Emails | [https://www.kaggle.com/datasets/yoadjei/adversarial-bec-email-dataset](https://www.kaggle.com/datasets/yoadjei/adversarial-bec-email-dataset) |

#### 1. Enron Email Dataset (Legitimate Emails)
Processed file:

```text
enron/enron_clean.csv
```

Preprocessing statistics:

```text
Original emails: 517401
Successfully parsed: 517401
Remove empty subject: 33236
Remove empty body: 0
Remove short emails: 814
Remove duplicates: 255130

Final emails: 228221
```

Label assignment:

```text
label = 0
dataset = 1
```

---

#### 2. Nazario Phishing Corpus

Processed file:

```text
nazario/nazario_clean.csv
```

Preprocessing statistics:

```text
Original emails: 2276
Successfully parsed: 2276
Remove empty subject: 20
Remove empty body: 970
Remove short emails: 0
Remove duplicates: 336

Final emails: 950
```

Label assignment:

```text
label = 1
dataset = 2
```

---

#### 3. Adversarial BEC Email Dataset

Processed file:

```text
Adversarial_BEC_Email/bec_clean.csv
```

Preprocessing statistics:

```text
Original emails: 8422
Successfully parsed: 8422
Remove empty subject: 60
Remove empty body: 60
Remove short emails: 0
Remove duplicates: 2964

Final emails: 5338
```

Label assignment:

```text
label = 1
dataset = 3 (clean BEC)
dataset = 4 (adversarial BEC)
```

---

### Dataset Balancing

After preprocessing:

#### Phishing Emails

```text
Nazario Corpus      950
BEC Dataset        5338
-----------------------
Total              6288
```

#### Legitimate Emails

To avoid severe class imbalance, all phishing emails were retained and an equal number of legitimate emails were randomly sampled from the Enron dataset.

seed = 42
```text
Sampled Enron Emails: 6288
```

Final class distribution:

```text
Phishing (label=1):      6288
Legitimate (label=0):    6288
```

Total dataset size:

```text
12576 emails
```

Random seed:

```text
seed = 42
```

---

### Train / Validation / Test Split

The dataset was shuffled using a fixed random seed and split using stratified sampling.

Split ratio:

```text
Train : 80%
Valid : 10%
Test  : 10%
```

Dataset statistics:

```text
Train : 10060
Valid : 1258
Test  : 1258
```

#### Train Label Distribution

```text
label=1 : 5030
label=0 : 5030
```

#### Validation Label Distribution

```text
label=1 : 629
label=0 : 629
```

#### Test Label Distribution

```text
label=1 : 629
label=0 : 629
```

### Dataset Schema

Each email record contains the following fields:

| Column    | Type    | Description                                                                                            |
| --------- | ------- | ------------------------------------------------------------------------------------------------------ |
| `subject` | String  | The email subject line after preprocessing and normalization.                                          |
| `body`    | String  | The email body content after HTML removal, whitespace normalization, and text cleaning.                |
| `label`   | Integer | Binary classification label. `0` represents a legitimate email, while `1` represents a phishing email. |
| `dataset` | Integer | Source dataset identifier used for analysis and error tracking.                                        |

#### Label Definition

| Label | Description      |
| ----- | ---------------- |
| `0`   | Legitimate Email |
| `1`   | Phishing Email   |

#### Dataset Identifier

| Dataset ID | Source                        |
| ---------- | ----------------------------- |
| `1`        | Enron Email Dataset           |
| `2`        | Nazario Phishing Corpus       |
| `3`        | Synthetic BEC Email Dataset   |
| `4`        | Adversarial BEC Email Dataset |

#### Example

```csv
subject,body,label,dataset
Verify your PayPal Account,"We recently detected unusual activity in your account. Please verify your information.",1,2
```

In this example:

* `subject` contains the email title.
* `body` contains the email content.
* `label=1` indicates a phishing email.
* `dataset=2` indicates that the sample originates from the Nazario Phishing Corpus.
