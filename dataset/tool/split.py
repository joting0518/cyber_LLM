import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
# Load
enron = pd.read_csv(
    "enron/enron_clean.csv"
)

nazario = pd.read_csv(
    "nazario/nazario_clean.csv"
)

bec = pd.read_csv(
    "Adversarial_BEC_Email/bec_clean.csv"
)

# Phishing
phishing = pd.concat(
    [nazario, bec],
    ignore_index=True
)

print(
    f"Phishing emails: "
    f"{len(phishing)}"
)

# Sample Legitimate
enron_sample = enron.sample(
    n=len(phishing),
    random_state=SEED
)

print(
    f"Legitimate emails: "
    f"{len(enron_sample)}"
)


# Merge
dataset = pd.concat(
    [enron_sample, phishing],
    ignore_index=True
)

dataset = dataset.sample(
    frac=1,
    random_state=SEED
).reset_index(
    drop=True
)

print(
    f"Total emails: "
    f"{len(dataset)}"
)

print()

print(
    dataset["label"]
    .value_counts()
)


# Train / Temp
train_df, temp_df = train_test_split(
    dataset,
    test_size=0.2,
    random_state=SEED,
    stratify=dataset["label"]
)


# Valid / Test
valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=SEED,
    stratify=temp_df["label"]
)


# Statistics
print()
print(
    f"Train: {len(train_df)}"
)

print(
    f"Valid: {len(valid_df)}"
)

print(
    f"Test : {len(test_df)}"
)

print()

print("Train Label Distribution")
print(
    train_df["label"]
    .value_counts()
)

print()

print("Valid Label Distribution")
print(
    valid_df["label"]
    .value_counts()
)

print()

print("Test Label Distribution")
print(
    test_df["label"]
    .value_counts()
)


# Save
train_df.to_csv(
    "train.csv",
    index=False
)

valid_df.to_csv(
    "valid.csv",
    index=False
)

test_df.to_csv(
    "test.csv",
    index=False
)

print()
print("Saved:")
print("train.csv")
print("valid.csv")
print("test.csv")

# Phishing emails: 6288
# Legitimate emails: 6288
# Total emails: 12576

# label
# 1    6288
# 0    6288
# Name: count, dtype: int64

# Train: 10060
# Valid: 1258
# Test : 1258

# Train Label Distribution
# label
# 1    5030
# 0    5030
# Name: count, dtype: int64

# Valid Label Distribution
# label
# 0    629
# 1    629
# Name: count, dtype: int64

# Test Label Distribution
# label
# 0    629
# 1    629
