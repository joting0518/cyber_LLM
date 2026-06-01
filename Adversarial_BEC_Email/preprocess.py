import pandas as pd
import re

# =========================
# Load CSV
# =========================

clean_df = pd.read_csv(
    "synthetic_emails.csv"
)

poison_df = pd.read_csv(
    "synthetic_emails_poisoned.csv"
)

print(
    f"Original emails: "
    f"{len(clean_df) + len(poison_df)}"
)

# =========================
# Add Labels
# =========================

clean_df["label"] = 1
clean_df["dataset"] = 3

poison_df["label"] = 1
poison_df["dataset"] = 4

# =========================
# Merge
# =========================

df = pd.concat(
    [clean_df, poison_df],
    ignore_index=True
)

print(
    f"Successfully parsed: "
    f"{len(df)}"
)

# =========================
# Normalize Text
# =========================

def normalize(text):

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

df["subject"] = (
    df["subject"]
    .fillna("")
    .apply(normalize)
)

df["body"] = (
    df["body"]
    .fillna("")
    .apply(normalize)
)

# =========================
# Remove Empty Subject
# =========================

before = len(df)

df = df[
    df["subject"] != ""
]

after = len(df)

print(
    f"Remove empty subject: "
    f"{before-after}"
)

# =========================
# Remove Empty Body
# =========================

before = len(df)

df = df[
    df["body"] != ""
]

after = len(df)

print(
    f"Remove empty body: "
    f"{before-after}"
)

# =========================
# Remove Short Emails
# =========================

before = len(df)

df = df[
    (
        df["subject"].str.len()
        +
        df["body"].str.len()
    ) >= 20
]

after = len(df)

print(
    f"Remove short emails: "
    f"{before-after}"
)

# =========================
# Remove Duplicates
# =========================

before = len(df)

df = df.drop_duplicates(
    subset=["subject", "body"]
)

after = len(df)

print(
    f"Remove duplicates: "
    f"{before-after}"
)

# =========================
# Final Statistics
# =========================

print()
print(
    f"Final emails: {len(df)}"
)

# =========================
# Keep Columns
# =========================

df = df[
    [
        "subject",
        "body",
        "label",
        "dataset"
    ]
]

# =========================
# Save
# =========================

df.to_csv(
    "bec_clean.csv",
    index=False
)

print("\nSaved: bec_clean.csv")

# =========================
# Sample
# =========================

print("\nSample:\n")

for i in range(min(3, len(df))):

    print("=" * 80)

    print("SUBJECT:")
    print(df.iloc[i]["subject"])

    print("\nBODY:")
    print(df.iloc[i]["body"][:500])

    print("\nLABEL:")
    print(df.iloc[i]["label"])

    print("\nDATASET:")
    print(df.iloc[i]["dataset"])

# Original emails: 8422
# Successfully parsed: 8422
# Remove empty subject: 60
# Remove empty body: 60
# Remove short emails: 0
# Remove duplicates: 2964

# Final emails: 5338