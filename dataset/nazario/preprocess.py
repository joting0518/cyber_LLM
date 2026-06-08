import pandas as pd
import re
from bs4 import BeautifulSoup

# =========================
# Load CSV
# =========================

df = pd.read_csv("nazario.csv")

original_count = len(df)

print(f"Original emails: {original_count}")

# =========================
# Fill NaN
# =========================

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

print(f"Successfully parsed: {len(df)}")
df["dataset"] = 2
# =========================
# Remove HTML
# =========================

def clean_html(text):

    try:

        soup = BeautifulSoup(
            str(text),
            "html.parser"
        )

        return soup.get_text(
            separator=" "
        )

    except:

        return ""

df["body"] = df["body"].apply(
    clean_html
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

df["subject"] = df["subject"].apply(
    normalize
)

df["body"] = df["body"].apply(
    normalize
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
    df["body"].str.len() >= 20
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

final_count = len(df)

print()
print(f"Final emails: {final_count}")

# =========================
# Keep Required Columns
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
    "nazario_clean.csv",
    index=False
)

print("\nSaved: nazario_clean.csv")

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

# Original emails: 2276
# Successfully parsed: 2276
# Remove empty subject: 20
# Remove empty body: 970
# Remove short emails: 0
# Remove duplicates: 336

# Final emails: 950