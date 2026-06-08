from email import message_from_string
import pandas as pd
import re


# Load Enron Dataset
df = pd.read_csv("enron.csv")

print(f"Original emails: {len(df)}")

emails = []


# Parse Email
for raw_email in df["message"]:

    try:

        msg = message_from_string(raw_email)

        subject = msg.get("Subject", "")

        # Extract Body
        if msg.is_multipart():

            body = ""

            for part in msg.walk():

                if part.get_content_type() == "text/plain":

                    payload = part.get_payload(decode=True)

                    if payload:

                        body += payload.decode(
                            errors="ignore"
                        )

        else:

            body = msg.get_payload()

            if body is None:
                body = ""

            body = str(body)

        emails.append({
            "subject": subject,
            "body": body,
            "label": 0,
            "dataset": 1
        })

    except Exception:
        continue

enron_df = pd.DataFrame(emails)

print(f"Successfully parsed: {len(enron_df)}")


# Clean Subject
def clean_subject(subject):

    if subject is None:
        return ""

    subject = str(subject)

    # Remove Re:, RE:, FW:, FWD:
    while True:

        new_subject = re.sub(
            r'^(re|fw|fwd)\s*:\s*',
            '',
            subject,
            flags=re.IGNORECASE
        )

        if new_subject == subject:
            break

        subject = new_subject

    subject = re.sub(r'\s+', ' ', subject)

    return subject.strip()


# Clean Body
def clean_body(text):

    if text is None:
        return ""

    text = str(text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

enron_df["subject"] = (
    enron_df["subject"]
    .fillna("")
    .apply(clean_subject)
)

enron_df["body"] = (
    enron_df["body"]
    .fillna("")
    .apply(clean_body)
)


# Remove Empty Subject
before = len(enron_df)

enron_df = enron_df[
    enron_df["subject"] != ""
]

print(
    f"Remove empty subject: "
    f"{before-len(enron_df)}"
)


# Remove Empty Body
before = len(enron_df)

enron_df = enron_df[
    enron_df["body"] != ""
]

print(
    f"Remove empty body: "
    f"{before-len(enron_df)}"
)


# Remove Very Short Emails
before = len(enron_df)

enron_df = enron_df[
    (
        enron_df["subject"].str.len()
        +
        enron_df["body"].str.len()
    ) >= 20
]

print(
    f"Remove short emails: "
    f"{before-len(enron_df)}"
)


# Remove Duplicates
before = len(enron_df)

enron_df = enron_df.drop_duplicates(
    subset=["subject", "body"]
)

print(
    f"Remove duplicates: "
    f"{before-len(enron_df)}"
)


# Reset Index
enron_df = enron_df.reset_index(
    drop=True
)


# Final Statistics
print()
print(f"Final emails: {len(enron_df)}")

print()
print("Sample:")

for i in range(5):

    print("=" * 80)

    print("SUBJECT:")
    print(enron_df.iloc[i]["subject"])

    print()

    print("BODY:")
    print(
        enron_df.iloc[i]["body"][:300]
    )

    print()

    print("LABEL:")
    print(
        enron_df.iloc[i]["label"]
    )


# Save
enron_df.to_csv(
    "enron_clean.csv",
    index=False
)

print()
print("Saved: enron_clean.csv")

# Original emails: 517401
# Successfully parsed: 517401
# Remove empty subject: 33236
# Remove empty body: 0
# Remove short emails: 814
# Remove duplicates: 255130

# Final emails: 228221