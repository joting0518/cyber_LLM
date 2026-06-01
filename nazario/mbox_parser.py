import mailbox
import pandas as pd
import csv
# total = 2276
mbox = mailbox.mbox("Nazario_Phishing_Corpus.mbox")
emails = []

for msg in mbox:

    subject = msg["subject"]

    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(
                    decode=True
                ).decode(errors="ignore")
    else:
        body = msg.get_payload(
            decode=True
        ).decode(errors="ignore")

    emails.append({
        "subject": subject,
        "body": body,
        "label": 1
    })

df = pd.DataFrame(emails)

print(df.head())
print("email amount",len(df))
print(df.columns)

df = df.drop([1685, 1703, 1711])

df.to_csv(
    "nazario.csv",
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_ALL
)

print(len(df))  # 2276