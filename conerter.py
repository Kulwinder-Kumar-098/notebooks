import pandas as pd

# 🔹 File path
file_path = r"C:\Users\hp\OneDrive\Desktop\desktop files\notebooks\bail.jsonl"

# 🔹 Load JSONL
df = pd.read_json(file_path, lines=True)

print("Columns:", df.columns)
print("Rows:", len(df))

# 🔹 Save directly to CSV
df.to_csv("bail_corpus.csv", index=False, encoding="utf-8")

print("✅ Converted to bail_corpus_raw.csv")