import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
FEATURES_ROOT = "data/features_csv"
LABEL_CSV = "data/speech_dataset.csv"
OUTPUT_CSV = "final_dataset.csv"

# === Step 1: Load Labels ===
print("🔄 Loading labels from speech_dataset.csv...")
label_df = pd.read_csv(LABEL_CSV)
label_df["Filename"] = label_df["Filename"].str.strip().str.lower()
label_map = dict(zip(label_df["Filename"].str.replace(".wav", "", regex=False), label_df["Emotion"]))
print(f"✅ Loaded {len(label_map)} labels.")

# === Step 2: Walk through features directory ===
data = []
skipped = 0

print("🔍 Reading feature vectors and matching labels...")
for root, dirs, files in os.walk(FEATURES_ROOT):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            file_id = Path(file).stem.strip().lower()  # Remove '.csv' and normalize

            label = label_map.get(file_id)
            if label:
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        features = df.iloc[-1].values  # Get the last row
                        data.append([file_id, label] + list(features))
                except Exception as e:
                    print(f"[ERROR] Could not read {file}: {e}")
                    skipped += 1
            else:
                print(f"[SKIPPED] No label found for {file}")
                skipped += 1

# === Step 3: Save final dataset ===
if data:
    print("💾 Saving to final_dataset.csv...")
    feature_count = len(data[0]) - 2
    columns = ["file", "label"] + [f"f{i}" for i in range(feature_count)]
    final_df = pd.DataFrame(data, columns=columns)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Done. {len(data)} samples saved to {OUTPUT_CSV}")
else:
    print("⚠️ No data collected. Check for filename mismatches or missing features.")

print(f"🧾 Total skipped files: {skipped}")
