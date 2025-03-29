import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Update these paths as needed
project_root = "/Users/kenlim/Documents/SUTD/Term 7&8/50.021 Artificial Intelligence/AI-Project--Speech-Emotion-Recognition"
wav_root = os.path.join(project_root, "data", "dataset")
output_root = os.path.join(project_root, "data", "features_csv")
config_path = "/Users/kenlim/Documents/SUTD/Term 7&8/50.021 Artificial Intelligence/opensmile/config/emobase/emobase_copy.conf"
smilextract_bin = "/usr/local/bin/SMILExtract"  # already works globally

os.makedirs(output_root, exist_ok=True)

# Step 1: Gather all .wav files
wav_files = []
for root, _, files in os.walk(wav_root):
    for file in files:
        if file.endswith(".wav"):
            wav_path = os.path.join(root, file)
            rel_path = os.path.relpath(wav_path, wav_root)  # keeps subfolder structure
            csv_path = os.path.join(output_root, rel_path.replace(".wav", ".csv"))
            wav_files.append((wav_path, csv_path, rel_path))


# Step 2: Define function to extract one file
def extract_feature(wav_path, csv_path, rel_path):
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        cmd = [
            smilextract_bin,
            "-C", config_path,
            "-I", wav_path,
            "-O", csv_path,
        ]
        print(f"Extracting: {rel_path}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"✅ Success: {rel_path}"
    except subprocess.CalledProcessError:
        return f"❌ Failed (openSMILE error): {rel_path}"
    except Exception as e:
        return f"❌ Failed (other error): {rel_path} ({e})"


# Step 3: Run in parallel (up to 8 cores)
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(extract_feature, wav, csv, rel) for wav, csv, rel in wav_files]
        for future in as_completed(futures):
            print(future.result())
