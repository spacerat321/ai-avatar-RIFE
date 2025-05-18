import os
import shutil
from tqdm import tqdm
from pathlib import Path

# === Resolve project root ===
project_root = Path(__file__).resolve().parent

# === Define absolute paths ===
dataset_root = project_root / "data/vimeo_septuplet/sequences"
trainlist_file = project_root / "data/vimeo_septuplet/sep_trainlist.txt"
output_root = project_root / "train"

# === Load training list ===
with open(trainlist_file, 'r') as f:
    clip_keys = [line.strip() for line in f.readlines() if line.strip()]

print(f"Found {len(clip_keys)} valid clip entries in {trainlist_file}")
os.makedirs(output_root, exist_ok=True)

skipped = 0

for key in tqdm(clip_keys):
    input_dir = os.path.join(dataset_root, key)
    out_name = key.replace("/", "_")
    out_dir = os.path.join(output_root, out_name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        shutil.copy(os.path.join(input_dir, "im1.png"), os.path.join(out_dir, "00000.png"))
        shutil.copy(os.path.join(input_dir, "im4.png"), os.path.join(out_dir, "00001.png"))  # GT
        shutil.copy(os.path.join(input_dir, "im7.png"), os.path.join(out_dir, "00002.png"))
    except FileNotFoundError:
        skipped += 1
        shutil.rmtree(out_dir, ignore_errors=True)

print(f"Conversion complete. Skipped {skipped} clips due to missing frames.")
print(f"Ready-to-train RIFE dataset created in: {output_root}/")
