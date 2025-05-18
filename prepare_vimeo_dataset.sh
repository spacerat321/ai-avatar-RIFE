#!/bin/bash

set -e

# Step 1: Download Vimeo-90K Septuplet Dataset
echo "Checking for aria2..."
if ! command -v aria2c &> /dev/null; then
    echo "Installing aria2..."
    sudo apt update && sudo apt install -y aria2
fi

#echo "Downloading Vimeo-90K dataset..."
#aria2c -x 16 -s 16 -o vimeo_septuplet.zip https://data.csail.mit.edu/toflow/vimeo_septuplet.zip


# Download Vimeo-90K Septuplet via Google Drive
echo "Downloading Vimeo-90K zip dataset from Google Drive..."
gdown https://drive.google.com/uc?id=1G8kA7G5X8P1rNfbsnFPyLbHrm_Vsvv7m -O vimeo_septuplet.zip
unzip vimeo_septuplet.zip

echo "Extracting dataset..."
unzip vimeo_septuplet.zip
rm vimeo_septuplet.zip

# Step 2: Convert Septuplets to RIFE-compatible Triplets
echo "Converting septuplets to triplets..."

# Embedded Python script for conversion
python3 - <<EOF
import os, shutil
from tqdm import tqdm
from glob import glob

input_root = "vimeo_septuplet/sequences"
output_root = "train"
os.makedirs(output_root, exist_ok=True)

clips = glob(f"{input_root}/**/im1.png", recursive=True)
print(f"Converting {len(clips)} sequences...")

for im1_path in tqdm(clips):
    clip_dir = os.path.dirname(im1_path)
    name = clip_dir.replace(f"{input_root}/", "").replace("/", "_")
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        shutil.copy(os.path.join(clip_dir, "im1.png"), os.path.join(out_dir, "00000.png"))
        shutil.copy(os.path.join(clip_dir, "im4.png"), os.path.join(out_dir, "00001.png"))  # GT frame
        shutil.copy(os.path.join(clip_dir, "im7.png"), os.path.join(out_dir, "00002.png"))
    except FileNotFoundError:
        continue
EOF

echo "Dataset prepared in ./train/"
