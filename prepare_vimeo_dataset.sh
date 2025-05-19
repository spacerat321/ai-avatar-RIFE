#!/bin/bash

set -e

DATA_ROOT="/ai-avatar-data"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
TRAINLIST_FILE="$DATA_ROOT/tri_trainlist.txt"
TESTLIST_FILE="$DATA_ROOT/tri_testlist.txt"

# Select dataset: septuplet or triplet
DATASET=${1:-septuplet}
echo "Selected dataset: $DATASET"
echo "Using data root: $DATA_ROOT"

pip install --quiet tqdm Pillow

# Create central dataset directory
sudo mkdir -p "$DATA_ROOT"
sudo chown "$USER":"$USER" "$DATA_ROOT"

# Download and unzip helper
download_and_extract() {
  local url=$1
  local filename=$2
  local extract_dir=$3

  cd "$DATA_ROOT"
  if [ ! -f "$filename" ]; then
    echo "Downloading $filename ..."
    wget "$url" -O "$filename"
  else
    echo "$filename already exists. Skipping download."
  fi

  if [ ! -d "$extract_dir" ]; then
    echo "Extracting $filename ..."
    unzip -q "$filename"
  else
    echo "$extract_dir already extracted. Skipping unzip."
  fi
  cd "$PROJECT_ROOT"
}

# Septuplet: requires conversion
if [[ "$DATASET" == "septuplet" ]]; then
  download_and_extract "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip" "vimeo_septuplet.zip" "vimeo_septuplet"

  if [ ! -f "$TRAINLIST_FILE" ]; then
    echo "Error: tri_trainlist.txt not found in $DATA_ROOT"
    exit 1
  fi

  echo "Converting Vimeo-Septuplet to RIFE triplets using tri_trainlist.txt..."

  python3 - <<EOF
import os, shutil
from tqdm import tqdm
from pathlib import Path

data_root = Path("$DATA_ROOT")
project_root = Path("$PROJECT_ROOT")
dataset_root = data_root / "vimeo_septuplet/sequences"
trainlist_file = data_root / "tri_trainlist.txt"
output_root = project_root / "train"

with open(trainlist_file, 'r') as f:
    clip_keys = [line.strip() for line in f if line.strip()]

print(f"Found {len(clip_keys)} valid training clips.")
os.makedirs(output_root, exist_ok=True)
skipped = 0

for key in tqdm(clip_keys):
    input_dir = dataset_root / key
    out_name = key.replace("/", "_")
    out_dir = output_root / out_name
    os.makedirs(out_dir, exist_ok=True)

    try:
        shutil.copy(input_dir / "im1.png", out_dir / "00000.png")
        shutil.copy(input_dir / "im4.png", out_dir / "00001.png")
        shutil.copy(input_dir / "im7.png", out_dir / "00002.png")
    except FileNotFoundError:
        skipped += 1
        shutil.rmtree(out_dir, ignore_errors=True)

print(f"Triplet conversion complete. Skipped {skipped} clips.")
print(f"Triplets stored in: {output_root}")
EOF

# Triplet: direct copy
elif [[ "$DATASET" == "triplet" ]]; then
  download_and_extract "http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip" "vimeo_triplet.zip" "vimeo_triplet"

  echo "Copying triplet sequences to project train/ folder..."
  mkdir -p "$PROJECT_ROOT/train"
  cp -r "$DATA_ROOT/vimeo_triplet/sequences/"* "$PROJECT_ROOT/train/"
  echo "Triplet dataset ready at $PROJECT_ROOT/train/"

else
  echo "Unknown dataset: $DATASET"
  echo "Usage: $0 [septuplet|triplet]"
  exit 1
fi

echo "Done. You may now begin training using: python3 train.py --dataset ./train"
