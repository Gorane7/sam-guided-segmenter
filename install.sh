#!/bin/bash

# Exit on error
set -e

# Create python venv
python3 -m venv .venv
source .venv/bin/activate

# Install opencv into venv
python -m pip install opencv-python

# Install sam2 into venv
if [ -d "sam2" ]; then
    rm -rf sam2
fi

git clone https://github.com/facebookresearch/sam2.git
cd sam2
# If you get an error about filesystem running out of space, then uncomment the following line
# export TMPDIR='/var/tmp'

sed -i '3s/"setuptools>=61.0",/"setuptools>=62.3.0,<75.9",/' pyproject.toml # Fix setuptools version https://github.com/facebookresearch/sam2/issues/611#issuecomment-2741690522
python -m pip install -e .

# Download checkpoints for sam2
cd checkpoints
./download_ckpts.sh

# Exit back into original directory
cd ../..
