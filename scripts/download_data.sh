#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
echo "Dataset downloaded to data/raw."
