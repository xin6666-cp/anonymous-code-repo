#!/usr/bin/env bash
# Download public backbones into checkpoints/. Run from the project root:
#   bash scripts/download_weights.sh
# Requires: pip install -U "huggingface_hub[cli]"
#
# NOTE: CheXbert (chexbert.pth) is distributed by its authors under their own terms and is
#       NOT auto-downloaded here. Obtain it from the official CheXbert release and place it
#       at checkpoints/evaluator/chexbert.pth.
set -euo pipefail

PRE=checkpoints/pretrained
EVAL=checkpoints/evaluator
mkdir -p "$PRE" "$EVAL"

dl () {  # dl <hf_repo_id> <target_dir>
  echo ">> $1 -> $2"
  huggingface-cli download "$1" --local-dir "$2" --local-dir-use-symlinks False
}

dl microsoft/rad-dino                         "$PRE/rad-dino"
dl microsoft/BiomedVLP-CXR-BERT-specialized   "$PRE/BiomedVLP-CXR-BERT-specialized"
dl distilbert/distilgpt2                       "$PRE/distilgpt2"
dl google-bert/bert-base-uncased               "$PRE/bert-base-uncased"

echo
echo "All HF backbones downloaded."
echo "Remaining manual step: place CheXbert weights at $EVAL/chexbert.pth"
