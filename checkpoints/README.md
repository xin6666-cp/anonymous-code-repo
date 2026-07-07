# Checkpoints

Weights are **not** committed to git. Download public backbones and place model files here.

```
checkpoints/
├── pretrained/
│   ├── rad-dino/
│   ├── BiomedVLP-CXR-BERT-specialized/
│   ├── distilgpt2/
│   └── bert-base-uncased/
├── evaluator/
│   └── chexbert.pth
└── final_model/
    └── put_final_checkpoint_here
```

- Public backbones: `bash scripts/download_weights.sh`.
- CheXbert (`chexbert.pth`): obtain from the official CheXbert release; place under `evaluator/`.
- Stage 1 checkpoint: produced by the `pretrain` phase (found under `outputs/`); pass it to
  Stage 2 via `--load`.
- Paths can be overridden on the command line (e.g. `--rad_dino_path`, `--chexbert_path`).
