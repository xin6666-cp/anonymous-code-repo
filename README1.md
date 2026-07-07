<div align="center">

# 🩺 ECCTL: Explicit Comparative Change Token Learning for Longitudinal Chest X-ray Report Generation

![Status](https://img.shields.io/badge/Status-Under%20Review-orange)
![Anonymous](https://img.shields.io/badge/Double--Blind-Anonymous-lightgrey)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1-ee4c2c)
![Lightning](https://img.shields.io/badge/Lightning-%E2%89%A52.1-792ee5)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- Place the framework overview (paper Fig. 2) at assets/framework.png -->
<img src="assets/framework.png" alt="ECCTL Framework Overview" width="88%">

*Explicit current–prior comparison for change-aware radiology report generation.*

</div>

---

## 📰 News

- **Code released for anonymous review.**
- Two-stage training and evaluation scripts are provided.
- Backbone / pretrained weights are fetched via `scripts/download_weights.sh`.

---

## 🔎 Overview

Longitudinal chest X-ray report generation must describe how findings **change** relative to
a prior study — *new*, *worsened*, *improved*, or *stable*. Most prior-aware methods fuse or
jointly encode current and prior images, which (i) is sensitive to differences in
positioning / acquisition / spatial appearance, and (ii) mixes change information into
general visual features, forcing the decoder to infer temporal change implicitly.

**ECCTL** models disease change through **explicit comparison**, in two stages:

- **Stage 1 — Vision-Language Alignment Pretraining.** Image–text contrastive alignment
  between longitudinally enhanced visual features and report text.
- **Stage 2 — Temporal Change-Aware Report Generation** with two dedicated modules:
  - 🧷 **Soft Prior Registration (SPR)** — softly aligns prior tokens to current tokens in
    feature space (cross-attention with a learnable residual gate `α = σ(α_raw)`), then
    builds explicit change features from the signed residual, absolute residual, and
    element-wise interaction.
  - 🧠 **Comparison Reasoning Module (CRM)** — learnable comparison queries summarize the
    three-stream pool `[current; aligned-prior; change]` into compact **change-aware
    latents**, supplied directly to the decoder. A prior-availability gate zeroes them for
    no-prior cases to avoid temporal hallucination.

An auxiliary weak temporal-change loss (`L_tc`) supervises the comparison latents using
change categories mined from reference reports.

**Input:** current X-ray + optional prior X-ray (if exist)+ clinical context (indication / history).
**Output:** a free-text *Findings* report.

### ✨ Highlights

- 🔬 **Explicit change evidence** for the decoder, not just fused history.
- 🧷 **Soft prior alignment** improves current–prior comparability without anatomical registration.
- 🧠 **Compact comparison latents** with weak *new / worsened / improved / stable* supervision.
- 🛡️ **Prior-aware gating** preserves single-study behavior when no prior exists.

---

## ⚙️ Installation

```bash
conda create -n ecctl python=3.10
conda activate ecctl
pip install -r requirements.txt
```

> **GPU / CUDA.** Install a `torch` build matching your CUDA driver (see https://pytorch.org).
> Versions pinned in `requirements.txt` are a reference and may need adjustment.

---

## 📁 Repository Structure

```
project-root/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── main_github.py                
├── models/
│   ├── model_github.py            
│   ├── comparison_reasoning.py   
│   ├── soft_prior_registration.py 
│   ├── bert_model.py              
│   └── perceiver_pytorch.py       
├── src/preprocessing/
│   └── section_split.py       
├── scripts/
│   ├── prepare_data.sh  
├── tools/                        
├── data/          
└── checkpoints/  
```

> `outputs/` and `logs/` are created at run time and git-ignored. Run parameters live in
> `scripts/*.sh` (single source of truth) and can be overridden via environment variables or
> CLI flags. `models/bert_model.py`, `models/perceiver_pytorch.py`, and `tools/` are part of
> your existing codebase.

---

## 🗂️ Dataset Preparation

We use **MIMIC-CXR** and **MIMIC-ABN** from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/).
Both are credentialed and **not redistributed** here.

**1) Raw annotation.** A per-study annotation JSON keyed by split:

```json
{ "train": [ {"id": "...", "findings": "...", "impression": "...", "prior_study": {...}|null, ...}, ... ],
  "val":   [ ... ],
  "test":  [ ... ] }
```

The *Findings* section is the generation target. Fields consumed by the pipeline:

| Field              | Meaning                                                                    |
|--------------------|----------------------------------------------------------------------------|
| `findings`         | Current findings (reference target).                                       |
| `impression`       | Impression (scanned with findings for change keywords).                    |
| `prior_study`      | Prior study reference, or `null` if none.                                  |
| `view_position`    | Projection view (PA / AP / LATERAL / …), mapped via `view_position_dict`.  |
| clinical context   | Indication / history text used as decoder-side context.                    |
| image path         | Current (and optional prior) image path, relative to `--images_dir`.       |

**2) Add weak temporal-change labels** (`src/preprocessing/section_split.py`):

```bash
python src/preprocessing/section_split.py \
    --input  data/raw/mimic_cxr_annotation.json \
    --output data/processed/mimic_cxr_annotation_labeled.json
```

| Field          | Values                                                                       |
|----------------|------------------------------------------------------------------------------|
| `has_prior`    | `1` if a prior study exists, else `0` (prior-availability mask).             |
| `change_label` | `0=no_prior`, `1=new`, `2=worsened`, `3=improved`, `4=stable`, `5=none`.     |

> Only `change_label ∈ {1,2,3,4}` are supervised by `L_tc` (remapped to `{0,1,2,3}`);
> `{0, 5}` (no-prior / no keyword) are excluded. Keyword priority:
> `new > worsened > improved > stable`.

**3) View-position dictionary** — a JSON mapping view strings to integer ids, passed via
`--view_position_dict`.

**Processed layout**
```
data/
├── raw/         mimic_cxr_annotation.json  ·  mimic_abn_annotation.json
├── processed/   *_annotation_labeled.json  ·  view_position_dict.json
└── images/      (MIMIC images — not redistributed)
```
Splits (`train`/`val`/`test`) are defined inside the annotation JSON, following the official
MIMIC split.

---

## 🧩 Checkpoints & Pretrained Weights

Fetch public backbones and place model files under `checkpoints/` (nothing large is committed).

```bash
bash scripts/download_weights.sh
```

```
checkpoints/
├── pretrained/
│   ├── rad-dino/                        # vision encoder (frozen)
│   ├── BiomedVLP-CXR-BERT-specialized/  # text encoder
│   ├── distilgpt2/                      # decoder base
│   └── bert-base-uncased/               # tokenizer for the CheXbert metric
├── evaluator/
│   └── chexbert.pth                     # CheXbert labeler (CE metrics) — download manually
└── final_model/
    └── best_model.ckpt                  # your trained Stage 2 checkpoint
```

> `chexbert.pth` is obtained from the official CheXbert release. Paths can be overridden on
> the command line (`--rad_dino_path`, `--cxr_bert_path`, `--distilgpt2_path`,
> `--chexbert_path`, `--bert_path`).

---

## 🚀 Training

ECCTL is trained in two stages (both via `main_github.py`).

```bash
# Stage 1 — vision-language alignment pretraining
STAGE=pretrain bash scripts/m-cxr-pretraining-finetune.sh

# Stage 2 — temporal change-aware generation (default)
bash scripts/m-cxr-report-generation-finetune.sh
```



## 💬stage1

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main_github.py \
--data_name "mimic_cxr" \
--version "best" \
--task "pretraining" \
--phase "inference" \
--ann_path "Data/priorrg_mimic_cxr_annotation.json" \
--view_position_dict "/Data/view-positions-dict-mimic.json" \
--images_dir "./images" \
--max_length 100 \
--encoder_max_length 300 \
--num_workers 6 \
--is_save_checkpoint "yes" \
--ckpt_zoo_dir "" \
--test_ckpt_path "./checkpoint/best_model.ckpt" \
--temporal_fusion_num_blocks 3 \
--perceiver_num_blocks 3 \
--num_latents 128 \
--patience 10 \
--pt_lr 5.0e-5 \
--epochs 30 \
--batch_size 32

```

💬stage2

---

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2 python main_github.py \
--data_name "mimic_cxr" \
--version "best" \
--task "report-generation-gpt2" \
--phase "finetune" \
--distilgpt2_path  /models/distilgpt2/ \
--rad_dino_path  /models/rad-dino \
--cxr_bert_path  models/BiomedVLP-CXR-BERT-specialized \
--bert_path /models/bert-base-uncased \
--chexbert_path /models/RRG-metrics-pretrained-model/chexbert.pth \
--save_dir  ./outputs \
--ann_path "/Data/priorrg_mimic_cxr_annotation_sec.json" \
--view_position_dict "/Data/view-positions-dict-mimic.json" \
--images_dir "./images" \
--max_length 100 \
--encoder_max_length 300 \
--num_workers 6 \
--is_save_checkpoint "yes" \
--load "./outputs/<stage1_run>/checkpoint/best_model.ckpt" \
--ckpt_zoo_dir "" \
--temporal_fusion_num_blocks 3 \
--perceiver_num_blocks 3 \
--num_latents 128 \
--patience 5 \
--pt_lr 5.0e-6 \
--ft_lr 5.0e-5 \
--monitor_metric "RCB" \
--epochs 30 \
--batch_size 8 \
--use_crm "yes" \

```

---

## 🙏 Acknowledgements

This project builds on public datasets, foundation models, and evaluation tools, including
**MIMIC-CXR**, **MIMIC-ABN**, **RAD-DINO**, **CXR-BERT (BiomedVLP)**, **distilGPT-2**, the
**Perceiver** architecture, **CheXbert**, and **RadGraph / RaTEScore**. We thank the
maintainers of these public resources. All acknowledgements are kept free of
author-identifying information for anonymous review.

