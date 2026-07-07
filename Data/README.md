# Data

Datasets are **not** included in this repository. MIMIC-CXR and MIMIC-ABN are credentialed
and must be obtained from their official providers (PhysioNet).

Expected layout after preparation:

```
data/
├── raw/
│   ├── mimic_cxr_annotation.json
│   └── mimic_abn_annotation.json
├── processed/
│   ├── mimic_cxr_annotation_labeled.json   # output of src/preprocessing/section_split.py
│   ├── mimic_abn_annotation_labeled.json
│   └── view_position_dict.json
└── images/                                   # MIMIC images (not redistributed)
```

Steps:
1. Place the raw annotation JSON(s) under `data/raw/`. Each file is a dict keyed by split
   (`train` / `val` / `test`); each split is a list of study records.
2. Run `bash scripts/prepare_data.sh` to add `change_label` / `has_prior`.
3. Provide `view_position_dict.json` mapping view positions to integer ids.
4. Point `--images_dir` at your image root.

See the main README (Dataset Preparation) for field definitions and label semantics.



### Change Label Definition

The processed annotation adds the following fields to each sample:

| Field                       | Type | Description                                                  |
| --------------------------- | ---: | ------------------------------------------------------------ |
| `has_prior`                 |  int | `1` if a previous study exists; otherwise `0`.               |
| `change_label`              |  int | Temporal change category derived from report text.           |
| `_text_has_temporal_signal` |  int | Diagnostic-only field showing whether temporal trigger words were found. |

The `change_label` values are:

| Label | Name       | Meaning                                                      | Used for temporal-change supervision |
| ----: | ---------- | ------------------------------------------------------------ | ------------------------------------ |
|   `0` | `no_prior` | No previous study is available.                              | No                                   |
|   `1` | `new`      | A new finding is described.                                  | Yes                                  |
|   `2` | `worsened` | A finding is described as worse or increased.                | Yes                                  |
|   `3` | `improved` | A finding is described as improved, decreased, or resolved.  | Yes                                  |
|   `4` | `stable`   | A finding is explicitly described as stable, unchanged, or similar. | Yes                                  |
|   `5` | `none`     | A previous study exists, but no temporal-change keyword is matched. | No                                   |

During training, labels `1` to `4` are supervised. Labels `0` and `5` should be ignored for the temporal-change loss. This avoids treating samples with previous studies but no explicit change description as artificially stable.

### Run Data Preprocessing

From the project root, run:

```bash
python section_split.py  
INPUT_ANN=Data/raw/mimic_cxr_annotation.json \
OUTPUT_ANN=Data/processed/mimic_cxr_annotation_labeled.json \
```

You can override paths without modifying the code:

```bash
INPUT_ANN=Data/raw/mimic_cxr_annotation.json \
OUTPUT_ANN=Data/processed/mimic_cxr_annotation_labeled.json \
bash scripts/prepare_data.sh
```

### Toy Processed Demo

This file is fully synthetic and does not contain real patient data. It is included only to illustrate the expected annotation format.

Example item:

```json
{
  "id": "demo_000001",
  "image_path": "data/images/demo_000001.jpg",
  "prior_study": {
    "study_id": "demo_000000",
    "image_path": "data/images/demo_000000.jpg"
  },
  "view_position": "PA",
  "findings": "Compared with the previous study, the left basilar opacity has decreased.",
  "impression": "Interval improvement of left basilar opacity.",
  "has_prior": 1,
  "change_label": 3,
  "_text_has_temporal_signal": 1,
  "split": "train"
}
```

## Checkpoints and Pretrained Models

Large model files are not included in this repository. Please place them under:

```text
checkpoints/
├── pretrained/
│   ├── distilgpt2/
│   ├── rad-dino/
│   ├── BiomedVLP-CXR-BERT-specialized/
│   └── bert-base-uncased/
├── evaluator/
│   └── chexbert.pth
└── final_model/
    └── best_model.ckpt
```

## 
