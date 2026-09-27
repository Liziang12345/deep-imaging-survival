# Multi-Stage Imaging Models

Research code for a generic two-stage learning pipeline using paired 3D volumes and variable-length instance sets.

This repository is intentionally dataset-agnostic. It does **not** include source data, subject identifiers, institution names, local file paths, cohort sizes, study titles, project acronyms, model results, or trained checkpoints.

## Repository structure

```text
.
├── stage1_paired_3d_classifier.py
├── stage2_bag_survival.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Stage 1

`stage1_paired_3d_classifier.py` implements a paired-volume 3D classification pipeline with:

- two co-registered 3D image inputs and binary masks;
- mask-guided fixed-size ROI extraction;
- a shared-weight 3D Swin Transformer encoder;
- gated feature fusion across the paired inputs;
- an internal subject-disjoint development split created only from the training split;
- final refitting on the complete training split before holdout evaluation;
- export of probabilities, metrics, and deep features.

### Metadata format

The metadata CSV must contain:

```text
instance_id,subject_id,label,split
```

`label` must be binary. `split` accepts `train`, `validation`, and `test` (or `0`, `1`, and `2`).

Each image or mask directory should contain files named:

```text
<instance_id>.nii.gz
```

### Example

```bash
python stage1_paired_3d_classifier.py \
  --metadata /path/to/metadata.csv \
  --first-volume-dir /path/to/volume_a \
  --first-mask-dir /path/to/mask_a \
  --second-volume-dir /path/to/volume_b \
  --second-mask-dir /path/to/mask_b \
  --output-dir outputs/stage1
```

Use `--no-pretrained` to initialize the backbone without torchvision pretrained weights.

## Stage 2

`stage2_bag_survival.py` implements subject-level multiple-instance time-to-event modeling with:

- variable-length bags of instance features;
- a shared instance encoder;
- gated-attention pooling;
- three generic distribution descriptors derived from instance metadata;
- Cox partial likelihood with Breslow handling of tied event times;
- ranking, attention-entropy, and risk-score regularization;
- train-only preprocessing and model selection;
- post-lock evaluation on holdout splits.

### Input tables

**Instance feature table**

```text
instance_id,subject_id,feat_0,feat_1,...
```

**Subject table**

```text
subject_id,time,event,split
```

**Instance metadata table**

```text
instance_id,subject_id,location,region_group
```

The default feature prefix is `feat_`. Use `--feature-prefix ""` only when every non-key column in the feature table is an intended numeric feature.

### Example

```bash
python stage2_bag_survival.py \
  --instance-features /path/to/instance_features.csv \
  --subjects /path/to/subjects.csv \
  --instance-metadata /path/to/instance_metadata.csv \
  --output-dir outputs/stage2
```

## Installation

A recent Python 3 environment with PyTorch is recommended.

```bash
pip install -r requirements.txt
```

GPU acceleration is used automatically when CUDA is available.

## Data and privacy

No data are distributed with this repository. Do not commit raw images, masks, tabular data, generated predictions, checkpoints, logs, or files containing direct or indirect identifiers. The provided `.gitignore` excludes common local data and output locations, but it should be reviewed before every public release.

## Reproducibility notes

- Random seeds are fixed in both scripts.
- Preprocessing statistics are fit on the training split only.
- Holdout splits are not used for gradient updates.
- Stage 1 classification thresholds are selected from training predictions and then fixed for holdout evaluation.
- Input identifiers are used only for data linkage and output bookkeeping; they are not model features.

## Citation

A citation section can be added after the associated work is public. No manuscript title, author list, institution, dataset size, or unpublished result is included here.
