# HINGE

Official implementation placeholder for **HINGE**, an image-conditioned masked reconstruction model for spatial transcriptomics gene-expression prediction.

HINGE learns to reconstruct masked spot-level gene expression from histology image embeddings. The current codebase includes the PyTorch model, CellFM/MindSpore weight-loading utilities, data assembly, masking scheduler, validation, checkpointing, and runnable command-line entry points.

> Paper, pretrained checkpoints, processed features, and final benchmark numbers will be linked here after release.

## Method Overview



HINGE uses a progressive mask chain over selected genes. At each training step, the model receives masked expression values, a timestep, a mask state, and image-derived conditioning features, then predicts the original gene-expression vector.

## News

- `2026-05-25`: Repository cleanup, packaging metadata, data checker, and training entry point added.

## Installation

```bash
conda create -n hinge python=3.10 -y
conda activate hinge

pip install -e .
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build that matches your GPU and driver if the default wheel is not appropriate for your machine.

Optional CellFM checkpoint loading requires MindSpore:

```bash
pip install -e ".[mindspore]"
```

The HEST download examples use the Hugging Face CLI:

```bash
pip install -e ".[data]"
```

## Repository Layout

```text
Hinge/
├── check_data.py        # Validate data layout, feature files, and gene metadata
├── dataloader.py        # AnnData and image-embedding dataset assembly
├── load_weight.py       # MindSpore CellFM checkpoint conversion helpers
├── mask_scheduler.py    # Mask-chain schedule used during training
├── models.py            # HINGE model definition
├── pipeline.py          # Trainer and validation/checkpoint loop
├── retentionlayer.py    # Retention, LoRA, AdaLN, and decoder layers
├── train.py             # Command-line training entry point
├── train_helper.py      # Logging, seeding, EMA, and helper utilities
└── utils.py             # CLI arguments and runtime utilities
```

## Data Sources

This project is designed around HEST-style spatial transcriptomics data.

- HEST-1k dataset: [MahmoodLab/hest on Hugging Face](https://huggingface.co/datasets/MahmoodLab/hest)
- HEST code and download tutorials: [mahmoodlab/HEST](https://github.com/mahmoodlab/HEST)
- UNI image encoder: [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)
- CONCH image encoder: [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)
- CellFM source and metadata: [biomed-AI/CellFM](https://github.com/biomed-AI/CellFM)
- CellFM pretrained checkpoint mirror: [ShangguanNingyuan/CellFM](https://huggingface.co/ShangguanNingyuan/CellFM)

HEST, UNI, CONCH, and CellFM may require separate license review, Hugging Face login, and approval of gated model or dataset terms. This repository does not redistribute those files.

## Expected Data Layout

`hinge-train` expects processed data under `--data_path`:

```text
Data/
└── hest1k_datasets/CSCC/
    ├── st/
    │   └── <slide>.h5ad
    └── processed_data/
        ├── all_slide_lst.txt
        ├── used_selected_gene_list.txt
        ├── 1spot_uni_ebd/
        │   └── <slide>_uni.pt
        ├── 1spot_conch_ebd/
        │   └── <slide>_conch.pt
        ├── 1spot_uni_ebd_aug/
        │   └── <slide>_uni_aug.pt
        └── 1spot_conch_ebd_aug/
            └── <slide>_conch_aug.pt
```

Required files:

- `st/<slide>.h5ad`: AnnData object containing spot-by-gene expression.
- `processed_data/all_slide_lst.txt`: one slide id per line, without extension.
- `processed_data/used_selected_gene_list.txt`: one selected gene name per line.
- `processed_data/1spot_uni_ebd/<slide>_uni.pt`: tensor with shape `[spots, d_uni]`.
- `processed_data/1spot_conch_ebd/<slide>_conch.pt`: tensor with shape `[spots, d_conch]`.
- `--gene_info_path`: CellFM-style gene metadata CSV. It must contain a gene-name column such as `gene_name`, `gene`, `symbol`, or `feature_name`, and a gene-id column such as `gene_id`, `id`, `index`, or `idx`.

Optional augmentation files are required only when `--num_aug_ratio > 0`:

- `processed_data/1spot_uni_ebd_aug/<slide>_uni_aug.pt`: tensor with shape `[spots, K, d_uni]`.
- `processed_data/1spot_conch_ebd_aug/<slide>_conch_aug.pt`: tensor with shape `[spots, K, d_conch]`.

If you do not have augmented patch features, run with `--num_aug_ratio 0`.

## Data Preparation

### 1. Download HEST source data

First request access to the HEST Hugging Face dataset, then download the slide ids you plan to use. The full dataset is large, so prefer a filtered download.

Example with the Hugging Face CLI:

```bash
pip install -U huggingface_hub
huggingface-cli login

huggingface-cli download MahmoodLab/hest \
  --repo-type dataset \
  --local-dir Data/hest_raw \
  --include "HEST_v1_1_0.csv" "st/*" "wsis/*" "metadata/*"
```

For selective organ/cancer downloads, use the official HEST tutorial in `mahmoodlab/HEST`, filter `HEST_v1_1_0.csv`, and download only the desired ids.

### 2. Build the HINGE folder

Copy or symlink selected `.h5ad` files into `st/`, then create the slide list and selected-gene list:

```bash
mkdir -p Data/hest1k_datasets/CSCC/st
mkdir -p Data/hest1k_datasets/CSCC/processed_data

# Example: one slide id per line, matching st/<slide>.h5ad
printf "NCBI758\nNCBI759\n" > Data/hest1k_datasets/CSCC/processed_data/all_slide_lst.txt

# Example: one selected gene per line
printf "ACTB\nGAPDH\nMKI67\n" > Data/hest1k_datasets/CSCC/processed_data/used_selected_gene_list.txt
```

Use your experiment's real slide ids and gene panel. The default config holds out `NCBI759` with `--slide_out NCBI759`.

### 3. Generate UNI and CONCH spot features

For each slide, crop one patch centered at each spatial spot, run the patch through UNI and CONCH, and save tensors in the expected folders. The tensor row order must match `adata.obs` in `st/<slide>.h5ad`.

The current repository does **not** include the WSI patch-extraction script or feature-generation script. This is intentional for now because HEST, UNI, and CONCH are gated or separately licensed. Before public release, add your preprocessing script or provide processed `.pt` feature downloads here.

Feature contract:

```python
torch.save(uni_features, "processed_data/1spot_uni_ebd/<slide>_uni.pt")
torch.save(conch_features, "processed_data/1spot_conch_ebd/<slide>_conch.pt")

# Optional augmentation: K augmented crops/views per spot.
torch.save(uni_aug_features, "processed_data/1spot_uni_ebd_aug/<slide>_uni_aug.pt")
torch.save(conch_aug_features, "processed_data/1spot_conch_ebd_aug/<slide>_conch_aug.pt")
```

Expected shapes:

```text
uni_features:        [n_spots, d_uni]
conch_features:      [n_spots, d_conch]
uni_aug_features:    [n_spots, K, d_uni]
conch_aug_features:  [n_spots, K, d_conch]
```

## Checkpoints

Place pretrained or released checkpoints under a local `checkpoints/` or `weights/` directory. These files are ignored by Git.

```text
weights/
├── cellfm_base_weight.ckpt      # optional CellFM MindSpore checkpoint
└── hinge_cscc_best.pt           # HINGE checkpoint placeholder
```

Download placeholders:

- HINGE pretrained weights: `TODO: add Hugging Face / Zenodo / Google Drive link`
- Processed CSCC UNI/CONCH features: `TODO: add link`
- CellFM base weight: see the CellFM links above and confirm the license before use.

## Validate Data Before Training

Run this before launching a long experiment:

```bash
hinge-check-data \
  --data_path Data/hest1k_datasets/CSCC/ \
  --gene_info_path CellFM-main/csv/gene_info.csv \
  --slide_out NCBI759 \
  --num_aug_ratio 0
```

The checker verifies:

- every slide in `all_slide_lst.txt` exists under `st/`;
- selected genes exist in each `.h5ad`;
- gene names map to CellFM gene ids;
- UNI and CONCH feature tensors exist and match the number of spots;
- augmentation tensors exist only when requested.

## Training

Single-GPU or CPU smoke run without augmentation:

```bash
hinge-train \
  --data_path Data/hest1k_datasets/CSCC/ \
  --gene_info_path CellFM-main/csv/gene_info.csv \
  --MS_CKPT_PATH weights/cellfm_base_weight.ckpt \
  --results_dir results \
  --expr_name cSCC \
  --slide_out NCBI759 \
  --num_aug_ratio 0 \
  --global_batch_size 8 \
  --total_epochs 40 \
  --amp
```

Training with augmented features:

```bash
hinge-train \
  --data_path Data/hest1k_datasets/CSCC/ \
  --gene_info_path CellFM-main/csv/gene_info.csv \
  --MS_CKPT_PATH weights/cellfm_base_weight.ckpt \
  --results_dir results \
  --expr_name cSCC_aug \
  --slide_out NCBI759 \
  --num_aug_ratio 7 \
  --global_batch_size 32 \
  --total_epochs 40 \
  --amp
```

Resume from a HINGE checkpoint:

```bash
hinge-train \
  --data_path Data/hest1k_datasets/CSCC/ \
  --gene_info_path CellFM-main/csv/gene_info.csv \
  --resume results/cSCC/checkpoints/best_weight.pt \
  --results_dir results \
  --expr_name cSCC_resume \
  --num_aug_ratio 0
```

## Reproducing the Main Experiment

The intended CSCC workflow is:

1. Download selected CSCC slides from HEST.
2. Create `all_slide_lst.txt` and `used_selected_gene_list.txt`.
3. Generate one-spot UNI and CONCH embeddings for every slide.
4. Run `hinge-check-data`.
5. Train with `hinge-train`.
6. Evaluate with the released evaluation script. `TODO: add eval.py and metric commands`.

Current reproducibility status:

- Training code: included.
- Data validation: included.
- HEST raw data download instructions: included.
- UNI/CONCH feature extraction script: not included yet.
- Evaluation script and released model checkpoints: not included yet.

## Citation

If this code is useful, please cite HINGE and the upstream resources used by your experiment.

```bibtex
@inproceedings{hinge2026,
  title     = {Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene  Expression Generation from Histology Images},
  author    = {Fang, Donghai and Li, Yongheng and Wang, Zhen znd Zeng, Yuansong and Min, Wenwen},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

```bibtex
@article{zeng2025cellfm,
  title   = {CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells},
  author  = {Zeng, Yuansong and others},
  journal = {Nature Communications},
  year    = {2025}
}
```

