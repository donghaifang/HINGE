"""Validate HINGE training data before launching a run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as anndata
import numpy as np
import pandas as pd
import torch

from Hinge.train import _load_gene_indices
from Hinge.utils import build_parser, normalize_args


def _read_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.genfromtxt(path, dtype=str)
    if values.ndim == 0:
        return [str(values)]
    return [str(v) for v in values.tolist()]


def _check_tensor(path: Path, expected_spots: int, expected_rank: int) -> tuple[int, ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} must contain a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != expected_rank:
        raise ValueError(f"{path} must have rank {expected_rank}, got shape {tuple(tensor.shape)}")
    if int(tensor.shape[0]) != expected_spots:
        raise ValueError(f"{path} spot count mismatch: expected {expected_spots}, got {int(tensor.shape[0])}")
    return tuple(int(v) for v in tensor.shape)


def validate(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    processed = data_path / "processed_data"
    st_dir = data_path / "st"

    slide_file = processed / args.folder_list_filename
    slides = _read_list(slide_file)
    holdouts = {s.strip() for s in args.slide_out.split(",") if s.strip()}
    train_slides = [slide for slide in slides if slide not in holdouts]
    if not train_slides:
        raise ValueError("No training slides remain after applying --slide_out.")

    gene_file = processed / f"used_{args.gene_list_filename}"
    selected_genes = _read_list(gene_file)
    _load_gene_indices(args.gene_info_path, selected_genes)

    num_aug_ratio = int(args.num_aug_ratio)
    failures: list[str] = []
    total_spots = 0

    for slide in train_slides:
        try:
            h5ad_path = st_dir / f"{slide}.h5ad"
            if not h5ad_path.exists():
                raise FileNotFoundError(h5ad_path)
            adata = anndata.read_h5ad(h5ad_path, backed="r")
            missing_genes = [gene for gene in selected_genes if gene not in adata.var_names]
            if missing_genes:
                preview = ", ".join(missing_genes[:8])
                raise KeyError(f"{len(missing_genes)} selected genes missing from {h5ad_path}: {preview}")
            n_spots = int(adata.n_obs)
            total_spots += n_spots

            uni_shape = _check_tensor(processed / "1spot_uni_ebd" / f"{slide}_uni.pt", n_spots, 2)
            conch_shape = _check_tensor(processed / "1spot_conch_ebd" / f"{slide}_conch.pt", n_spots, 2)

            if num_aug_ratio > 0:
                uni_aug_shape = _check_tensor(processed / "1spot_uni_ebd_aug" / f"{slide}_uni_aug.pt", n_spots, 3)
                conch_aug_shape = _check_tensor(processed / "1spot_conch_ebd_aug" / f"{slide}_conch_aug.pt", n_spots, 3)
                if uni_aug_shape[1] < num_aug_ratio or conch_aug_shape[1] < num_aug_ratio:
                    raise ValueError(
                        f"{slide} has fewer augmentation views than --num_aug_ratio={num_aug_ratio}: "
                        f"UNI {uni_aug_shape}, CONCH {conch_aug_shape}"
                    )

            print(f"[OK] {slide}: spots={n_spots}, UNI={uni_shape}, CONCH={conch_shape}")
        except Exception as exc:
            failures.append(f"[FAIL] {slide}: {exc}")

    if failures:
        print("\n".join(failures))
        raise SystemExit(1)

    print(
        f"Data check passed: {len(train_slides)} training slides, "
        f"{len(selected_genes)} genes, {total_spots} original spots."
    )


def main() -> None:
    parser = build_parser()
    args = normalize_args(parser.parse_args())
    if not os.path.exists(args.gene_info_path):
        raise FileNotFoundError(args.gene_info_path)
    pd.read_csv(args.gene_info_path, nrows=1)
    validate(args)


if __name__ == "__main__":
    main()

