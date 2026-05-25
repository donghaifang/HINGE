"""Command-line training entry point for HINGE."""

from __future__ import annotations

import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import torch

from Hinge.dataloader import assemble_dataset, prepare_dataloaders_groupwise
from Hinge.models import Hinge_models
from Hinge.pipeline import Trainer
from Hinge.train_helper import create_logger
from Hinge.utils import build_parser, normalize_args, set_seed


def _load_gene_indices(gene_info_path: str, selected_genes: list[str]) -> np.ndarray:
    """Map selected gene names to CellFM gene ids when metadata is available."""
    gene_info = pd.read_csv(gene_info_path)
    name_candidates = ("gene_name", "gene", "symbol", "feature_name")
    id_candidates = ("gene_id", "id", "index", "idx")

    name_col = next((col for col in name_candidates if col in gene_info.columns), None)
    id_col = next((col for col in id_candidates if col in gene_info.columns), None)
    if name_col is None or id_col is None:
        raise ValueError(
            f"Could not infer gene name/id columns from {gene_info_path}. "
            f"Expected one of {name_candidates} and one of {id_candidates}."
        )

    gene_to_id = dict(zip(gene_info[name_col].astype(str), gene_info[id_col].astype(int)))
    missing = [gene for gene in selected_genes if gene not in gene_to_id]
    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"{len(missing)} selected genes are missing from gene_info_path: {preview}")

    return np.asarray([gene_to_id[gene] for gene in selected_genes], dtype=np.int64)


def main() -> None:
    parser = build_parser()
    args = normalize_args(parser.parse_args())

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", str(args.num_workers_gpu)))
    set_seed(args.global_seed, rank=rank, world_size=world_size)

    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir)
    else:
        experiment_dir = Path(args.results_dir) / args.expr_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = str(experiment_dir / args.checkpoint_dir)
    args.logger = create_logger(str(experiment_dir), level=getattr(logging, args.log_level))

    dataset, args = assemble_dataset(args, rank=rank)
    batch_size = max(1, args.global_batch_size // max(1, world_size))
    train_loader, val_loader = prepare_dataloaders_groupwise(
        args,
        dataset,
        batch_size=batch_size,
        val_ratio=args.val_ratio,
    )

    ggidx = _load_gene_indices(args.gene_info_path, args.selected_genes)
    model = Hinge_models[args.model](
        ggidx=ggidx,
        input_size=args.input_gene_size,
        label_size=args.cond_size,
        enc_nlayers=args.enc_nlayers,
    )

    if args.MS_CKPT_PATH and os.path.exists(args.MS_CKPT_PATH):
        from Hinge.load_weight import load_ms_to_torch

        load_ms_to_torch(model, args.MS_CKPT_PATH, device="cpu", pretrained_layers=set(range(args.enc_nlayers)))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=args.resume_strict)

    gpu_id = 0 if torch.cuda.is_available() else -1
    trainer = Trainer(model, train_loader, val_loader, rank=rank, gpu_id=gpu_id, model_args=args)
    trainer.train(args.total_epochs)


if __name__ == "__main__":
    main()
