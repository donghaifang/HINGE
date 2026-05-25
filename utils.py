import argparse
import ast
import random
import numpy as np
from typing import List
import torch
import torch.distributed as dist


# =========================
# Argument Parser
# =========================
def build_parser():
    parser = argparse.ArgumentParser(description="Training entry")

    # =========================
    # Basic/Path/Experiment
    # =========================
    parser.add_argument("--expr_name", type=str, default="cSCC")
    parser.add_argument("--data_path", type=str, default="Data/hest1k_datasets/CSCC/")
    parser.add_argument("--results_dir", type=str, default="cSCC_results/")
    parser.add_argument("--gene_info_path", type=str, default="CellFM-main/csv/gene_info.csv")
    parser.add_argument("--MS_CKPT_PATH", type=str, default="CellFM-main/base_weight.ckpt")
    parser.add_argument("--experiment_dir", type=str, default="", help="If not empty, use this fixed directory, otherwise auto-increment")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint save directory")

    # =========================
    # Dataset and Assembly (assemble_dataset)
    # =========================
    parser.add_argument("--slide_out", type=str, default="NCBI759", help="Test exclusion: comma-separated list")
    parser.add_argument("--folder_list_filename", type=str, default="all_slide_lst.txt")
    parser.add_argument("--gene_list_filename", type=str, default="selected_gene_list.txt")
    parser.add_argument("--num_aug_ratio", type=int, default=7, help="Number of augmented samples per original spot; 0 means no augmentation")

    # Quantities to be written back by assemble_dataset (do not need to be provided in advance but can be manually overridden)
    parser.add_argument("--input_gene_size", type=int, default=-1, help="Automatically inferred from gene table, typically no need to manually input")
    parser.add_argument("--cond_size", type=int, default=-1, help="Automatically inferred from image feature dimension, typically no need to manually input")

    # =========================
    # Model Structure
    # =========================
    parser.add_argument("--model", type=str, default="Hinge", choices=["Hinge"])
    parser.add_argument("--enc_nlayers", type=int, default=40)

    # =========================
    # Mask Scheduler
    # =========================
    parser.add_argument("--num_timesteps", type=int, default=50, help="Mask steps T")
    parser.add_argument("--mask_fill", type=str, default="gauss", choices=["zero", "gauss"], help="Mask fill method")
    parser.add_argument("--idx_mode", type=str, default="random", choices=["random", "variance", "group"], help="Index mode for the mask")
    parser.add_argument("--gamma_mode", type=str, default="poly_auto", choices=["linear", "cosine", "sqrt", "exp", "poly", "poly_auto"],
                        help="Gamma scheduling mode for mask generation")
    parser.add_argument("--gamma_param", type=str, default="{'K_tail': 1}", help="Additional parameters for gamma_mode (e.g., {'alpha': 0.13} for poly, {'k_init': 1} for poly_auto)")

    # =========================
    # Training Hyperparameters (aligned with Trainer)
    # =========================
    parser.add_argument("--total_epochs", type=int, default=40)
    parser.add_argument("--global_batch_size", type=int, default=32, help="Total batch size, will be divided by world_size for each GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="<=0 means no gradient clipping")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="step", choices=["none", "cosine", "step"])
    parser.add_argument("--lr_schedule_by", type=str, default="epoch", choices=["step", "epoch"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--step_decay_milestones", type=int, nargs="*", default=[20, 35])
    parser.add_argument("--step_decay_gamma", type=float, default=0.2)

    # Logging/Validation/Checkpointing (per epoch)
    parser.add_argument("--log_every_epochs", type=int, default=1)
    parser.add_argument("--val_warmup_epochs", type=int, default=15)
    parser.add_argument("--val_every_epochs", type=int, default=1)
    parser.add_argument("--ckpt_every_epochs", type=int, default=0, help="0 means no periodic checkpointing")
    parser.add_argument("--early_stop_patience_epochs", type=int, default=8)
    parser.add_argument("--save_best_mode", type=str, default="min", choices=["min", "max"])

    # Deprecated arguments (will be mapped to the newer ones in normalize_args)
    parser.add_argument("--ckpt_every", type=int, default=0, help="(deprecated) checkpoint interval in epochs, replaced by --ckpt_every_epochs")

    # =========================
    # DataLoader / Splitting
    # =========================
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader worker threads")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--drop_last_train", action="store_true")
    parser.add_argument("--shuffle_train", action="store_true", help="Whether to shuffle during single-card training; controlled by sampler for DDP")
    parser.add_argument("--val_batch_size", type=int, default=0, help="0 means same as train batch size per card")

    # =========================
    # Randomness
    # =========================
    parser.add_argument("--global_seed", type=int, default=42)

    # =========================
    # Distributed/Device Configuration
    # =========================
    parser.add_argument("--num_workers_gpu", type=int, default=1, help="World size: number of parallel GPUs/Processes")
    parser.add_argument("--available_gpus", type=str, default="cuda:0", help="Comma-separated list, e.g., 'cuda:0,cuda:1'")
    parser.add_argument("--ddp_backend", type=str, default="", help='Default "nccl" for CUDA, else "gloo"')
    parser.add_argument("--ddp_init_method", type=str, default="env", choices=["env", "file"])
    parser.add_argument("--ddp_file_dir", type=str, default="./tmp", help="Directory for shared files when ddp_init_method='file'")

    # =========================
    # Recovery/Loading
    # =========================
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path")
    parser.add_argument("--resume_strict", action="store_true")

    # =========================
    # Curriculum Learning
    # =========================
    parser.add_argument("--curriculum_enabled", action=argparse.BooleanOptionalAction, default=True, help="Enable warmup curriculum")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_mask_ratio", type=float, default=0.20)

    return parser


def normalize_args(args):
    """
    Compatibility for older flags and safety corrections:
    - Maps the old --ckpt_every to --ckpt_every_epochs (only when the new value is not explicitly set)
    - Ensures 'betas' is a tuple
    - Corrects some edge cases
    """
    # Compatibility for old names
    if getattr(args, "ckpt_every_epochs", 0) == 0 and getattr(args, "ckpt_every", 0) > 0:
        args.ckpt_every_epochs = args.ckpt_every

    # Convert betas to tuple if it's a list
    if isinstance(args.betas, list):
        args.betas = tuple(args.betas)

    if isinstance(args.gamma_param, str):
        try:
            parsed_gamma_param = ast.literal_eval(args.gamma_param)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("--gamma_param must be a Python-style dictionary string, e.g. \"{'K_tail': 1}\"") from exc
        if not isinstance(parsed_gamma_param, dict):
            raise ValueError("--gamma_param must evaluate to a dictionary")
        args.gamma_param = parsed_gamma_param

    # Ensure world_size is valid
    args.num_workers_gpu = max(1, int(args.num_workers_gpu))

    # Ensure global_batch_size is valid
    args.global_batch_size = max(1, int(args.global_batch_size))

    # Default values for missing fields
    if not hasattr(args, "lr_schedule_by"):
        args.lr_schedule_by = "step"
    
    # # Dynamically set results_dir based on slide_out
    # if hasattr(args, "slide_out") and hasattr(args, "results_dir") :
    #     args.results_dir = os.path.join(args.results_dir, args.slide_out)
    #     os.makedirs(args.results_dir, exist_ok=True)
    
    return args


# -----------------------------
# Utils
# -----------------------------
def set_seed(base_seed: int, rank: int, world_size: int):
    seed = base_seed * world_size + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_device(rank: int, available_gpus: List[str]):
    # Supports "cuda:0" / "cpu"; the input available_gpus is like ["cuda:0", "cuda:1", ...]
    return available_gpus[rank]

def printfreeze(model=None):
    if model is None:
        return
    frozen, trainable = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    print("====== Freeze Check After Loading ======")
    print(f"Total params: {len(frozen) + len(trainable)}")
    print(f"Frozen: {len(frozen)} | Trainable: {len(trainable)}")

    print("-- Frozen sample --")
    for n in frozen:
        print("   ", n)

    print("-- Trainable sample --")
    for n in trainable:
        print("   ", n)
