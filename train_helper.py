import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from typing import Optional

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def _safe_rank(default=0) -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return default

def create_logger(logging_dir: Optional[str],
                  name: str = "train",
                  level: int = logging.INFO,
                  to_stdout: bool = True):
    rank = _safe_rank(default=0)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if getattr(logger, "_configured", False):
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if rank == 0:
        if logging_dir:
            os.makedirs(logging_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(logging_dir, "log.txt"), encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        if to_stdout:
            sh = logging.StreamHandler()
            sh.setLevel(level)
            sh.setFormatter(fmt)
            logger.addHandler(sh)
    else:
        logger.addHandler(logging.NullHandler())

    logger._configured = True
    return logger

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
