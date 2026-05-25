import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Literal
from mindspore import load_checkpoint


def _as_torch_tensor(ms_obj, dtype=None, device="cpu"):
    """MindSpore Tensor/Param/ndarray -> torch.Tensor"""
    if hasattr(ms_obj, "asnumpy"):
        arr = ms_obj.asnumpy()
    elif isinstance(ms_obj, np.ndarray):
        arr = ms_obj
    else:
        arr = np.array(ms_obj)
    t = torch.from_numpy(arr)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.to(device)


def _try_copy(param: torch.nn.Parameter,
              src_t: torch.Tensor,
              allow_partial: bool) -> str:
    dst_shape = tuple(param.shape)
    src_shape = tuple(src_t.shape)

    if src_shape == dst_shape:
        param.copy_(src_t.to(param.dtype))
        return "loaded"

    if src_t.T.shape == param.shape:
        param.copy_(src_t.T.to(param.dtype))
        return "loaded(transposed)"

    if not allow_partial:
        return f"shape_mismatch({src_shape}->{dst_shape})"

    if param.dim() == 2 and src_t.dim() == 2:
        h = min(src_t.shape[0], dst_shape[0])
        w = min(src_t.shape[1], dst_shape[1])
        pad = torch.zeros(dst_shape, dtype=param.dtype, device=param.device)
        pad[:h, :w] = src_t[:h, :w].to(param.dtype)
        param.copy_(pad)
        tag = "padded" if (h, w) != dst_shape else "loaded"
        return f"loaded({tag})"
    if param.dim() == 1 and src_t.dim() == 1:
        h = min(src_t.shape[0], dst_shape[0])
        pad = torch.zeros(dst_shape, dtype=param.dtype, device=param.device)
        pad[:h] = src_t[:h].to(param.dtype)
        param.copy_(pad)
        tag = "padded" if h != dst_shape[0] else "loaded"
        return f"loaded({tag})"

    return f"shape_mismatch({src_shape}->{dst_shape})"


def build_full_name_map(model: torch.nn.Module,
                        enc_layers_with_ckpt: Set[int] = {0, 1},
                        skip_cls_token: bool = False,
                        skip_mask_token: bool = False) -> Dict[str, Optional[str]]:
    param_names = set(dict(model.named_parameters()).keys())

    def pick_existing(*cands):
        for name in cands:
            if name in param_names:
                return name
        return None 

    name_map: Dict[str, Optional[str]] = {}

    # ---- gene_emb ----
    ge_torch = pick_existing("gene_emb.weight", "gene_emb")  
    if ge_torch is not None:
        name_map["gene_emb"] = ge_torch

    name_map["cls_token"] = None if skip_cls_token else pick_existing("cls_token")
    name_map["zero_emb"] = None if skip_mask_token else pick_existing("zero_emb")

    for ms_k, pt_k in [
        ("value_enc.value_enc.a",              "value_enc.value_enc.a"),
        ("value_enc.value_enc.w1.weight",      "value_enc.value_enc.w1.weight"),
        ("value_enc.value_enc.w3.weight",      "value_enc.value_enc.w3.weight"),
        ("value_enc.value_enc.table.weight",   "value_enc.value_enc.table.weight"),
    ]:
        if pt_k in param_names:
            name_map[ms_k] = pt_k

    # ---- ValueDecoder（MS: value_dec → PT: final_layer）----
    pt_w1 = pick_existing("final_layer.w1.weight", "value_dec.w1.weight")
    pt_w2 = pick_existing("final_layer.w2.weight", "value_dec.w2.weight")
    if pt_w1 is not None:
        name_map["value_dec.w1.weight"] = pt_w1
    if pt_w2 is not None:
        name_map["value_dec.w2.weight"] = pt_w2

    # ---- CellwiseDecoder（MS: cellwise_dec → PT: cellwise_dec）----
    pt_map_w = pick_existing("cellwise_dec.map.weight")
    if pt_map_w is not None:
        name_map["cellwise_dec.map.weight"] = pt_map_w

    for i in sorted(enc_layers_with_ckpt):
        # attn
        for ms_s, pt_s in [
            (f"encoder.{i}.attn.q_proj.weight", f"retentionblocks.{i}.attn.q_proj.weight"),
            (f"encoder.{i}.attn.k_proj.weight", f"retentionblocks.{i}.attn.k_proj.weight"),
            (f"encoder.{i}.attn.v_proj.weight", f"retentionblocks.{i}.attn.v_proj.weight"),
            (f"encoder.{i}.attn.u_proj.weight", f"retentionblocks.{i}.attn.u_proj.weight"),
            (f"encoder.{i}.attn.o_proj.weight", f"retentionblocks.{i}.attn.o_proj.weight"),
        ]:
            if pt_s in param_names:
                name_map[ms_s] = pt_s
        # ffn
        for ms_s, pt_s in [
            (f"encoder.{i}.ffn.u_proj.weight",  f"retentionblocks.{i}.ffn.u_proj.weight"),
            (f"encoder.{i}.ffn.v_proj.weight",  f"retentionblocks.{i}.ffn.v_proj.weight"),
            (f"encoder.{i}.ffn.o_proj.weight",  f"retentionblocks.{i}.ffn.o_proj.weight"),
        ]:
            if pt_s in param_names:
                name_map[ms_s] = pt_s
        # post-norm（gamma/beta -> weight/bias）
        for ms_s, pt_s in [
            (f"encoder.{i}.post_norm1.gamma",   f"retentionblocks.{i}.post_norm1.weight"),
            (f"encoder.{i}.post_norm1.beta",    f"retentionblocks.{i}.post_norm1.bias"),
            (f"encoder.{i}.post_norm2.gamma",   f"retentionblocks.{i}.post_norm2.weight"),
            (f"encoder.{i}.post_norm2.beta",    f"retentionblocks.{i}.post_norm2.bias"),
        ]:
            if pt_s in param_names:
                name_map[ms_s] = pt_s

    skip_keys = [
        "ST_emb", "value_enc.mask_emb",
        "ST_enc.a", "ST_enc.w1.weight", "ST_enc.w3.weight", "ST_enc.table.weight",
        # "cellwise_dec.map.weight",
        "global_step", "learning_rate", "beta1_power", "beta2_power",
        "current_iterator_step", "last_overflow_iterator_step",
        "current_iterator_step",
        "last_overflow_iterator_step",
    ]
    for k in skip_keys:
        name_map[k] = None

    return name_map


def load_ms_to_torch(model: torch.nn.Module,
                     ms_ckpt_path: str,
                     device: str = "cpu",
                     allow_partial: bool = False,
                     freeze_preset: str = "conservative",
                     pretrained_layers: Set[int] = set(range(40)),
                     also_freeze_value_encoder: bool = True,
                     also_freeze_embeddings: bool = True,
                     also_freeze_value_decoder: bool = True) -> Dict[str, List]:
    model.to(device)
    ms_params = load_checkpoint(ms_ckpt_path)

    name_map = build_full_name_map(model, enc_layers_with_ckpt=pretrained_layers, skip_cls_token=False, skip_mask_token=False)
    model_params = dict(model.named_parameters())

    loaded, transposed, padded, sliced = [], [], [], []
    skipped_optimizer, missing_in_ms, missing_in_torch = [], [], []
    shape_mismatch = []

    with torch.no_grad():
        for ms_name, pt_name in name_map.items():
            if ms_name.startswith("moment1.") or ms_name.startswith("moment2."):
                skipped_optimizer.append(ms_name); continue

            if ms_name not in ms_params:
                missing_in_ms.append(ms_name); continue
            if pt_name is None:
                continue
            if pt_name not in model_params:
                missing_in_torch.append((ms_name, pt_name)); continue

            src = ms_params[ms_name]
            t = _as_torch_tensor(src, dtype=model_params[pt_name].dtype, device=device)
            status = _try_copy(model_params[pt_name], t, allow_partial)

            if status == "loaded":
                loaded.append((ms_name, pt_name))
            elif status == "loaded(transposed)":
                transposed.append((ms_name, pt_name))
            elif status == "loaded(padded)" or status == "loaded(sliced)":
                (padded if "padded" in status else sliced).append((ms_name, pt_name))
            else:
                shape_mismatch.append((ms_name, pt_name, status))

    freeze_with_preset(model,
                       pretrained_layers=pretrained_layers,
                       preset=freeze_preset,
                       also_freeze_value_encoder=also_freeze_value_encoder,
                       also_freeze_embeddings=also_freeze_embeddings,
                       also_freeze_value_decoder=also_freeze_value_decoder)

    print("====== MS -> Torch Load Report ======")
    print(f"Loaded: {len(loaded)} | Transposed: {len(transposed)} | Padded: {len(padded)} | Sliced: {len(sliced)}")
    print(f"Shape mismatch: {len(shape_mismatch)} | Missing in MS: {len(missing_in_ms)} | "
          f"Missing in Torch: {len(missing_in_torch)} | Skipped optimizer/moment: {len(skipped_optimizer)}")

    if shape_mismatch:
        print("-- Shape mismatch sample (up to 8) --")
        for item in shape_mismatch[:8]:
            print(item)
    if missing_in_torch:
        print("-- In mapping but torch param not found (up to 8) --")
        for item in missing_in_torch[:8]:
            print(item)

    return {
        "loaded": loaded,
        "transposed": transposed,
        "padded": padded,
        "sliced": sliced,
        "shape_mismatch": shape_mismatch,
        "missing_in_ms": missing_in_ms,
        "missing_in_torch": missing_in_torch,
        "skipped_optimizer": skipped_optimizer,
    }


# ==========
def freeze_with_preset(model: torch.nn.Module,
                       pretrained_layers: Set[int] = {0, 1},
                       preset: str = "conservative",
                       also_freeze_value_encoder: bool = False,
                       also_freeze_embeddings: bool = False,
                       also_freeze_value_decoder: bool = False) -> None:

    for p in model.parameters():
        p.requires_grad = True

    BASE_LINEAR_SUFFIXES = (
        ".attn.q_proj.weight", ".attn.k_proj.weight", ".attn.v_proj.weight",
        ".attn.u_proj.weight", ".attn.o_proj.weight",
        ".ffn.u_proj.weight",  ".ffn.v_proj.weight",  ".ffn.o_proj.weight",
    )

    for i in pretrained_layers:
        prefix = f"retentionblocks.{i}."

        for name, p in model.named_parameters():
            if not name.startswith(prefix):
                continue

            if name.endswith(BASE_LINEAR_SUFFIXES):
                p.requires_grad = False

            if name in {
                f"{prefix}post_norm1.weight", f"{prefix}post_norm1.bias",
                f"{prefix}post_norm2.weight", f"{prefix}post_norm2.bias",
            }:
                p.requires_grad = False

            if (f"{prefix}adaln_attn.proj" in name or
                f"{prefix}adaln_ffn.proj"  in name or
                f"{prefix}pre_norm1"       in name or
                f"{prefix}pre_norm2"       in name):
                p.requires_grad = True

    if preset in ("moderate", "aggressive") or also_freeze_value_encoder:
        for name, p in model.named_parameters():
            if name.startswith("value_enc."):
                p.requires_grad = False

    if preset in ("moderate", "aggressive") or also_freeze_value_decoder:
        for name, p in model.named_parameters():
            if name.startswith("final_layer.") or name.startswith("value_dec."):
                if name.endswith(".w1.weight") or name.endswith(".w1.bias") \
                   or name.endswith(".w2.weight") or name.endswith(".w2.bias"):
                    p.requires_grad = False
                elif name.endswith("pre_norm._lambda_bias"):
                    p.requires_grad = True
                elif ".proj." in name:
                    p.requires_grad = True

        for name, p in model.named_parameters():
            if name.startswith("cellwise_dec."):
                if name.endswith("map.weight"):
                    p.requires_grad = False
                elif name.endswith("pre_norm._lambda_bias"):
                    p.requires_grad = True
                elif ".proj." in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    if preset == "aggressive" or also_freeze_embeddings:
        for name, p in model.named_parameters():
            if name.startswith("gene_emb") or name.startswith("cls_token") or name.startswith("zero_emb"):
                p.requires_grad = False

    for name, p in model.named_parameters():
        if ".lora_" in name:
            p.requires_grad = True