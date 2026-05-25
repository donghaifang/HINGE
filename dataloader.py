import os

import pandas as pd
import anndata as anndata
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset, Subset

def is_dist():
    return dist.is_available() and dist.is_initialized()


class CustomDataset(Dataset):
    def __init__(self, x, y, group_ids=None):
        self.data = x
        self.label = y
        if group_ids is None:
            group_ids = np.arange(len(x))
        self.group_ids = np.asarray(group_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def groups(self):
        return self.group_ids


def _log(args, msg, rank):
    if getattr(args, "logger", None) is not None:
        if rank == 0:
            args.logger.info(msg)
    else:
        if rank == 0:
            print(msg)


def assemble_dataset(input_args, rank: int = 0):
    args = input_args  
    rs = np.random.RandomState(getattr(args, "global_seed", 42))  

    folder_list_path = os.path.join(args.data_path, "processed_data", args.folder_list_filename)
    slidename_lst = list(np.genfromtxt(folder_list_path, dtype=str))
    holdouts = [s.strip() for s in args.slide_out.split(",") if s.strip()]

    for slide_out in holdouts:
        if slide_out in slidename_lst:
            slidename_lst.remove(slide_out)
            _log(args, f"{slide_out} is held out for testing.", rank)
        else:
            _log(args, f"[WARN] slide_out '{slide_out}' not found in folder list; skip removing.", rank)
    _log(args, f"Remaining {len(slidename_lst)} slides: {slidename_lst}", rank)

    gene_list_path = os.path.join(args.data_path, "processed_data", "used_" + args.gene_list_filename)
    selected_genes = list(np.genfromtxt(gene_list_path, dtype=str))
    args.input_gene_size = len(selected_genes)
    args.selected_genes = selected_genes
    _log(args, f"Selected genes filename: {args.gene_list_filename} | len: {len(selected_genes)}", rank)

    first = True
    all_img_ebd_ori = None  # torch.Tensor [N0, D]
    all_count_mtx_ori = None  # numpy array [N0, G]

    _log(args, "Loading original data...", rank)
    for sample_name in slidename_lst:
        h5_path = os.path.join(args.data_path, "st", f"{sample_name}.h5ad")
        ad = anndata.read_h5ad(h5_path)
        sub = ad[:, selected_genes]
        counts_np = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        this_count = counts_np  # (spots, G)

        # image embeddings: uni + conch 
        uni_path = os.path.join(args.data_path, "processed_data", "1spot_uni_ebd", f"{sample_name}_uni.pt")
        conch_path = os.path.join(args.data_path, "processed_data", "1spot_conch_ebd", f"{sample_name}_conch.pt")
        img_ebd_uni = torch.load(uni_path, map_location="cpu")    # (spots, d1)
        img_ebd_conch = torch.load(conch_path, map_location="cpu")# (spots, d2)
        this_img = torch.cat([img_ebd_uni, img_ebd_conch], dim=1) # (spots, D)

        if first:
            all_count_mtx_ori = this_count
            all_img_ebd_ori = this_img
            first = False
        else:
            all_count_mtx_ori = np.concatenate([all_count_mtx_ori, this_count], axis=0)
            all_img_ebd_ori = torch.cat([all_img_ebd_ori, this_img], dim=0)

        _log(
            args,
            f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape} | img ebd shape: {tuple(all_img_ebd_ori.shape)}",
            rank
        )

    N0 = int(all_count_mtx_ori.shape[0])
    D = int(all_img_ebd_ori.shape[1])
    args.cond_size = D

    num_aug_ratio = int(getattr(args, "num_aug_ratio", 0))
    if num_aug_ratio > 0:
        first = True
        all_img_ebd_aug = None  # torch.Tensor [N0, K, D]
        _log(args, "Augmentation data loading...", rank)
        for sample_name in slidename_lst:
            uni_aug_path = os.path.join(args.data_path, "processed_data", "1spot_uni_ebd_aug", f"{sample_name}_uni_aug.pt")
            conch_aug_path = os.path.join(args.data_path, "processed_data", "1spot_conch_ebd_aug", f"{sample_name}_conch_aug.pt")
            img_ebd_uni_aug = torch.load(uni_aug_path, map_location="cpu")      # (spots, K, d1)
            img_ebd_conch_aug = torch.load(conch_aug_path, map_location="cpu")  # (spots, K, d2)
            cat_aug = torch.cat([img_ebd_uni_aug, img_ebd_conch_aug], dim=-1)   # (spots, K, D)

            if first:
                all_img_ebd_aug = cat_aug
                first = False
            else:
                all_img_ebd_aug = torch.cat([all_img_ebd_aug, cat_aug], dim=0)

            _log(
                args,
                f"With augmentation {sample_name} loaded, img_ebd_mtx shape: {tuple(cat_aug.shape)}, all_img_ebd_aug shape: {tuple(all_img_ebd_aug.shape)}",
                rank
            )

        N0_check, K, D_check = all_img_ebd_aug.shape
        assert N0_check == N0 and D_check == D, "Aug shape not aligned with originals."
        if num_aug_ratio > K:
            raise ValueError(f"num_aug_ratio({num_aug_ratio}) > K({K}) for augmentation choices.")

        all_count_mtx_aug = np.repeat(all_count_mtx_ori, num_aug_ratio, axis=0)  # (N0*num_aug_ratio, G)

        select_idx = np.stack(
            [rs.choice(K, size=num_aug_ratio, replace=False) for _ in range(N0)],
            axis=0
        )  # (N0, num_aug_ratio)

        sel = torch.as_tensor(select_idx, device=all_img_ebd_aug.device, dtype=torch.long)
        selected = torch.take_along_dim(
            all_img_ebd_aug,                      # [N0, K, D]
            sel[..., None].expand(-1, -1, D),     # [N0, R, D]
            dim=1
        )  # [N0, R, D]
        selected_img_ebd_aug = selected.reshape(-1, D)  # [N0*R, D]

        all_img_ebd = torch.cat([all_img_ebd_ori, selected_img_ebd_aug], dim=0)           # (N0 + N0*R, D)
        all_count_mtx = np.concatenate([all_count_mtx_ori, all_count_mtx_aug], axis=0)    # (N0 + N0*R, G)

        group_ids = np.concatenate([
            np.arange(N0, dtype=np.int64),
            np.repeat(np.arange(N0, dtype=np.int64), repeats=num_aug_ratio)
        ], axis=0)  # (N0 + N0*R,)
        _log(args, f"{num_aug_ratio}:1 augmentation. final count_mtx: {all_count_mtx.shape} | final img_ebd: {tuple(all_img_ebd.shape)}", rank)
    else:
        all_img_ebd = all_img_ebd_ori
        all_count_mtx = all_count_mtx_ori
        group_ids = np.arange(N0, dtype=np.int64)
        _log(args, f"No augmentation. final count_mtx: {all_count_mtx.shape} | final img_ebd: {tuple(all_img_ebd.shape)}", rank)

    df = pd.DataFrame(all_count_mtx, columns=selected_genes)
    nan_idx = df.index[df.isnull().all(axis=1)]
    zero_idx = df.index[(df.fillna(0).sum(axis=1) == 0)]
    rm_idx = sorted(set(nan_idx.tolist()) | set(zero_idx.tolist()))
    if len(rm_idx) > 0:
        keep_mask = np.ones(df.shape[0], dtype=bool)
        keep_mask[rm_idx] = False
        all_count_mtx = df.fillna(0.0).to_numpy()[keep_mask, :]  # (N_keep, G)
        all_img_ebd = all_img_ebd[keep_mask, :]                  # (N_keep, D)
        group_ids = group_ids[keep_mask]                          # (N_keep,)
        _log(args, f"After exclude rows with all nan/zeros: {all_count_mtx.shape}, {tuple(all_img_ebd.shape)}", rank)
    else:
        all_count_mtx = df.fillna(0.0).to_numpy()

    all_count_mtx_selected_genes = np.log2(all_count_mtx + 1.0).astype(np.float32)
    _log(args, f"Selected genes count matrix shape: {all_count_mtx_selected_genes.shape}", rank)

    all_img_ebd = all_img_ebd.detach().float()
    all_img_ebd.requires_grad_(False)

    X = torch.from_numpy(all_count_mtx_selected_genes)  # (N, G) float32
    Y = all_img_ebd                                     # (N, D) float32

    dataset = CustomDataset(X, Y, group_ids=group_ids)
    return dataset, args


def _groupwise_split_indices(groups: np.ndarray, val_ratio: float, seed: int):
    groups = np.asarray(groups).astype(np.int64)
    unique_groups = np.unique(groups)
    g_rng = np.random.RandomState(seed)

    g_rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_val_groups = max(1, int(round(n_groups * val_ratio)))
    n_val_groups = min(n_val_groups, n_groups - 1) if n_groups > 1 else 1

    val_group_set = set(unique_groups[:n_val_groups])
    all_idx = np.arange(len(groups))
    val_mask = np.isin(groups, list(val_group_set))
    val_idx = all_idx[val_mask]
    train_idx = all_idx[~val_mask]
    return train_idx, val_idx

def prepare_dataloaders_groupwise(args, dataset: CustomDataset, batch_size: int, val_ratio: float = 0.1):
    train_idx, val_idx = _groupwise_split_indices(dataset.groups(), val_ratio, seed=args.global_seed)

    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())

    if is_dist():
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, seed=args.global_seed, drop_last=args.drop_last_train
        )
        val_sampler = DistributedSampler(
            val_dataset, shuffle=False, seed=args.global_seed, drop_last=False
        )
        shuffle_train = False
        shuffle_val = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = getattr(args, "shuffle_train", True)
        shuffle_val = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=getattr(args, "pin_memory", True),
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=getattr(args, "num_workers", 4),
        drop_last=getattr(args, "drop_last_train", True),
        persistent_workers=(getattr(args, "persistent_workers", False) if getattr(args, "num_workers", 0) > 0 else False),
        prefetch_factor=(getattr(args, "prefetch_factor", None) if getattr(args, "num_workers", 0) > 0 else None),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8*min(batch_size, getattr(args, "val_batch_size", batch_size) if getattr(args, "val_batch_size", 0) > 0 else batch_size),
        pin_memory=getattr(args, "pin_memory", True),
        shuffle=shuffle_val,
        sampler=val_sampler,
        num_workers=max(0, getattr(args, "num_workers", 4) // 2),
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=None,
    )

    if not is_dist() or dist.get_rank() == 0:
        n_train_groups = len(np.unique(dataset.groups()[train_idx]))
        n_val_groups = len(np.unique(dataset.groups()[val_idx]))
        args.logger.info(f"[GroupSplit] train: {len(train_idx)} samples / {n_train_groups} groups | val: {len(val_idx)} samples / {n_val_groups} groups")

    return train_loader, val_loader
