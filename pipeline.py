from tqdm import tqdm
from copy import deepcopy
import argparse
import math
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from Hinge.mask_scheduler import MaskScheduler
from Hinge.train_helper import requires_grad, update_ema


def _is_dist():
    return dist.is_available() and dist.is_initialized()

# =========================
# Trainer
# =========================
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data,
        val_data,
        rank: int,
        gpu_id: int,
        model_args: argparse.Namespace,
    ) -> None:
    
        self.T = getattr(model_args, "num_timesteps", 50)
        self.rank = rank
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.val_data = val_data
        self.args = model_args
        self.mask_fill = getattr(self.args, "mask_fill", "gauss")
        self.gamma_mode = getattr(self.args, "gamma_mode", "poly_auto")  
        self.gamma_param = getattr(self.args, "gamma_param", {"K_tail": 1})  


        self.log_every_epochs = getattr(self.args, "log_every_epochs", 1)
        self.val_warmup_epochs = getattr(self.args, "val_warmup_epochs", 3)
        self.val_every_epochs = getattr(self.args, "val_every_epochs", 2)
        self.ckpt_every_epochs = getattr(self.args, "ckpt_every_epochs", 0)
        self.early_stop_patience_epochs = getattr(self.args, "early_stop_patience_epochs", 10)
        self.save_best_mode = getattr(self.args, "save_best_mode", "min")


        self.lr = getattr(self.args, "lr", 1e-4)
        self.weight_decay = getattr(self.args, "weight_decay", 0.0)
        self.betas = tuple(getattr(self.args, "betas", (0.9, 0.999)))
        self.eps = getattr(self.args, "eps", 1e-8)
        self.grad_accum_steps = max(1, getattr(self.args, "grad_accum_steps", 1))
        device = torch.device(f"cuda:{gpu_id}" if str(gpu_id) != "-1" and torch.cuda.is_available() else "cpu")
        self.device = device
        self.grad_clip_norm = float(getattr(self.args, "grad_clip_norm", 0.0))
        self.amp = bool(getattr(self.args, "amp", False)) and device.type == "cuda"
        self.model = model.to(device)

        if _is_dist():
            self.model = DDP(
                self.model,
                device_ids=[gpu_id] if device.type == "cuda" else None,
                output_device=gpu_id if device.type == "cuda" else None,
                find_unused_parameters=False,
            )


        self.scheduler = MaskScheduler(
            G=self.args.input_gene_size, 
            T=self.T, 
            device=device, 
            mask_fill=self.mask_fill,
            gamma_mode=self.gamma_mode,   
            gamma_param=self.gamma_param  
        )

        model_ref = self.model.module if _is_dist() else self.model


        trainable_params = [p for p in model_ref.parameters() if p.requires_grad]
        opt_params = trainable_params

        self.optimizer = torch.optim.AdamW(
            opt_params,
            lr=self.lr, weight_decay=self.weight_decay, betas=self.betas, eps=self.eps
        )


        sch_name = getattr(self.args, "scheduler", "none")
        if sch_name == "cosine":
            warmup = int(getattr(self.args, "warmup_steps", 0))
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, getattr(self.args, "total_epochs", 100))
            )
            self.warmup_steps = warmup
            self.global_step = 0
        elif sch_name == "step":
            milestones = list(getattr(self.args, "step_decay_milestones", [25, 35, 40]))
            gamma = float(getattr(self.args, "step_decay_gamma", 0.2))
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
            self.warmup_steps = 0
            self.global_step = 0
            self.step_milestones = sorted(milestones)
            self._final_unfrozen = False
        else:
            self.lr_scheduler = None
            self.warmup_steps = 0
            self.global_step = 0


        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)


        self.global_epoch = 0
        self.best_dev = float("inf") if self.save_best_mode == "min" else -float("inf")
        self.no_improve_epochs = 0


        self.curriculum_enabled  = bool(getattr(self.args, "curriculum_enabled", False))
        self.warmup_epochs       = int(getattr(self.args, "warmup_epochs", 5))     
        self.warmup_mask_ratio   = float(getattr(self.args, "warmup_mask_ratio", 0.20))  


        self.ckpt_dir = getattr(self.args, "checkpoint_dir", "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._log_config()


    def elboloss(self, x0, t, y):
        B = x0.size(0)
        idx = self.scheduler.batch_generate_mask_chain(B)  
        x_t, m_prev, d_t = self.scheduler.batch_sample_mask_and_forward(x0, idx, t)
        pred_x0 = (self.model if not _is_dist() else self.model.module)(x_t.unsqueeze(1), t, m_prev, m_prev - d_t, y)
        loss = ((pred_x0 - x0) * d_t).pow(2)
        return loss.sum() / d_t.sum()
        
        
    def _run_batch(self, x, t, y):
        with torch.amp.autocast('cuda', enabled=self.amp):
            loss = self.elboloss(x, t, y) / self.grad_accum_steps

        self.scaler.scale(loss).backward()

        step_now = ((self.global_step + 1) % self.grad_accum_steps == 0)
        if step_now:
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                if _is_dist():
                    torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.grad_clip_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)


            if self.lr_scheduler is not None and getattr(self.args, "lr_schedule_by", "step") == "step":
                if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
                    warm_ratio = (self.global_step + 1) / float(self.warmup_steps)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.lr * warm_ratio
                else:
                    self.lr_scheduler.step()

        return loss.detach()


    def _run_epoch(self, epoch):
        if isinstance(self.train_data.sampler, DistributedSampler):
            self.train_data.sampler.set_epoch(epoch)

        self.model.train()

        local_loss_sum = 0.0
        local_count = 0
        progress_bar = tqdm(enumerate(self.train_data), 
                            total=len(self.train_data),
                            disable=(self.rank != 0), 
                            desc=f"Epoch {epoch} [GPU{self.gpu_id}]")

        for step, (x, y) in progress_bar:
            x = x.to(self.device)
            y = y.to(self.device)

            if self.curriculum_enabled and (self.global_epoch <= self.warmup_epochs):
                t = self._sample_t_warmup(B=x.size(0), device=x.device)
            else:
                t = torch.randint(1, self.T + 1, (x.size(0),), device=x.device)

            loss = self._run_batch(x, t, y)
            bs = x.size(0)
            local_loss_sum += (loss.item() * bs * self.grad_accum_steps)
            local_count += bs
            self.global_step += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        stats = torch.tensor([local_loss_sum, float(local_count)], device=self.device)
        if _is_dist():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_loss_sum, global_count = stats.tolist()
        epoch_avg_loss = global_loss_sum / max(1.0, global_count)

        if self.lr_scheduler is not None and getattr(self.args, "lr_schedule_by", "step") != "step":
            self.lr_scheduler.step()

        return epoch_avg_loss


    def train(self, max_epochs: int):
        self.model.train()

        world_size = dist.get_world_size() if _is_dist() else 1
        self.max_epochs = int(max_epochs)

        for epoch in range(max_epochs):
            self.global_epoch = epoch + 1
            epoch_avg_loss = self._run_epoch(epoch=self.global_epoch)
            if self.rank == 0 and (self.global_epoch % self.log_every_epochs == 0) and hasattr(self.args, "logger"):
                self.args.logger.info(
                    f"Epoch={self.global_epoch:05d} | Training Loss (avg over epoch): {epoch_avg_loss:.5f}"
                )

            if self.ckpt_every_epochs and self.ckpt_every_epochs > 0:
                if self.rank == 0 and (self.global_epoch % self.ckpt_every_epochs == 0):
                    self.save_checkpoint(suffix=f"epoch_{self.global_epoch:05d}")

            do_val = (self.global_epoch >= self.val_warmup_epochs) and \
                     (self.global_epoch % self.val_every_epochs == 0)
            if do_val:
                dev_mse = None
                if self.rank == 0:
                    dev_mse = self._run_val_epoch()

                    improved = (dev_mse < self.best_dev) if self.save_best_mode == "min" else (dev_mse > self.best_dev)
                    if improved:
                        if hasattr(self.args, "logger"):
                            self.args.logger.info(
                                f"Epoch={self.global_epoch:05d} | Dev MSE: {dev_mse:.5f} better than Best({self.best_dev:.5f}) --> SAVE"
                            )
                        self.best_dev = dev_mse
                        self.save_checkpoint(suffix="best_weight")

                if _is_dist():
                    dist.barrier()

                if self._early_stop_gate_open() and (self.no_improve_epochs > self.early_stop_patience_epochs):
                    if self.rank == 0 and hasattr(self.args, "logger"):
                        self.args.logger.info(
                            f"Early stopping at epoch {self.global_epoch} "
                            f"(no improvement for {self.no_improve_epochs} epochs after gate opened)."
                        )
                    break

    def _early_stop_gate_open(self) -> bool:
        return self.global_epoch >= self.val_warmup_epochs

    def _ensure_splits_cache(self):
        if getattr(self, "_splits_all", None) is not None:
            return
        dev = self.scheduler.device
        with torch.no_grad():
            t_all = torch.arange(0, self.T + 1, device=dev, dtype=torch.long)
            splits = self.scheduler._splits(t_all).to("cpu")
            self._splits_all = splits
            self._G = int(self.scheduler.G)

    def _t_cap_for_ratio(self, r: float) -> int:
        r = float(max(0.0, min(1.0, r)))
        if r <= 0:
            return 1
        self._ensure_splits_cache()
        splits = self._splits_all
        G = self._G
        target = int(r * G + 1e-9)
        idx = torch.searchsorted(splits, target, right=True).item() - 1
        return max(1, min(self.T, int(idx)))

    def _sample_t_warmup(self, B: int, device) -> torch.Tensor:
        t_cap = getattr(self, "_warmup_t_cap", None)
        if t_cap is None:
            t_cap = self._t_cap_for_ratio(self.warmup_mask_ratio)
            self._warmup_t_cap = t_cap
            if hasattr(self.args, "logger"):
                self.args.logger.info(f"[Warmup] mask_ratio≤{self.warmup_mask_ratio:.2f} -> t_cap={t_cap}/{self.T}")
        return torch.randint(1, max(2, t_cap + 1), (B,), device=device)  
        
    @torch.no_grad()
    def _run_val_epoch(self):
        self.model.eval()
        running_mse, count = 0.0, 0
        progress_bar = tqdm(enumerate(self.val_data), total=len(self.val_data),
                            disable=(self.rank != 0), desc=f"Validation [GPU{self.gpu_id}]")
        for _, (gt_x, y) in progress_bar:
            gt_x = gt_x.to(self.device); y = y.to(self.device)
            B = gt_x.size(0)
            t = torch.randint(1, self.T + 1, (B,), device=gt_x.device)
            with torch.amp.autocast('cuda', enabled=self.amp):
                loss = self.elboloss(gt_x, t, y) / self.grad_accum_steps
            running_mse += loss.item() * B
            count += B
            progress_bar.set_postfix(loss=f"{loss:.4f}")
        avg = running_mse / max(1, count)
        return avg


    def save_checkpoint(self, suffix="best_weight"):
        if self.rank != 0:
            return
        checkpoint = {
            "model": (self.model.module if _is_dist() else self.model).state_dict(),
            # "ema": self.ema.state_dict(),
            "opt": self.optimizer.state_dict(),
            "epoch": self.global_epoch,
        }
        checkpoint_path = os.path.join(self.ckpt_dir, f"{suffix}.pt")
        torch.save(checkpoint, checkpoint_path)
        if hasattr(self.args, "logger"):
            self.args.logger.info(f"Saved checkpoint to {checkpoint_path}")
        else:
            print(f"Saved checkpoint to {checkpoint_path}")

    def _log_config(self):
        if hasattr(self.args, "logger"):
            self.args.logger.info("----- Training Configuration -----")
            self.args.logger.info(f"num_timesteps (T): {self.T}")
            self.args.logger.info(f"mask_fill: {self.mask_fill}")
            self.args.logger.info(f"gamma_mode: {self.gamma_mode}")
            self.args.logger.info(f"gamma_param: {self.gamma_param}")
            self.args.logger.info(f"lr: {self.lr}")
            self.args.logger.info(f"weight_decay: {self.weight_decay}")
            self.args.logger.info(f"betas: {self.betas}")
            self.args.logger.info(f"eps: {self.eps}")
            self.args.logger.info(f"grad_accum_steps: {self.grad_accum_steps}")
            self.args.logger.info(f"grad_clip_norm: {self.grad_clip_norm}")
            self.args.logger.info(f"amp: {self.amp}")
            self.args.logger.info(f"log_every_epochs: {self.log_every_epochs}")
            self.args.logger.info(f"val_warmup_epochs: {self.val_warmup_epochs}")
            self.args.logger.info(f"val_every_epochs: {self.val_every_epochs}")
            self.args.logger.info(f"ckpt_every_epochs: {self.ckpt_every_epochs}")
            self.args.logger.info(f"early_stop_patience_epochs: {self.early_stop_patience_epochs}")
            self.args.logger.info(f"ckpt_dir: {self.ckpt_dir}")
            self.args.logger.info(f"curriculum_enabled: {self.curriculum_enabled}")
            self.args.logger.info(f"warmup_epochs: {self.warmup_epochs}")
            self.args.logger.info(f"warmup_mask_ratio: {self.warmup_mask_ratio}")
            self.args.logger.info("---------------------------------")
            self.args.logger.info(
                f"Initializing Trainer... "
                f"Params: {sum(p.numel() for p in (self.model.module.parameters() if _is_dist() else self.model.parameters())):,}"
            )
            
