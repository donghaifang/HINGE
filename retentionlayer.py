import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SRMSNorm(nn.Module):
    def __init__(self, emb_dims: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(emb_dims)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        n = torch.linalg.vector_norm(x.to(torch.float32), ord=2, dim=-1, keepdim=True)
        n = torch.clamp(self.scale * n, min=self.eps)
        return (x / n).to(dtype)

class Kernel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)

class LoraBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int) -> None:
        super().__init__()
        assert r >= 0, "LoRA rank r must be non-negative."
        self.r = r
        if r == 0:
            self.A = None
            self.B = None
            self.scaling = 0.0
            return

        self.scaling = 1.0 / float(r)

        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r == 0:
            return torch.zeros(x.shape[0], self.B.out_features if self.B is not None else x.shape[-1],
                               device=x.device, dtype=x.dtype)
        return self.B(self.A(x)) * self.scaling

    def update_weight(self) -> torch.Tensor:
        if self.r == 0:
            return None
        return (self.B.weight @ self.A.weight) * self.scaling


class MHRetention(nn.Module):
    def __init__(self, emb_dims: int, num_heads: int, lth: Optional[float] = None, lora: int = 0) -> None:
        super().__init__()
        assert emb_dims % num_heads == 0, "emb_dims must be divisible by num_heads."
        self.emb_dims = emb_dims
        self.num_heads = num_heads
        self.head_dim = emb_dims // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.lora = lora

        beta = 1.0 if lth is None else (lth * 8) ** -0.25

        self.q_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.k_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)

        nn.init.xavier_normal_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)

        self.kernelQ = Kernel()
        self.kernelK = Kernel()
        self.kernelV = nn.Identity()
        self.kernelU = SiLU()
        self.inner_norm = SRMSNorm(self.head_dim)

        if self.lora > 0:
            self.lora_q = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_k = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)

    # ========== helpers ==========
    @staticmethod
    def _expand_global_y(y: torch.Tensor, L1: int) -> torch.Tensor:
        return y.unsqueeze(1).expand(y.size(0), L1, y.size(-1))

    def _reshape_heads(self, t: torch.Tensor, B: int) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Dh)
        return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _lengths_to_mask(
        lengths: Union[int, torch.Tensor],
        L: int,
        device: torch.device,
        dtype: torch.dtype,
        B: int,
    ) -> torch.Tensor:
        if isinstance(lengths, int):
            base = torch.arange(L, device=device).view(1, L)      # (1,L)
            valid = (base < lengths).to(dtype).expand(B, L)       # (B,L)
        elif isinstance(lengths, torch.Tensor):
            assert lengths.numel() == B, "valid_len must be (B,)"
            base = torch.arange(L, device=device).view(1, L)      # (1,L)
            valid = (base < lengths.view(B, 1)).to(dtype)         # (B,L)
        else:
            raise TypeError("lengths must be int or (B,) tensor")
        return valid.view(B, 1, L, 1)                             # (B,1,L,1)

    @staticmethod
    def _ensure_mask_shape(mask: torch.Tensor, B: int, L: int, device, dtype) -> torch.Tensor:
        m = mask.to(device=device, dtype=dtype)
        if m.dim() == 2:              # (B,L)
            m = m.view(B, 1, L, 1)
        elif m.dim() == 3:            # (B,L,1)
            m = m.view(B, 1, L, 1)
        elif m.dim() == 4:
            m = m.view(B, -1, L, 1)[:, :1]   
        else:
            raise ValueError("mask must be broadcastable to (B,1,L,1)")
        return torch.clamp(m, 0, 1)

    # ========== forward ==========
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        v_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B, L1, D = x.shape
        assert D == self.emb_dims
        device, dtype = x.device, x.dtype
        H, Dh = self.num_heads, self.head_dim

        if y is None:
            y = x
        elif y.dim() == 2:
            y = self._expand_global_y(y, L1)
        elif y.dim() != 3:
            raise ValueError("y must be None, (B,D) or (B,L2,D)")
        assert y.shape[0] == B and y.shape[-1] == D
        L2 = y.shape[1]
        
        self_mode = (L1 == L2) and ((y is x) or (y.data_ptr() == x.data_ptr()))
        if self_mode:
            # (4D, D) @ (B,L1,D)^T
            W_all = torch.cat(
                [self.q_proj.weight, self.u_proj.weight,
                 self.k_proj.weight, self.v_proj.weight],
                dim=0
            )  # (4D, D)
            x_all = F.linear(x, W_all)  # (B, L1, 4D)
            q, u, k, v = x_all.split(D, dim=-1)  # (B, L1, D)
        else:
            # GEMM
            W_x = torch.cat([self.q_proj.weight, self.u_proj.weight], dim=0)  # (2D, D)
            xu = F.linear(x, W_x)  # (B,L1,2D)
            q, u = xu.split(D, dim=-1)

            W_y = torch.cat([self.k_proj.weight, self.v_proj.weight], dim=0)  # (2D, D)
            ykv = F.linear(y, W_y)  # (B,L2,2D)
            k, v = ykv.split(D, dim=-1)

        if self.lora > 0:
            q = q + self.lora_q(x)
            u = u + self.lora_u(x)
            if self_mode:
                k = k + self.lora_k(x)
                v = v + self.lora_v(x)
            else:
                k = k + self.lora_k(y)
                v = v + self.lora_v(y)

        # (B,H,L, Dh)
        Q = q.view(B, L1, H, Dh).transpose(1, 2).contiguous()  # (B,H,L1,Dh)
        U = u.view(B, L1, H, Dh).transpose(1, 2).contiguous()
        K = k.view(B, L2, H, Dh).transpose(1, 2).contiguous()  # (B,H,L2,Dh)
        V = v.view(B, L2, H, Dh).transpose(1, 2).contiguous()

        # kernels + scaling
        Q = self.kernelQ(Q) / self.scale
        K = self.kernelK(K) / self.scale
        # V = Id
        U = self.kernelU(U)

        if seq_mask is None and ("x_valid_len" in kwargs):
            seq_mask = self._lengths_to_mask(kwargs["x_valid_len"], L1, device, dtype, B)
        elif seq_mask is not None:
            seq_mask = self._ensure_mask_shape(seq_mask, B, L1, device, dtype)

        if attn_mask is None and ("y_valid_len" in kwargs):
            attn_mask = self._lengths_to_mask(kwargs["y_valid_len"], L2, device, dtype, B)
        elif attn_mask is not None:
            attn_mask = self._ensure_mask_shape(attn_mask, B, L2, device, dtype)

        if v_pos is not None:
            V = V * v_pos.to(device=device, dtype=V.dtype)  

        if seq_mask is not None:
            sm = seq_mask.to(device=Q.device, dtype=Q.dtype)  # (B,1,L1,1)
            Q.mul_(sm)
            U.mul_(sm)
        if attn_mask is not None:
            am = attn_mask.to(device=K.device, dtype=K.dtype) # (B,1,L2,1)
            K.mul_(am)
            V.mul_(am)

        # Retention: KV:(B,H,Dh,Dh), O:(B,H,L1,Dh)
        B, H = Q.shape[0], Q.shape[1]
        L1, L2, Dh = Q.shape[2], K.shape[2], Q.shape[3]

        # (B*H, Dh, L2) @ (B*H, L2, Dh) -> (B*H, Dh, Dh)
        K_bh = K.reshape(B * H, L2, Dh).transpose(1, 2).contiguous()
        V_bh = V.reshape(B * H, L2, Dh).contiguous()
        KV_bh = torch.bmm(K_bh, V_bh)  # (B*H, Dh, Dh)

        # (B*H, L1, Dh) @ (B*H, Dh, Dh) -> (B*H, L1, Dh)
        Q_bh = Q.reshape(B * H, L1, Dh).contiguous()
        O_bh = torch.bmm(Q_bh, KV_bh)  # (B*H, L1, Dh)

        # (B,H,L1,Dh)
        O = O_bh.view(B, H, L1, Dh)
        O = self.inner_norm(O) * U  # (B,H,L1,Dh)
        O = O.transpose(1, 2).reshape(B, L1, H * Dh).contiguous()
        O_in = O
        out = self.o_proj(O_in)
        if self.lora > 0:
            out = out + self.lora_o(O_in) 
        return out

class GatedLinearUnit(nn.Module):
    def __init__(self, emb_dims: int, lth: Optional[float] = None, lora: int = 0) -> None:
        super().__init__()
        assert lora >= 0, "LoRA rank must be non-negative."
        beta = 1.0 if lth is None else (lth * 8) ** -0.25
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.norm = SRMSNorm(emb_dims)
        self.lora = lora
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)
        if self.lora > 0:
            self.lora_u = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_v = LoraBlock(emb_dims, emb_dims, lora)
            self.lora_o = LoraBlock(emb_dims, emb_dims, lora)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        W_uv = torch.cat([self.u_proj.weight, self.v_proj.weight], dim=0)  # (2D, D)
        uv = torch.nn.functional.linear(x, W_uv)  # (B,L,2D)
        u, v = uv.split(D, dim=-1)  # (B,L,D), (B,L,D)

        if self.lora > 0:
            x_flat = x.reshape(-1, D)  # (B*L, D)
            u = u + self.lora_u(x_flat).view(B, L, D)
            v = v + self.lora_v(x_flat).view(B, L, D)

        z = u * v    # out-of-place
        out = self.o_proj(z)  # (B,L,D)
        if self.lora > 0:
            z_flat = z.reshape(-1, D)
            out = out + self.lora_o(z_flat).view(B, L, D)
        return out

class SoftPreNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, init_lambda: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        bias = torch.logit(torch.tensor(min(max(init_lambda, 1e-6), 0.999)))
        self._lambda_bias = nn.Parameter(bias.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = torch.sigmoid(self._lambda_bias)
        return (1.0 - lam) * x + lam * self.ln(x)

def _broadcast_c(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if c.dim() == 2:  # [B,C] -> [B,1,C]
        c = c.unsqueeze(1)
    if x.dim() == 2:  # [B,D] -> [B,1,D]
        x = x.unsqueeze(1)
    if c.size(1) == 1 and x.size(1) > 1:
        c = c.expand(-1, x.size(1), -1)
    return c

class Ada6Head(nn.Module):
    def __init__(self, cond_dim: int, hidden_size: int, use_sigmoid: bool = True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.D = hidden_size
        self.proj = nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        setattr(self.proj, "_skip_basic_init", True)
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias.zero_()
            D = hidden_size
            gate_bias_init = 6.0 if self.use_sigmoid else 1.0
            # gate_a 
            self.proj.bias[2*D:3*D].fill_(gate_bias_init)
            # gate_f 
            self.proj.bias[5*D:6*D].fill_(gate_bias_init)

    def forward(self, c_bt: torch.Tensor, B: int, T: int, D: int):
        c_bt = F.silu(c_bt)
        s = self.proj(c_bt)  # [B*T, 6D]
        sh_a, sc_a, g_a, sh_f, sc_f, g_f = s.split(D, dim=-1)
        def V(u): return u.view(B, T, D)
        if self.use_sigmoid:
            return V(sh_a), V(sc_a) + 1.0, torch.sigmoid(V(g_a)), V(sh_f), V(sc_f) + 1.0, torch.sigmoid(V(g_f))
        else:
            return V(sh_a), V(sc_a) + 1.0, V(g_a), V(sh_f), V(sc_f) + 1.0, V(g_f)

    def project_bd(self, c_b: torch.Tensor, D: int):
        c_b = F.silu(c_b)
        s = self.proj(c_b)  # [B, 6D]
        sh_a, sc_a, g_a, sh_f, sc_f, g_f = s.split(D, dim=-1)
        if self.use_sigmoid:
            return (
                sh_a.unsqueeze(1), (sc_a + 1.0).unsqueeze(1), torch.sigmoid(g_a).unsqueeze(1),
                sh_f.unsqueeze(1), (sc_f + 1.0).unsqueeze(1), torch.sigmoid(g_f).unsqueeze(1),
            )
        else:
            return (
                sh_a.unsqueeze(1), (sc_a + 1.0).unsqueeze(1), g_a.unsqueeze(1),
                sh_f.unsqueeze(1), (sc_f + 1.0).unsqueeze(1), g_f.unsqueeze(1),
            )

class RetentionLayer(nn.Module):
    def __init__(self,
                 cond_dim: int,
                 emb_dims: int,
                 num_heads: int,
                 lth: float,
                 dropout: float = 0.0,
                 lora: int = 0,
                 base_frozen: bool = False,
                 use_checkpoint: bool = False,
                 use_sigmoid: bool = True):
        super().__init__()
        self.alpha = (2 * lth) ** 0.25
        self.use_checkpoint = use_checkpoint

        self.attn = MHRetention(emb_dims, num_heads, lth, lora)
        self.ffn  = GatedLinearUnit(emb_dims, lth, lora)
        self.dropout = nn.Dropout(p=dropout)

        self.pre_norm1 = SoftPreNorm(emb_dims, eps=1e-6, init_lambda=0.0)
        self.pre_norm2 = SoftPreNorm(emb_dims, eps=1e-6, init_lambda=0.0)

        self.post_norm1 = nn.LayerNorm(emb_dims, eps=1e-6, elementwise_affine=True)
        self.post_norm2 = nn.LayerNorm(emb_dims, eps=1e-6, elementwise_affine=True)

        self.adaln = Ada6Head(cond_dim, emb_dims, use_sigmoid=use_sigmoid)

        if base_frozen:
            for m in [self.attn, self.ffn, self.post_norm1, self.post_norm2]:
                for p in m.parameters():
                    p.requires_grad = False
            self.pre_norm1._lambda_bias.requires_grad = True
            self.pre_norm2._lambda_bias.requires_grad = True
            for p in self.adaln.parameters(): p.requires_grad = True

    def _maybe_ckpt(self, fn, *args):
        if self.use_checkpoint and any(t.requires_grad for t in args if isinstance(t, torch.Tensor)):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x: torch.Tensor, c: torch.Tensor, **kwargs) -> torch.Tensor:
        B = x.size(0)
        T = 1 if x.dim() == 2 else x.size(1)
        D = x.size(-1)

        if c.dim() == 2:
            sh_a, sc1_a, g_a, sh_f, sc1_f, g_f = self.adaln.project_bd(c, D)  # [B,1,D] ×6
        else:
            c_bt = _broadcast_c(c, x).reshape(B * T, -1)
            sh_a, sc1_a, g_a, sh_f, sc1_f, g_f = self.adaln(c_bt, B, T, D)  # [B,T,D] ×6

        z = self.pre_norm1(x)                                     # [B,T,D]
        x_attn_in = torch.addcmul(sh_a, z, sc1_a, value=1.0)      # [B,T,D]  x_attn_in = shift + (1+scale) * z
        def _attn(inp): return self.attn(inp, y=None, **kwargs)
        out_attn = self._maybe_ckpt(_attn, x_attn_in)
        out_attn = self.dropout(out_attn)
        res1 = torch.add(out_attn * g_a, x, alpha=self.alpha)
        y1   = self.post_norm1(res1)

        z2       = self.pre_norm2(y1)
        x_mlp_in = torch.addcmul(sh_f, z2, sc1_f, value=1.0)
        def _ffn(inp): return self.ffn(inp)
        out_ffn = self._maybe_ckpt(_ffn, x_mlp_in)
        out_ffn = self.dropout(out_ffn)
        res2 = torch.add(out_ffn * g_f, y1, alpha=self.alpha)
        x    = self.post_norm2(res2)
        return x

class ValueDecoder(nn.Module):
    def __init__(self,
                 cond_dim: int,
                 emb_dims: int,
                 dropout: float = 0.0):
        super().__init__()
        self.pre_norm = SoftPreNorm(emb_dims, eps=1e-6, init_lambda=0.0)
        self.w1  = nn.Linear(emb_dims, emb_dims, bias=False)
        self.act = nn.LeakyReLU()
        self.w2  = nn.Linear(emb_dims, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(cond_dim, 2 * emb_dims, bias=True)
        setattr(self.proj, "_skip_basic_init", True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        T = 1 if x.dim() == 2 else x.size(1)
        D = x.size(-1)
        c = F.silu(c)
        if c.dim() == 2:
            s = self.proj(c)  # [B, 2D]
            sh, sc = s.split(D, dim=-1)  # [B,D] ×2
            sh_d, sc1_d = sh.unsqueeze(1), (sc + 1.0).unsqueeze(1)
        else:
            c_bt = _broadcast_c(c, x).reshape(B * T, -1)
            s = self.proj(c_bt)  # [B*T, 2D]
            sh, sc = s.split(D, dim=-1)  # [B*T,D] ×2
            def V(u): return u.view(B, T, D)
            sh_d, sc1_d = V(sh), V(sc) + 1.0

        z = self.pre_norm(x)
        x_in = torch.addcmul(sh_d, z, sc1_d, value=1.0)

        h = self.dropout(self.act(self.w1(x_in)))
        pred = self.w2(h).view(B, T)
        # pred = self.w2(self.act(self.w1(x_in))).view(B, T)
        return pred # [B,T(L)]


class CellwiseDecoder(nn.Module):
    def __init__(
        self,
        in_dims: int,
        emb_dims: int = None,
        dropout: float = 0.0,
        cond_dim: int = None,
    ) -> None:
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.in_dims   = in_dims
        self.emb_dims  = emb_dims

        self.map = nn.Linear(in_dims, emb_dims, bias=False)

        self.pre_norm = SoftPreNorm(emb_dims, eps=1e-6, init_lambda=0.0)
        self.dropout  = nn.Dropout(p=dropout)

        self.has_cond = cond_dim is not None
        if self.has_cond:
            self.proj = nn.Linear(cond_dim, 2 * emb_dims, bias=True)
            setattr(self.proj, "_skip_basic_init", True)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        cell_emb: torch.Tensor,  # (B, D)
        gene_emb: torch.Tensor,   # (B, L, in_dims)
        c: torch.Tensor = None,  # (B, cond_dim) or (B, L, cond_dim)
    ) -> torch.Tensor:
        B = gene_emb.size(0)
        # 1) query = sigmoid(W·gene_emb)
        q = torch.sigmoid(self.map(gene_emb))        # (B, L, D)

        # 2) AdaLN :query <- shift + (1+scale) * pre_norm(query)
        if self.has_cond:
            if c is None:
                raise ValueError("cond_dim was set, but no condition `c` was provided.")
            D = q.size(-1)
            T = q.size(1)
            c = F.silu(c)
            if c.dim() == 2:
                s = self.proj(c)                    # (B, 2D)
                sh, sc = s.split(D, dim=-1)         # (B,D) ×2
                sh_q, sc1_q = sh.unsqueeze(1), (sc + 1.0).unsqueeze(1)  # (B,1,D)
            else:
                c_bt = _broadcast_c(c, q).reshape(B * T, -1)
                s = self.proj(c_bt)                 # (B*T, 2D)
                sh, sc = s.split(D, dim=-1)         # (B*T, D)
                def V(u): return u.view(B, T, D)
                sh_q, sc1_q = V(sh), V(sc) + 1.0    # (B,L,D)

            z = self.pre_norm(q)                    # (B,L,D)
            q = torch.addcmul(sh_q, z, sc1_q, value=1.0)

        # 3) dropout on query
        q = self.dropout(q)                         # (B,L,D)

        # 4) key = cell_emb.view(B, D, 1)
        if cell_emb.dim() != 2 or cell_emb.size(-1) != self.emb_dims:
            raise ValueError(f"Expected cell_emb shape (B,{self.emb_dims}), got {tuple(cell_emb.shape)}")
        key = cell_emb.view(B, self.emb_dims, 1)    # (B,D,1)

        # 5) pred = bmm(query, key).view(B, L)
        pred = torch.bmm(q, key).view(B, -1)        # (B,L)
        return pred