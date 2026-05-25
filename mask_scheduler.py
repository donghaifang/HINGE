import torch
import math

class MaskScheduler:
    def __init__(self, G, T, device="cpu",
                 mask_fill="gauss",      # 'zero'   | 'gauss'
                 idx_mode="random",      # 'random' | 'variance' | 'group' 
                 gamma_mode="linear",    # 'linear' | 'cosine' | 'sqrt' | 'exp' | ('poly', alpha) | 'poly_auto'
                 gamma_param=None,       # e.g., {'alpha':1.42} or {'beta':3.0} or {'K_tail':1}
                 gene_scores=None,       # [G] for idx_mode!='random' (e.g., variance)
                 gene_groups=None,       # list[list[int]] groups if idx_mode=='group'
                 seed=None):
        self.G, self.T, self.device = G, T, device
        self.mask_fill = mask_fill
        self.idx_mode = idx_mode
        self.gamma_mode = gamma_mode
        self.gamma_param = gamma_param or {}
        self.gene_scores = gene_scores
        self.gene_groups = gene_groups
        self.seed = seed
        self._B = None
        # deterministic generator (optional)
        self._gen = torch.Generator(device=device)
        if seed is not None:
            self._gen.manual_seed(seed)

        if self.gamma_mode == "poly_auto":
            k_init = int((self.gamma_param or {}).get("K_tail", 1))
            T = max(2, int(self.T));  G = max(1, int(self.G))
            import math
            # alpha = log(K/G) / log(1/T) 
            alpha = math.log(k_init / float(G)) / math.log(1.0 / float(T))
            alpha = float(max(1.05, min(alpha, 4.0)))  
            self.gamma_mode = "poly"
            self.gamma_param = {"alpha": alpha}
            
    # ---------- gamma: #masked at t ----------
    def _gamma(self, t):  # t: [B]
        t = t.float()
        T = float(self.T)
        if self.gamma_mode == "linear":
            g = t / T
        elif self.gamma_mode == "cosine":
            g = 0.5 * (1 - torch.cos(math.pi * t / T))
        elif self.gamma_mode == "sqrt":
            g = torch.sqrt(t / T)
        elif self.gamma_mode == "exp":
            beta = float(self.gamma_param.get("beta", 3.0))
            g = (torch.exp(beta * t / T) - 1.0) / (math.e**beta - 1.0)
        elif self.gamma_mode == "poly":
            alpha = float(self.gamma_param.get("alpha", 1.42))
            g = 1 - (1 - t / T) ** alpha
        else:
            raise ValueError(f"Unknown gamma_mode: {self.gamma_mode}")
        return g.clamp_(0, 1)

    def _splits(self, t):   # -> [B] long
        t = t.clamp(min=0, max=self.T)
        g = self._gamma(t)
        return (g * self.G).long()

    # ---------- idx (order) ----------
    def _build_idx(self, B):
        if self.idx_mode == "random":
            return torch.argsort(torch.rand(B, self.G, device=self.device, generator=self._gen), dim=-1)
        elif self.idx_mode == "variance":
            assert self.gene_scores is not None and self.gene_scores.numel() == self.G
            # higher score earlier or later? choose 'descend' for early decode of important ones
            base = torch.argsort(self.gene_scores.to(self.device), dim=-1, descending=True)
            return base.unsqueeze(0).expand(B, -1).clone()
        elif self.idx_mode == "group":
            # flatten groups in chosen order; e.g., housekeeping first, then HVGs
            assert self.gene_groups is not None and sum(len(g) for g in self.gene_groups) == self.G
            flat = torch.tensor([i for grp in self.gene_groups for i in grp], device=self.device, dtype=torch.long)
            return flat.unsqueeze(0).expand(B, -1).clone()
        else:
            raise ValueError(f"Unknown idx_mode: {self.idx_mode}")

    # ---------- public APIs (compatible with your training/infer loop) ----------
    def batch_generate_mask_chain(self, B, seed=None):
        self._B = B
        if seed is not None:
            self._gen.manual_seed(int(seed))
        return self._build_idx(B)   # [B,G]

    def _m_from_splits(self, idx, splits_vec):
        # idx: [B,G], splits_vec: [B]
        B, G = idx.shape
        m = torch.ones(B, G, device=self.device)
        row = torch.arange(B, device=self.device).unsqueeze(1).expand(B, G)
        col = torch.arange(G, device=self.device).unsqueeze(0).expand(B, G)
        mask_pos = col < splits_vec.unsqueeze(1)
        m[row[mask_pos], idx[mask_pos]] = 0.0
        return m

    def batch_sample_mask_and_forward(self, x0, idx, t):
        # x0:[B,G], t:[B]
        splits = self._splits(t)
        splits_prev = self._splits(t - 1)
        m_t   = self._m_from_splits(idx, splits)
        m_prev= self._m_from_splits(idx, splits_prev)
        if self.mask_fill == "zero":
            noise = torch.zeros_like(x0)
        else:  # "gauss"
            noise = torch.randn_like(x0)
        x_t = m_t * x0 + (1 - m_t) * noise
        d_t = m_prev - m_t
        return x_t, m_prev, d_t



