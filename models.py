import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from Hinge.retentionlayer import RetentionLayer, ValueDecoder, CellwiseDecoder

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return t_freq

class CondNet(nn.Module):
    def __init__(self, label_size, mask_size, cond_dim=384, t_freq=256):
        super().__init__()
        self.time = nn.Sequential(
            nn.Linear(t_freq, cond_dim, bias=True), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True)
        )
        self.label = nn.Sequential(
            nn.Linear(label_size, cond_dim, bias=True), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True)
        )
        self.mask = nn.Sequential(
            nn.Linear(mask_size, cond_dim, bias=True), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True)
        )

        nn.init.normal_(self.time[0].weight, std=0.02)
        nn.init.normal_(self.time[2].weight, std=0.02)
        nn.init.normal_(self.label[0].weight, std=0.02)
        nn.init.normal_(self.label[2].weight, std=0.02)
        nn.init.normal_(self.mask[0].weight, std=0.02)
        nn.init.normal_(self.mask[2].weight, std=0.02)

    def forward(self, t_feat, y, mp): 
        z = self.time(t_feat) + self.label(y) + self.mask(mp)
        return z  # (B, cond_dim)

class FFN(nn.Module):
    def __init__(self, in_dims, emb_dims, b=256):
        super().__init__()
        self.w1 = nn.Linear(in_dims, b, bias=False)
        self.act1 = nn.LeakyReLU()
        self.w3 = nn.Linear(b, b, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.table = nn.Linear(b, emb_dims, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        b, l, d = x.shape
        v = x.view(-1, d)
        v = self.act1(self.w1(v))
        v = self.w3(v) + v * self.a
        v = self.softmax(v)
        v = self.table(v)
        return v.view(b, l, -1)

class ValueEncoder(nn.Module):
    def __init__(self, emb_dims):
        super().__init__()
        self.value_enc = FFN(1, emb_dims)

    def forward(self, x):
        expr = x.unsqueeze(-1)
        expr_emb = self.value_enc(expr)
        return expr_emb

class HingeModel(nn.Module):
    def __init__(self,
                 ggidx,
                 n_genes=27855,
                 input_size=200,
                 label_size=512,
                 learn_sigma=False,
                 enc_dims=1536,
                 enc_nlayers=40,
                 enc_num_heads=48,
                 lora=0,
                 enc_dropout=0.1, 
                 use_sigmoid=True):
        super().__init__()
        assert enc_nlayers==40, "enc_nlayers != 40"
        self.use_sigmoid = use_sigmoid
        self.register_buffer("gg", torch.from_numpy(ggidx).long(), persistent=True)

        self.gene_emb = nn.Parameter(torch.empty(n_genes + 1 + (-n_genes - 1) % 8, enc_dims))
        nn.init.xavier_normal_(self.gene_emb)
        with torch.no_grad():
            self.gene_emb[0, :] = 0

        self.cls_token = nn.Parameter(torch.empty(1, 1, enc_dims))
        nn.init.xavier_normal_(self.cls_token)
        self.zero_emb = nn.Parameter(torch.empty(1, 1, enc_dims))
        nn.init.normal_(self.zero_emb, std=0.02)

        self.learn_sigma  = learn_sigma
        self.input_size   = input_size
        self.enc_nlayers  = enc_nlayers
        self.enc_dims     = enc_dims
        cond_dim          = enc_dims

        self.value_enc   = ValueEncoder(enc_dims)
        self.time_embed  = TimestepEmbedder()
        self.cond_embed  = CondNet(label_size=label_size, mask_size=input_size, cond_dim=enc_dims)

        lth_cfg = 40
        frozen_idxs = set(range(40))
        self.retentionblocks = nn.ModuleList([
            RetentionLayer(
                cond_dim=cond_dim,
                emb_dims=enc_dims,
                num_heads=enc_num_heads,
                lth=lth_cfg,
                dropout=enc_dropout * i / enc_nlayers,
                lora=lora,
                base_frozen=(i in frozen_idxs),
                use_checkpoint=False,
                use_sigmoid=self.use_sigmoid
            )
            for i in range(enc_nlayers)
        ])

        self.final_layer = ValueDecoder(cond_dim, enc_dims)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                if getattr(module, "_skip_basic_init", False):
                    return
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        with torch.no_grad():
            for i, blk in enumerate(self.retentionblocks):
                if isinstance(blk.post_norm1, nn.LayerNorm):
                    blk.post_norm1.eps = 1e-6
                if isinstance(blk.post_norm2, nn.LayerNorm):
                    blk.post_norm2.eps = 1e-6

                # 2.2 SoftPreNorm：λ ≈ 0
                if hasattr(blk, "pre_norm1"):
                    blk.pre_norm1._lambda_bias.copy_(torch.tensor(-12.0))  # sigmoid(-12) ~ 6e-6
                if hasattr(blk, "pre_norm2"):
                    blk.pre_norm2._lambda_bias.copy_(torch.tensor(-12.0))

                # ============ 6*D（[sh_a, sc_a, g_a, sh_f, sc_f, g_f]）===========
                if hasattr(blk, "adaln") and hasattr(blk.adaln, "proj"):
                    proj = blk.adaln.proj  # Linear(cond_dim -> 6*D)
                    setattr(proj, "_skip_basic_init", True)  
                    with torch.no_grad():
                        proj.weight.zero_()
                        proj.bias.zero_()
                        D = proj.out_features // 6
                        gate_init_value = 6.0 if self.use_sigmoid else 1.0
                        proj.bias[2 * D:3 * D].fill_(gate_init_value)
                        proj.bias[5 * D:6 * D].fill_(gate_init_value)

                for head_name in ("adaln_attn", "adaln_ffn"):
                    head = getattr(blk, head_name, None)
                    if head is None:
                        continue
                    proj = head.proj  # Linear(cond_dim -> 3*D)
                    setattr(proj, "_skip_basic_init", True)  
                    D = proj.out_features // 3
                    proj.weight.zero_()
                    proj.bias.zero_()
                    gate_init_value = 6.0 if self.use_sigmoid else 1.0
                    proj.bias[2*D:3*D].fill_(gate_init_value)

        if hasattr(self, "final_layer") and isinstance(self.final_layer, nn.Module):
            with torch.no_grad():
                if hasattr(self.final_layer, "pre_norm"):
                    self.final_layer.pre_norm._lambda_bias.copy_(torch.tensor(-12.0))

                if hasattr(self.final_layer, "proj"):
                    setattr(self.final_layer.proj, "_skip_basic_init", True)
                    self.final_layer.proj.weight.zero_()
                    self.final_layer.proj.bias.zero_()

    def cellfm_encode(self, x, gene, cond, mt=None):
        if x.dim() == 3:
            x_raw = x.squeeze(1)  # (B, L_eff)
        else:
            x_raw = x
        B, L_eff = x_raw.shape
        D = self.cls_token.size(-1)

        # ---- gene embedding ----
        gene_idx = gene.long()[..., :L_eff]  # (L_eff,)
        gene_emb = self.gene_emb[gene_idx]  # (L_eff, D)
        gene_emb = gene_emb.unsqueeze(0).expand(B, L_eff, D)  # (B, L_eff, D)

        # ---- ValueEncoder ----
        expr_emb = self.value_enc(x_raw)  # (B, L_eff, D)

        if mt is not None:
            mt_slice = mt[..., :L_eff].to(expr_emb.dtype)  # (B, L_eff)
            mbar = (1.0 - mt_slice)  
            if torch.any(mbar > 0):
                token = self.zero_emb.expand(B, L_eff, D)  # (B, L_eff, D)
                expr_emb = expr_emb + token * mbar.unsqueeze(-1)  

        expr_emb = expr_emb + gene_emb  # (B, L_eff, D)
        cls_token = self.cls_token.expand(B, 1, D)  # (B,1,D)
        expr_emb = torch.cat([cls_token, expr_emb], dim=1)  # (B, L, D)  L=1+L_eff

        len_scale = torch.rsqrt(torch.tensor(L_eff, device=x_raw.device, dtype=torch.float32) + 1e-6)
        len_scale = len_scale.view(1, 1, 1, 1)  # (1,1,1,1)

        attn_mask_vec = None
        seq_mask_vec = None
        if mt is not None:
            ones_cls = torch.ones(B, 1, device=x_raw.device, dtype=expr_emb.dtype)  # CLS=1
            mt_full = torch.cat([ones_cls, mt_slice], dim=1)  # (B, L)
            attn_mask_vec = mt_full  
            
        half = self.enc_nlayers // 2  # 20
        for i in range(self.enc_nlayers):
            if (attn_mask_vec is not None) and (i < half):
                expr_emb = self.retentionblocks[i](
                    x=expr_emb,
                    c=cond,
                    v_pos=len_scale,
                    attn_mask=attn_mask_vec,  # (B, L)
                    seq_mask=seq_mask_vec,  # (B, L)
                )
            else:
                expr_emb = self.retentionblocks[i](
                    x=expr_emb,
                    c=cond,
                    v_pos=len_scale
                )

        return expr_emb[:, 0, :], expr_emb[:, 1:, :], gene_emb

    def forward(self, x, t, mp, mt, y):
        """
        x: (N, NumGene)
        t: (N,)
        y: (N, 512)
        """
        t_feat = self.time_embed(t)
        c = self.cond_embed(t_feat, y, mp)      # -> (N, D)
        _, x, gene = self.cellfm_encode(x, self.gg, c, mt=mt)   # -> (N, L_eff, D)
        x = self.final_layer(x, c)              # -> (N, 1, NumGene)
        return x

def Hinge(**kwargs):
    return HingeModel(**kwargs)


Hinge_models = {"Hinge": Hinge}
# could set models with different sizes as default options in dictionary "Hinge_models".
