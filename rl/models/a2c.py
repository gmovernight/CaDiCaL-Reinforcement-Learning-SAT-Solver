
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CNet(nn.Module):
    """A2C feature encoder + heads (candidate-aware policy).

    - Policy: per-candidate scoring with shared MLP over [cand_embed, global_embed]
    - Value: pooled candidate context + global
    """

    def __init__(self, global_dim: int = 11, cand_dim: int = 5,
                 K: int = 16, cand_hidden: int = 64, hidden: int = 128):
        super().__init__()
        self.K = int(K)
        self.cand_hidden = int(cand_hidden)
        self.g_hidden = 64

        # Encoders
        self.cand_phi = nn.Sequential(
            nn.Linear(cand_dim, cand_hidden),
            nn.ReLU(),
            nn.Linear(cand_hidden, cand_hidden),
            nn.ReLU(),
        )
        self.g_enc = nn.Sequential(
            nn.Linear(global_dim, self.g_hidden),
            nn.ReLU(),
        )

        # Policy head: shared over candidates
        pi_in = cand_hidden + self.g_hidden
        self.pi_head = nn.Sequential(
            nn.Linear(pi_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Value head: pooled candidate context + global
        v_in = cand_hidden * 2 + self.g_hidden  # mean + max + global
        self.v_torso = nn.Sequential(
            nn.Linear(v_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.v = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)

    def forward(self, global_feats: torch.Tensor, cand_feats: torch.Tensor):
        # Shapes
        B, K, _ = cand_feats.shape

        # Encode
        g = self.g_enc(global_feats)                        # [B, g_hidden]
        c = self.cand_phi(cand_feats.view(B * K, -1))       # [B*K, cand_hidden]
        c = c.view(B, K, -1)                                # [B, K, cand_hidden]

        # Policy logits per candidate
        g_tile = g.unsqueeze(1).expand(-1, K, -1)           # [B, K, g_hidden]
        hc = torch.cat([c, g_tile], dim=-1)                 # [B, K, cand_hidden+g_hidden]
        logits = self.pi_head(hc).squeeze(-1)               # [B, K]

        # Value from pooled candidate context + global
        c_mean = c.mean(dim=1)
        c_max, _ = c.max(dim=1)
        hv = torch.cat([c_mean, c_max, g], dim=-1)          # [B, v_in]
        hv = self.v_torso(hv)
        value = self.v(hv)                                  # [B, 1]
        return logits, value


class A2C(nn.Module):
    """Wrapper that applies candidate masking and exposes a clean API.

    forward(global: [B, G], cands: [B, K, C], mask: [B, K] or None)
      -> logits [B, K] (masked), value [B, 1]
    """

    def __init__(self, global_dim: int = 11, cand_dim: int = 5,
                 K: int = 16, hidden: int = 128, cand_hidden: int = 64):
        super().__init__()
        self.K = int(K)
        self.net = A2CNet(global_dim=global_dim, cand_dim=cand_dim,
                          K=K, cand_hidden=cand_hidden, hidden=hidden)

    def forward(self, global_feats: torch.Tensor, cand_feats: torch.Tensor,
                mask: torch.Tensor | None = None):
        logits, value = self.net(global_feats, cand_feats)
        if mask is not None:
            # mask: 1 for valid, 0 for invalid; shape [B,K]
            # set invalid logits to large negative value
            mask = mask.to(dtype=logits.dtype, device=logits.device)
            invalid = (mask <= 0)
            if invalid.any():
                logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        return logits, value
