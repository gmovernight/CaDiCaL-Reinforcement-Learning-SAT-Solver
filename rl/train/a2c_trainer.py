
from __future__ import annotations
import torch, torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Any

class A2CTrainer:
    def __init__(self, net, lr: float = 3e-4, value_coef: float = 0.5, entropy_coef: float = 0.01, grad_clip: float = 0.7, device: str = "cpu"):
        self.net = net.to(device)
        self.opt = Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999))
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.device = device

    def step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        g = torch.as_tensor(batch["global"], dtype=torch.float32, device=self.device)
        c = torch.as_tensor(batch["cands"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        adv = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        m = batch.get("mask", None)
        if m is not None:
            m = torch.as_tensor(m, dtype=torch.bool, device=self.device)

        # Forward (with mask if provided)
        if m is not None:
            logits, values = self.net(g, c, m)
        else:
            logits, values = self.net(g, c)
        logp_all = torch.log_softmax(logits, dim=-1)
        logp_a = logp_all.gather(1, a.view(-1,1)).squeeze(1)
        entropy = -(logp_all.exp() * logp_all).sum(dim=-1).mean()

        # Advantage normalization for stability
        adv_n = adv
        if adv_n.numel() > 1:
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        policy_loss = -(logp_a * adv_n).mean()
        # Huber loss (smooth L1) for value to reduce sensitivity to outliers
        value_loss = F.smooth_l1_loss(values.squeeze(1), ret)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip).item())
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "grad_norm": grad_norm,
        }
