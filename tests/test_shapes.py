
import numpy as np
from rl.models.a2c import A2CNet

def test_a2c_forward_shapes():
    net = A2CNet(K=16, cand_hidden=64, hidden=128)
    import torch
    g = torch.zeros(2, 11)
    c = torch.zeros(2, 16, 5)
    logits, value = net(g, c)
    assert logits.shape == (2, 16)
    assert value.shape == (2, 1)
