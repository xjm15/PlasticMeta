import torch
from torch import nn

def linear(out_dim, in_dim, adaptation, meta):
    w = nn.Parameter(torch.ones(out_dim, in_dim))
    nn.init.kaiming_normal_(w)
    b = nn.Parameter(torch.zeros(out_dim))
    w.meta, b.meta = meta, meta
    w.adaptation, b.adaptation = adaptation, adaptation
    return w, b

def layer_norm(d_model, adaptation, meta):
    gamma = nn.Parameter(torch.ones(d_model))
    beta = nn.Parameter(torch.zeros(d_model))
    gamma.meta, beta.meta = meta, meta
    gamma.adaptation, beta.adaptation = adaptation, adaptation
    return gamma, beta

def attention(d_model, num_heads, adaptation, meta):
    q_w = nn.Parameter(torch.ones(d_model, d_model))
    nn.init.kaiming_normal_(q_w)
    q_b = nn.Parameter(torch.zeros(d_model))
    k_w = nn.Parameter(torch.ones(d_model, d_model))
    nn.init.kaiming_normal_(k_w)
    k_b = nn.Parameter(torch.zeros(d_model))
    v_w = nn.Parameter(torch.ones(d_model, d_model))
    nn.init.kaiming_normal_(v_w)
    v_b = nn.Parameter(torch.zeros(d_model))
    fc_w = nn.Parameter(torch.ones(d_model, d_model))
    nn.init.kaiming_normal_(fc_w)
    fc_b = nn.Parameter(torch.zeros(d_model))
    q_w.meta, q_b.meta, k_w.meta, k_b.meta, v_w.meta, v_b.meta, fc_w.meta, fc_b.meta = (
        meta, meta, meta, meta, meta, meta, meta, meta)
    q_w.adaptation, q_b.adaptation, k_w.adaptation, k_b.adaptation, v_w.adaptation, v_b.adaptation, \
        fc_w.adaptation, fc_b.adaptation = (
        adaptation, adaptation, adaptation, adaptation, adaptation, adaptation, adaptation, adaptation)

    return q_w, q_b, k_w, k_b, v_w, v_b, fc_w, fc_b