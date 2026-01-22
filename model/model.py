# model.py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CrossAttentionEM_MultiHeadCDR(nn.Module):
    """
    Multi-head EM+CDR cross-attention.

    Inputs:
      ab_emb: (L_ab, embed_dim)
      ag_emb: (L_ag, embed_dim)
      imgt_cat: (L_ab,) in {0..6}

    Output:
      S: (L_ab, L_ag)
    """
    def __init__(
        self,
        embed_dim: int = 1536,
        imgt_vocab: int = 7,
        cdr_emb_dim: int = 32,
        proj_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.ln = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, proj_dim)
        self.k_proj = nn.Linear(embed_dim, proj_dim)

        self.imgt_embed = nn.Embedding(imgt_vocab, cdr_emb_dim)
        self.cdr_to_q = nn.Linear(cdr_emb_dim, proj_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, ab_emb: torch.Tensor, ag_emb: torch.Tensor, imgt_cat: torch.Tensor) -> torch.Tensor:
        device = ab_emb.device
        imgt_cat = imgt_cat.to(device)

        # NOTE: original code had ag_ln = ab_ln bug; fixed here.
        ab_n = self.ln(ab_emb)  # (L_ab, d)
        ag_n = self.ln(ag_emb)  # (L_ag, d)

        cdr_e = self.imgt_embed(imgt_cat)  # (L_ab, cdr_emb_dim)
        cdr_q = self.cdr_to_q(cdr_e)       # (L_ab, proj_dim)

        Q = self.q_proj(ab_n) + cdr_q      # (L_ab, proj_dim)
        K = self.k_proj(ag_n)              # (L_ag, proj_dim)

        Q = self.dropout(Q)
        K = self.dropout(K)

        L_ab = Q.size(0)
        L_ag = K.size(0)

        # (L_ab, proj_dim) -> (H, L_ab, head_dim)
        Qh = Q.view(L_ab, self.num_heads, self.head_dim).permute(1, 0, 2).contiguous()
        # (L_ag, proj_dim) -> (H, head_dim, L_ag)
        Kh = K.view(L_ag, self.num_heads, self.head_dim).permute(1, 2, 0).contiguous()

        Sh = torch.bmm(Qh, Kh) / self.scale  # (H, L_ab, L_ag)
        S = Sh.mean(dim=0)                   # (L_ab, L_ag)
        return S


@torch.no_grad()
def predict_hatp(model: nn.Module, ab_emb: torch.Tensor, ag_emb: torch.Tensor, imgt_cat: torch.Tensor) -> torch.Tensor:
    """
    hat_p[i] = sum_j pi[i,j] * sigmoid(S[i,j]), where pi = softmax(S over antigen residues).
    Returns:
      hat_p: (L_ab,)
    """
    S = model(ab_emb, ag_emb, imgt_cat)
    pi = F.softmax(S, dim=1)       # (L_ab, L_ag)
    p_pos = torch.sigmoid(S)       # (L_ab, L_ag)
    hat_p = (pi * p_pos).sum(dim=1)
    return hat_p
