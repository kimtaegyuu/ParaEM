#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle

import numpy as np
import torch

from model import CrossAttentionEM_MultiHeadCDR, predict_hatp


# ---------------- config (edit here if needed) ----------------
EMBED_DIM = 1536
IMGT_VOCAB = 7
CDR_EMB_DIM = 32
PROJ_DIM = 256
NUM_HEADS = 4
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------------------------------


def read_imgt_string(path: str) -> np.ndarray:
    """
    imgt.txt: single line like '000112233...'
    Returns int64 numpy array shape (L_ab,)
    """
    with open(path, "r") as f:
        s = f.read().strip()
    if not s:
        raise ValueError(f"Empty imgt file: {path}")
    arr = np.fromiter((int(ch) for ch in s), dtype=np.int64)
    return arr


def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
    model.load_state_dict(state, strict=True)


def main():
    p = argparse.ArgumentParser(
        description="Predict residue-wise probabilities from ab/ag ESM3 embeddings + imgt, and write per-residue TSV."
    )
    p.add_argument("--ab_esm3", required=True, help="Path to antibody ESM3 embedding pickle (numpy [L_ab,D])")
    p.add_argument("--ag_esm3", required=True, help="Path to antigen  ESM3 embedding pickle (numpy [L_ag,D])")
    p.add_argument("--imgt_txt", required=True, help="Path to imgt.txt (single line, e.g., 000112..., len == L_ab)")
    p.add_argument("--model_pt", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--output", required=True, help="Output directory")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # load inputs
        
    with open(args.ab_esm3,"rb") as f:
        ab = pickle.load(f).to(DEVICE)
    with open(args.ag_esm3,"rb") as f:
        ag = pickle.load(f).to(DEVICE)   
    
    imgt_np = read_imgt_string(args.imgt_txt)    # [L_ab]
    imgt = torch.from_numpy(imgt_np).to(DEVICE).long()

    # init + load model
    model = CrossAttentionEM_MultiHeadCDR(
        embed_dim=EMBED_DIM,
        imgt_vocab=IMGT_VOCAB,
        cdr_emb_dim=CDR_EMB_DIM,
        proj_dim=PROJ_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    load_checkpoint(model, args.model_pt)
    model.eval()

    # predict
    with torch.no_grad():
        hat_p = predict_hatp(model, ab, ag, imgt).detach().cpu().numpy().astype(np.float32)  # [L_ab]

    # write TSV (one residue per row)
    out_path = os.path.join(args.output, "pred.tsv")
    with open(out_path, "w") as f:
        f.write("idx\tprob\n")
        for i, v in enumerate(hat_p):
            f.write(f"{i}\t{float(v):.6f}\n")

    print(f"Saved: {out_path}  (L_ab={hat_p.shape[0]})")


if __name__ == "__main__":
    main()
