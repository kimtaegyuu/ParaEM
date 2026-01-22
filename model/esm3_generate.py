#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
from typing import List

import torch
from Bio import SeqIO
from huggingface_hub import login

from esm.models.esm3 import ESM3
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.sdk.api import ESMProtein, SamplingConfig


def load_fasta_sequences(fasta_file: str) -> List[str]:
    records = list(SeqIO.parse(fasta_file, "fasta"))
    if len(records) == 0:
        raise ValueError(f"No FASTA records found in: {fasta_file}")
    return [str(rec.seq).strip() for rec in records]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Encode a single FASTA sequence using ESM3 and save per-residue embeddings (ONLY) as a pickle.\n\n"
            "Hugging Face token note:\n"
            "- You must create a Hugging Face access token to download ESM3 weights.\n"
            "- Create it on Hugging Face: Settings -> Access Tokens -> New token.\n"
            "- Use a token with permission to read/download models.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--hugging_token", required=True, help="Hugging Face access token (e.g., 'hf_...').")
    p.add_argument("--fasta_file", required=True, help="Input FASTA file path (expects at least 1 record).")
    p.add_argument("--output", required=True, help="Output directory to write {fasta_base}_esm3.pkl")
    p.add_argument(
        "--model_id",
        default=ESM3_OPEN_SMALL,
        help=f"ESM3 model identifier (default: {ESM3_OPEN_SMALL})",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on: cuda or cpu (default: auto).",
    )
    return p


def main():
    args = build_argparser().parse_args()

    fasta_file = args.fasta_file
    out_dir = args.output
    model_id = args.model_id
    device = args.device

    os.makedirs(out_dir, exist_ok=True)

    # Output name: {fasta_base}_esm3.pkl
    fasta_base = os.path.splitext(os.path.basename(fasta_file))[0]
    out_path = os.path.join(out_dir, f"{fasta_base}_esm3.pkl")

    # Login to HF to allow downloading weights
    login(token=args.hugging_token)

    seqs = load_fasta_sequences(fasta_file)
    if len(seqs) > 1:
        print(f"[WARN] FASTA has {len(seqs)} records; using the first one only.")
    sequence = seqs[0]

    # Load model/client
    client = ESM3.from_pretrained(model_id).to(device)
    client.eval()

    with torch.no_grad():
        protein_tensor = client.encode(ESMProtein(sequence=sequence))
        # Depending on SDK version, this may or may not be a torch Tensor-like object
        try:
            protein_tensor = protein_tensor.to(device)
        except Exception:
            pass

        output = client.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True),
        )

        protein_esm3 = output.per_residue_embedding  # typically torch.Tensor [L, D]

    # Save ONLY the embedding
    if not isinstance(protein_esm3, torch.Tensor):
        # Fallback: dump as-is
        emb_to_save = protein_esm3
    else:
        emb_to_save = protein_esm3.detach().to("cpu").float().numpy()

    with open(out_path, "wb") as f:
        pickle.dump(emb_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved embedding only: {out_path}")
    try:
        print(f"Embedding shape: {emb_to_save.shape}")
    except Exception:
        print(f"Embedding type: {type(emb_to_save)}")


if __name__ == "__main__":
    main()
