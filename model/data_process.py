#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List, Tuple

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


# ---------------- defaults / config ----------------
# Fv only (IMGT position slicing)
FV_MIN_IMGT = 1
FV_MAX_IMGT = 128

# IMGT CDR ranges (common IMGT positions)
H_CDR1 = (27, 38)
H_CDR2 = (56, 65)
H_CDR3 = (105, 117)

L_CDR1 = (27, 38)
L_CDR2 = (56, 65)
L_CDR3 = (105, 117)


# ---------------- helpers ----------------
def normalize_id(x: str) -> str:
    return str(x).strip()


def parse_chain_list(chain_field: str) -> List[str]:
    """
    chain_field can be "A" or "A;B"
    """
    if chain_field is None:
        return []
    s = str(chain_field).strip()
    if not s:
        return []
    return [c.strip() for c in s.split(";") if c.strip()]


def is_protein_residue(res) -> bool:
    hetflag, _, _ = res.id
    return hetflag == " "


def aa1(resname: str) -> str:
    try:
        return seq1(resname)
    except Exception:
        if resname == "MSE":
            return "M"
        return "X"


def extract_atom_seq_and_map(chain, min_imgt: int, max_imgt: int):
    """
    ATOM-based sequence and mapping, filtered by IMGT position range [min_imgt, max_imgt].
    Keeps residues with coordinates (CA exists).
    """
    seq = []
    mapping = []
    seq_idx = 0

    for res in chain:
        if not is_protein_residue(res):
            continue
        if not res.has_id("CA"):
            continue

        _, resseq, icode = res.id
        if resseq < min_imgt or resseq > max_imgt:
            continue

        aa = aa1(res.get_resname())
        icode = (icode.strip() if isinstance(icode, str) else "")
        seq.append(aa)
        mapping.append(
            {
                "seq_idx": seq_idx,
                "resseq": int(resseq),
                "icode": icode,
                "aa": aa,
                "res": res,
            }
        )
        seq_idx += 1

    return "".join(seq), mapping


def extract_atom_seq(chain):
    """
    ATOM-based sequence without IMGT slicing (used for antigen concat).
    """
    seq = []
    for res in chain:
        if not is_protein_residue(res):
            continue
        if not res.has_id("CA"):
            continue
        seq.append(aa1(res.get_resname()))
    return "".join(seq)


def imgt_category_for_resseq(resseq: int, chain_type: str) -> int:
    """
    frame=0
    H: CDR-H1=1, CDR-H2=2, CDR-H3=3
    L: CDR-L1=4, CDR-L2=5, CDR-L3=6
    """
    if chain_type == "H":
        if H_CDR1[0] <= resseq <= H_CDR1[1]:
            return 1
        if H_CDR2[0] <= resseq <= H_CDR2[1]:
            return 2
        if H_CDR3[0] <= resseq <= H_CDR3[1]:
            return 3
        return 0

    if chain_type == "L":
        if L_CDR1[0] <= resseq <= L_CDR1[1]:
            return 4
        if L_CDR2[0] <= resseq <= L_CDR2[1]:
            return 5
        if L_CDR3[0] <= resseq <= L_CDR3[1]:
            return 6
        return 0

    return 0


def imgt_categories_for_mapping(mapping, chain_type: str) -> List[int]:
    return [imgt_category_for_resseq(item["resseq"], chain_type) for item in mapping]


# ---------------- core ----------------
def process_single_pdb(
    pdb_path: str,
    vh_chain: str,
    vl_chain: str,
    ag_chains: List[str],
) -> Tuple[str, str, str, List[int]]:
    """
    Returns:
      pdb_id, ab_seq (VH+VL), ag_seq, imgt_cat (len == len(ab_seq))
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")

    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0].strip().lower()

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = next(structure.get_models())

    vh_chain = normalize_id(vh_chain)
    vl_chain = normalize_id(vl_chain)

    if not ag_chains:
        raise ValueError("Empty antigen chain list (use --antigen_chain like 'A' or 'A;B').")

    present = {ch.id for ch in model}
    for cid in [vh_chain, vl_chain] + ag_chains:
        if cid not in present:
            raise ValueError(
                f"{pdb_id}: chain '{cid}' not found in {os.path.basename(pdb_path)}. present={sorted(present)}"
            )

    # Fv-only antibody sequences + mapping
    H_seq, H_map = extract_atom_seq_and_map(model[vh_chain], FV_MIN_IMGT, FV_MAX_IMGT)
    L_seq, L_map = extract_atom_seq_and_map(model[vl_chain], FV_MIN_IMGT, FV_MAX_IMGT)

    ab_seq = H_seq + L_seq

    # IMGT categories (H then L), aligned with ab_seq
    imgt_H = imgt_categories_for_mapping(H_map, "H")
    imgt_L = imgt_categories_for_mapping(L_map, "L")
    imgt_cat = imgt_H + imgt_L

    if len(imgt_cat) != len(ab_seq):
        raise RuntimeError(f"{pdb_id}: imgt length mismatch (imgt={len(imgt_cat)}, ab_seq={len(ab_seq)})")

    # Antigen sequence (concat in given order)
    ag_seq = "".join([extract_atom_seq(model[cid]) for cid in ag_chains])

    return pdb_id, ab_seq, ag_seq, imgt_cat


def write_outputs(
    out_dir: str,
    pdb_id: str,
    ab_seq: str,
    ag_seq: str,
    imgt_cat: List[int],
):
    os.makedirs(out_dir, exist_ok=True)
    prefix = pdb_id

    out_ab_fa = os.path.join(out_dir, f"{prefix}_antibody.fasta")
    out_ag_fa = os.path.join(out_dir, f"{prefix}_antigen.fasta")
    out_imgt_txt = os.path.join(out_dir, f"{prefix}_antibody_imgt.txt")

    with open(out_ab_fa, "w") as f:
        f.write(f">{pdb_id}\n{ab_seq}\n")

    with open(out_ag_fa, "w") as f:
        f.write(f">{pdb_id}\n{ag_seq}\n")

    # One line, space-separated ints (0..6), aligned with antibody.fasta sequence
    with open(out_imgt_txt, "w") as f:
        f.write("".join(map(str, imgt_cat)) + "\n")

    print("Wrote:")
    print(f"  {out_ab_fa}")
    print(f"  {out_ag_fa}")
    print(f"  {out_imgt_txt}")


def build_argparser():
    p = argparse.ArgumentParser(
        description="Generate antibody/antigen FASTA and IMGT (0~6) categories from a single IMGT-renumbered PDB."
    )
    p.add_argument("--pdb_name", required=True, help="Input PDB path (e.g., /path/to/xxxx_imgt.pdb)")
    p.add_argument("--VH_chain", required=True, help="VH chain ID (e.g., H)")
    p.add_argument("--VL_chain", required=True, help="VL chain ID (e.g., L)")
    p.add_argument(
        "--antigen_chain",
        required=True,
        help="Antigen chain ID(s). Use 'A' or 'A;B' (semicolon-separated; need quotes).",
    )
    p.add_argument("--output", required=True, help="Output directory")
    return p


def main():
    args = build_argparser().parse_args()

    pdb_path = args.pdb_name
    vh = args.VH_chain
    vl = args.VL_chain
    ag_chains = parse_chain_list(args.antigen_chain)

    pdb_id, ab_seq, ag_seq, imgt_cat = process_single_pdb(
        pdb_path=pdb_path,
        vh_chain=vh,
        vl_chain=vl,
        ag_chains=ag_chains,
    )

    write_outputs(
        out_dir=args.output,
        pdb_id=pdb_id,
        ab_seq=ab_seq,
        ag_seq=ag_seq,
        imgt_cat=imgt_cat,
    )


if __name__ == "__main__":
    main()
