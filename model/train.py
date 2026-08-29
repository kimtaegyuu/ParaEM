import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CrossAttentionEM_MultiHeadCDR, predict_hatp


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_digit_lines(path):
    """Each line is one sample: e.g., 000101 or 000111222."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append([int(ch) for ch in line])
    return data


def to_tensor_list(x):
    return [
        item.float().cpu() if isinstance(item, torch.Tensor)
        else torch.tensor(item, dtype=torch.float32)
        for item in x
    ]


def load_dataset(ab_path, ag_path, label_path, imgt_path):
    ab_embs = to_tensor_list(load_pickle(ab_path))
    ag_embs = to_tensor_list(load_pickle(ag_path))
    labels = load_digit_lines(label_path)
    imgt = load_digit_lines(imgt_path)

    assert len(ab_embs) == len(ag_embs) == len(labels) == len(imgt), \
        "Number of samples does not match."

    for i, (ab, y, m) in enumerate(zip(ab_embs, labels, imgt)):
        assert len(y) == ab.shape[0], f"Label length mismatch at sample {i}"
        assert len(m) == ab.shape[0], f"IMGT length mismatch at sample {i}"

    return ab_embs, ag_embs, labels, imgt


def evaluate_macro(model, ab_embs, ag_embs, labels, imgt_masks, device):
    model.eval()
    ap_list, auc_list = [], []

    with torch.no_grad():
        for ab_emb, ag_emb, y, imgt in zip(ab_embs, ag_embs, labels, imgt_masks):
            y_true = np.asarray(y, dtype=np.float32)

            ab = ab_emb.to(device)
            ag = ag_emb.to(device)
            imgt_cat = torch.tensor(imgt, dtype=torch.long, device=device)

            pred = predict_hatp(model, ab, ag, imgt_cat).detach().cpu().numpy()

            n_pos = int(y_true.sum())

            if n_pos > 0:
                ap_list.append(float(average_precision_score(y_true, pred)))
            else:
                ap_list.append(0.0)

            if 0 < n_pos < len(y_true):
                auc_list.append(float(roc_auc_score(y_true, pred)))

    return {
        "macro_auc_pr_mean": float(np.mean(ap_list)) if ap_list else None,
        "macro_auc_pr_std": float(np.std(ap_list)) if ap_list else None,
        "macro_auc_roc_mean": float(np.mean(auc_list)) if auc_list else None,
        "macro_auc_roc_std": float(np.std(auc_list)) if auc_list else None,
        "n_used_ap": len(ap_list),
        "n_used_auroc": len(auc_list),
    }


def train_em(
    model,
    train_ab, train_ag, train_y, train_imgt,
    val_ab, val_ag, val_y, val_imgt,
    em_iters,
    m_epochs,
    lr,
    patience,
    save_path,
    device,
):
    model.to(device)
    opt = Adam(model.parameters(), lr=lr)

    best_ap = -1.0
    best_iter = 0
    no_improve = 0

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for it in range(1, em_iters + 1):
        # E-step
        model.eval()
        q_list = []

        with torch.no_grad():
            for ab_emb, ag_emb, y, imgt in zip(train_ab, train_ag, train_y, train_imgt):
                ab = ab_emb.to(device)
                ag = ag_emb.to(device)
                y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
                imgt_t = torch.tensor(imgt, dtype=torch.long, device=device)

                S = model(ab, ag, imgt_t)
                pi = F.softmax(S, dim=1)
                p = torch.sigmoid(S)

                like = (p ** y_t) * ((1.0 - p) ** (1.0 - y_t))
                q = pi * like
                q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-12)

                q_list.append(q.detach())

        # M-step
        model.train()
        for me in range(1, m_epochs + 1):
            total_loss = 0.0

            for ab_emb, ag_emb, y, imgt, q in zip(train_ab, train_ag, train_y, train_imgt, q_list):
                ab = ab_emb.to(device)
                ag = ag_emb.to(device)
                y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
                imgt_t = torch.tensor(imgt, dtype=torch.long, device=device)

                opt.zero_grad()

                S = model(ab, ag, imgt_t)
                p = torch.sigmoid(S)

                eps = 1e-12
                label_loss = -(
                    y_t * p.clamp(min=eps).log()
                    + (1.0 - y_t) * (1.0 - p).clamp(min=eps).log()
                )
                label_loss = (q * label_loss).sum(dim=1).mean()

                align_loss = -(q * F.log_softmax(S, dim=1)).sum(dim=1).mean()

                loss = label_loss + align_loss
                loss.backward()
                opt.step()

                total_loss += loss.item()

            print(f"EM {it}/{em_iters} | M-epoch {me}/{m_epochs} | loss={total_loss / len(train_ab):.6f}")

        # Validation
        val_metrics = evaluate_macro(model, val_ab, val_ag, val_y, val_imgt, device)
        val_ap = val_metrics["macro_auc_pr_mean"]

        print(f"[VAL] EM {it} | AUPRC={val_ap:.6f} | AUROC={val_metrics['macro_auc_roc_mean']}")

        if val_ap > best_ap:
            best_ap = val_ap
            best_iter = it
            no_improve = 0

            torch.save(model.state_dict(), save_path)
            print(f"[SAVE] best model saved: {save_path}")

        else:
            no_improve += 1
            print(f"[EARLY STOP] no improvement {no_improve}/{patience}")

            if no_improve >= patience:
                print(f"[STOP] best EM iter={best_iter}, best val AUPRC={best_ap:.6f}")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_antibody", required=True)
    p.add_argument("--train_antigen", required=True)
    p.add_argument("--train_labels", required=True)
    p.add_argument("--train_imgt", required=True)

    p.add_argument("--valid_antibody", required=True)
    p.add_argument("--valid_antigen", required=True)
    p.add_argument("--valid_labels", required=True)
    p.add_argument("--valid_imgt", required=True)

    p.add_argument("--test_antibody", default=None)
    p.add_argument("--test_antigen", default=None)
    p.add_argument("--test_labels", default=None)
    p.add_argument("--test_imgt", default=None)

    p.add_argument("--embed_dim", type=int, default=1536)
    p.add_argument("--imgt_vocab", type=int, default=7)
    p.add_argument("--cdr_emb_dim", type=int, default=32)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--em_iters", type=int, default=50)
    p.add_argument("--m_epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--save_path", default="model_weight/best_paraem.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)

    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] device={device}")

    train_data = load_dataset(args.train_antibody, args.train_antigen, args.train_labels, args.train_imgt)
    val_data = load_dataset(args.valid_antibody, args.valid_antigen, args.valid_labels, args.valid_imgt)

    model = CrossAttentionEM_MultiHeadCDR(
        embed_dim=args.embed_dim,
        imgt_vocab=args.imgt_vocab,
        cdr_emb_dim=args.cdr_emb_dim,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    model = train_em(
        model,
        *train_data,
        *val_data,
        em_iters=args.em_iters,
        m_epochs=args.m_epochs,
        lr=args.lr,
        patience=args.patience,
        save_path=args.save_path,
        device=device,
    )

    val_metrics = evaluate_macro(model, *val_data, device=device)
    print("[FINAL VAL]", val_metrics)

    if all([args.test_antibody, args.test_antigen, args.test_labels, args.test_imgt]):
        test_data = load_dataset(args.test_antibody, args.test_antigen, args.test_labels, args.test_imgt)
        test_metrics = evaluate_macro(model, *test_data, device=device)
        print("[FINAL TEST]", test_metrics)


if __name__ == "__main__":
    main()