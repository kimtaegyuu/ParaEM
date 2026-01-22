import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CrossAttentionEM_MultiHeadCDR, predict_hatp


def make_dummy_lists(
    n_samples: int,
    embed_dim: int,
    ab_len_range: tuple,
    ag_len_range: tuple,
    pos_rate: float,
    imgt_vocab: int,
):
    """
    Returns:
      ab_embs:  list[Tensor(L_ab, D)]
      ag_embs:  list[Tensor(L_ag, D)]
      labels:   list[list[int]] (0/1)
      imgt:     list[list[int]] (0..imgt_vocab-1)
    """
    ab_embs, ag_embs, labels, imgt = [], [], [], []

    for _ in range(n_samples):
        L_ab = random.randint(ab_len_range[0], ab_len_range[1])
        L_ag = random.randint(ag_len_range[0], ag_len_range[1])

        ab = torch.randn(L_ab, embed_dim)
        ag = torch.randn(L_ag, embed_dim)

        y = torch.bernoulli(torch.full((L_ab,), float(pos_rate))).to(torch.int64).tolist()
        m = torch.randint(0, imgt_vocab, (L_ab,), dtype=torch.int64).tolist()

        ab_embs.append(ab)
        ag_embs.append(ag)
        labels.append(y)
        imgt.append(m)

    return ab_embs, ag_embs, labels, imgt


def evaluate_macro(
    model,
    ab_embs, ag_embs, labels, imgt_masks,
    device=None,
    ap_no_pos_value=0.0,
    skip_no_pos=False,
    skip_single_class_for_auroc=True,
):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ap_list = []
    auc_list = []

    with torch.no_grad():
        for ab_emb, ag_emb, y, imgt in zip(ab_embs, ag_embs, labels, imgt_masks):
            y_true = np.asarray(y, dtype=np.float32)
            ab = ab_emb.to(device)
            ag = ag_emb.to(device)
            imgt_cat = torch.as_tensor(imgt, dtype=torch.long, device=device)

            hat_p = predict_hatp(model, ab, ag, imgt_cat).detach().cpu().numpy()

            n_pos = int(y_true.sum())
            if n_pos == 0:
                if not skip_no_pos:
                    ap_list.append(float(ap_no_pos_value))
            else:
                ap_list.append(float(average_precision_score(y_true, hat_p)))

            if skip_single_class_for_auroc:
                if n_pos == 0 or n_pos == len(y_true):
                    pass
                else:
                    auc_list.append(float(roc_auc_score(y_true, hat_p)))
            else:
                try:
                    auc_list.append(float(roc_auc_score(y_true, hat_p)))
                except Exception:
                    pass

    return {
        "macro_auc_pr_mean": float(np.mean(ap_list)) if len(ap_list) else None,
        "macro_auc_pr_std": float(np.std(ap_list)) if len(ap_list) else None,
        "macro_auc_roc_mean": float(np.mean(auc_list)) if len(auc_list) else None,
        "macro_auc_roc_std": float(np.std(auc_list)) if len(auc_list) else None,
        "n_used_ap": int(len(ap_list)),
        "n_used_auroc": int(len(auc_list)),
    }


def train_em(
    model,
    train_ab_embs, train_ag_embs, train_labels, train_imgt,
    val_ab_embs, val_ag_embs, val_labels, val_imgt,
    em_iters=15, m_epochs=2, lr=1e-5,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    N = len(train_ab_embs)

    for it in range(1, em_iters + 1):
        # ---------- E-step ----------
        model.eval()
        all_q = []
        with torch.no_grad():
            for ab_emb, ag_emb, y, imgt in zip(train_ab_embs, train_ag_embs, train_labels, train_imgt):
                ab = ab_emb.to(device)
                ag = ag_emb.to(device)
                y_true = torch.as_tensor(y, dtype=torch.float, device=device)
                imgt_cat = torch.as_tensor(imgt, dtype=torch.long, device=device)

                y_mat = y_true.unsqueeze(1)

                S = model(ab, ag, imgt_cat)
                pi = F.softmax(S, dim=1)
                p_pos = torch.sigmoid(S)

                like = (p_pos ** y_mat) * ((1.0 - p_pos) ** (1.0 - y_mat))
                q = pi * like
                q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-12)
                all_q.append(q)

        # ---------- M-step ----------
        model.train()
        for me in range(1, m_epochs + 1):
            total_loss = 0.0

            for (ab_emb, ag_emb, y, imgt), q in zip(
                zip(train_ab_embs, train_ag_embs, train_labels, train_imgt), all_q
            ):
                ab = ab_emb.to(device)
                ag = ag_emb.to(device)
                y_true = torch.as_tensor(y, dtype=torch.float, device=device)
                imgt_cat = torch.as_tensor(imgt, dtype=torch.long, device=device)

                y_mat = y_true.unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)

                S = model(ab, ag, imgt_cat)
                p_pos = torch.sigmoid(S)

                eps = 1e-12
                label_ce = -(
                    y_mat * (p_pos.clamp(min=eps)).log()
                    + (1.0 - y_mat) * (1.0 - p_pos).clamp(min=eps).log()
                )
                L_label = (q.detach() * label_ce).sum(dim=1).mean()

                log_pi = F.log_softmax(S, dim=1)
                L_align = -(q.detach() * log_pi).sum(dim=1).mean()

                loss = L_label + L_align
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            print(f"EM {it}/{em_iters} | M-epoch {me}/{m_epochs} | avg_loss={total_loss / N:.6f}")

        metrics = evaluate_macro(model, val_ab_embs, val_ag_embs, val_labels, val_imgt, device=device)
        print(
            f"[VAL] after EM {it}: macro-AUPRC={metrics['macro_auc_pr_mean']:.6f} "
            f"(std={metrics['macro_auc_pr_std']:.6f}, n={metrics['n_used_ap']}), "
            f"macro-AUROC={metrics['macro_auc_roc_mean'] if metrics['macro_auc_roc_mean'] is None else round(metrics['macro_auc_roc_mean'], 6)}"
        )

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dummy train
    train_ab_embs, train_ag_embs, train_labels, train_imgt = make_dummy_lists(
        n_samples=40,
        embed_dim=1536,
        ab_len_range=(220, 260),
        ag_len_range=(150, 230),
        pos_rate=0.12,
        imgt_vocab=7,
    )

    # dummy val
    val_ab_embs, val_ag_embs, val_labels, val_imgt = make_dummy_lists(
        n_samples=15,
        embed_dim=1536,
        ab_len_range=(220, 260),
        ag_len_range=(150, 230),
        pos_rate=0.12,
        imgt_vocab=7,
    )

    model = CrossAttentionEM_MultiHeadCDR(
        embed_dim=1536,
        imgt_vocab=7,
        cdr_emb_dim=32,
        proj_dim=256,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    model = train_em(
        model,
        train_ab_embs, train_ag_embs, train_labels, train_imgt,
        val_ab_embs, val_ag_embs, val_labels, val_imgt,
        em_iters=15, m_epochs=2, lr=1e-5,
        device=device,
    )

    final_metrics = evaluate_macro(model, val_ab_embs, val_ag_embs, val_labels, val_imgt, device=device)
    print("[FINAL VAL macro]", final_metrics)


if __name__ == "__main__":
    main()
