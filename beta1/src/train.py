"""
Training 3 kelas fundus: 0 = normal, 1 = katarak, 2 = penyakit lain.

Jalankan dari folder Cataract (bukan dari src):
    python -m src.train --epochs 2 --warmup 1            # tes cepat
    python -m src.train                                  # training penuh EfficientNet-B0
    python -m src.train --model resnet50 --name resnet_3c
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, f1_score,
)

from src.data.dataset import FundusDataset, CLASS_NAMES, NUM_CLASSES
from src.data.text_rules import split_tokens, token_category
from src.transforms import get_transforms, MEAN, STD

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficientnet_b0", help="nama model timm")
    p.add_argument("--name", default="efficientnet_3c", help="folder output di models/final/")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3, help="epoch awal: hanya latih head")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_full", type=float, default=5e-5)
    p.add_argument("--wpower", type=float, default=0.5,
                   help="bobot kelas = (N/(K*n_kelas)) ** wpower. 1.0 penuh, 0.5 akar, 0 mati")
    p.add_argument("--smooth", type=float, default=0.05, help="label smoothing")
    p.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    p.add_argument("--drop", type=float, default=0.3)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--aux_binary", action="store_true",
                   help="tambah kepala biner normal-vs-bukan-normal (anchor loss) "
                        "untuk memaksa representasi normal lebih tegas")
    p.add_argument("--aux_weight", type=float, default=0.3,
                   help="bobot suku loss biner terhadap loss 3-kelas")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class MultiHeadModel(nn.Module):
    """Backbone timm + kepala 3-kelas, opsional + kepala biner
    normal (0) vs bukan-normal (1=katarak, 2=other). Kedua kepala berbagi
    fitur yang sama dari backbone; kepala biner hanya dipakai sebagai sinyal
    supervisi tambahan saat training (anchor), keputusan akhir tetap dari
    kepala 3-kelas (argmax)."""

    def __init__(self, arch, pretrained, drop_rate, drop_path_rate, use_aux):
        super().__init__()
        self.backbone = timm.create_model(
            arch, pretrained=pretrained, num_classes=0,   # 0 = hanya fitur (sudah di-pool)
            drop_rate=drop_rate, drop_path_rate=drop_path_rate,
        )
        nf = self.backbone.num_features
        self.fc3 = nn.Linear(nf, NUM_CLASSES)
        self.use_aux = use_aux
        if use_aux:
            self.fc_bin = nn.Linear(nf, 1)

    def forward(self, x):
        feat = self.backbone(x)
        out3 = self.fc3(feat)
        out_bin = self.fc_bin(feat).squeeze(1) if self.use_aux else None
        return out3, out_bin

    def head_parameters(self):
        params = list(self.fc3.parameters())
        if self.use_aux:
            params += list(self.fc_bin.parameters())
        return params


def macro_ap(y, p):
    """Rata-rata PR-AUC (average precision) satu-lawan-sisa untuk tiap kelas."""
    Y = np.eye(NUM_CLASSES)[y]
    return float(np.mean([average_precision_score(Y[:, k], p[:, k]) for k in range(NUM_CLASSES)]))


def per_class_recall(y, pred):
    cm = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))
    return cm.diagonal() / np.maximum(cm.sum(1), 1)


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda"
    torch.backends.cudnn.benchmark = True
    print("Device:", dev)
    if dev == "cuda":
        print("GPU   :", torch.cuda.get_device_name(0))

    # ---------- Data ----------
    tf_train, tf_eval = get_transforms(a.size)
    train_ds = FundusDataset("train", tf_train)
    val_ds = FundusDataset("val", tf_eval)
    test_ds = FundusDataset("test", tf_eval)

    def make_loader(ds, shuffle):
        kw = dict(batch_size=a.bs, shuffle=shuffle,
                  num_workers=a.workers, pin_memory=(dev == "cuda"))
        if a.workers > 0:
            kw["persistent_workers"] = True
        return DataLoader(ds, **kw)

    train_dl = make_loader(train_ds, True)
    val_dl = make_loader(val_ds, False)
    test_dl = make_loader(test_ds, False)
    for nm, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        print(f"{nm:5s} {len(ds):5d} | " +
              " | ".join(f"{c} {n}" for c, n in zip(CLASS_NAMES, ds.class_counts())))

    # ---------- Model ----------
    model = MultiHeadModel(
        a.model, pretrained=True, drop_rate=a.drop, drop_path_rate=a.drop_path,
        use_aux=a.aux_binary,
    ).to(dev)

    cw = train_ds.class_weights(a.wpower).to(dev)
    print("bobot kelas:", {c: round(float(w), 2) for c, w in zip(CLASS_NAMES, cw)},
          f"| smoothing {a.smooth} | wd {a.wd}")
    if a.aux_binary:
        print(f"kepala biner (normal vs bukan-normal) AKTIF | bobot aux {a.aux_weight}")

    def train_loss(logits3, logit_bin, y):
        loss = F.cross_entropy(logits3, y, weight=cw, label_smoothing=a.smooth)
        if a.aux_binary:
            y_bin = (y != 0).float()          # 0 = normal, 1 = katarak atau other
            loss_bin = F.binary_cross_entropy_with_logits(logit_bin, y_bin)
            loss = loss + a.aux_weight * loss_bin
        return loss

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    head_ids = {id(p) for p in model.head_parameters()}

    def set_backbone(trainable):
        for prm in model.parameters():
            prm.requires_grad = True if trainable else (id(prm) in head_ids)

    @torch.no_grad()
    def predict(dl, desc):
        model.eval()
        logits, ys = [], []
        for x, y in tqdm(dl, desc=desc, leave=False):
            with torch.autocast(dev, enabled=use_amp):
                out3, _ = model(x.to(dev, non_blocking=True))
            logits.append(out3.float().cpu())
            ys.append(y)
        logits, y = torch.cat(logits), torch.cat(ys)
        loss = F.cross_entropy(logits, y).item()          # polos, tanpa bobot/aux
        return torch.softmax(logits, 1).numpy(), y.numpy(), loss

    out = Path("models/final") / a.name
    out.mkdir(parents=True, exist_ok=True)

    # ---------- Training ----------
    set_backbone(False)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=a.lr_head, weight_decay=a.wd)
    best, bad = -1.0, 0
    history = []

    for ep in range(1, a.epochs + 1):
        if ep == a.warmup + 1:
            print("--> Tahap 2: fine-tuning semua layer")
            set_backbone(True)
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr_full, weight_decay=a.wd)

        model.train()
        total = 0.0
        for x, y in tqdm(train_dl, desc=f"ep {ep:02d}/{a.epochs}"):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=use_amp):
                logits3, logit_bin = model(x)
            loss = train_loss(logits3.float(), logit_bin.float() if logit_bin is not None else None, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()

        pv, yv, vloss = predict(val_dl, "val")
        pred_v = pv.argmax(1)
        score = macro_ap(yv, pv)                               # kriteria model terbaik
        mf1 = f1_score(yv, pred_v, average="macro")
        rec = per_class_recall(yv, pred_v)
        history.append({"epoch": ep, "train_loss": total / len(train_dl), "val_loss": vloss,
                        "val_macro_ap": score, "val_macro_f1": float(mf1),
                        "val_recall": {c: float(r) for c, r in zip(CLASS_NAMES, rec)}})
        print(f"ep {ep:02d} train_loss {total/len(train_dl):.4f} | val_loss {vloss:.4f} | "
              f"macroAP {score:.3f} macroF1 {mf1:.3f} | recall N/C/O "
              f"{rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f}")

        if score > best:
            best, bad = score, 0
            torch.save(model.state_dict(), out / "model.pt")
            print("    model terbaik disimpan")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping")
                break

    # ---------- Evaluasi akhir (test dipakai sekali) ----------
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    pv, yv, _ = predict(val_dl, "val")
    pt, yt, _ = predict(test_dl, "test")
    pred = pt.argmax(1)

    cm = confusion_matrix(yt, pred, labels=list(range(NUM_CLASSES)))
    rep = classification_report(yt, pred, target_names=CLASS_NAMES, digits=3,
                                output_dict=True, zero_division=0)
    print("\n=== HASIL TEST ===")
    print(classification_report(yt, pred, target_names=CLASS_NAMES, digits=3, zero_division=0))
    print("Confusion matrix (baris = sebenarnya, kolom = prediksi)")
    print(pd.DataFrame(cm, index=[f"asli {c}" for c in CLASS_NAMES],
                       columns=[f"pred {c}" for c in CLASS_NAMES]).to_string())

    res = {
        "macro_f1": float(f1_score(yt, pred, average="macro")),
        "balanced_accuracy": float(per_class_recall(yt, pred).mean()),
        "macro_ovr_auc": float(roc_auc_score(yt, pt, multi_class="ovr", average="macro")),
        "macro_ap": macro_ap(yt, pt),
        "cataract_vs_rest_auc": float(roc_auc_score(yt == 1, pt[:, 1])),
        "cataract_vs_rest_ap": float(average_precision_score(yt == 1, pt[:, 1])),
        "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score")}
                      for c in CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
    }
    print(f"\nmacro-F1 {res['macro_f1']:.3f} | balanced acc {res['balanced_accuracy']:.3f} | "
          f"macro OVR-AUC {res['macro_ovr_auc']:.3f}")
    print(f"Katarak vs sisanya: AUC {res['cataract_vs_rest_auc']:.3f} | "
          f"PR-AUC {res['cataract_vs_rest_ap']:.3f}")

    # ---------- Rincian kelas "other" per jenis penyakit ----------
    te = test_ds.df.copy()
    te["pred"] = pred
    oth = te[te.label == 2].copy()

    def subtype(t):
        toks = split_tokens(t)
        return token_category(toks[0]) if toks else "?"

    oth["jenis"] = oth.keywords.apply(subtype)
    sub = (oth.assign(benar=(oth.pred == 2)).groupby("jenis").benar
           .agg(["size", "mean"]).rename(columns={"size": "jumlah", "mean": "recall_other"}))
    print("\nRecall kelas 'other' per jenis (D=diabetes, G=glaukoma, A=AMD, H=hipertensi, "
          "M=miopia, O=lain):")
    print(sub.round(3).to_string())
    res["other_recall_by_type"] = {k: float(v) for k, v in sub.recall_other.items()}

    # ---------- Simpan ----------
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    rows = []
    for split, ds, p in [("val", val_ds, pv), ("test", test_ds, pt)]:
        d = ds.df[["path", "label", "keywords"]].copy()
        d["split"] = split
        for k, c in enumerate(CLASS_NAMES):
            d[f"p_{c}"] = p[:, k]
        d["pred"] = p.argmax(1)
        rows.append(d)
    pd.concat(rows).to_csv(f"outputs/reports/{a.name}_probs.csv", index=False)

    dump = lambda o: o.item() if hasattr(o, "item") else str(o)
    with open(out / "meta.json", "w") as f:
        json.dump({
            "name": a.name, "arch": a.model, "image_size": a.size,
            "mean": MEAN, "std": STD, "num_classes": NUM_CLASSES,
            "label_map": {str(i): c for i, c in enumerate(CLASS_NAMES)},
            "decision": "argmax", "seed": a.seed,
            "config": {"wpower": a.wpower, "smooth": a.smooth, "wd": a.wd,
                       "drop": a.drop, "drop_path": a.drop_path, "lr_full": a.lr_full,
                       "aux_binary": a.aux_binary, "aux_weight": a.aux_weight},
            "val_macro_ap_best": float(best),
            **res,
        }, f, indent=2, default=dump)
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=dump)
    print("\nTersimpan di", out, "dan outputs/reports/" + a.name + "_probs.csv")


if __name__ == "__main__":   # wajib di Windows agar num_workers > 0 tidak error
    main()