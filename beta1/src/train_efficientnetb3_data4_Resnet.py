"""
ResNet50 training on the Kaggle fundus dataset that is ALREADY split:

    dataset/
      train/ cataract, diabetic_retinopathy, glaucoma, normal
      val/   cataract, diabetic_retinopathy, glaucoma, normal
      test/  cataract, diabetic_retinopathy, glaucoma, normal

Same concept / same pattern as train_efficientnetb3_data4_CL3.py:
- PyTorch + timm EfficientNetB3, .pth output
- SquarePad -> Resize -> RetinaCLAHE (deterministic, identical in train & eval)
- Internal 4-class training (Normal / Cataract / Other-DR / Other-Glaucoma),
  collapsed to 3 classes (Normal / Cataract / Other) at evaluation
  (P(Other) = P(DR) + P(Glaucoma)). Set USE_FINE_LABELS = False for plain 3-class training.
- Two-stage training: head only -> unfreeze last N blocks
- AdamW, discriminative LR, cosine schedule with warmup, EMA weights,
  frozen BatchNorm stats, gradient accumulation, grad clipping, NaN-safe loop
- Anti-overfitting: drop_path, head dropout, light MixUp, RandomErasing,
  label smoothing, loss weights (no undersampling)
- Diagnostics: per-class recall/precision per epoch, 3-class and 4-class
  confusion matrices, misclassified_test.csv, 2x2 training curves

WHAT CHANGED vs the CL3 script:
- No more scanning one folder + StratifiedGroupKFold split. The dataset already
  provides train/val/test, so each split is read directly from its own folder.
- No patient-id grouping (this Kaggle set has no left/right pairing info in the
  file names), so the group-leak asserts are removed.
- No undersampling / water_fill (it was not used in main() anyway).

Run from the project root (CopyFileDataset):
    python src/train_resnet50_kaggle4.py
"""
import copy
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, f1_score,
)
import matplotlib.pyplot as plt

# ==========================================
# 0. Configuration
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent.parent      # CopyFileDataset/
ROOT_DIR = PROJECT_DIR / "dataset"                        # contains train/ val/ test/

# 3-class evaluation (same as before)
CLASS_NAMES = ["Normal", "Cataract", "Other"]
# folder name (lowercase) -> (coarse 3-class id, subtype)
FOLDER_MAP = {
    "normal":               (0, "normal"),
    "cataract":             (1, "cataract"),
    "diabetic_retinopathy": (2, "diabetic_retinopathy"),
    "glaucoma":             (2, "glaucoma"),
}

ARCH = "resnet50"
PRETRAINED_NAME = "resnet50.tv_in1k"   # timm weight tag; falls back to ARCH if it fails to load
HEAD_ATTR = "fc"                       # head attribute name in timm ResNet
IMG_SIZE = 300          # same as the EfficientNet run (identical CLAHE pipeline)
BATCH_SIZE = 12         # reduce if OOM
SEED = 42

# Two-stage training: head only -> unfreeze part of the backbone -> fine-tune
WARMUP_EPOCHS = 5         # stage 1: backbone frozen, only the head is trained
MAX_EPOCHS = 36           # cosine schedule FINISHES here
PATIENCE = 12             # early stopping on val_macro_f1 (EMA), counted only in stage 2
MIN_DELTA = 1e-3
LR_HEAD = 3e-4
LR_HEAD_STAGE2 = 1e-4
LR_FINETUNE = 5e-5        # backbone LR (discriminative LR: backbone < head)
WEIGHT_DECAY = 1e-2       # AdamW (decoupled)
LABEL_SMOOTHING = 0.1
UNFREEZE_LAST_N_BLOCKS = 3   # of the 16 ResNet50 bottlenecks: the last 3 = layer4
ACCUM_STEPS = 2           # effective batch 12x2=24
EMA_DECAY = 0.998
FREEZE_BN_STATS = True    # backbone BatchNorm always in eval mode

# CLAHE (deterministic preprocessing, identical in train & eval)
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
RETINA_THRESH = 10        # gray <= this = black background outside the retina

# ---- Anti-overfitting ----
USE_FINE_LABELS = True        # train 4 classes (Other split into DR / Glaucoma), evaluate 3 classes
CLASS_WEIGHT_MODE = "coarse_balanced"   # "coarse_balanced" | "fine_balanced" | "none"
DROP_PATH_RATE = 0.1          # ResNet50 has only 16 blocks -> 0.05-0.1 is enough
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.4
ERASE_PROB = 0.25

RUN_NAME = "resnet50_kaggle4_CL3"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "reports3" / RUN_NAME
MODEL_DIR = PROJECT_DIR / "models" / "final3" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==========================================
# 1. Transform
# ==========================================
class SquarePad:
    """Black padding to a square shape BEFORE resize so the round retina
    is not squashed into an oval."""
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        pad_w, pad_h = s - w, s - h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return ImageOps.expand(img, padding, fill=(0, 0, 0))


def retina_mask(rgb: np.ndarray, thresh: int = RETINA_THRESH):
    """Boolean mask of the retina area (True = retina) from a uint8 RGB image.
    Largest contour, hole-filled, slightly eroded. None if no retina detected."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (gray > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [biggest], -1, 1, thickness=cv2.FILLED)
    filled = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=2)
    if filled.mean() < 0.05:
        return None
    return filled.astype(bool)


class RetinaCLAHE:
    """CLAHE on the L channel (LAB), retina area only. DETERMINISTIC ->
    must be identical in train, val, test and serving."""
    def __init__(self, clip_limit=CLAHE_CLIP_LIMIT, tile_grid=CLAHE_TILE_GRID,
                 thresh=RETINA_THRESH):
        self.clip_limit = clip_limit
        self.tile_grid = tuple(tile_grid)
        self.thresh = thresh

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"))
        mask = retina_mask(rgb, self.thresh)
        if mask is None:
            return img
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L = lab[..., 0].copy()
        L[~mask] = int(L[mask].mean())         # neutralize the background before CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        lab[..., 0] = clahe.apply(L)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        out[~mask] = rgb[~mask]                # restore the black area outside the retina
        return Image.fromarray(out)


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Single source of truth for train & eval: SquarePad -> Resize -> CLAHE
PREPROCESS_STEPS = [SquarePad(), transforms.Resize((IMG_SIZE, IMG_SIZE))]
if USE_CLAHE:
    PREPROCESS_STEPS.append(RetinaCLAHE())

train_tf = transforms.Compose([
    *PREPROCESS_STEPS,
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # hue is NOT jittered (retina color carries clinical information)
    transforms.RandomAffine(degrees=25, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8, fill=0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=ERASE_PROB, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
])
eval_tf = transforms.Compose([
    *PREPROCESS_STEPS,          # IDENTICAL to train_tf preprocessing
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ==========================================
# 2. Dataset (pre-split: dataset/<split>/<class>/*.jpg)
# ==========================================
IMG_EXT = (".jpg", ".jpeg", ".png")


def scan_split(root: Path, split: str) -> pd.DataFrame:
    """Scan root/<split>/<cataract|diabetic_retinopathy|glaucoma|normal>/*.
    Columns: path, label (3-class id), subtype (folder name)."""
    split_dir = root / split
    if not split_dir.is_dir():
        raise RuntimeError(f"Split folder not found: {split_dir}")
    rows = []
    for folder in sorted(split_dir.iterdir()):
        if not folder.is_dir():
            continue
        key = folder.name.strip().lower()
        if key not in FOLDER_MAP:
            print(f"  (skipping unknown folder: {split}/{folder.name})")
            continue
        label, subtype = FOLDER_MAP[key]
        files = sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
        for f in files:
            rows.append({"path": str(f), "label": label, "subtype": subtype})
        print(f"  {split:5s}/{folder.name:22s} -> {len(files):5d} images "
              f"(label {label} = {CLASS_NAMES[label]})")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found in {split_dir}.")
    return df


def build_fine_labels(dfs: dict):
    """Add the 'fine' column to every split using ONE shared mapping.
    Normal & Cataract = 1 class each; Other is split per subtype (DR, Glaucoma)
    when USE_FINE_LABELS. Returns (dfs, fine_names, f2c)."""
    all_df = pd.concat(dfs.values())
    fine_names, f2c, key2id = [], [], {}
    other = CLASS_NAMES.index("Other")
    for coarse, cname in enumerate(CLASS_NAMES):
        subs = sorted(all_df[all_df.label == coarse].subtype.unique())
        split_it = USE_FINE_LABELS and coarse == other and len(subs) > 1
        for k in (subs if split_it else [None]):
            key2id[(coarse, k)] = len(fine_names)
            fine_names.append(cname if k is None else f"{cname}/{k}")
            f2c.append(coarse)
    out = {}
    for name, d in dfs.items():
        d = d.copy()
        d["fine"] = [key2id.get((c, s_), key2id.get((c, None))) for c, s_ in zip(d.label, d.subtype)]
        out[name] = d.reset_index(drop=True)
    return out, fine_names, np.array(f2c)


class FundusDF(Dataset):
    """Returns (image, 3-class label, fine label)."""
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        return self.transform(img), int(row.label), int(row.fine)


# ==========================================
# 3. Model
# ==========================================
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.5

PRETRAINED_USED = {"name": None}


def is_head_param(name: str) -> bool:
    return name.startswith(HEAD_ATTR + ".")


def get_model(num_out: int):
    """ResNet50 backbone (timm, num_classes=0 -> 2048-d features after global avg pool)
    + new head: Linear(2048,256) -> LayerNorm -> ReLU -> Dropout -> Linear(256,num_out).
    The head is attached to the `fc` attribute so that timm's built-in forward uses it."""
    try:
        m = timm.create_model(PRETRAINED_NAME, pretrained=True, num_classes=0,
                              drop_path_rate=DROP_PATH_RATE)
        PRETRAINED_USED["name"] = PRETRAINED_NAME
    except Exception as e:
        print(f"  Warning: weights '{PRETRAINED_NAME}' failed to load ({type(e).__name__}: {e}); "
              f"falling back to timm default '{ARCH}'.")
        m = timm.create_model(ARCH, pretrained=True, num_classes=0, drop_path_rate=DROP_PATH_RATE)
        PRETRAINED_USED["name"] = ARCH
    in_features = m.num_features
    setattr(m, HEAD_ATTR, nn.Sequential(
        nn.Linear(in_features, HEAD_HIDDEN),
        nn.LayerNorm(HEAD_HIDDEN),   # NOT BatchNorm1d: batch 12 is too small
        nn.ReLU(inplace=True),
        nn.Dropout(HEAD_DROPOUT),
        nn.Linear(HEAD_HIDDEN, num_out),   # num_out = number of FINE classes
    ))
    return m.to(DEVICE)


def _block_key(name: str):
    """'layer4.2.conv3.weight' -> (4, 2); None if not part of a bottleneck (conv1/bn1/fc)."""
    p = name.split(".")
    if len(p) > 2 and p[0].startswith("layer") and p[0][5:].isdigit() and p[1].isdigit():
        return (int(p[0][5:]), int(p[1]))
    return None


def set_trainable(model, mode: str):
    """mode='head': head only (fc). mode='partial': head + the last N bottlenecks
    (N=3 -> layer4). conv1/bn1 and layer1-3 stay frozen."""
    names = [n for n, _ in model.named_parameters()]
    if mode == "head":
        for n, p in model.named_parameters():
            p.requires_grad = is_head_param(n)
    elif mode == "partial":
        blocks = sorted({k for k in map(_block_key, names) if k is not None})
        unfreeze = set(blocks[-UNFREEZE_LAST_N_BLOCKS:])
        for n, p in model.named_parameters():
            p.requires_grad = is_head_param(n) or (_block_key(n) in unfreeze)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  unfreeze: {['layer%d.%d' % k for k in sorted(unfreeze)]} | "
              f"trainable parameters: {n_tr:,}")
    else:
        raise ValueError(mode)


# ==========================================
# 3b. Stability helpers
# ==========================================
def set_bn_eval(model):
    """Keep BatchNorm in eval mode even under model.train() so running stats
    do not drift on small batches + augmentation."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


class ModelEMA:
    """Exponential Moving Average of the weights; the EMA model is evaluated and saved."""
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad = False
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def make_optimizer(model, lr_head, lr_backbone):
    """AdamW with different LRs: head vs backbone."""
    head = [p for n, p in model.named_parameters() if p.requires_grad and is_head_param(n)]
    back = [p for n, p in model.named_parameters() if p.requires_grad and not is_head_param(n)]
    groups = [{"params": head, "lr": lr_head}]
    if back:
        groups.append({"params": back, "lr": lr_backbone})
    return optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def cosine_with_warmup(opt, total_steps, warmup_steps, min_ratio=0.05):
    """Linear warmup then cosine decay, per STEP."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return optim.lr_scheduler.LambdaLR(opt, f)


def loss_from_probs(probs, labels, eps=LABEL_SMOOTHING):
    """Loss (label smoothing, same scale as train loss) from evaluate() probabilities.
    Returns (loss_smoothed, pure_nll)."""
    logp = np.log(np.clip(probs, 1e-8, 1.0))
    nll = -logp[np.arange(len(labels)), labels]
    smooth = -logp.mean(axis=1)
    return float(((1 - eps) * nll + eps * smooth).mean()), float(nll.mean())


# ==========================================
# 4. Evaluation
# ==========================================
@torch.no_grad()
def evaluate(model, loader, f2c_mat, return_fine=False):
    """Model outputs FINE-class logits; probabilities are collapsed to 3 classes
    through f2c_mat [n_fine x 3]. Metrics are computed on the 3 classes."""
    model.eval()
    all_probs, all_labels, all_fine = [], [], []
    n_fine = f2c_mat.shape[0]
    for x, y, yf in loader:
        x = x.to(DEVICE)
        with torch.autocast(DEVICE, enabled=USE_AMP):
            logits = model(x)
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs = np.nan_to_num(probs, nan=1.0 / n_fine)
        all_probs.append(probs)
        all_labels.append(y.numpy())
        all_fine.append(yf.numpy())
    fprobs = np.concatenate(all_probs).astype(np.float64)
    fprobs /= fprobs.sum(1, keepdims=True)
    labels = np.concatenate(all_labels)
    flabels = np.concatenate(all_fine)
    probs = fprobs @ f2c_mat
    preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    auc = None
    try:
        if len(np.unique(labels)) > 1:
            auc = float(roc_auc_score(labels, probs, average="macro", multi_class="ovr"))
    except Exception as e:
        print(f"  AUC warning: {e}")
    if return_fine:
        return acc, mf1, auc, labels, preds, probs, flabels, fprobs
    return acc, mf1, auc, labels, preds, probs


def save_confusion_matrix(y_true, y_pred, title, path):
    """Saves raw-count and per-row-percentage (recall) versions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Confusion matrix (counts) saved: {path}")

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_pct, display_labels=CLASS_NAMES).plot(
        cmap="Blues", ax=ax, values_format=".0%")
    plt.title(title + " (percent per row / recall)")
    plt.tight_layout()
    pct_path = path.parent / (path.stem + "_percent" + path.suffix)
    plt.savefig(pct_path, dpi=200)
    plt.close(fig)
    print(f"  Confusion matrix (percent) saved: {pct_path}")
    return cm


def save_clahe_preview(df: pd.DataFrame, path: Path, n: int = 4):
    """Before/after CLAHE comparison for a visual check."""
    base = transforms.Compose([SquarePad(), transforms.Resize((IMG_SIZE, IMG_SIZE))])
    clahe = RetinaCLAHE()
    sample = df.sample(n=min(n, len(df)), random_state=SEED)
    fig, axes = plt.subplots(2, len(sample), figsize=(3 * len(sample), 6), squeeze=False)
    for j, (_, row) in enumerate(sample.iterrows()):
        im = base(Image.open(row.path).convert("RGB"))
        axes[0, j].imshow(im); axes[0, j].set_title(f"original ({row.subtype})")
        axes[1, j].imshow(clahe(im)); axes[1, j].set_title("CLAHE")
        axes[0, j].axis("off"); axes[1, j].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  CLAHE preview saved: {path}")


# ==========================================
# 5. Main
# ==========================================
def main():
    print("Device:", DEVICE, "| AMP:", USE_AMP)
    if DEVICE == "cuda":
        print("GPU   :", torch.cuda.get_device_name(0))

    print(f"\n== Scanning dataset (pre-split) at {ROOT_DIR} ==")
    raw = {s: scan_split(ROOT_DIR, s) for s in ("train", "val", "test")}
    dfs, FINE_NAMES, F2C = build_fine_labels(raw)
    train_df, val_df, test_df = dfs["train"], dfs["val"], dfs["test"]
    n_fine = len(FINE_NAMES)
    F2C_MAT = np.zeros((n_fine, len(CLASS_NAMES)))
    F2C_MAT[np.arange(n_fine), F2C] = 1.0
    print("\nFine classes (trained):", {i: f"{n} -> {CLASS_NAMES[F2C[i]]}" for i, n in enumerate(FINE_NAMES)})
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:5s} {len(d):5d} images | fine {dict(d.fine.value_counts().sort_index())}")

    if USE_CLAHE:
        save_clahe_preview(train_df, OUTPUT_DIR / "clahe_preview.png")

    # ---- No undersampling: ALL train data is used. Balance comes from the loss weights ----
    fine_counts = train_df.fine.value_counts().reindex(range(n_fine)).fillna(0).to_numpy()
    coarse_counts = np.bincount(F2C, weights=fine_counts, minlength=len(CLASS_NAMES))
    N = fine_counts.sum()
    if CLASS_WEIGHT_MODE == "coarse_balanced":
        w = N / (len(CLASS_NAMES) * coarse_counts[F2C])
    elif CLASS_WEIGHT_MODE == "fine_balanced":
        w = N / (n_fine * fine_counts.clip(min=1))
    else:
        w = np.ones(n_fine)
    w = w / (w * fine_counts).sum() * N          # average weight per sample = 1
    print(f"\n== Loss weights ({CLASS_WEIGHT_MODE}) ==")
    for i, n in enumerate(FINE_NAMES):
        print(f"  {n:30s} n_train={int(fine_counts[i]):5d}  weight={w[i]:.3f}")

    train_dl = DataLoader(FundusDF(train_df, train_tf), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(FundusDF(val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(FundusDF(test_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(n_fine)
    ema = ModelEMA(model, EMA_DECAY)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE),
                                    label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    steps_per_epoch = max(1, math.ceil(len(train_dl) / ACCUM_STEPS))

    print(f"\n== Stage 1: warmup {WARMUP_EPOCHS} epochs (backbone frozen, head only) ==")
    set_trainable(model, "head")
    opt = make_optimizer(model, LR_HEAD, LR_FINETUNE)
    sched = cosine_with_warmup(opt, WARMUP_EPOCHS * steps_per_epoch, warmup_steps=steps_per_epoch)

    history = []
    best_val_mf1, best_epoch, bad_epochs, best_state = -1.0, 0, 0, None
    stage2_started = False

    for ep in range(1, MAX_EPOCHS + 1):
        if ep == WARMUP_EPOCHS + 1 and not stage2_started:
            print(f"\n== Stage 2: unfreeze the last {UNFREEZE_LAST_N_BLOCKS} blocks, fine-tune ==")
            set_trainable(model, "partial")
            opt = make_optimizer(model, LR_HEAD_STAGE2, LR_FINETUNE)
            sched = cosine_with_warmup(opt, (MAX_EPOCHS - WARMUP_EPOCHS) * steps_per_epoch,
                                       warmup_steps=steps_per_epoch)
            stage2_started = True

        model.train()
        if FREEZE_BN_STATS:
            set_bn_eval(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        total, n_ok, n_skip, accum = 0.0, 0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, _, yf) in enumerate(train_dl):
            x, yf = x.to(DEVICE), yf.to(DEVICE)
            use_mix = MIXUP_PROB > 0 and np.random.rand() < MIXUP_PROB
            if use_mix:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1 - lam) * x[perm]
                y_b = yf[perm]
            with torch.autocast(DEVICE, enabled=USE_AMP):
                logits = model(x)
            logits = logits.float()
            if use_mix:
                loss = lam * criterion(logits, yf) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, yf)
            if not torch.isfinite(loss):
                n_skip += 1
                continue
            scaler.scale(loss / ACCUM_STEPS).backward()
            total += loss.item()
            n_ok += 1
            accum += 1
            if accum % ACCUM_STEPS == 0 or i == len(train_dl) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                ema.update(model)
                opt.zero_grad(set_to_none=True)
        if n_skip:
            print(f"  (epoch {ep}: {n_skip} batches skipped due to NaN/inf loss)")
        train_loss = total / max(1, n_ok)

        # Evaluation uses the EMA weights; val loss is in the FINE-class space
        (val_acc, val_mf1, val_auc, v_true, v_pred, _,
         v_ftrue, v_fprob) = evaluate(ema.module, val_dl, F2C_MAT, return_fine=True)
        val_loss, val_nll = loss_from_probs(v_fprob, v_ftrue)
        cm_v = confusion_matrix(v_true, v_pred, labels=list(range(len(CLASS_NAMES))))
        rec = cm_v.diagonal() / cm_v.sum(1).clip(min=1)
        prec = cm_v.diagonal() / cm_v.sum(0).clip(min=1)
        lr_head_now = opt.param_groups[0]["lr"]

        row = {"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "val_nll": val_nll,
               "gap": val_loss - train_loss, "val_acc": val_acc, "val_macro_f1": val_mf1,
               "val_auc": val_auc, "lr_head": lr_head_now}
        for j, c in enumerate(CLASS_NAMES):
            row[f"recall_{c}"], row[f"precision_{c}"] = float(rec[j]), float(prec[j])
        history.append(row)
        print(f"ep {ep:02d} train {train_loss:.4f} | val {val_loss:.4f} | gap {val_loss - train_loss:+.3f} | "
              f"acc {val_acc:.3f} | mf1 {val_mf1:.3f} | recall N/C/O "
              f"{rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f} | lr {lr_head_now:.1e}")

        # Checkpoint & early stopping on val_macro_f1 (EMA); patience counts only in stage 2
        if val_mf1 > best_val_mf1 + MIN_DELTA:
            best_val_mf1, best_epoch, bad_epochs = val_mf1, ep, 0
            best_state = {k: v.detach().clone() for k, v in ema.module.state_dict().items()}
            print(f"    best model saved (val_macro_f1={val_mf1:.3f})")
        elif stage2_started:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"EarlyStopping at epoch {ep} (val_macro_f1 has not improved for {PATIENCE} epochs)")
                break

    # ---------- Final evaluation ----------
    model.load_state_dict(best_state)
    (test_acc, test_mf1, test_auc, y_true, y_pred, y_prob,
     f_true, f_prob) = evaluate(model, test_dl, F2C_MAT, return_fine=True)
    print(f"\n=== TEST RESULTS (3 classes, model from epoch {best_epoch}) ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0))
    cm = save_confusion_matrix(y_true, y_pred, f"{ARCH} | Test Acc {test_acc*100:.1f}%",
                               OUTPUT_DIR / "confusion_matrix.png")
    print(pd.DataFrame(cm, index=[f"true {c}" for c in CLASS_NAMES],
                       columns=[f"pred {c}" for c in CLASS_NAMES]).to_string())
    print(f"\nTest acc {test_acc:.3f} | macro-F1 {test_mf1:.3f} | macro AUC {test_auc}")

    # ---- Fine-class diagnostics ----
    f_pred = f_prob.argmax(1)
    fine_ids = list(range(n_fine))
    print(f"\n=== Per FINE-CLASS report ({n_fine} classes) ===")
    print(classification_report(f_true, f_pred, labels=fine_ids, target_names=FINE_NAMES,
                                digits=3, zero_division=0))
    cm_f = confusion_matrix(f_true, f_pred, labels=fine_ids)
    fine_acc = float((f_pred == f_true).mean())
    print(f"Fine-class ({n_fine}-class) test ACC: {fine_acc:.3f}")
    fig, ax = plt.subplots(figsize=(1.6 * n_fine + 2, 1.4 * n_fine + 2))
    ConfusionMatrixDisplay(cm_f, display_labels=FINE_NAMES).plot(
        cmap="Blues", ax=ax, values_format="d", xticks_rotation=30)
    plt.title(f"{ARCH} | Fine-class confusion matrix (counts)\n"
              f"Fine-class ACC {fine_acc*100:.2f}% | {len(CLASS_NAMES)}-class Test ACC {test_acc*100:.2f}%")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_fine.png", dpi=200)
    plt.close(fig)

    # ---- Misclassified files ----
    mis = test_df.reset_index(drop=True).copy()
    mis["true"] = [CLASS_NAMES[i] for i in y_true]
    mis["pred"] = [CLASS_NAMES[i] for i in y_pred]
    mis["true_fine"] = [FINE_NAMES[i] for i in f_true]
    for j, c in enumerate(CLASS_NAMES):
        mis[f"p_{c}"] = y_prob[:, j].round(4)
    mis = mis[y_true != y_pred].sort_values("true")
    mis.to_csv(OUTPUT_DIR / "misclassified_test.csv", index=False)
    print(f"\nTest misclassifications: {len(mis)} files -> {OUTPUT_DIR / 'misclassified_test.csv'}")
    if len(mis):
        print(pd.crosstab(mis.true_fine, mis.pred).to_string())

    train_acc, train_mf1, *_ = evaluate(
        model, DataLoader(FundusDF(train_df, eval_tf), batch_size=BATCH_SIZE, num_workers=0), F2C_MAT)
    print(f"\nTrain acc (overfitting check) : {train_acc:.3f}  | Test acc: {test_acc:.3f} "
          f"| gap {train_acc - test_acc:+.3f}")
    print(f"Best val acc                  : {max(h['val_acc'] for h in history):.3f}")

    # ---------- Save ----------
    torch.save(model.state_dict(), MODEL_DIR / "model.pth")
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3,
                                output_dict=True, zero_division=0)
    meta = {
        "name": RUN_NAME, "arch": ARCH, "pretrained_weights": PRETRAINED_USED["name"], "image_size": IMG_SIZE,
        "mean": MEAN, "std": STD, "num_classes": len(CLASS_NAMES),
        "num_logits": n_fine, "fine_names": FINE_NAMES, "fine_to_coarse": F2C.tolist(),
        "serving_note": "model.pth has num_logits outputs. softmax -> sum the probabilities of the "
                        "fine classes into 3 classes using fine_to_coarse (Other = DR + Glaucoma).",
        "label_map": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "dataset": "kaggle cataract-dr-normal-glaucoma (pre-split train/val/test)",
        "square_pad": True, "training": "centralized_two_stage",
        "preprocessing_order": ["square_pad", "resize", "clahe" if USE_CLAHE else None],
        "clahe": {"enabled": USE_CLAHE, "clip_limit": CLAHE_CLIP_LIMIT,
                  "tile_grid": list(CLAHE_TILE_GRID), "channel": "L (LAB)",
                  "retina_mask_thresh": RETINA_THRESH,
                  "note": "serving MUST use identical preprocessing"},
        "balance": f"no_undersampling; loss_weights={CLASS_WEIGHT_MODE}",
        "stability": {"optimizer": "AdamW", "weight_decay": WEIGHT_DECAY, "lr_head": LR_HEAD,
                      "lr_head_stage2": LR_HEAD_STAGE2, "lr_backbone": LR_FINETUNE,
                      "schedule": "cosine_with_warmup", "ema_decay": EMA_DECAY,
                      "accum_steps": ACCUM_STEPS, "freeze_bn_stats": FREEZE_BN_STATS,
                      "head_norm": "LayerNorm", "unfreeze_last_n_blocks": UNFREEZE_LAST_N_BLOCKS,
                      "weights_saved": "EMA"},
        "regularization": {"drop_path_rate": DROP_PATH_RATE, "head_dropout": HEAD_DROPOUT,
                           "mixup_alpha": MIXUP_ALPHA, "mixup_prob": MIXUP_PROB,
                           "erase_prob": ERASE_PROB, "label_smoothing": LABEL_SMOOTHING},
        "best_epoch": best_epoch,
        "test_accuracy": test_acc, "test_accuracy_fine": fine_acc, "test_macro_f1": test_mf1, "test_macro_auc": test_auc,
        "train_accuracy_check_overfit": train_acc,
        "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score")}
                      for c in CLASS_NAMES},
        "confusion_matrix": cm.tolist(), "confusion_matrix_fine": cm_f.tolist(), "seed": SEED,
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "epoch_history.csv", index=False)

    # ---------- 2x2 curves ----------
    h = pd.DataFrame(history)

    def vlines(a):
        a.axvline(WARMUP_EPOCHS + 0.5, color="gray", ls=":", lw=1)
        a.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best epoch ({best_epoch})")

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].plot(h.epoch, h.train_loss, label="train loss")
    ax[0, 0].plot(h.epoch, h.val_loss, label="val loss")
    vlines(ax[0, 0]); ax[0, 0].legend(); ax[0, 0].set_title("Loss (CE + label smoothing, fine classes)")
    ax[0, 1].plot(h.epoch, h.val_acc, label="val acc")
    ax[0, 1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
    vlines(ax[0, 1]); ax[0, 1].legend(); ax[0, 1].set_title("Validation metrics (3 classes)")
    for c in CLASS_NAMES:
        ax[1, 0].plot(h.epoch, h[f"recall_{c}"], label=f"recall {c}")
    vlines(ax[1, 0]); ax[1, 0].legend(); ax[1, 0].set_title("Per-class recall (who is failing?)")
    ax[1, 1].plot(h.epoch, h.gap, color="tab:red", label="gap = val_loss - train_loss")
    ax[1, 1].axhline(0, color="gray", lw=0.8)
    vlines(ax[1, 1]); ax[1, 1].set_title("Generalization gap & learning rate")
    ax_lr = ax[1, 1].twinx()
    ax_lr.plot(h.epoch, h.lr_head, color="tab:green", ls=":", label="lr head")
    l1, n1 = ax[1, 1].get_legend_handles_labels(); l2, n2 = ax_lr.get_legend_handles_labels()
    ax[1, 1].legend(l1 + l2, n1 + n2, loc="upper right")
    for a in ax.ravel():
        a.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=200)
    plt.close(fig)
    print(f"\n✓ All artifacts saved in {MODEL_DIR} and {OUTPUT_DIR}")


if __name__ == "__main__":
    main()