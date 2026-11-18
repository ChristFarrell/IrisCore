"""
Three-class training (Normal / Cataract / Other) with EfficientNetB3,
PyTorch (not Keras/TF), .pth output.

Built to be COMPARED with the notebook "eye-diseases-classification-cnn.ipynb"
(Keras, EfficientNetB3, 4-class dataset: cataract/diabetic_retinopathy/glaucoma/
normal, clearly SEPARATED). The training structure here is deliberately kept as
close as possible to that notebook's style (freeze backbone -> partially unfreeze -> fine-tune,
EarlyStopping, ReduceLROnPlateau), so that the comparison is fair in terms of METHOD.

IMPORTANT DIFFERENCES TO BE AWARE OF (read before judging the results):
- That notebook has 4 classes that are EACH a single, distinctive disease.
  We have 3 classes, but "Other" is a MIXTURE of many different diseases
  (diabetes, glaucoma, AMD, myopia, etc. merged into one label). Separating "normal"
  from a pile of mixed diseases is much harder than separating
  4 diseases that each stand on their own. If our accuracy is not
  as high as that notebook's, this does NOT mean the model is worse -- it is
  structurally harder. This is important to state in the report.
- FedAvg is not used here (plain, centralized training) -- for a pure comparison
  of "standard EfficientNetB3", without federated learning as a confounding variable.
- We still use a group-safe + stratified split (StratifiedGroupKFold), NOT a plain
  train_test_split like the notebook, because our data has left/right eye pairs
  per patient that must not leak across splits. The notebook
  does not need this because its dataset has no patient-pairing information.
- The dataset is assumed to be in dataokdir/Normal, dataokdir/Cataract, dataokdir/Other
  (change ROOT_DIR if different).

UPDATE (stability fixes, lessons learned from the FedAvg script):
- evaluate() now uses nan_to_num -- if the model happens to produce
  NaN/inf logits, the output is cleaned before metrics are computed, so that
  there is no "Input contains NaN" error and no cascade into empty metrics.
- roc_auc_score is called with the ORIGINAL labels (not label_binarize first) --
  the old pattern triggered the error "multi_class must be in ('ovo','ovr')" in
  some cases/sklearn versions.
- The training loop skips batches whose loss is NaN/inf + gradient clipping, so that
  the model weights are not permanently corrupted if a single batch blows up
  numerically (common with the combination of AMP + high resolution + BatchNorm).

UPDATE (CLAHE preprocessing for fundus images):
- Added RetinaCLAHE: CLAHE on the L channel (LAB color space) so that
  low-contrast blood vessels & small lesions become clearer, without
  changing the colors.
- 3 conditions that are enforced:
  1) DETERMINISTIC & IDENTICAL in train AND eval (val/test/serving). Not
     random augmentation -- same lesson as SquarePad. The preprocessing pipeline
     is shared through a single list, PREPROCESS_STEPS.
  2) The black area outside the retina is NOT part of the CLAHE statistics: the retina mask
     is computed first, pixels outside the mask are filled with the retina mean value while
     CLAHE is computed (so that tile histograms at the edge are not dominated by 0),
     and after CLAHE the outside area is restored to the original black.
  3) Conservative clipLimit (CLAHE_CLIP_LIMIT=2.0) so that it does not introduce
     noise/artifacts that could bring back an unstable val_loss.
- Requires: pip install opencv-python

UPDATE (anti-overfitting + Normal-vs-Other; previous run: train loss kept falling to 0.31
but val loss stalled at ~0.45 and val acc peaked at epoch ~18 and then declined):
- INTERNAL 4-CLASS (Normal / Cataract / Other-Diabetic / Other-Glaucoma), collapsed to
  3 classes at evaluation (P(Other) = P(Diabetic)+P(Glaucoma)). "Other" is not a single
  visual pattern but 2 different diseases; forcing 1 label makes the network build
  a branched decision region that interferes with the boundary against Normal. With sub-labels, each
  class = 1 clear pattern and the dataset is exactly balanced at 1k/1k/1k/1k.
- NO more undersampling: all data is used (4k, not 3k) -> the cheapest overfitting
  remedy. Balance is maintained through loss weights (coarse_balanced) so that Normal vs
  Other stays 1:1 in gradient weight.
- Regularization: stochastic depth (drop_path), head dropout 0.5, light MixUp,
  small RandomErasing, moderate geometric/color augmentation, unfreeze 3 blocks (not 4),
  a cosine LR schedule that FINISHES within MAX_EPOCHS=36 (previously the LR was still high when
  val started getting worse).
- Diagnostics: per-class recall/precision per epoch, 4-class confusion matrix,
  list of misclassified files (misclassified_test.csv), 2x2 curves.
- IMPORTANT for serving: model.pth now has 4 logits. Take the softmax of the 4 logits and then
  sum the columns into 3 classes (see meta.json -> fine_to_coarse).

Run from the Cataract folder:
    python -m src.train_efficientnetb3
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, f1_score,
)
import matplotlib.pyplot as plt

# ==========================================
# 0. Configuration
# ==========================================
ROOT_DIR = Path(r"/home/project04/Iris1003/data4categorize")
CLASS_MAP = {"normal": 0, "cataract": 1, "other": 2}
CLASS_NAMES = ["Normal", "Cataract", "Other"]

ARCH = "efficientnet_b3"
IMG_SIZE = 300          # native EfficientNetB3 resolution (same as the comparison notebook)
BATCH_SIZE = 12          # B3 @300 is heavier than B0 @384 -- reduce if OOM
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15

# Two-stage training: head only -> unfreeze part of the backbone -> fine-tune
# ---- STABILITY settings (val_loss previously spiked up to 3.2 while val_acc was smooth) ----
WARMUP_EPOCHS = 5         # stage 1: backbone frozen, only the head is trained
MAX_EPOCHS = 36          # cosine schedule FINISHES here (val peak was previously at epoch ~18 with the LR still high)
PATIENCE = 12             # early stopping based on val_macro_f1 (EMA), counted only in stage 2
MIN_DELTA = 1e-3          # a val_macro_f1 improvement smaller than this counts as no improvement
LR_HEAD = 3e-4            # lowered from 1e-3: LR 1e-3 on a fresh head = head weights jumping around
LR_HEAD_STAGE2 = 1e-4     # head in stage 2 (already converged, a gentler LR is enough)
LR_FINETUNE = 2e-5        # backbone LR (discriminative LR: backbone < head)
WEIGHT_DECAY = 1e-2       # AdamW (decoupled). Previously Adam + 1e-4 = almost no effect
LABEL_SMOOTHING = 0.1
UNFREEZE_LAST_N_BLOCKS = 3   # back to 3: fine-tuning 4 blocks on small data memorizes easily
ACCUM_STEPS = 2           # gradient accumulation: effective batch 12x2=24, calmer gradients
EMA_DECAY = 0.998         # evaluated/saved weights = moving average (EMA), damps epoch-to-epoch spikes
FREEZE_BN_STATS = True    # backbone BatchNorm always in eval mode (running stats do not follow small batches/augmentation)

# CLAHE (deterministic preprocessing, identical in train & eval)
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0        # conservative; 1.5-3.0 is reasonable for fundus. >4 is prone to noise/artifacts
CLAHE_TILE_GRID = (8, 8)
RETINA_THRESH = 10            # pixels with gray <= this are treated as black background outside the retina

# ---- Anti-overfitting ----
USE_FINE_LABELS = True        # train 4 classes (Other split per subfolder), evaluation stays 3 classes
CLASS_WEIGHT_MODE = "coarse_balanced"   # "coarse_balanced" | "fine_balanced" | "none"
DROP_PATH_RATE = 0.2          # timm stochastic depth; raise to 0.3 if still overfitting
MIXUP_ALPHA = 0.2             # Beta(a,a); small = gentle mixing
MIXUP_PROB = 0.4              # probability that a batch is mixed up; 0 = disabled
ERASE_PROB = 0.25             # small RandomErasing patch (0.02-0.08 of the image area)

RUN_NAME = "efficientnetb3_data4_CL3PURE"
OUTPUT_DIR = Path("outputs/reports3") / RUN_NAME
MODEL_DIR = Path("models/final3") / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
torch.manual_seed(SEED)
np.random.seed(SEED)

PID_RE = re.compile(r"(\d+)_(?:left|right)", re.IGNORECASE)


# ==========================================
# 1. Transform
# ==========================================
class SquarePad:
    """Black padding to a square shape BEFORE resize -- the comparison notebook
    does not do this (it resizes straight to a square), but for our fundus images
    this matters so that the round retina is not squashed into an oval."""
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        pad_w, pad_h = s - w, s - h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return ImageOps.expand(img, padding, fill=(0, 0, 0))


def retina_mask(rgb: np.ndarray, thresh: int = RETINA_THRESH):
    """Boolean mask of the retina area (True = retina) from a uint8 RGB image.
    Take the largest contour and fill it completely (hole-filling) so that lesions/dark
    areas INSIDE the retina are not dropped from the mask. Slightly eroded so that the
    dark transition ring at the retina edge is not processed.
    Returns None if no retina is detected (nearly black image)."""
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
    if filled.mean() < 0.05:        # retina < 5% of image area -> implausible
        return None
    return filled.astype(bool)


class RetinaCLAHE:
    """CLAHE on the L channel (LAB), for the retina area only.
    DETERMINISTIC (no randomness) -> MUST be used identically in train, val, test,
    and serving. The black area outside the retina:
      - during CLAHE: filled with the retina mean L, so it does not distort
        the histogram/clip of tiles near the edge;
      - after CLAHE: restored to the original pixels (black)."""
    def __init__(self, clip_limit=CLAHE_CLIP_LIMIT, tile_grid=CLAHE_TILE_GRID,
                 thresh=RETINA_THRESH):
        self.clip_limit = clip_limit
        self.tile_grid = tuple(tile_grid)
        self.thresh = thresh

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"))
        mask = retina_mask(rgb, self.thresh)
        if mask is None:                       # deterministic fallback: leave as is
            return img
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L = lab[..., 0].copy()
        L[~mask] = int(L[mask].mean())         # neutralize the background before computing CLAHE
        # the CLAHE object is created per call (cheap) so it is safe for a multi-worker DataLoader
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        lab[..., 0] = clahe.apply(L)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        out[~mask] = rgb[~mask]                # restore the area outside the retina (original black)
        return Image.fromarray(out)


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# DETERMINISTIC preprocessing: a single source of truth for train & eval.
# Order: SquarePad -> Resize -> CLAHE. CLAHE after Resize (cheap & tile size
# is consistent relative to the 300x300 image) and BEFORE geometric augmentation,
# so that the retina mask is not disturbed by black pixels produced by RandomAffine.
PREPROCESS_STEPS = [SquarePad(), transforms.Resize((IMG_SIZE, IMG_SIZE))]
if USE_CLAHE:
    PREPROCESS_STEPS.append(RetinaCLAHE())

train_tf = transforms.Compose([
    *PREPROCESS_STEPS,
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # fundus is fairly invariant to small rotations/shifts; hue is NOT jittered
    # (the red/orange color of the retina carries clinical information)
    transforms.RandomAffine(degrees=25, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8, fill=0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=ERASE_PROB, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
])
eval_tf = transforms.Compose([
    *PREPROCESS_STEPS,          # IDENTICAL to train_tf (SquarePad -> Resize -> CLAHE)
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ==========================================
# 2. Dataset
# ==========================================
IMG_EXT = (".jpg", ".jpeg", ".png")


def scan_dataokdir(root: Path) -> pd.DataFrame:
    """Scan dataokdir/<Normal|Cataract|Other>/*.jpg. If a folder
    has subfolders (e.g. Other/Diabetic/, Other/Glaucoma/), each subfolder is
    scanned separately and its name is recorded in the 'subtype' column -- used later
    to balance Diabetic vs Glaucoma inside the Other class."""
    rows = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        label = CLASS_MAP.get(folder.name.strip().lower())
        if label is None:
            print(f"  (skipping unknown folder: {folder.name})")
            continue

        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        if subdirs:
            for sd in subdirs:
                files = [f for f in sd.iterdir() if f.suffix.lower() in IMG_EXT]
                for f in files:
                    m = PID_RE.search(f.name)
                    pid = m.group(1) if m else f.stem
                    rows.append({"path": str(f), "label": label, "pid": pid, "subtype": sd.name})
                print(f"  {folder.name}/{sd.name:12s} -> {len(files):5d} images (label {label}, subtype={sd.name})")
            direct_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]
            if direct_files:
                print(f"  Warning: {len(direct_files)} files directly in {folder.name}/ "
                      f"(not in a subfolder) -- marked subtype='unspecified', NOT included in "
                      f"the Diabetic/Glaucoma balancing.")
                for f in direct_files:
                    m = PID_RE.search(f.name)
                    pid = m.group(1) if m else f.stem
                    rows.append({"path": str(f), "label": label, "pid": pid, "subtype": "unspecified"})
        else:
            files = [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT]
            for f in files:
                m = PID_RE.search(f.name)
                pid = m.group(1) if m else f.stem
                rows.append({"path": str(f), "label": label, "pid": pid, "subtype": "unspecified"})
            print(f"  {folder.name:10s} -> {len(files):5d} images (label {label})")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found in {root}. Check ROOT_DIR and the subfolder names.")

    other_subtypes = df[df.label == 2].subtype.unique().tolist()
    if other_subtypes == ["unspecified"]:
        print("\n  !! IMPORTANT WARNING: the 'Other' class has no detected Diabetic/Glaucoma subfolders "
              "-- everything is marked 'unspecified'. The 50:50 balancing will NOT "
              "apply. Check the folder structure of dataokdir/Other/.")
    else:
        print(f"\n  'Other' subtypes detected: {other_subtypes}")
    return df


def build_fine_labels(df: pd.DataFrame):
    """Add the 'fine' column. Normal & Cataract = 1 class each; Other
    is split per subtype (Diabetic, Glaucoma, ...) if subfolders are detected.
    Returns (df, fine_names, f2c) where f2c[i] = the 3-class id of fine class i."""
    fine_names, f2c, key2id = [], [], {}
    other = CLASS_MAP["other"]
    for coarse, cname in enumerate(CLASS_NAMES):
        subs = set(df[df.label == coarse].subtype.unique())
        split_it = USE_FINE_LABELS and coarse == other and subs != {"unspecified"}
        for k in (sorted(subs) if split_it else [None]):
            key2id[(coarse, k)] = len(fine_names)
            fine_names.append(cname if k is None else f"{cname}/{k}")
            f2c.append(coarse)
    df = df.copy()
    df["fine"] = [key2id.get((c, s_), key2id.get((c, None))) for c, s_ in zip(df.label, df.subtype)]
    return df, fine_names, np.array(f2c)


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
# 3. Group-safe split + stratification
# ==========================================
def group_safe_stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    n_folds = int(round(1 / min(val_frac, test_frac)))
    n_test_folds = max(1, round(n_folds * test_frac))
    n_val_folds = max(1, round(n_folds * val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df = df.reset_index(drop=True)
    df["fold"] = -1
    for k, (_, idx) in enumerate(sgkf.split(df, df.fine, df.pid)):   # stratification per FINE class
        df.loc[idx, "fold"] = k
    test_df = df[df.fold < n_test_folds].drop(columns="fold").reset_index(drop=True)
    val_df = df[(df.fold >= n_test_folds) & (df.fold < n_test_folds + n_val_folds)] \
        .drop(columns="fold").reset_index(drop=True)
    train_df = df[df.fold >= n_test_folds + n_val_folds].drop(columns="fold").reset_index(drop=True)
    assert not (set(train_df.pid) & set(val_df.pid)), "LEAK: train & val patients overlap!"
    assert not (set(train_df.pid) & set(test_df.pid)), "LEAK: train & test patients overlap!"
    assert not (set(val_df.pid) & set(test_df.pid)), "LEAK: val & test patients overlap!"
    return train_df, val_df, test_df


def water_fill(counts: dict, cap: int) -> dict:
    """Divide 'cap' slots among several groups as fairly as possible. Groups with little
    data are taken in full first; leftover quota that cannot be absorbed is passed
    on to other groups that still have stock (so the total stays = cap)."""
    remaining = dict(counts)
    alloc = {k: 0 for k in counts}
    left = int(cap)
    while left > 0 and remaining:
        share = max(1, left // len(remaining))
        for k in list(remaining):
            take = min(share, remaining[k], left)
            alloc[k] += take
            remaining[k] -= take
            left -= take
            if remaining[k] <= 0:
                del remaining[k]
            if left <= 0:
                break
    return alloc


def undersample_to_cataract(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Normal & Cataract: old logic (undersample to n_cat). Other: SPLIT
    EVENLY across subtypes (Diabetic/Glaucoma) via water_fill, instead of drawing
    randomly from the mixed pool -- so that no single subtype dominates."""
    n_cat = int((df.label == 1).sum())
    print(f"  Target per class (= number of cataract): {n_cat}")

    parts = [df[df.label == 1]]   # cataract: use all

    normal_pool = df[df.label == 0]
    n_take = min(len(normal_pool), n_cat)
    if n_take < n_cat:
        print(f"  Warning: normal has only {n_take} < target {n_cat}, using all.")
    parts.append(normal_pool.sample(n=n_take, random_state=seed))

    other_pool = df[df.label == 2]
    counts = other_pool.subtype.value_counts().to_dict()
    alloc = water_fill(counts, n_cat)
    print(f"  'Other' allocation per subtype (total target {n_cat}, split EVENLY):")
    for k in counts:
        available, taken = counts[k], alloc.get(k, 0)
        flag = "  <- below the even target, remainder shifted to other subtypes" \
            if available < n_cat / len(counts) else ""
        print(f"    {k:12s}: available {available:5d} -> taken {taken:5d}{flag}")
    other_parts = [other_pool[other_pool.subtype == k].sample(n=n, random_state=seed)
                   for k, n in alloc.items() if n > 0]
    parts.extend(other_parts)

    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ==========================================
# 4. Model
# ==========================================
HEAD_HIDDEN = 256     # hidden layer size, same as the comparison notebook (Dense(256))
HEAD_DROPOUT = 0.5    # raised from 0.4 (overfitting)


def get_model(num_out: int):
    """The classifier head is replaced from a plain Linear (timm default) to
    GlobalAvgPool (already handled by timm when num_classes>0) -> Dense(256) ->
    LayerNorm -> ReLU -> Dropout(HEAD_DROPOUT) -> Dense(num_out), imitating the head structure in the
    comparison notebook. This dropout is the most relevant fix for the overfitting symptom
    (train 98.7% vs test 90.1%) -- the old head had no regularization at all."""
    m = timm.create_model(ARCH, pretrained=True, num_classes=0, drop_path_rate=DROP_PATH_RATE)   # num_classes=0: backbone+pool only, no head
    in_features = m.num_features
    m.classifier = nn.Sequential(
        nn.Linear(in_features, HEAD_HIDDEN),
        nn.LayerNorm(HEAD_HIDDEN),   # NOT BatchNorm1d: batch 12 is too small, its statistics are noisy
        nn.ReLU(inplace=True),
        nn.Dropout(HEAD_DROPOUT),
        nn.Linear(HEAD_HIDDEN, num_out),   # num_out = number of FINE classes
    )
    return m.to(DEVICE)


def set_trainable(model, mode: str):
    """mode='head': classifier head only. mode='partial': head + the last N
    backbone blocks (like 'unfreeze last 40 layers' in the notebook)."""
    head_names = set()
    for n, _ in model.named_parameters():
        if "classifier" in n:
            head_names.add(n)

    if mode == "head":
        for n, p in model.named_parameters():
            p.requires_grad = n in head_names
    elif mode == "partial":
        block_ids = sorted({n.split(".")[1] for n, _ in model.named_parameters()
                            if n.startswith("blocks.")}, key=int)
        unfreeze_blocks = set(block_ids[-UNFREEZE_LAST_N_BLOCKS:])
        for n, p in model.named_parameters():
            in_unfrozen_block = n.startswith("blocks.") and n.split(".")[1] in unfreeze_blocks
            p.requires_grad = (n in head_names) or in_unfrozen_block or n.startswith("conv_head") \
                or n.startswith("bn2")
    else:
        raise ValueError(mode)


# ==========================================
# 4b. Stability helpers
# ==========================================
def set_bn_eval(model):
    """Lock the backbone BatchNorm in eval mode even under model.train(). Without this,
    BN layers that are 'frozen' (requires_grad=False) STILL update running
    mean/var from small batches + augmentation -> statistics at val differ from
    those at train -> val_loss spikes. Called after model.train() every epoch."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


class ModelEMA:
    """Exponential Moving Average of the weights. The EMA model is the one that is evaluated and
    saved as the checkpoint -- the averaged weights are much smoother than the
    raw weights at the end of each epoch."""
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad = False
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))   # EMA is "warmed up" at the start
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def make_optimizer(model, lr_head, lr_backbone):
    """AdamW with different LRs: head vs backbone (discriminative LR)."""
    head = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" in n]
    back = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]
    groups = [{"params": head, "lr": lr_head}]
    if back:
        groups.append({"params": back, "lr": lr_backbone})
    return optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def cosine_with_warmup(opt, total_steps, warmup_steps, min_ratio=0.05):
    """LR rises linearly during warmup and then decays by cosine (per STEP). Replaces
    ReduceLROnPlateau, which reacts to a noisy val_loss (it can cut the LR
    because of a momentary spike)."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return optim.lr_scheduler.LambdaLR(opt, f)


def loss_from_probs(probs, labels, eps=LABEL_SMOOTHING):
    """Compute the loss (with label smoothing, same scale as the train loss) from the
    probabilities returned by evaluate() -- a single forward pass, computed in
    float32 (not fp16 autocast). Returns (loss_smoothed, pure_nll)."""
    logp = np.log(np.clip(probs, 1e-8, 1.0))
    nll = -logp[np.arange(len(labels)), labels]
    smooth = -logp.mean(axis=1)
    return float(((1 - eps) * nll + eps * smooth).mean()), float(nll.mean())


# ==========================================
# 5. Evaluation
# ==========================================
@torch.no_grad()
def evaluate(model, loader, f2c_mat, return_fine=False):
    """The model outputs FINE-class logits; probabilities are collapsed to 3 classes
    through f2c_mat [n_fine x 3] (Diabetic+Glaucoma -> Other). All metrics
    (acc/F1/AUC/confusion) are computed on the 3 classes."""
    model.eval()
    all_probs, all_labels, all_fine = [], [], []
    n_fine = f2c_mat.shape[0]
    for x, y, yf in loader:
        x = x.to(DEVICE)
        with torch.autocast(DEVICE, enabled=USE_AMP):
            logits = model(x)
        # clean NaN/inf BEFORE softmax
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
    """Save 2 versions: raw counts AND per-row percentages (per-class
    recall). The test set is NOT balanced -- the percentage version makes every class equal
    visually without throwing away test data (see the earlier discussion)."""
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
    """Save a before/after CLAHE comparison (visual check: the retina edge
    must be clean with no rings/noise, the black area stays black). If it
    looks excessive/speckled, lower CLAHE_CLIP_LIMIT."""
    base = transforms.Compose([SquarePad(), transforms.Resize((IMG_SIZE, IMG_SIZE))])
    clahe = RetinaCLAHE()
    sample = df.sample(n=min(n, len(df)), random_state=SEED)
    fig, axes = plt.subplots(2, len(sample), figsize=(3 * len(sample), 6), squeeze=False)
    for j, (_, row) in enumerate(sample.iterrows()):
        im = base(Image.open(row.path).convert("RGB"))
        axes[0, j].imshow(im); axes[0, j].set_title(f"original ({CLASS_NAMES[int(row.label)]})")
        axes[1, j].imshow(clahe(im)); axes[1, j].set_title("CLAHE")
        axes[0, j].axis("off"); axes[1, j].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  CLAHE preview saved: {path}")


# ==========================================
# 6. Main
# ==========================================
def main():
    print("Device:", DEVICE, "| AMP:", USE_AMP)
    if DEVICE == "cuda":
        print("GPU   :", torch.cuda.get_device_name(0))

    print("\n== Scanning dataset ==")
    df = scan_dataokdir(ROOT_DIR)
    df, FINE_NAMES, F2C = build_fine_labels(df)
    n_fine = len(FINE_NAMES)
    F2C_MAT = np.zeros((n_fine, len(CLASS_NAMES)))
    F2C_MAT[np.arange(n_fine), F2C] = 1.0
    print(f"Total images: {len(df)} | unique patients: {df.pid.nunique()}")
    print("Fine classes (trained):", {i: f"{n} -> {CLASS_NAMES[F2C[i]]}" for i, n in enumerate(FINE_NAMES)})
    print("Raw distribution (fine):", dict(df.fine.value_counts().sort_index()))

    print("\n== Group-safe split + fine-class stratification ==")
    train_df, val_df, test_df = group_safe_stratified_split(df, VAL_FRAC, TEST_FRAC, SEED)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:5s} {len(d):5d} images | fine {dict(d.fine.value_counts().sort_index())}")

    if USE_CLAHE:
        save_clahe_preview(train_df, OUTPUT_DIR / "clahe_preview.png")

    # ---- No undersampling: ALL train data is used. Balance comes from the loss weights ----
    fine_counts = train_df.fine.value_counts().reindex(range(n_fine)).fillna(0).to_numpy()
    coarse_counts = np.bincount(F2C, weights=fine_counts, minlength=len(CLASS_NAMES))
    N = fine_counts.sum()
    if CLASS_WEIGHT_MODE == "coarse_balanced":
        # each 3-CLASS class (Normal/Cataract/Other) gets the same total gradient weight;
        # inside Other, Diabetic & Glaucoma share it equally
        w = N / (len(CLASS_NAMES) * coarse_counts[F2C])
    elif CLASS_WEIGHT_MODE == "fine_balanced":
        w = N / (n_fine * fine_counts.clip(min=1))
    else:
        w = np.ones(n_fine)
    w = w / (w * fine_counts).sum() * N          # average weight per sample = 1
    print(f"\n== Loss weights ({CLASS_WEIGHT_MODE}) ==")
    for i, n in enumerate(FINE_NAMES):
        print(f"  {n:20s} n_train={int(fine_counts[i]):5d}  weight={w[i]:.3f}")

    train_dl = DataLoader(FundusDF(train_df, train_tf), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, drop_last=True)
    # drop_last=True: keeps the batch size uniform (more consistent gradients)
    val_dl = DataLoader(FundusDF(val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(FundusDF(test_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(n_fine)
    ema = ModelEMA(model, EMA_DECAY)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE),
                                    label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    steps_per_epoch = max(1, math.ceil(len(train_dl) / ACCUM_STEPS))   # optimizer steps per epoch

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
            if use_mix:                              # gentle MixUp: damps overconfidence at the Normal/Other boundary
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1 - lam) * x[perm]
                y_b = yf[perm]
            with torch.autocast(DEVICE, enabled=USE_AMP):
                logits = model(x)
            logits = logits.float()                  # loss is computed in float32
            if use_mix:
                loss = lam * criterion(logits, yf) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, yf)
            if not torch.isfinite(loss):
                n_skip += 1                          # skip batches that blow up numerically
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

        # Evaluation uses the EMA weights. Val loss is computed in the FINE-class space so that it is
        # comparable to the train loss (both are 4-class CE + label smoothing).
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

        # Checkpoint & early stopping are both based on val_macro_f1 (EMA).
        # Patience only starts counting in stage 2 so that warmup is not cut short.
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

    # ---- Fine-class diagnostics: do Diabetic or Glaucoma samples escape to Normal? ----
    f_pred = f_prob.argmax(1)
    fine_ids = list(range(n_fine))
    print("\n=== Per FINE-CLASS report (4 classes) ===")
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

    # ---- Misclassified files (for manual review / Grad-CAM) ----
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
        "name": RUN_NAME, "arch": ARCH, "image_size": IMG_SIZE,
        "mean": MEAN, "std": STD, "num_classes": len(CLASS_NAMES),
        "num_logits": n_fine, "fine_names": FINE_NAMES, "fine_to_coarse": F2C.tolist(),
        "serving_note": "model.pth has num_logits outputs. softmax -> sum the probabilities of the "
                        "fine classes into 3 classes using fine_to_coarse (Other = Diabetic + Glaucoma).",
        "label_map": {str(i): c for i, c in enumerate(CLASS_NAMES)},
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
        a.axvline(WARMUP_EPOCHS + 0.5, color="gray", ls=":", lw=1)            # stage 1 | 2 boundary
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