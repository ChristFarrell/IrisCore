"""
Tarik foto test yang SALAH ditebak stage 2 (normal vs other) dari model
cascade FedAvg, lalu buat Grad-CAM-nya. Fokus ke stage 2 karena itu yang
bertanggung jawab atas keputusan normal-vs-other (lihat diskusi di chat).

Pakai ulang split yang SAMA PERSIS dengan training (seed & logika sama),
jadi test set di sini dijamin identik dengan test set yang dipakai evaluasi
cascade kemarin -- tidak ada kebocoran atau perbedaan data.

Jalankan dari folder Cataract:
    python -m src.export_misclassified_gradcam
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.fedavg_train_efficientnet import (
    ROOT_DIR, VAL_FRAC, TEST_FRAC, SEED, IMG_SIZE, MEAN, STD, BATCH_SIZE,
    DEVICE, USE_AMP, MODEL_DIR,
    scan_dataokdir, group_safe_stratified_split, get_model, eval_tf,
)
from src.gradcam import GradCAM, overlay, to_b64_png  # sudah ada dari sebelumnya

TOP_N = 30   # jumlah foto salah yang diambil (campuran normal->other dan other->normal)
OUT_DIR = Path("outputs/reports2/efficientnet_fedavg_cascade/misclassified_gradcam")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tensor_to_rgb(x: torch.Tensor) -> np.ndarray:
    """Tensor ternormalisasi (3,H,W) -> uint8 HxWx3, gambar yg BENAR dilihat model."""
    a = x.cpu().numpy() * np.array(STD).reshape(3, 1, 1) + np.array(MEAN).reshape(3, 1, 1)
    return (a.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


def main():
    print("Device:", DEVICE)

    print("\n== Membangun ulang split yang identik dengan training ==")
    df = scan_dataokdir(ROOT_DIR)
    _, _, test_df = group_safe_stratified_split(df, VAL_FRAC, TEST_FRAC, SEED)

    # Hanya foto yang label ASLINYA bukan cataract -- inilah subset yang jadi
    # tanggung jawab stage 2 (sama seperti saat training/evaluasi stage 2).
    test2 = test_df[test_df.label != 1].copy()
    test2["y_other"] = (test2.label == 2).astype(int)
    print(f"Test subset non-cataract: {len(test2)} foto "
          f"(normal {( test2.y_other==0).sum()}, other {(test2.y_other==1).sum()})")

    print("\n== Memuat model stage 2 (normal vs other) ==")
    model = get_model(2)
    state = torch.load(MODEL_DIR / "stage2_normal_other" / "model.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    cam = GradCAM(model)

    print("\n== Menjalankan prediksi + Grad-CAM per foto ==")
    rows = []
    for _, r in test2.iterrows():
        img = Image.open(r.path).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits.float(), 1)[0].cpu().numpy()
        pred = int(prob.argmax())
        rows.append({"path": r.path, "pid": r.pid, "y_other": int(r.y_other), "pred": pred,
                     "p_normal": float(prob[0]), "p_other": float(prob[1])})
    res = pd.DataFrame(rows)
    res["benar"] = res.y_other == res.pred
    wrong = res[~res.benar].copy()
    wrong["selisih"] = (wrong.p_normal - wrong.p_other).abs()   # makin kecil = makin "ragu-ragu"
    wrong = wrong.sort_values("selisih")   # tampilkan yang paling membingungkan model duluan

    print(f"Total salah: {len(wrong)} dari {len(res)} "
          f"({(wrong.y_other==0).sum()} normal->other, {(wrong.y_other==1).sum()} other->normal)")
    wrong.to_csv(OUT_DIR / "misclassified_list.csv", index=False)

    picked = wrong.head(TOP_N).reset_index(drop=True)
    print(f"\n== Membuat Grad-CAM untuk {len(picked)} foto ==")
    n = len(picked)
    cols = 5
    rows_n = -(-n // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.2, rows_n * 3.6))
    axes = np.array(axes).reshape(-1)

    label_name = {0: "normal", 1: "other"}
    for i, rec in picked.iterrows():
        img = Image.open(rec.path).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(DEVICE)
        c, _, _ = cam(x, int(rec.pred))          # heatmap untuk kelas yg DIPREDIKSI (salah)
        view = tensor_to_rgb(x[0])
        blended = overlay(view, c)
        Image.fromarray(blended).save(OUT_DIR / f"{Path(rec.path).stem}_gradcam.png")

        ax = axes[i]
        ax.imshow(blended)
        ax.set_title(f"asli: {label_name[rec.y_other]} | pred: {label_name[rec.pred]}\n"
                     f"P(other)={rec.p_other:.2f}", fontsize=8)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Stage 2 (normal vs other) -- {n} foto paling membingungkan model", fontsize=12)
    plt.tight_layout()
    contact_sheet = OUT_DIR / "contact_sheet.png"
    plt.savefig(contact_sheet, dpi=150)
    plt.close(fig)

    print(f"\n✓ Daftar lengkap     : {OUT_DIR / 'misclassified_list.csv'}")
    print(f"✓ Grad-CAM per foto  : {OUT_DIR}/<nama_foto>_gradcam.png")
    print(f"✓ Contact sheet      : {contact_sheet}")


if __name__ == "__main__":
    main()