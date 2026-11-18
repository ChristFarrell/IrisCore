"""
Cek apakah gaya crop/vignette foto (bulat polos vs ada border terang/hexagonal)
berkorelasi dengan label. Kalau iya, itu bukti Clever Hans: model bisa belajar
"gaya foto" sebagai jalan pintas, bukan tanda medis sungguhan.

Jalankan dari folder Cataract:
    python -m src.check_crop_style
"""
import numpy as np
import pandas as pd
import cv2

from src.fedavg_train_efficientnet import ROOT_DIR, scan_dataokdir, CLASS_NAMES

def crop_signature(path: str) -> dict:
    """Ukur seberapa 'bersih' crop fundus: seharusnya bulat gelap di luar."""
    img = cv2.imread(path)
    if img is None:
        return {"ok": False}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) rasio aspek foto asli: fundus biasanya mendekati 1:1
    aspect = max(w, h) / min(w, h)

    # 2) kecerahan di 4 sudut (seharusnya hitam pekat kalau crop-nya bersih)
    m = min(h, w) // 12
    corners = [gray[:m, :m], gray[:m, -m:], gray[-m:, :m], gray[-m:, -m:]]
    corner_brightness = float(np.mean([c.mean() for c in corners]))

    # 3) seberapa dekat bentuk terang ke lingkaran sempurna
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    circularity = np.nan
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        if perim > 0:
            circularity = float(4 * np.pi * area / (perim ** 2))  # 1.0 = lingkaran sempurna

    return {"ok": True, "aspect": aspect, "corner_brightness": corner_brightness,
            "circularity": circularity}


def main():
    df = scan_dataokdir(ROOT_DIR)
    print(f"\nMenganalisis gaya crop {len(df)} foto (bisa beberapa menit)...")
    sigs = [crop_signature(p) for p in df.path]
    sig_df = pd.DataFrame(sigs)
    df = pd.concat([df.reset_index(drop=True), sig_df], axis=1)
    df = df[df.ok].copy()

    # tandai "mencurigakan": sudut tidak gelap (vignette/border terang) ATAU
    # bentuknya jauh dari lingkaran (crop hexagonal/kotak seperti di contoh)
    df["border_terang"] = df.corner_brightness > 15
    df["bukan_lingkaran"] = df.circularity < 0.80

    print("\n=== Rata-rata corner_brightness per kelas (makin tinggi = makin mencurigakan) ===")
    print(df.groupby("label").corner_brightness.agg(["mean", "median", "std"]).rename(
        index=dict(enumerate(CLASS_NAMES))))

    print("\n=== Rata-rata circularity per kelas (makin jauh dari 1.0 = makin mencurigakan) ===")
    print(df.groupby("label").circularity.agg(["mean", "median", "std"]).rename(
        index=dict(enumerate(CLASS_NAMES))))

    print("\n=== Proporsi foto 'border_terang' per kelas ===")
    print((df.groupby("label").border_terang.mean() * 100).round(1).rename(
        index=dict(enumerate(CLASS_NAMES))).astype(str) + "%")

    print("\n=== Proporsi foto 'bukan_lingkaran' per kelas ===")
    print((df.groupby("label").bukan_lingkaran.mean() * 100).round(1).rename(
        index=dict(enumerate(CLASS_NAMES))).astype(str) + "%")

    out = "outputs/reports2/crop_style_check.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ Detail lengkap disimpan: {out}")
    print("\nCara baca: kalau salah satu kelas (terutama 'other') proporsinya jauh")
    print("lebih tinggi dari kelas lain di border_terang/bukan_lingkaran, itu bukti")
    print("sumber datanya tidak seragam dan model mungkin memakai itu sebagai jalan pintas.")


if __name__ == "__main__":
    main()