"""
manifest.py (VERSI 3 KELAS, DIBATASI/BALANCED) - klasifikasi PER MATA.

Label:  0 = normal | 1 = katarak | 2 = penyakit lain (D, G, A, H, M, O digabung)

Perubahan dari versi sebelumnya:
  1. Gambar dengan keyword kualitas ambigu (lens dust, refractive media opacity,
     low image quality, dst.) TIDAK lagi otomatis dibuang. Sekarang diperlakukan
     seperti keyword lain: kalau berdiri sendiri -> masuk kelas "other" (kategori O),
     kalau bercampur dengan "normal"/"cataract" -> tetap dibuang (aturan campuran).
  2. Jumlah data dibatasi supaya seimbang:
       - SEMUA gambar katarak dipakai. Jumlahnya (N_cat) menjadi batas atas.
       - Normal dibatasi maksimum N_cat (diambil acak, seed tetap).
       - Other dibatasi maksimum N_cat TOTAL, dibagi dulu ke tiap subtipe
         (D, G, A, H, M, O) dengan "water-filling": subtipe yang datanya sedikit
         (glaukoma, hipertensi, AMD, miopia) diambil semaksimal mungkin, sisa
         kuota baru dilimpahkan ke subtipe yang datanya banyak (diabetes, other-O).
         Ini mencegah diabetes mendominasi kelas "other" seperti versi sebelumnya.
  3. Mata dengan 2 kategori penyakit berbeda, atau katarak + penyakit lain,
     TETAP dibuang (tidak berubah dari versi sebelumnya).
  4. Subtipe di EXCLUDED_OTHER_SUBTYPES dikeluarkan dari kelas "other". Kuota
     yang tadinya untuk subtipe itu (lewat water-filling) dilimpahkan ke
     subtipe lain yang tersisa, supaya total kelas "other" tetap = N_cat.
     PENTING: exclude ini harus diputuskan berdasarkan alasan klinis/domain
     SEBELUM melihat hasil model, bukan berdasarkan subtipe mana yang recall-nya
     rendah di suatu run. Memilih exclude dari hasil model = data leakage
     (menyesuaikan data training memakai jawaban dari evaluasi test).
  5. Gambar katarak tambahan dari sumber eksternal (di EXTERNAL_CATARACT_DIR,
     tanpa CSV/keyword) ikut ditambahkan ke pool katarak SEBELUM N_cat dihitung.
     Jadi N_cat = katarak ODIR + katarak eksternal, dan normal/other otomatis
     ikut naik batasnya mengikuti N_cat yang baru. Kolom "source" menandai
     asal tiap baris (odir / external), supaya bisa dicek terpisah nanti
     (misalnya apakah model belajar gaya foto, bukan penyakitnya).

Konsekuensi: dataset akhir jauh lebih kecil dan seimbang (kira-kira 1:1:1),
bukan lagi mencerminkan proporsi asli ODIR. Ini pilihan sadar untuk mengatasi
ketimpangan kelas, dan perlu disebutkan sebagai keterbatasan/metodologi di laporan.

Jalankan dari folder Cataract:
    python -m src.data.manifest
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.text_rules import classify_eye, LABEL_NAMES, OTHER_SUBTYPES

# ------------------------------------------------------------------ pengaturan
IMG_DIR = Path("data/raw")
CSV_PATH = Path("data/full_df.csv")
OUT_PATH = Path("data/split.csv")
REPORT_DIR = Path("outputs/reports")

# Folder gambar katarak tambahan (tanpa CSV, cuma isi gambar katarak_001.png dst.).
# Kosongkan (None) kalau tidak ingin memakai sumber tambahan ini.
EXTERNAL_CATARACT_DIR = Path("data/external/cataractdataset/2_cataract")
EXTERNAL_IMG_EXT = {".jpg", ".jpeg", ".png"}
EXTERNAL_EXCLUDE_SUBSTR = ["_aug", "aug_"]   # jaga-jaga, buang kalau ada file augmentasi ikut kebawa

# Subtipe yang DIKELUARKAN dari kelas "other". Alasan klinis (bukan performa model):
# glaukoma (G) dan hipertensi (H) tidak bisa didiagnosis andal hanya dari satu foto
# fundus tunggal -- glaukoma perlu tekanan bola mata dan rasio cup-to-disc, hipertensi
# perlu data tekanan darah -- sehingga di luar cakupan wajar untuk model berbasis satu
# foto. Kosongkan list ini (EXCLUDED_OTHER_SUBTYPES = []) untuk memakai semua subtipe.
EXCLUDED_OTHER_SUBTYPES = ["G", "H", "D", "O"]

SEED = 42
N_FOLDS = 20        # 20 lipatan: 3 test (15%), 3 val (15%), 14 train (70%)
TEST_FOLDS = 3
VAL_FOLDS = 3


# ------------------------------------------------------------------ pembatasan jumlah
def water_fill(counts, cap):
    """Bagi 'cap' slot ke beberapa subtipe seadil mungkin.
    Subtipe yang datanya sedikit diambil semua duluan; sisa kuota yang tidak
    terserap dilimpahkan ke subtipe lain yang masih punya stok."""
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


def scan_external_cataract(folder: Path):
    """Pindai folder gambar katarak eksternal (tanpa CSV). Setiap file jadi satu
    baris label=1, group unik per file (bukan per pasien, karena tidak ada info
    pasangan kiri-kanan di sumber ini)."""
    if folder is None:
        return pd.DataFrame(columns=["path", "label", "label_name", "subtype",
                                     "group", "keywords", "source"])
    if not folder.exists():
        print(f"[!] EXTERNAL_CATARACT_DIR tidak ditemukan: {folder} -- dilewati")
        return pd.DataFrame(columns=["path", "label", "label_name", "subtype",
                                     "group", "keywords", "source"])

    files = [f for f in sorted(folder.iterdir())
             if f.suffix.lower() in EXTERNAL_IMG_EXT
             and not any(s in f.stem.lower() for s in EXTERNAL_EXCLUDE_SUBSTR)]
    skipped = [f for f in folder.iterdir()
              if f.suffix.lower() in EXTERNAL_IMG_EXT
              and any(s in f.stem.lower() for s in EXTERNAL_EXCLUDE_SUBSTR)]
    if skipped:
        print(f"Sumber eksternal: {len(skipped)} file augmentasi ('_aug') dilewati")

    rows = [{"path": str(f), "label": 1, "label_name": LABEL_NAMES[1], "subtype": "",
            "group": f"ext_{f.stem}", "keywords": "cataract (sumber eksternal)",
            "source": "external"} for f in files]
    print(f"Sumber eksternal katarak: {len(rows)} gambar asli dipakai dari {folder}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ proses utama
def main():
    df = pd.read_csv(CSV_PATH)
    n_before = len(df)
    df = df.drop_duplicates(subset="ID").reset_index(drop=True)
    n_dup = n_before - len(df)
    print(f"Jumlah pasien di CSV: {len(df)}"
          + (f"  ({n_dup} baris ID duplikat dibuang)" if n_dup else ""))

    rows, excluded = [], []
    for _, r in df.iterrows():
        for side in ("Left", "Right"):
            fname = str(r[f"{side}-Fundus"])
            text = r[f"{side}-Diagnostic Keywords"]
            path = IMG_DIR / fname
            if not path.exists():
                excluded.append({"ID": r["ID"], "file": fname, "alasan": "file tidak ada",
                                 "keywords": text})
                continue

            label, subtype, reason = classify_eye(text)
            if label is None:
                excluded.append({"ID": r["ID"], "file": fname, "alasan": reason,
                                 "keywords": text})
                continue

            rows.append({"path": str(path), "label": label, "label_name": LABEL_NAMES[label],
                         "subtype": subtype or "", "group": r["ID"], "keywords": text,
                         "source": "odir"})

    full = pd.DataFrame(rows)
    if full.empty:
        raise SystemExit("Tidak ada gambar yang lolos. Cek IMG_DIR dan nama kolom CSV.")
    ex = pd.DataFrame(excluded)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ex.to_csv(REPORT_DIR / "mata_dibuang.csv", index=False)

    print("\n=== Sebelum dibatasi (semua mata yang lolos klasifikasi) ===")
    print(full.label_name.value_counts().to_string())
    print("\nAlasan dibuang (mata multi-penyakit / katarak campuran / tanpa keyword):")
    print(ex.alasan.value_counts().to_string() if not ex.empty else "(tidak ada)")

    # ---------- gabungkan katarak eksternal SEBELUM n_cat dihitung ----------
    external_cataract = scan_external_cataract(EXTERNAL_CATARACT_DIR)
    n_odir_cat = int((full.label == 1).sum())
    n_ext_cat = len(external_cataract)

    # ---------- pembatasan jumlah ----------
    n_cat = n_odir_cat + n_ext_cat
    print(f"\nBatas atas jumlah data (= jumlah katarak) = {n_cat} "
          f"({n_odir_cat} dari ODIR + {n_ext_cat} dari sumber eksternal)")
    if n_cat == 0:
        raise SystemExit("Tidak ada gambar katarak yang lolos. Cek keyword 'cataract' di CSV.")

    cataract_rows = pd.concat([full[full.label == 1], external_cataract], ignore_index=True)

    normal_pool = full[full.label == 0]
    n_take_normal = min(len(normal_pool), n_cat)
    normal_rows = normal_pool.sample(n=n_take_normal, random_state=SEED)
    if n_take_normal < n_cat:
        print(f"Peringatan: normal hanya tersedia {n_take_normal} (< {n_cat})")

    other_pool_all = full[full.label == 2]
    other_pool = other_pool_all[~other_pool_all.subtype.isin(EXCLUDED_OTHER_SUBTYPES)]
    if EXCLUDED_OTHER_SUBTYPES:
        n_excl = len(other_pool_all) - len(other_pool)
        print(f"\nSubtipe dikeluarkan dari 'other' (alasan klinis, ditentukan sebelum "
              f"melihat hasil model): {EXCLUDED_OTHER_SUBTYPES} -> {n_excl} mata tidak diikutkan")
        other_pool_all[other_pool_all.subtype.isin(EXCLUDED_OTHER_SUBTYPES)].to_csv(
            REPORT_DIR / "subtipe_dikecualikan.csv", index=False)

    counts = other_pool.subtype.value_counts().to_dict()
    alloc = water_fill(counts, n_cat)
    print(f"\nAlokasi kelas 'other' per subtipe (water-filling, target total {n_cat}, "
          f"subtipe dikecualikan tidak diikutkan):")
    for k in OTHER_SUBTYPES:
        if k in EXCLUDED_OTHER_SUBTYPES:
            print(f"  {k}: DIKECUALIKAN (alasan klinis)")
            continue
        print(f"  {k}: tersedia {counts.get(k, 0):5d} -> diambil {alloc.get(k, 0):5d}")
    if sum(alloc.values()) < n_cat:
        print(f"Peringatan: total 'other' tersedia ({sum(counts.values())}) < target ({n_cat}), "
              f"hanya {sum(alloc.values())} yang bisa diambil.")

    other_parts = [other_pool[other_pool.subtype == k].sample(n=n, random_state=SEED)
                   for k, n in alloc.items() if n > 0]
    other_rows = pd.concat(other_parts) if other_parts else other_pool.iloc[0:0]

    m = pd.concat([cataract_rows, normal_rows, other_rows]).reset_index(drop=True)
    m = m.drop(columns="subtype")

    print("\n=== Setelah dibatasi (dataset final) ===")
    print(m.label_name.value_counts().to_string())
    print(f"Total dipakai: {len(m)} dari {len(full)} yang lolos klasifikasi "
          f"({len(df)*2} mata sebelum klasifikasi)")

    # ---------- split per pasien, distratifikasi per kelas ----------
    # group harus satu tipe: ODIR pakai ID (angka), eksternal pakai teks "ext_...".
    # Tanpa ini, StratifiedGroupKFold error saat mengurutkan campuran int dan str.
    m["group"] = m["group"].astype(str)
    # kalau external_cataract kosong (folder tidak ditemukan/0 file), kolom "label" hasil
    # concat bisa berubah jadi dtype 'object' (bukan int), dan itu membuat sklearn gagal
    # mengenali tipe target ("unknown" bukan "multiclass"). Paksa balik ke int di sini.
    m["label"] = m["label"].astype(int)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    m["fold"] = -1
    for k, (_, idx) in enumerate(sgkf.split(m, m.label, m.group)):
        m.loc[idx, "fold"] = k
    m["split"] = "train"
    m.loc[m.fold < TEST_FOLDS, "split"] = "test"
    m.loc[(m.fold >= TEST_FOLDS) & (m.fold < TEST_FOLDS + VAL_FOLDS), "split"] = "val"
    m = m.drop(columns="fold")

    print("\n=== Jumlah per split dan kelas ===")
    tabel = m.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    print(tabel.loc[["train", "val", "test"]])

    ids = {s: set(m[m.split == s].group) for s in ("train", "val", "test")}
    print("\n=== Cek pasien bocor ===")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = len(ids[a] & ids[b])
        print(f"{a} vs {b}: {n} pasien tumpang tindih", "(OK)" if n == 0 else "(BOCOR!)")

    # ---------- simpan (selalu cadangkan split lama, jangan pernah menimpa diam-diam) ----------
    if OUT_PATH.exists():
        backup = OUT_PATH.with_name("split_backup.csv")
        i = 1
        while backup.exists():
            i += 1
            backup = OUT_PATH.with_name(f"split_backup_{i}.csv")
        OUT_PATH.rename(backup)
        print(f"\nsplit.csv lama dicadangkan ke {backup}")
    m.to_csv(OUT_PATH, index=False)
    print(f"Tersimpan: {OUT_PATH}")


if __name__ == "__main__":
    main()