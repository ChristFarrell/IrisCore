"""
Aturan klasifikasi keyword ODIR per mata. Dipakai bersama oleh manifest.py
(untuk membuat split.csv) dan train.py (untuk analisis recall per subtipe),
supaya definisinya selalu sinkron.
"""
import re

LABEL_NAMES = {0: "normal", 1: "cataract", 2: "other"}
OTHER_SUBTYPES = ["D", "G", "A", "H", "M", "O"]

# Boleh diubah kalau perlu, tapi jaga tetap sama antara manifest.py dan train.py
ALLOW_MULTI_CATEGORY_OTHER = False   # True = mata dgn >1 kategori penyakit boleh masuk "other"
STRICT_MULTI_TOKEN = False           # True = mata dgn >1 keyword (walau 1 kategori) dibuang


def split_tokens(text):
    if not isinstance(text, str):
        return []
    parts = re.split(r"[,，;；、]+", text.lower())
    return [p.strip() for p in parts if p.strip()]


def token_category(tok):
    """Kode kategori ODIR untuk satu keyword.
    Keyword yang tidak cocok kategori manapun (termasuk keyword kualitas
    gambar seperti 'lens dust') jatuh ke 'O' (other/lainnya)."""
    if "normal fundus" in tok:
        return "N"
    if "cataract" in tok:
        return "C"
    if "glaucoma" in tok:
        return "G"
    if "hypertensi" in tok:
        return "H"
    if "myopi" in tok:
        return "M"
    if "diabet" in tok or "proliferative" in tok:
        return "D"
    if "macular degeneration" in tok:
        return "A"
    return "O"


def classify_eye(text):
    """Klasifikasi satu mata dari teks Diagnostic Keywords.
    Return: (label, subtype, alasan_dibuang)
      label   : 0 normal | 1 cataract | 2 other | None jika dibuang
      subtype : salah satu dari OTHER_SUBTYPES jika label == 2, selain itu None
    """
    toks = split_tokens(text)
    if not toks:
        return None, None, "tanpa keyword"
    cats = {token_category(t) for t in toks}

    if cats == {"N"}:
        return 0, None, ""
    if "N" in cats:
        return None, None, "normal bercampur penyakit/keterangan lain"
    if cats == {"C"}:
        return 1, None, ""
    if "C" in cats:
        return None, None, "katarak + penyakit lain"

    if len(cats) > 1 and not ALLOW_MULTI_CATEGORY_OTHER:
        return None, None, "lebih dari 1 kategori penyakit"
    if STRICT_MULTI_TOKEN and len(set(toks)) > 1:
        return None, None, "lebih dari 1 keyword penyakit"

    subtype = sorted(cats)[0]
    return 2, subtype, ""