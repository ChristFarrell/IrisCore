import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["normal", "cataract", "other"]   # 0, 1, 2
NUM_CLASSES = len(CLASS_NAMES)


def pad_to_square(pil_img, background_color=(0, 0, 0)):
    """Menambah padding hitam supaya imej menjadi segi empat sama sempurna 
    tanpa merosakkan nisbah asal (mencegah imej fundus daripada leper/penyek)."""
    w, h = pil_img.size
    if w == h:
        return pil_img
    max_side = max(w, h)
    new_img = Image.new(pil_img.mode, (max_side, max_side), background_color)
    new_img.paste(pil_img, ((max_side - w) // 2, (max_side - h) // 2))
    return new_img


class FundusDataset(Dataset):
    """Membaca data/split.csv. Label: 0 = normal, 1 = katarak, 2 = penyakit lain."""

    def __init__(self, split, transform=None, csv_path="data/split.csv"):
        assert split in ("train", "val", "test"), "split harus train/val/test"
        df = pd.read_csv(csv_path)
        self.df = df[df.split == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path).convert("RGB")
        
        # 1. Gunakan Square Padding terlebih dahulu
        img = pad_to_square(img)
        
        # 2. Jalankan transformasi torchvision jika ada
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(int(r.label), dtype=torch.long)

    @property
    def labels(self):
        return self.df.label.values

    def class_counts(self):
        return np.bincount(self.df.label.values, minlength=NUM_CLASSES)

    def class_weights(self, power=0.5):
        """Bobot kelas untuk CrossEntropyLoss: (N / (K * n_kelas)) ** power.
        power 1.0 = bobot penuh, 0.5 = akar (lebih lembut), 0 = tanpa bobot."""
        c = self.class_counts().astype(float)
        w = (c.sum() / (NUM_CLASSES * np.maximum(c, 1))) ** power
        return torch.tensor(w, dtype=torch.float32)


if __name__ == "__main__":
    # Tes cepat dari folder Cataract:  python -m src.data.dataset
    from torchvision import transforms
    from torch.utils.data import DataLoader

    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    for s in ["train", "val", "test"]:
        ds = FundusDataset(s, tf)
        print(s, "jumlah:", len(ds), "| per kelas:", dict(zip(CLASS_NAMES, ds.class_counts())))

    train_ds = FundusDataset("train", tf)
    print("bobot kelas (power 0.5):", train_ds.class_weights(0.5).tolist())
    x, y = next(iter(DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)))
    print("bentuk batch:", x.shape, "| label:", y.tolist(), "| dtype:", y.dtype)