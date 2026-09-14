import zipfile
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


PATHMNIST_LABELS = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}


def load_npz_slice(npz_path, key, n):
    with zipfile.ZipFile(npz_path) as z:
        with z.open(f"{key}.npy") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
            assert not fortran_order
            n = min(n, shape[0])
            item_size = dtype.itemsize * int(np.prod(shape[1:]))
            data = f.read(n * item_size)
            arr = np.frombuffer(data, dtype=dtype).reshape((n,) + shape[1:])
            return arr.copy()


class PathMNISTSubset(Dataset):
    def __init__(self, npz_path, split, n_samples, transform=None):
        self.images = load_npz_slice(npz_path, f"{split}_images", n_samples)
        self.labels = load_npz_slice(npz_path, f"{split}_labels", n_samples)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_pathmnist_loader(npz_path, n_samples, split="train", batch_size=16, shuffle=True):
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    dataset = PathMNISTSubset(npz_path, split, n_samples, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def translate_label(labels):
    if isinstance(labels, int):
        labels = [labels]
    return [PATHMNIST_LABELS[int(lbl)] for lbl in labels]
