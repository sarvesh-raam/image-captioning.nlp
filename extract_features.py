"""
Runs the frozen backbone over every image once and caches its output tokens, so training only
has to run the (much smaller) encoder head and decoder.

    python extract_features.py --encoder clip-vit-b16      # recommended
    python extract_features.py --encoder resnet50

Writes data/features_<encoder>.npy (float16, memory-mapped) and data/features_<encoder>.json.
Disk usage per image: clip-vit-b16 ~300 KB, resnet50 ~200 KB, clip-vit-b32 ~77 KB.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader import get_transforms
from model import ENCODERS, CNNEncoder


class ImageFiles(Dataset):
    def __init__(self, image_dir, files, transform):
        self.image_dir, self.files, self.transform = image_dir, files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            image = Image.open(os.path.join(self.image_dir, self.files[i])).convert("RGB")
            return self.transform(image), True
        except OSError:
            return torch.zeros(3, 224, 224), False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="clip-vit-b16", choices=list(ENCODERS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.data_dir, "dataset.json")) as f:
        dataset = json.load(f)
    files = [img["file"] for img in dataset["images"]]

    encoder = CNNEncoder(encoder_type=args.encoder, pretrained=True).to(device).eval()
    spec = ENCODERS[args.encoder]
    out_path = os.path.join(args.data_dir, f"features_{args.encoder}.npy")
    features = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float16, shape=(len(files), spec["tokens"], spec["feature_dim"])
    )

    transform = get_transforms(encoder_type=args.encoder)
    loader = DataLoader(ImageFiles(dataset["image_dir"], files, transform),
                        batch_size=args.batch_size, num_workers=args.workers)

    valid = []
    row = 0
    with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for images, ok in tqdm(loader, desc=f"Extracting {args.encoder} features"):
            feats = encoder.extract_features(images.to(device))
            features[row: row + len(images)] = feats.float().cpu().numpy().astype(np.float16)
            valid.extend(ok.tolist())
            row += len(images)
    features.flush()

    # Rows of unreadable images are recorded as null so they are skipped at training time
    meta = {
        "encoder": args.encoder,
        "shape": list(features.shape),
        "images": [f if ok else None for f, ok in zip(files, valid)],
    }
    with open(os.path.join(args.data_dir, f"features_{args.encoder}.json"), "w") as f:
        json.dump(meta, f)
    print(f"Saved {sum(valid)} / {len(files)} feature maps to {out_path}")


if __name__ == "__main__":
    main()
