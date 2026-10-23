"""
Prepares the COCO Karpathy split (113k train / 5k val / 5k test images, ~5 captions each).

    python prepare_data.py                      # full dataset (~19 GB of images)
    python prepare_data.py --max-train 20000    # smaller subset for quicker experiments
    python prepare_data.py --only-local --holdout 250
        # use only images already in coco_images/ and hold some of them out as val/test
        # (handy for smoke tests on a laptop; not comparable to published results)

Writes data/dataset.json and data/vocab.pkl.
"""
import argparse
import concurrent.futures
import json
import os
import pickle
import random

import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from data_loader import Vocabulary

KARPATHY_REPO = "yerevann/coco-karpathy"
PARTS = {"train": "train", "restval": "train", "validation": "val", "test": "test"}


def load_karpathy():
    frames = []
    for part, split in PARTS.items():
        path = hf_hub_download(KARPATHY_REPO, f"data/{part}-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(path)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def download_image(row, image_dir):
    path = os.path.join(image_dir, row["filepath"], row["filename"])
    if os.path.exists(path):
        return True
    try:
        response = requests.get(row["url"], timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(response.content)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", default="coco_images")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--max-train", type=int, default=None, help="limit the number of training images")
    parser.add_argument("--max-eval", type=int, default=None, help="limit the number of val/test images each")
    parser.add_argument("--only-local", action="store_true", help="skip downloading; keep images already on disk")
    parser.add_argument("--holdout", type=int, default=0,
                        help="move this many training images into each of val and test (for --only-local runs)")
    parser.add_argument("--freq-threshold", type=int, default=5)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    df = load_karpathy()
    print(f"Karpathy split: {df['split'].value_counts().to_dict()}")

    if args.only_local:
        exists = df.apply(lambda r: os.path.exists(os.path.join(args.image_dir, r["filepath"], r["filename"])), axis=1)
        df = df[exists]
        print(f"Images found locally: {len(df)}")
    else:
        subsets = []
        for split, group in df.groupby("split"):
            limit = args.max_train if split == "train" else args.max_eval
            subsets.append(group.sample(n=min(limit, len(group)), random_state=0) if limit else group)
        df = pd.concat(subsets)
        print(f"Downloading up to {len(df)} images to {args.image_dir}/ ...")
        rows = df.to_dict("records")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            ok = list(tqdm(pool.map(lambda r: download_image(r, args.image_dir), rows), total=len(rows)))
        df = df[ok]

    images = [
        {
            "file": f"{r['filepath']}/{r['filename']}",
            "cocoid": int(r["cocoid"]),
            "split": r["split"],
            "captions": [s.strip() for s in r["sentences"]],
        }
        for r in df.to_dict("records")
    ]

    if args.holdout:
        train = [img for img in images if img["split"] == "train"]
        random.Random(0).shuffle(train)
        for img in train[: args.holdout]:
            img["split"] = "val"
        for img in train[args.holdout: 2 * args.holdout]:
            img["split"] = "test"

    counts = {s: sum(img["split"] == s for img in images) for s in ("train", "val", "test")}
    print(f"Final dataset: {counts}")

    # The vocabulary is built from training captions only, so val/test stay unseen
    vocab = Vocabulary(freq_threshold=args.freq_threshold)
    vocab.build_vocabulary(cap for img in images if img["split"] == "train" for cap in img["captions"])
    print(f"Vocabulary size: {len(vocab)}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dataset.json"), "w") as f:
        json.dump({"image_dir": args.image_dir, "images": images}, f)
    with open(os.path.join(args.out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    print(f"Wrote {args.out_dir}/dataset.json and {args.out_dir}/vocab.pkl")


if __name__ == "__main__":
    main()
