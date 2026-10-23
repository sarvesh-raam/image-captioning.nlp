import json
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Image normalisation statistics expected by each backbone
NORMALIZATION = {
    "resnet50": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}
MAX_CAPTION_TOKENS = 30


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Lowercase and split punctuation into its own tokens ("kitchen." -> "kitchen", ".")
        return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?]", str(text).lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                for token in tokenized_text]


def load_split(data_dir, encoder_type):
    """Loads the cached feature matrix and the dataset index written by prepare_data.py."""
    with open(f"{data_dir}/dataset.json") as f:
        images = json.load(f)["images"]
    with open(f"{data_dir}/features_{encoder_type}.json") as f:
        meta = json.load(f)
    features = np.lib.format.open_memmap(f"{data_dir}/features_{encoder_type}.npy", mode="r")
    row_of = {name: i for i, name in enumerate(meta["images"])}
    # Only images whose features were extracted successfully are usable
    images = [img for img in images if img["file"] in row_of]
    for img in images:
        img["row"] = row_of[img["file"]]
    return images, features


class FeatureCaptionDataset(Dataset):
    """One sample per (image, caption) pair, reading backbone features from a memory-mapped array."""

    def __init__(self, images, features, vocab):
        self.features = features
        self.features_path = getattr(features, "filename", None)
        self.vocab = vocab
        self.samples = [(img["row"], cap) for img in images for cap in img["captions"]]

    def __getstate__(self):
        # DataLoader workers on Windows receive a pickled copy of the dataset. Send only the path of
        # the memory-mapped feature file (pickling the memmap would copy the whole array into RAM);
        # each worker reopens it on first use.
        state = self.__dict__.copy()
        if self.features_path is not None:
            state["features"] = None
        return state

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.features is None:
            self.features = np.load(self.features_path, mmap_mode="r")
        row, caption = self.samples[index]
        feats = torch.from_numpy(np.asarray(self.features[row], dtype=np.float32))
        tokens = self.vocab.numericalize(caption)[:MAX_CAPTION_TOKENS]
        ids = [self.vocab.stoi["<SOS>"]] + tokens + [self.vocab.stoi["<EOS>"]]
        return feats, torch.tensor(ids)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        features = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return features, targets


def get_feature_loader(images, features, vocab, batch_size=64, shuffle=True, num_workers=0):
    dataset = FeatureCaptionDataset(images, features, vocab)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=MyCollate(pad_idx=vocab.stoi["<PAD>"]),
    )


def get_transforms(image_size=224, encoder_type="resnet50"):
    """Inference/feature-extraction preprocessing for the given backbone."""
    if encoder_type == "resnet50":
        mean, std = NORMALIZATION["resnet50"]
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    mean, std = NORMALIZATION["clip"]
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
