import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
from collections import Counter
from torchvision import transforms

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Basic word-based tokenizer (can use spacy for better quality)
        return str(text).lower().split()

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

class COCODataset(Dataset):
    def __init__(self, root_dir, list_file, vocab, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab

        with open(list_file, 'r') as f:
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_name = self.data_list[index]["image"]
        caption = self.data_list[index]["caption"]
        
        img_path = os.path.join(self.root_dir, image_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to first image if current is corrupted
            print(f"Skipping corrupted image {image_name}: {e}")
            return self.__getitem__(0)

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return images, targets

def get_loader(root_folder, list_file, vocab, transform, batch_size=8, num_workers=0, shuffle=True, pin_memory=True):
    dataset = COCODataset(root_folder, list_file, vocab, transform=transform)
    pad_idx = vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset

def get_transforms(image_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
