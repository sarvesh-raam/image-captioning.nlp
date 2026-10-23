"""
Trains the captioner on cached backbone features (run prepare_data.py and extract_features.py first).

Stage 1 - cross-entropy (teacher forcing):
    python train.py --encoder clip-vit-b16
Stage 2 - self-critical sequence training, optimises CIDEr directly (starts from the stage 1 best):
    python train.py --encoder clip-vit-b16 --scst

Both stages resume automatically, select the best epoch by validation CIDEr and stop early.
Checkpoints: checkpoints/<encoder>_<stage>_best.pth / _last.pth; metrics: checkpoints/<encoder>_history.json
"""
import argparse
import json
import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader import Vocabulary, get_feature_loader, load_split
from inference import greedy_decode, ids_to_words, sample_decode
from metrics import CiderD
from model import ImageCaptioningModel


class ImageFeatures(Dataset):
    """One sample per image (used for validation and SCST)."""

    def __init__(self, images, features):
        self.images, self.features = images, features

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.features[self.images[i]["row"]], dtype=np.float32)), i


def tokenized_refs(images):
    return [[Vocabulary.tokenizer_eng(c) for c in img["captions"]] for img in images]


@torch.no_grad()
def validate(model, images, features, vocab, device, batch_size=128):
    """Greedy-decodes every validation image and returns CIDEr-D (df computed on the val references)."""
    model.eval()
    refs = tokenized_refs(images)
    hyps = [None] * len(images)
    loader = DataLoader(ImageFeatures(images, features), batch_size=batch_size)
    for feats, idx in tqdm(loader, desc="Validating", leave=False):
        memory = model.encoder.encode_features(feats.to(device))
        ids = greedy_decode(model, memory, vocab.stoi["<SOS>"], vocab.stoi["<EOS>"])
        for i, row in zip(idx.tolist(), ids.tolist()):
            hyps[i] = ids_to_words(row, vocab)
    cider = CiderD(refs).corpus_score(hyps, refs)
    return cider, hyps[:3]


def xe_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp, desc):
    model.train()
    total = 0.0
    loop = tqdm(loader, desc=desc, leave=False)
    for feats, captions in loop:
        feats, captions = feats.to(device, non_blocking=True), captions.to(device, non_blocking=True)
        # Teacher forcing: input is everything except the last token, target is shifted by one
        inputs, targets = captions[:, :-1], captions[:, 1:]
        with torch.autocast(device.type, enabled=use_amp):
            logits = model.decoder(inputs, model.encoder.encode_features(feats))
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
        loop.set_postfix(loss=f"{loss.item():.3f}")
    return total / max(len(loader), 1)


def scst_epoch(model, images, features, vocab, cider, optimizer, device, batch_size, num_samples, desc):
    """
    Self-critical sequence training (Rennie et al., 2017) with the "mean of other samples" baseline:
    each image gets `num_samples` sampled captions, and each one is rewarded by how much its CIDEr-D
    beats the average of the others.
    """
    sos, eos, pad = vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]
    loader = DataLoader(ImageFeatures(images, features), batch_size=batch_size, shuffle=True)
    total_reward = 0.0
    loop = tqdm(loader, desc=desc, leave=False)
    for feats, idx in loop:
        # Dropout stays off so the sampling and gradient distributions match
        model.eval()
        memory = model.encoder.encode_features(feats.to(device)).repeat_interleave(num_samples, dim=0)
        with torch.no_grad():
            samples = sample_decode(model, memory, sos, eos)

        rewards = torch.zeros(samples.size(0))
        for j, row in enumerate(samples.tolist()):
            refs = images[idx[j // num_samples].item()]["captions"]
            ref_vecs = [cider.vector(Vocabulary.tokenizer_eng(r)) for r in refs]
            rewards[j] = cider.score(ids_to_words(row, vocab), ref_vecs)
        rewards = rewards.view(-1, num_samples)
        baseline = (rewards.sum(1, keepdim=True) - rewards) / (num_samples - 1)
        advantage = (rewards - baseline).view(-1).to(device)

        # Log-probabilities of the sampled tokens, up to and including the first <EOS>
        inputs = torch.cat([torch.full_like(samples[:, :1], sos), samples[:, :-1]], dim=1)
        log_probs = torch.log_softmax(model.decoder(inputs, memory), -1)
        token_lp = log_probs.gather(2, samples.unsqueeze(2)).squeeze(2)
        after_eos = (samples == eos).cumsum(1) - (samples == eos).long()
        mask = (after_eos == 0).float()
        loss = -(advantage[:, None] * token_lp * mask).sum() / mask.sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_reward += rewards.mean().item()
        loop.set_postfix(cider=f"{rewards.mean().item():.3f}")
    return total_reward / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default="clip-vit-b16")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--scst", action="store_true", help="run the SCST stage (needs a stage 1 checkpoint)")
    parser.add_argument("--epochs", type=int, default=None, help="default: 30 (XE) / 10 (SCST)")
    parser.add_argument("--batch-size", type=int, default=None, help="default: 64 (XE) / 32 images (SCST)")
    parser.add_argument("--lr", type=float, default=None, help="default: 3e-4 (XE) / 1e-5 (SCST)")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--scst-samples", type=int, default=5)
    parser.add_argument("--val-images", type=int, default=1000, help="validation images used per epoch")
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    stage = "scst" if args.scst else "xe"
    epochs = args.epochs or (10 if args.scst else 30)
    batch_size = args.batch_size or (32 if args.scst else 64)
    lr = args.lr or (1e-5 if args.scst else 3e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[INFO] Stage {stage.upper()} on {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_amp else ""))

    with open(os.path.join(args.data_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    images, features = load_split(args.data_dir, args.encoder)
    train_images = [img for img in images if img["split"] == "train"]
    val_images = [img for img in images if img["split"] == "val"][: args.val_images]
    print(f"Images: {len(train_images)} train / {len(val_images)} val | vocab {len(vocab)}")

    model = ImageCaptioningModel(
        vocab_size=len(vocab), embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, encoder_type=args.encoder, load_backbone=False,
    ).to(device)

    os.makedirs("checkpoints", exist_ok=True)
    prefix = f"checkpoints/{args.encoder}_{stage}"
    best_path, last_path = f"{prefix}_best.pth", f"{prefix}_last.pth"
    history_path = f"checkpoints/{args.encoder}_history.json"
    history = json.load(open(history_path)) if os.path.exists(history_path) else []

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=lr, weight_decay=0.01 if not args.scst else 0.0)

    if args.scst:
        train_loader = None
        steps_per_epoch = math.ceil(len(train_images) / batch_size)
        cider_train = CiderD(tokenized_refs(train_images))  # df from the training corpus
    else:
        train_loader = get_feature_loader(train_images, features, vocab, batch_size=batch_size,
                                          num_workers=args.workers)
        steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    def lr_lambda(step):  # linear warmup, then cosine decay
        if args.scst:
            return 1.0
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"], label_smoothing=0.1)

    start_epoch, best_cider, bad_epochs = 0, -1.0, 0
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch, best_cider, bad_epochs = ckpt["epoch"], ckpt["best_cider"], ckpt.get("bad_epochs", 0)
        print(f"Resuming {stage} from epoch {start_epoch} (best val CIDEr {best_cider:.3f})")
    elif args.scst:
        xe_best = f"checkpoints/{args.encoder}_xe_best.pth"
        if not os.path.exists(xe_best):
            raise SystemExit(f"SCST needs a cross-entropy checkpoint first: {xe_best}")
        model.load_state_dict(torch.load(xe_best, map_location=device, weights_only=False)["model_state_dict"],
                              strict=False)
        best_cider, _ = validate(model, val_images, features, vocab, device)
        print(f"Starting SCST from {xe_best} (val CIDEr {best_cider:.3f})")

    for epoch in range(start_epoch, epochs):
        desc = f"{stage.upper()} epoch {epoch + 1}/{epochs}"
        if args.scst:
            train_metric = scst_epoch(model, train_images, features, vocab, cider_train, optimizer, device,
                                      batch_size, args.scst_samples, desc)
        else:
            train_metric = xe_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device,
                                    use_amp, desc)
        val_cider, examples = validate(model, val_images, features, vocab, device)
        label = "train CIDEr" if args.scst else "train loss"
        print(f"{desc}: {label} {train_metric:.4f} | val CIDEr {val_cider:.4f} | e.g. {' '.join(examples[0])}")
        history.append({"stage": stage, "epoch": epoch + 1, "train": train_metric, "val_cider": val_cider})
        with open(history_path, "w") as f:
            json.dump(history, f, indent=1)

        improved = val_cider > best_cider
        if improved:
            best_cider, bad_epochs = val_cider, 0
        else:
            bad_epochs += 1

        state = {
            "model_state_dict": model.trainable_state_dict(),
            "vocab": vocab,
            "encoder_type": args.encoder,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "stage": stage,
            "epoch": epoch + 1,
            "val_cider": val_cider,
            "best_cider": best_cider,
            "bad_epochs": bad_epochs,
        }
        if improved:
            torch.save(state, best_path)
            print(f"New best model saved to {best_path} (val CIDEr {best_cider:.4f})")
        torch.save({**state, "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()}, last_path)
        if bad_epochs >= args.patience:
            print(f"No validation improvement for {args.patience} epochs. Stopping early.")
            break


if __name__ == "__main__":
    main()
