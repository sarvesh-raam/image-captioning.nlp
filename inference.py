import math
import os
import pickle

import torch
import torch.nn as nn
from data_loader import get_transforms
from model import ImageCaptioningModel

SPECIAL_TOKENS = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}
# Function words that are allowed to repeat without penalty during beam search
GLUE_WORDS = {"a", "an", "the", "in", "and", "is", "of", "on", "to", "with", "at", "by", "for", ".", ","}
BACKBONE_PREFIXES = ("encoder.resnet.", "encoder.clip.")


def _infer_architecture(state_dict):
    """Recover embed_dim / num_layers from older checkpoints that did not store them."""
    embed_dim = state_dict["encoder.cnn_projection.weight"].shape[0]
    prefix = "decoder.transformer_decoder.layers."
    layer_ids = {int(k[len(prefix):].split(".")[0]) for k in state_dict if k.startswith(prefix)}
    num_layers = max(layer_ids) + 1 if layer_ids else 6
    return embed_dim, num_layers


def load_model(checkpoint_path, device, vocab_path="vocabulary.pkl", quantize=False, load_backbone=True):
    """
    Build an ImageCaptioningModel from a checkpoint. Returns (model, vocab).
    load_backbone=False skips the frozen backbone (for running on cached features).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # The vocab saved inside the checkpoint is the one the weights were trained with,
    # so it always takes priority over a standalone vocabulary.pkl.
    if "vocab" in checkpoint:
        vocab = checkpoint["vocab"]
    elif os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    else:
        raise FileNotFoundError("No vocabulary found in checkpoint or at " + vocab_path)

    embed_dim, num_layers = _infer_architecture(state_dict)
    encoder_type = checkpoint.get("encoder_type", "resnet50")

    expected_vocab = state_dict["decoder.fc_out.weight"].shape[0]
    if len(vocab) != expected_vocab:
        raise ValueError(f"Vocabulary size {len(vocab)} does not match checkpoint ({expected_vocab})")

    # Older checkpoints contain the ResNet weights; exported/new ones only store the trainable
    # parts, and the frozen backbone is loaded from its pretrained source instead.
    has_backbone = any(k.startswith(BACKBONE_PREFIXES) for k in state_dict)
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_heads=checkpoint.get("num_heads", 8),
        num_layers=num_layers,
        pretrained=not has_backbone,
        encoder_type=encoder_type,
        load_backbone=load_backbone,
    )
    if not load_backbone:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith(BACKBONE_PREFIXES)}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith(BACKBONE_PREFIXES)]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}")

    model.float().eval()
    if quantize:
        # int8 weights for Linear layers: smaller and faster on CPU, with a negligible quality cost
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return model.to(device), vocab


def format_caption(words):
    """Turn a token list into a readable sentence: 'A dog on a couch.'"""
    text = ""
    for w in words:
        if w in SPECIAL_TOKENS:
            continue
        text += w if (w in {".", ",", "!", "?"} or not text) else " " + w
    text = text.strip().rstrip(".,").strip()
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


# --- Batched decoding on encoder memory (used for validation and SCST) ---

def greedy_decode(model, memory, sos, eos, max_len=20):
    """memory: (B, T, D) -> token ids (B, <=max_len), each row cut after <EOS>."""
    ids = torch.full((memory.size(0), 1), sos, dtype=torch.long, device=memory.device)
    finished = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
    for _ in range(max_len):
        next_ids = model.decoder(ids, memory)[:, -1].argmax(-1)
        next_ids = next_ids.masked_fill(finished, eos)
        ids = torch.cat([ids, next_ids[:, None]], dim=1)
        finished |= next_ids == eos
        if finished.all():
            break
    return ids[:, 1:]


def sample_decode(model, memory, sos, eos, max_len=20, temperature=1.0):
    """Multinomial sampling (no gradients). Returns token ids (B, L) padded with <EOS>."""
    ids = torch.full((memory.size(0), 1), sos, dtype=torch.long, device=memory.device)
    finished = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
    for _ in range(max_len):
        logits = model.decoder(ids, memory)[:, -1] / temperature
        next_ids = torch.multinomial(torch.softmax(logits.float(), -1), 1).squeeze(1)
        next_ids = next_ids.masked_fill(finished, eos)
        ids = torch.cat([ids, next_ids[:, None]], dim=1)
        finished |= next_ids == eos
        if finished.all():
            break
    return ids[:, 1:]


def ids_to_words(ids, vocab):
    """One row of token ids -> list of words, stopping at <EOS> and dropping special tokens."""
    eos, words = vocab.stoi["<EOS>"], []
    for idx in ids:
        idx = int(idx)
        if idx == eos:
            break
        word = vocab.itos.get(idx, "")
        if word and word not in SPECIAL_TOKENS:
            words.append(word)
    return words


class CaptionGenerator:
    def __init__(self, model, vocab, device, image_size=224):
        self.model = model
        self.vocab = vocab
        self.device = device
        encoder = model.encoder
        self.transform = get_transforms(image_size=image_size, encoder_type=encoder.encoder_type)
        self.grid_size = encoder.grid_size
        self.num_prefix_tokens = encoder.num_prefix_tokens
        self.sos = vocab.stoi["<SOS>"]
        self.eos = vocab.stoi["<EOS>"]
        self.pad = vocab.stoi["<PAD>"]
        self.unk = vocab.stoi.get("<UNK>", 3)

    def encode(self, image):
        """PIL image -> encoder memory (1, T, D)."""
        return self.model.encoder(self.transform(image).unsqueeze(0).to(self.device))

    def _to_words(self, ids):
        return ids_to_words(ids, self.vocab)

    @torch.inference_mode()
    def generate_caption_greedy(self, image, max_len=20):
        self.model.eval()
        ids = greedy_decode(self.model, self.encode(image), self.sos, self.eos, max_len)
        return self._to_words(ids[0].tolist())

    @torch.inference_mode()
    def beam_search(self, memory, beam_size=3, max_len=20, alpha=0.7, num_results=1, repetition_penalty=2.0):
        """
        Beam search on encoder memory (1, T, D) with length normalisation and a repetition penalty.
        All live beams are decoded together in one batched forward pass per step.
        Returns up to `num_results` (token_ids, normalised_score) pairs, best first.
        """
        beams = [([self.sos], 0.0)]
        completed = []
        width = max(beam_size, num_results)

        for _ in range(max_len):
            batch = torch.tensor([ids for ids, _ in beams], device=self.device)
            log_probs = torch.log_softmax(self.model.decoder(batch, memory.expand(len(beams), -1, -1))[:, -1], dim=-1)
            topk_probs, topk_ids = log_probs.topk(width, dim=-1)

            candidates = []
            for b, (ids, score) in enumerate(beams):
                for k in range(width):
                    token_id = topk_ids[b, k].item()
                    penalty = 0.0
                    # Penalise repeated content words (stops "a man and a man and a man")
                    if token_id in ids and self.vocab.itos.get(token_id, "") not in GLUE_WORDS:
                        penalty += repetition_penalty
                    # UNK tokens get stripped later and leave holes in the sentence
                    if token_id == self.unk:
                        penalty += 5.0
                    candidates.append((ids + [token_id], score + topk_probs[b, k].item() - penalty))

            candidates.sort(key=lambda c: c[1] / (len(c[0]) ** alpha), reverse=True)
            beams = []
            for ids, score in candidates:
                if ids[-1] == self.eos:
                    completed.append((ids, score))
                else:
                    beams.append((ids, score))
                if len(beams) == width:
                    break

            if len(completed) >= width or not beams:
                break

        pool = sorted(completed + beams, key=lambda c: c[1] / (len(c[0]) ** alpha), reverse=True)
        results, seen = [], set()
        for ids, score in pool:
            text = format_caption(self._to_words(ids[1:]))
            if text and text not in seen:
                seen.add(text)
                results.append((ids[1:], score / (len(ids) ** alpha)))
            if len(results) == num_results:
                break
        return results

    @torch.inference_mode()
    def generate_caption_beam_search(self, image, beam_size=3, max_len=20):
        self.model.eval()
        best_ids, _ = self.beam_search(self.encode(image), beam_size, max_len)[0]
        return self._to_words(best_ids)

    @torch.inference_mode()
    def explain(self, memory, ids):
        """
        Teacher-forces `ids` once and returns, for each generated word, the model's probability for it
        and its cross-attention over the image grid (averaged over heads and decoder layers).
        """
        ids = [i for i in ids]
        if self.eos in ids:
            ids = ids[: ids.index(self.eos) + 1]
        inputs = torch.tensor([[self.sos] + ids[:-1]], device=self.device)

        captured = []
        layers = self.model.decoder.transformer_decoder.layers
        originals = [layer.multihead_attn.forward for layer in layers]

        def capture(original):
            def forward(*args, **kwargs):
                kwargs.update(need_weights=True, average_attn_weights=True)
                out, weights = original(*args, **kwargs)
                captured.append(weights)
                return out, weights
            return forward

        try:
            for layer, original in zip(layers, originals):
                layer.multihead_attn.forward = capture(original)
            logits = self.model.decoder(inputs, memory)
        finally:
            for layer, original in zip(layers, originals):
                layer.multihead_attn.forward = original

        probs = torch.softmax(logits[0].float(), -1)
        token_probs = probs[torch.arange(len(ids)), torch.tensor(ids, device=probs.device)]
        attention = torch.stack(captured).mean(0)[0]  # (L, T)
        attention = attention[:, self.num_prefix_tokens:]  # drop CLIP's CLS token
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-9)

        words, word_probs, maps = [], [], []
        for pos, idx in enumerate(ids):
            word = self.vocab.itos.get(idx, "")
            if idx == self.eos or word in SPECIAL_TOKENS:
                continue
            words.append(word)
            word_probs.append(round(token_probs[pos].item(), 4))
            maps.append([round(v, 4) for v in attention[pos].tolist()])
        # Confidence = geometric mean of the per-token probabilities (including <EOS>)
        confidence = math.exp(torch.log(token_probs.clamp_min(1e-9)).mean().item())
        return {"words": words, "word_probs": word_probs, "attention": maps, "confidence": confidence}

    @torch.inference_mode()
    def describe(self, image, beam_size=3, num_results=3, max_len=20):
        """Everything the API returns: top captions, confidence and per-word attention for the best one."""
        self.model.eval()
        memory = self.encode(image)
        results = self.beam_search(memory, beam_size, max_len, num_results=num_results)
        details = self.explain(memory, results[0][0])
        return {
            "captions": [{"text": format_caption(self._to_words(ids)), "score": round(score, 4)}
                         for ids, score in results],
            "confidence": round(details["confidence"], 4),
            "attention": {
                "grid": self.grid_size,
                # Region of the photo the grid covers: ResNet sees the whole (stretched) image,
                # CLIP sees a centred square crop
                "crop": "full" if self.model.encoder.encoder_type == "resnet50" else "center",
                "words": details["words"],
                "word_probs": details["word_probs"],
                "maps": details["attention"],
            },
        }
