"""
Shrinks a training checkpoint for deployment: drops optimizer/scheduler state, stores weights as
float16, and drops frozen backbone weights when they are identical to the pretrained ones
(they are then reloaded from torchvision / the Hugging Face Hub at startup).

    python export_model.py checkpoints/clip-vit-b16_scst_best.pth checkpoints/model.pth
"""
import argparse
import os

import torch

from inference import BACKBONE_PREFIXES, load_model

KEEP_KEYS = ("vocab", "encoder_type", "embed_dim", "num_heads", "num_layers", "stage", "epoch", "val_cider")


def backbone_matches_pretrained(state_dict, encoder_type):
    """True if the checkpoint's backbone weights (incl. BatchNorm statistics) equal the pretrained ones."""
    from model import CNNEncoder
    reference = CNNEncoder(encoder_type=encoder_type, pretrained=True).state_dict()
    for key, value in state_dict.items():
        if key.startswith(BACKBONE_PREFIXES):
            ref = reference.get(key[len("encoder."):])
            if ref is None or ref.shape != value.shape or not torch.equal(ref, value.to(ref.dtype)):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    encoder_type = checkpoint.get("encoder_type", "resnet50")

    if any(k.startswith(BACKBONE_PREFIXES) for k in state_dict):
        if backbone_matches_pretrained(state_dict, encoder_type):
            print("Backbone is unchanged from pretrained weights: dropping it.")
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(BACKBONE_PREFIXES)}
        else:
            print("Backbone differs from pretrained weights (e.g. updated BatchNorm statistics): keeping it.")

    exported = {k: checkpoint[k] for k in KEEP_KEYS if k in checkpoint}
    exported["encoder_type"] = encoder_type
    exported["model_state_dict"] = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    torch.save(exported, args.target)

    # Sanity check: the exported file must load and produce the same architecture
    load_model(args.target, torch.device("cpu"))
    mb = lambda p: os.path.getsize(p) / 1e6
    print(f"{args.source} ({mb(args.source):.0f} MB) -> {args.target} ({mb(args.target):.0f} MB)")


if __name__ == "__main__":
    main()
