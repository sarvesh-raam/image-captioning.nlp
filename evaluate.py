"""
Evaluates a checkpoint on the Karpathy test split with the standard COCO caption metrics
(BLEU-1..4, ROUGE-L, CIDEr; METEOR and SPICE too when Java is installed).

    python evaluate.py --checkpoint checkpoints/clip-vit-b16_scst_best.pth
    python evaluate.py --checkpoint ... --beam-sizes 1 3 5 --ablate-penalty   # ablation table

Runs on cached features (extract_features.py) for the checkpoint's encoder.
Writes results/<name>.json, results/<name>.md and results/<name>_captions.json (for error analysis).
"""
import argparse
import json
import os
import shutil

import torch
from PIL import Image
from tqdm import tqdm

from data_loader import Vocabulary, load_split
from inference import CaptionGenerator, load_model

PUNCTUATION = {".", ",", "!", "?"}


def coco_metrics(hypotheses, references):
    """hypotheses: {id: [str]}, references: {id: [str, ...]} -- already tokenised, lowercase."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    scorers = [(Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]), (Rouge(), "ROUGE-L"), (Cider(), "CIDEr")]
    if shutil.which("java"):
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice
        scorers += [(Meteor(), "METEOR"), (Spice(), "SPICE")]
    else:
        print("[INFO] Java not found: skipping METEOR and SPICE.")

    results = {}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(references, hypotheses)
        if isinstance(names, list):
            results.update(dict(zip(names, score)))
        else:
            results[names] = score
    return {k: round(float(v), 4) for k, v in results.items()}


def clean(text):
    return " ".join(t for t in Vocabulary.tokenizer_eng(text) if t not in PUNCTUATION)


def run(generator, model, images, features, beam_size, repetition_penalty, device, image_dir=None):
    """Captions every image, from cached features or (when image_dir is given) from the image files."""
    model.eval()
    hyps, refs, captions = {}, {}, []
    with torch.inference_mode():
        for img in tqdm(images, desc=f"beam={beam_size} penalty={repetition_penalty}", leave=False):
            if image_dir:
                try:
                    image = Image.open(os.path.join(image_dir, img["file"])).convert("RGB")
                except OSError:
                    continue  # unreadable file (also skipped by extract_features.py)
                memory = generator.encode(image)
            else:
                feats = torch.from_numpy(features[img["row"]].astype("float32"))[None].to(device)
                memory = model.encoder.encode_features(feats)
            ids, _ = generator.beam_search(memory, beam_size=beam_size, repetition_penalty=repetition_penalty)[0]
            text = " ".join(generator._to_words(ids))
            hyps[img["cocoid"]] = [clean(text) or "a"]
            refs[img["cocoid"]] = [clean(c) for c in img["captions"]]
            captions.append({"file": img["file"], "prediction": text, "references": img["captions"]})
    return coco_metrics(hyps, refs), captions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--beam-sizes", type=int, nargs="+", default=[3])
    parser.add_argument("--ablate-penalty", action="store_true", help="also run without the repetition penalty")
    parser.add_argument("--name", default=None, help="results file name (default: checkpoint name)")
    parser.add_argument("--from-images", action="store_true",
                        help="run the full model on the image files instead of cached features")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_model(args.checkpoint, device, load_backbone=args.from_images)
    generator = CaptionGenerator(model, vocab, device)
    encoder_type = model.encoder.encoder_type

    if args.from_images:
        with open(os.path.join(args.data_dir, "dataset.json")) as f:
            dataset = json.load(f)
        images, features, image_dir = dataset["images"], None, dataset["image_dir"]
        images = [img for img in images if os.path.exists(os.path.join(image_dir, img["file"]))]
    else:
        images, features = load_split(args.data_dir, encoder_type)
        image_dir = None
    images = [img for img in images if img["split"] == args.split][: args.max_images]
    print(f"Evaluating {args.checkpoint} ({encoder_type}) on {len(images)} {args.split} images")

    rows = []
    penalties = [2.0, 0.0] if args.ablate_penalty else [2.0]
    for beam_size in args.beam_sizes:
        for penalty in penalties:
            metrics, captions = run(generator, model, images, features, beam_size, penalty, device, image_dir)
            rows.append({"beam_size": beam_size, "repetition_penalty": penalty, **metrics})
            print(rows[-1])

    name = args.name or os.path.splitext(os.path.basename(args.checkpoint))[0]
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump({"checkpoint": args.checkpoint, "split": args.split, "images": len(images), "results": rows}, f,
                  indent=1)
    with open(f"results/{name}_captions.json", "w") as f:
        json.dump(captions, f, indent=1)  # predictions from the last configuration
    keys = list(rows[0].keys())
    with open(f"results/{name}.md", "w") as f:
        f.write(f"Results for `{args.checkpoint}` on {len(images)} {args.split} images\n\n")
        f.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[k]) for k in keys) + " |\n")
    print(f"Wrote results/{name}.json and results/{name}.md")


if __name__ == "__main__":
    main()
