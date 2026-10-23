import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loader import Vocabulary  # noqa: E402
from model import ImageCaptioningModel  # noqa: E402


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """A small, randomly initialised ResNet checkpoint: fast to build, no downloads needed."""
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(["a dog on a couch .", "a cat in a kitchen with a stove ."])
    model = ImageCaptioningModel(vocab_size=len(vocab), embed_dim=64, num_heads=4, num_layers=2, pretrained=False)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pth"
    torch.save({"model_state_dict": model.state_dict(), "vocab": vocab, "num_heads": 4}, path)
    return str(path)
