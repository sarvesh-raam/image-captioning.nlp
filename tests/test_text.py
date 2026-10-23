from data_loader import Vocabulary
from inference import format_caption
from metrics import CiderD


def test_tokenizer_splits_punctuation():
    assert Vocabulary.tokenizer_eng("A man's dog, on the couch.") == ["a", "man's", "dog", ",", "on", "the", "couch", "."]


def test_format_caption():
    assert format_caption(["a", "kitchen", "with", "a", "stove", "."]) == "A kitchen with a stove."
    assert format_caption(["a", "dog", ",", "a", "cat", "<EOS>"]) == "A dog, a cat."
    assert format_caption(["<UNK>"]) == ""


def test_cider_prefers_matching_caption():
    refs = [["a", "dog", "on", "a", "couch"], ["a", "brown", "dog", "lying", "on", "a", "sofa"]]
    corpus = [refs, [["a", "train", "at", "a", "station"]], [["a", "plate", "of", "food"]]]
    cider = CiderD(corpus)
    vecs = [cider.vector(r) for r in refs]
    good = cider.score(["a", "dog", "on", "a", "couch"], vecs)
    bad = cider.score(["a", "train", "at", "a", "station"], vecs)
    assert good > bad >= 0.0
