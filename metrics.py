"""
CIDEr-D (Vedantam et al., 2015), matching the coco-caption reference implementation.
Used for model selection and as the SCST reward; evaluate.py reports the official
pycocoevalcap numbers.
"""
import math
from collections import Counter, defaultdict

import numpy as np


def ngram_counts(tokens, n=4):
    return Counter(tuple(tokens[i:i + k]) for k in range(1, n + 1) for i in range(len(tokens) - k + 1))


class CiderD:
    def __init__(self, reference_corpus, n=4, sigma=6.0):
        """reference_corpus: list (one entry per image) of lists of tokenised reference captions."""
        self.n, self.sigma = n, sigma
        self.df = Counter()
        for refs in reference_corpus:
            for ngram in set(ng for ref in refs for ng in ngram_counts(ref, n)):
                self.df[ngram] += 1
        self.log_ref_len = math.log(float(len(reference_corpus)))

    def vector(self, tokens):
        counts = ngram_counts(tokens, self.n)
        vec = [defaultdict(float) for _ in range(self.n)]
        norm = [0.0] * self.n
        for ngram, tf in counts.items():
            k = len(ngram) - 1
            vec[k][ngram] = float(tf) * (self.log_ref_len - math.log(max(1.0, self.df[ngram])))
            norm[k] += vec[k][ngram] ** 2
        return vec, [math.sqrt(x) for x in norm], len(tokens)

    def _sim(self, hyp, ref):
        (vec_h, norm_h, len_h), (vec_r, norm_r, len_r) = hyp, ref
        penalty = math.exp(-((len_h - len_r) ** 2) / (2 * self.sigma ** 2))
        val = np.zeros(self.n)
        for k in range(self.n):
            for ngram, weight in vec_h[k].items():
                # CIDEr-D clips candidate counts to the reference counts
                val[k] += min(weight, vec_r[k].get(ngram, 0.0)) * vec_r[k].get(ngram, 0.0)
            if norm_h[k] != 0 and norm_r[k] != 0:
                val[k] /= norm_h[k] * norm_r[k]
            val[k] *= penalty
        return val

    def score(self, hyp_tokens, ref_vectors):
        """ref_vectors: list of self.vector(ref) for one image (precompute them once)."""
        hyp = self.vector(hyp_tokens)
        total = sum(self._sim(hyp, ref) for ref in ref_vectors)
        return float(np.mean(total) / len(ref_vectors) * 10.0)

    def corpus_score(self, hypotheses, references):
        """Mean CIDEr-D over images. hypotheses: token lists; references: lists of token lists."""
        scores = [self.score(h, [self.vector(r) for r in refs]) for h, refs in zip(hypotheses, references)]
        return float(np.mean(scores)) if scores else 0.0
