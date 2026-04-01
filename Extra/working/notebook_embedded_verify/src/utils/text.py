from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from", "has", "i", "if",
    "in", "is", "it", "my", "of", "on", "or", "so", "that", "the", "their", "then", "they", "this",
    "to", "was", "with", "you"
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./%-]+", normalize_text(text))


def extract_ngrams(text: str, n_values: tuple[int, ...] = (1, 2)) -> list[str]:
    tokens = [token for token in tokenize(text) if len(token) > 2 and token not in STOPWORDS]
    ngrams: list[str] = []
    for n_value in n_values:
        if len(tokens) < n_value:
            continue
        for index in range(len(tokens) - n_value + 1):
            ngrams.append(" ".join(tokens[index : index + n_value]))
    return ngrams


def top_ngrams(texts: Iterable[str], top_n: int = 5, min_count: int = 2) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(set(extract_ngrams(text)))
    ranked = [ngram for ngram, count in counter.most_common() if count >= min_count]
    return ranked[:top_n]


def lexical_overlap_score(text: str, patterns: Iterable[str]) -> tuple[float, list[str]]:
    pattern_list = list(patterns)
    source_tokens = set(tokenize(text))
    matches: list[str] = []
    for pattern in pattern_list:
        pattern_tokens = set(tokenize(pattern))
        if not pattern_tokens:
            continue
        if pattern_tokens.issubset(source_tokens) or len(pattern_tokens & source_tokens) >= max(1, len(pattern_tokens) // 2):
            matches.append(pattern)
    unique_matches = list(dict.fromkeys(matches))
    if not pattern_list:
        return 0.0, []
    return min(1.0, len(unique_matches) / max(1, len(pattern_list))), unique_matches
