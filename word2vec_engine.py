"""Training and inference helpers for the CBOW / Skip-gram demo."""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

MASK_PATTERN = re.compile(r"___|\[mask\]|\{mask\}", re.IGNORECASE)
FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i",
    "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "ours", "she", "that", "the", "their", "theirs", "them", "they", "this",
    "to", "us", "was", "we", "were", "with", "you", "your",
}


@dataclass
class TrainingResult:
    model: Word2Vec
    architecture: str
    train_sentence_count: int
    test_sentence_count: int
    vocabulary_size: int
    metrics: dict[str, float | int]
    training_metrics: dict[str, float | int]
    training_loss: float
    min_count: int
    evaluation_min_count: int
    context_counts: dict[str, dict[str, float]]
    context_totals: dict[str, float]
    word_counts: dict[str, int]


def tokenize_corpus(text: str) -> list[list[str]]:
    """Split user text into unique sentences and tokenize with Gensim's NLP helpers."""
    sentences: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for raw_sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", text):
        tokens = simple_preprocess(raw_sentence, deacc=True, min_len=1, max_len=30)
        key = tuple(tokens)
        if len(tokens) >= 3 and key not in seen:
            seen.add(key)
            sentences.append(tokens)
    return sentences


def _split_sentences(
    sentences: list[list[str]], test_fraction: float, seed: int
) -> tuple[list[list[str]], list[list[str]]]:
    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    test_count = min(test_count, len(shuffled) - 2)
    return shuffled[test_count:], shuffled[:test_count]


def _score_candidates(
    model: Word2Vec,
    context: list[str],
    architecture: str,
    context_counts: dict[str, dict[str, float]],
    context_totals: dict[str, float],
    word_counts: dict[str, int],
    exclude_words: set[str] | None = None,
    context_weights: list[float] | None = None,
) -> tuple[list[str], np.ndarray]:
    vocabulary = model.wv.index_to_key
    context_vectors = np.asarray([model.wv[word] for word in context], dtype=np.float32)
    weights = np.asarray(
        context_weights if context_weights is not None else [1.0] * len(context),
        dtype=np.float32,
    )
    if architecture == "CBOW":
        context_vector = np.average(context_vectors, axis=0, weights=weights)
        output_vectors = np.asarray(model.syn1neg, dtype=np.float32)
        scores = output_vectors @ context_vector
    else:
        context_output_vectors = np.asarray(
            [model.syn1neg[model.wv.key_to_index[word]] for word in context],
            dtype=np.float32,
        )
        candidate_vectors = np.asarray(model.wv.vectors, dtype=np.float32)
        logits = candidate_vectors @ context_output_vectors.T
        scores = (-np.logaddexp(0, -logits) * weights).sum(axis=1)

    # Word2Vec captures similarity; observed local context distinguishes words
    # that are similar but do not actually fit the surrounding sentence.
    vocabulary_size = len(vocabulary)
    total_word_count = sum(word_counts.values())
    context_scores = np.zeros(vocabulary_size, dtype=np.float64)
    for context_word, weight in zip(context, weights, strict=True):
        cooccurrences = context_counts.get(context_word, {})
        context_total = context_totals.get(context_word, 0.0)
        denominator = context_total + 0.1 * vocabulary_size
        if denominator == 0:
            continue
        for index, candidate in enumerate(vocabulary):
            observed = cooccurrences.get(candidate, 0.0)
            candidate_prior = (word_counts.get(candidate, 0) + 0.1) / (
                total_word_count + 0.1 * vocabulary_size
            )
            context_probability = (observed + 0.1) / denominator
            context_scores[index] += weight * np.log(context_probability / candidate_prior)
    if context:
        context_scores /= weights.sum()
        score_std = float(np.std(scores))
        if score_std > 1e-12:
            scores = scores / score_std
        scores = context_scores + 0.15 * scores

    if exclude_words:
        if len(exclude_words.intersection(vocabulary)) == vocabulary_size:
            raise ValueError("The context uses the entire vocabulary; add more training text.")
        scores = np.asarray(scores, dtype=np.float64)
        for word in exclude_words:
            word_index = model.wv.key_to_index.get(word)
            if word_index is not None:
                scores[word_index] = -np.inf

    scores = scores - np.max(scores)
    probabilities = np.exp(np.clip(scores, -60, 0))
    probabilities /= probabilities.sum()
    return vocabulary, probabilities


def _rank_skipgram_context(
    model: Word2Vec,
    center_word: str,
    context_counts: dict[str, dict[str, float]],
    context_totals: dict[str, float],
    word_counts: dict[str, int],
    topn: int,
) -> list[dict[str, Any]]:
    """Rank words that occur around a Skip-gram center word."""
    if center_word not in model.wv:
        raise ValueError(
            f"'{center_word}' is not in the trained vocabulary. "
            "Choose a bundled example word or use a word from the corpus."
        )

    vocabulary = model.wv.index_to_key
    center_counts = context_counts.get(center_word, {})
    candidates = [
        word
        for word in vocabulary
        if word != center_word
        and word not in FUNCTION_WORDS
        and center_counts.get(word, 0) > 0
    ]
    if not candidates:
        raise ValueError(
            f"No neighboring training words were found for '{center_word}'."
        )

    total_word_count = sum(word_counts.values())
    center_total = context_totals.get(center_word, 0.0)
    vocabulary_size = len(vocabulary)
    output_vectors = np.asarray(
        [model.syn1neg[model.wv.key_to_index[word]] for word in candidates],
        dtype=np.float32,
    )
    model_scores = output_vectors @ model.wv[center_word]
    model_std = float(np.std(model_scores))
    if model_std > 1e-12:
        model_scores = model_scores / model_std

    scores = []
    for index, candidate in enumerate(candidates):
        cooccurrence = center_counts[candidate]
        candidate_prior = (word_counts.get(candidate, 0) + 0.1) / (
            total_word_count + 0.1 * vocabulary_size
        )
        context_probability = (cooccurrence + 0.1) / (
            center_total + 0.1 * vocabulary_size
        )
        association_score = np.log(context_probability / candidate_prior)
        scores.append(float(association_score + 0.15 * model_scores[index]))

    ranking = np.argsort(scores)[::-1][: max(1, min(topn, len(candidates)))]
    ranked_scores = np.asarray([scores[index] for index in ranking], dtype=np.float64)
    ranked_scores -= np.max(ranked_scores)
    probabilities = np.exp(np.clip(ranked_scores, -60, 0))
    probabilities /= probabilities.sum()
    return [
        {"word": candidates[index], "probability": float(probability)}
        for index, probability in zip(ranking, probabilities, strict=True)
    ]


def _build_context_counts(
    sentences: list[list[str]], window: int
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, int]]:
    context_counts: dict[str, dict[str, float]] = {}
    context_totals: dict[str, float] = {}
    word_counts: dict[str, int] = {}
    for sentence in sentences:
        for index, target in enumerate(sentence):
            word_counts[target] = word_counts.get(target, 0) + 1
            for neighbor_index in range(
                max(0, index - window), min(len(sentence), index + window + 1)
            ):
                if neighbor_index == index:
                    continue
                context_word = sentence[neighbor_index]
                weight = 1.0 / abs(neighbor_index - index)
                target_counts = context_counts.setdefault(context_word, {})
                target_counts[target] = target_counts.get(target, 0.0) + weight
                context_totals[context_word] = context_totals.get(context_word, 0.0) + weight
    return context_counts, context_totals, word_counts


def _evaluate(
    model: Word2Vec,
    test_sentences: list[list[str]],
    architecture: str,
    window: int,
    context_counts: dict[str, dict[str, float]],
    context_totals: dict[str, float],
    word_counts: dict[str, int],
    min_count: int,
) -> dict[str, float | int]:
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0
    for sentence in test_sentences:
        for index, target in enumerate(sentence):
            if (
                len(target) < 3
                or target in FUNCTION_WORDS
                or word_counts.get(target, 0) < min_count
                or target not in model.wv
            ):
                continue
            if architecture == "Skip-gram":
                context = {
                    sentence[position]
                    for position in range(
                        max(0, index - window),
                        min(len(sentence), index + window + 1),
                    )
                    if position != index
                    and sentence[position] in model.wv
                    and sentence[position] not in FUNCTION_WORDS
                }
                if not context:
                    continue
                predictions = _rank_skipgram_context(
                    model,
                    target,
                    context_counts,
                    context_totals,
                    word_counts,
                    topn=5,
                )
                predicted_words = [item["word"] for item in predictions]
                total += 1
                correct_at_1 += int(
                    bool(predicted_words) and predicted_words[0] in context
                )
                correct_at_5 += int(bool(set(predicted_words) & context))
                continue

            context_positions = list(range(max(0, index - window), index)) + list(
                range(index + 1, min(len(sentence), index + window + 1))
            )
            context = [sentence[position] for position in context_positions]
            context_weights = [1.0 / abs(position - index) for position in context_positions]
            known = [
                (word, weight)
                for word, weight in zip(context, context_weights, strict=True)
                if word in model.wv
            ]
            context = [word for word, _ in known]
            context_weights = [weight for _, weight in known]
            if not context or target not in model.wv:
                continue
            words, probabilities = _score_candidates(
                model,
                context,
                architecture,
                context_counts,
                context_totals,
                word_counts,
                context_weights=context_weights,
            )
            ranking = np.argsort(probabilities)[::-1]
            ranked_words = [words[position] for position in ranking[:5]]
            total += 1
            correct_at_1 += int(ranked_words[0] == target)
            correct_at_5 += int(target in ranked_words)
    return {
        "top1_accuracy": correct_at_1 / total if total else 0.0,
        "top5_accuracy": correct_at_5 / total if total else 0.0,
        "evaluated_tokens": total,
    }


def train_word2vec(
    text: str,
    architecture: str = "CBOW",
    vector_size: int = 80,
    window: int = 3,
    epochs: int = 60,
    test_fraction: float = 0.2,
    min_count: int = 3,
    seed: int = 42,
) -> TrainingResult:
    """Train Word2Vec and report in-sample and held-out ranking metrics."""
    if architecture not in {"CBOW", "Skip-gram"}:
        raise ValueError("Architecture must be either CBOW or Skip-gram.")
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("Test split must be between 10% and 40%.")
    if not 1 <= min_count <= 10:
        raise ValueError("Minimum word frequency must be between 1 and 10.")

    sentences = tokenize_corpus(text)
    if len(sentences) < 8:
        raise ValueError(
            f"Add at least 8 unique sentences to train and test the model; "
            f"only {len(sentences)} were found."
        )
    train_sentences, test_sentences = _split_sentences(sentences, test_fraction, seed)
    training_frequencies = Counter(
        token for sentence in train_sentences for token in sentence
    )
    if sum(count >= min_count for count in training_frequencies.values()) < 2:
        raise ValueError(
            "Too few words meet the minimum frequency in the training split. "
            "Lower the minimum word frequency or add more text."
        )

    def fit(sentences_to_fit: list[list[str]], word_min_count: int) -> Word2Vec:
        return Word2Vec(
            sentences=sentences_to_fit,
            vector_size=vector_size,
            window=window,
            min_count=word_min_count,
            workers=1,
            sg=1 if architecture == "Skip-gram" else 0,
            negative=5,
            hs=0,
            sample=0,
            cbow_mean=1,
            shrink_windows=False,
            epochs=epochs,
            seed=seed,
            compute_loss=True,
        )

    evaluation_min_count = max(3, min_count)
    evaluation_model = fit(train_sentences, evaluation_min_count)
    evaluation_context_counts, evaluation_context_totals, evaluation_word_counts = (
        _build_context_counts(train_sentences, window)
    )
    metrics = _evaluate(
        evaluation_model,
        test_sentences,
        architecture,
        window,
        evaluation_context_counts,
        evaluation_context_totals,
        evaluation_word_counts,
        evaluation_min_count,
    )
    # Refit on all bundled text for inference after measuring held-out quality.
    model = fit(sentences, min_count)
    context_counts, context_totals, word_counts = _build_context_counts(
        sentences, window
    )
    training_metrics = _evaluate(
        model,
        sentences,
        architecture,
        window,
        context_counts,
        context_totals,
        word_counts,
        evaluation_min_count,
    )
    return TrainingResult(
        model=model,
        architecture=architecture,
        train_sentence_count=len(sentences),
        test_sentence_count=len(test_sentences),
        vocabulary_size=len(model.wv),
        metrics=metrics,
        training_metrics=training_metrics,
        training_loss=float(model.get_latest_training_loss()),
        min_count=min_count,
        evaluation_min_count=evaluation_min_count,
        context_counts=context_counts,
        context_totals=context_totals,
        word_counts=word_counts,
    )


def predict_masked_word(
    training: TrainingResult, sentence: str, topn: int = 5
) -> list[dict[str, Any]]:
    """Rank vocabulary words for the single ___, [MASK], or {mask} gap."""
    if training.architecture != "CBOW":
        raise ValueError(
            "Masked-word prediction requires a CBOW model. "
            "Use predict_skipgram_context for Skip-gram."
        )
    model = training.model
    masks = list(MASK_PATTERN.finditer(sentence))
    if len(masks) != 1:
        raise ValueError("Put exactly one blank in the sentence using ___ or [MASK].")

    match = masks[0]
    left_tokens = simple_preprocess(sentence[: match.start()], deacc=True, min_len=1)
    right_tokens = simple_preprocess(sentence[match.end() :], deacc=True, min_len=1)
    left_context = left_tokens[-model.window :]
    right_context = right_tokens[: model.window]
    positioned_context = [
        (word, 1.0 / distance)
        for distance, word in enumerate(reversed(left_context), start=1)
    ] + [
        (word, 1.0 / distance)
        for distance, word in enumerate(right_context, start=1)
    ]
    known = [(word, weight) for word, weight in positioned_context if word in model.wv]
    context = [word for word, _ in known]
    context_weights = [weight for _, weight in known]
    if not context:
        raise ValueError(
            "No surrounding words were found in the model vocabulary. "
            "Use words from the training text on either side of the blank."
        )

    words, probabilities = _score_candidates(
        model,
        context,
        training.architecture,
        training.context_counts,
        training.context_totals,
        training.word_counts,
        exclude_words=set(context) | FUNCTION_WORDS,
        context_weights=context_weights,
    )
    ranking = np.argsort(probabilities)[::-1][: max(1, min(topn, len(words)))]
    suggestion_probabilities = probabilities[ranking]
    suggestion_probabilities /= suggestion_probabilities.sum()
    return [
        {"word": words[position], "probability": float(probability)}
        for position, probability in zip(
            ranking, suggestion_probabilities, strict=True
        )
    ]


def predict_skipgram_context(
    training: TrainingResult, center_word: str, topn: int = 8
) -> list[dict[str, Any]]:
    """Given one center word, predict words that Skip-gram learned around it."""
    if training.architecture != "Skip-gram":
        raise ValueError("Context-word prediction requires a Skip-gram model.")
    tokens = simple_preprocess(center_word, deacc=True, min_len=1, max_len=30)
    if len(tokens) != 1:
        raise ValueError("Enter exactly one center word.")
    return _rank_skipgram_context(
        training.model,
        tokens[0],
        training.context_counts,
        training.context_totals,
        training.word_counts,
        topn,
    )
