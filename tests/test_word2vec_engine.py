import unittest
from pathlib import Path

from practice_examples import PRACTICE_EXAMPLES, SKIPGRAM_EXAMPLES
from word2vec_engine import (
    predict_masked_word,
    predict_skipgram_context,
    tokenize_corpus,
    train_word2vec,
)

SAMPLE_CORPUS = Path(__file__).resolve().parents[1] / "sample_corpus.txt"

CORPUS = """
the bright sun rose above the quiet hills.
golden sunlight spread across the green valley.
birds sang sweetly in the tall trees.
the gentle breeze moved through green grass.
dark clouds gathered over distant mountains.
fresh snow covered the quiet road.
the traveler packed a small bag for the journey.
she followed a narrow path through the forest.
the friendly guide showed us the way to the lake.
we walked along the quiet beach in the morning.
the cook added fresh herbs to the vegetable soup.
we baked warm bread in the small kitchen.
she sliced a red apple for an afternoon snack.
the ripe tomatoes made a tasty garden salad.
he poured hot tea into a blue cup.
the student opened a book in the quiet library.
the teacher explained the lesson with an example.
students asked thoughtful questions about science.
the computer stores information in a digital file.
developers tested the website on a mobile phone.
"""


class Word2VecEngineTests(unittest.TestCase):
    def test_tokenize_corpus_deduplicates_sentences(self):
        sentences = tokenize_corpus("Bright sun over the hills. Bright sun over the hills.")
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0][:2], ["bright", "sun"])

    def test_cbow_uses_average_context_and_negative_sampling(self):
        result = train_word2vec(
            CORPUS,
            architecture="CBOW",
            vector_size=16,
            window=3,
            epochs=2,
            min_count=1,
            seed=7,
        )
        self.assertEqual(result.model.sg, 0)
        self.assertEqual(result.model.cbow_mean, 1)
        self.assertEqual(result.model.negative, 5)
        self.assertEqual(result.model.hs, 0)
        self.assertEqual(result.model.sample, 0)
        self.assertEqual(result.model.window, 3)
        self.assertFalse(result.model.shrink_windows)

    def test_training_evaluates_held_out_tokens_and_predicts_mask(self):
        result = train_word2vec(
            CORPUS,
            architecture="CBOW",
            vector_size=40,
            window=3,
            epochs=20,
            seed=7,
        )
        self.assertGreater(result.metrics["evaluated_tokens"], 0)
        self.assertGreaterEqual(result.metrics["top5_accuracy"], result.metrics["top1_accuracy"])
        self.assertGreater(result.training_metrics["evaluated_tokens"], 0)
        self.assertGreaterEqual(
            result.training_metrics["top5_accuracy"],
            result.training_metrics["top1_accuracy"],
        )
        predictions = predict_masked_word(
            result, "the bright ___ rose above the quiet hills", topn=3
        )
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(item["word"] in result.model.wv for item in predictions))
        self.assertNotIn("bright", [item["word"] for item in predictions])
        self.assertAlmostEqual(sum(item["probability"] for item in predictions), 1.0)
        self.assertTrue(all(0 <= item["probability"] <= 1 for item in predictions))

    def test_skipgram_requires_center_word_context_prediction(self):
        result = train_word2vec(
            CORPUS,
            architecture="Skip-gram",
            vector_size=40,
            window=3,
            epochs=20,
            seed=7,
        )
        with self.assertRaisesRegex(ValueError, "requires a CBOW model"):
            predict_masked_word(result, "the bright [MASK] rose above the hills")

    def test_rejects_frequency_threshold_that_leaves_no_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "Too few words"):
            train_word2vec(CORPUS, min_count=10, vector_size=40, epochs=20)

    def test_bundled_cbow_practice_examples_are_suggested(self):
        corpus = SAMPLE_CORPUS.read_text(encoding="utf-8")
        self.assertLessEqual(sum(map(len, tokenize_corpus(corpus))), 10_000)
        expected_words = {
            "Feeling tired": "tired",
            "Alice's tears": "eyes",
            "The rabbit-hole": "rabbit",
            "The golden key": "key",
            "Alice's sister": "sister",
            "Alice's curiosity": "curiosity",
            "Feeling sleepy": "sleepy",
            "The river bank": "bank",
            "The Rabbit's watch": "curiosity",
            "Alice's tearful eyes": "eyes",
            "Sitting by her sister": "sister",
        }
        result = train_word2vec(
            corpus,
            architecture="CBOW",
            vector_size=80,
            window=3,
            epochs=60,
            min_count=1,
            seed=42,
        )
        for topic, prompt in PRACTICE_EXAMPLES.items():
            with self.subTest(topic=topic):
                suggestions = predict_masked_word(result, prompt, topn=5)
                self.assertEqual(len(suggestions), 5)
                self.assertTrue(
                    all(item["word"] in result.model.wv for item in suggestions)
                )
                self.assertIn(
                    expected_words[topic],
                    [item["word"] for item in suggestions],
                )

    def test_skipgram_predicts_multiple_context_words_from_center_word(self):
        corpus = SAMPLE_CORPUS.read_text(encoding="utf-8")
        result = train_word2vec(
            corpus,
            architecture="Skip-gram",
            vector_size=80,
            window=3,
            epochs=60,
            min_count=1,
            seed=42,
        )
        self.assertEqual(result.evaluation_min_count, 3)
        self.assertGreater(result.metrics["evaluated_tokens"], 0)
        self.assertGreater(result.metrics["top5_accuracy"], 0)
        expected_contexts = {
            "rabbit": {"hole"},
            "eyes": {"pink"},
            "sister": {"reading", "book"},
            "key": {"golden", "lock"},
            "door": {"opened"},
            "caterpillar": {"blue"},
            "watch": {"pocket"},
            "garden": {"loveliest"},
            "bottle": {"poison"},
            "table": {"glass"},
            "pool": {"tears"},
        }
        for center_word, expected in expected_contexts.items():
            with self.subTest(center_word=center_word):
                predictions = predict_skipgram_context(result, center_word, topn=8)
                predicted_words = {item["word"] for item in predictions}
                self.assertGreaterEqual(len(predictions), 2)
                self.assertTrue(expected.intersection(predicted_words))
                self.assertAlmostEqual(
                    sum(item["probability"] for item in predictions), 1.0
                )

    def test_all_skipgram_example_words_have_context_predictions(self):
        corpus = SAMPLE_CORPUS.read_text(encoding="utf-8")
        result = train_word2vec(
            corpus,
            architecture="Skip-gram",
            vector_size=80,
            window=3,
            epochs=60,
            min_count=1,
            seed=42,
        )
        for category, center_word in SKIPGRAM_EXAMPLES.items():
            with self.subTest(category=category, center_word=center_word):
                predictions = predict_skipgram_context(result, center_word, topn=8)
                self.assertTrue(predictions)
                self.assertNotIn(center_word, [item["word"] for item in predictions])

    def test_skipgram_context_prediction_rejects_invalid_or_wrong_architecture(self):
        cbow_result = train_word2vec(
            CORPUS,
            architecture="CBOW",
            vector_size=40,
            epochs=20,
            seed=7,
        )
        with self.assertRaisesRegex(ValueError, "requires a Skip-gram model"):
            predict_skipgram_context(cbow_result, "apple")

        skipgram_result = train_word2vec(
            CORPUS,
            architecture="Skip-gram",
            vector_size=40,
            epochs=20,
            seed=7,
        )
        with self.assertRaisesRegex(ValueError, "exactly one center word"):
            predict_skipgram_context(skipgram_result, "two words")


if __name__ == "__main__":
    unittest.main()
