import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def test_dataset_overview_matches_model_and_has_no_train_controls(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])

        content = "\n".join(element.value for element in app.markdown)
        button_labels = [button.label for button in app.button]

        for detail in (
            "Project Gutenberg",
            "fewer than 10,000",
            "80 dimensions",
            "3 words on the left and 3 on the right (7 positions including the center)",
            "averages the context-word vectors",
            "5 negative samples per positive pair; hierarchical softmax disabled",
            "Frequent-word subsampling:</strong> disabled",
            "20% are held out",
            "Model accuracy",
            "Training set — in-sample",
            "Held-out test set",
            "smoothed word co-occurrence counts",
        ):
            with self.subTest(detail=detail):
                self.assertIn(detail, content)

        self.assertTrue(any("Dataset overview" in label for label in button_labels))
        self.assertTrue(any("CBOW model" in label for label in button_labels))
        self.assertTrue(any("Skip-gram model" in label for label in button_labels))
        self.assertFalse(any(label.startswith("Train ") for label in button_labels))
        self.assertNotIn("ready to train", content)
        self.assertNotIn("Built-in dataset ·", content)
        self.assertNotIn("randomly shrinks", content)


if __name__ == "__main__":
    unittest.main()
