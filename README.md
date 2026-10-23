# WordLab: CBOW and Skip-gram

A small interactive Word2Vec project. The app automatically trains the selected CBOW or Skip-gram model on its included compact corpus, offers related practice sentences that are rephrased from the corpus, predicts masked words, and reports held-out top-1 / top-5 accuracy. No data upload or manual training button is required.

## Run locally on Windows

Double-click `run_app.bat`. It creates the project virtual environment if needed, installs the requirements if they are missing, and starts the app locally. Open the URL printed in the terminal (normally `http://localhost:8501`). Keep the terminal open while using the app; closing it stops the server.

To start it manually in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

The sidebar has three workspace cards: **Dataset overview**, **CBOW model**, and **Skip-gram model**. The overview describes the bundled corpus and displays both models' in-sample training scores and held-out test scores. It trains and caches both architectures on first use; switching to a model workspace reuses its cached result. There is no separate train button, dataset upload, or training-parameter control.

## Dataset and model configuration

- **Corpus:** `sample_corpus.txt` contains the first 9,999 Gensim-tokenized words from *Alice's Adventures in Wonderland* by Lewis Carroll (Project Gutenberg eBook 11: https://www.gutenberg.org/ebooks/11). The text is bundled locally, so training does not require a network connection. Gensim tokenization lowercases text, removes accents, keeps tokens 1–30 characters long, and discards duplicate sentences and sentences shorter than three tokens.
- **Model:** Gensim Word2Vec, 80-dimensional vectors, a fixed seven-position context window (three words on the left, the center word, and three words on the right), 60 epochs per fit, minimum word count 1 for the final model, one worker, and fixed seed 42. The effective context window is not randomly shrunk.
- **CBOW:** averages the context vectors. **Skip-gram:** learns center-word/context-word pairs.
- **Objective:** five negative samples per positive pair; hierarchical softmax and frequent-word subsampling are disabled. Training uses Gensim's built-in Word2Vec updates, not a separately configured Adam or Adagrad optimizer.
- **Evaluation:** a reproducible 80/20 sentence-level split. The evaluation model is trained only on the 80% training sentences; its metrics are calculated on the separate 20% held-out sentences, which are not included in that model's training. After scoring, a separate interactive model is refit on the full corpus for predictions; this refit does not change the reported held-out scores. The held-out evaluation model uses a minimum word count of three. The app also reports in-sample training scores for the full-corpus model using the same metrics. Training scores are expected to be optimistic because the model has seen those sentences and the prediction ranking uses co-occurrence counts built from the corpus. Prediction rankings combine learned vector scores with smoothed corpus co-occurrence counts.
- **Measured metrics (seed 42, current corpus and settings):** CBOW full-corpus training scores are 11.1% Top-1 and 23.4% Top-5; held-out test scores are 6.7% and 14.4%. Skip-gram full-corpus in-sample context hit@1/hit@5 scores are 13.3% and 44.9%; held-out scores are 6.3% and 19.6%. These are corpus-specific. Training scores are in-sample and optimistic, particularly because ranking uses co-occurrence counts from those same sentences; held-out results are the better estimate of performance on unseen sentences.

- **CBOW:** Choose a related example or enter a sentence with `___`, `[MASK]`, or `{mask}`. The model predicts the missing center word from surrounding words.
- **Skip-gram:** Choose or enter one center word (for example, `rabbit`). The model predicts multiple words that occur around it in the corpus.

Predictions run automatically for a selected example. Edit the input and click the predict button to try your own.

## Evaluation notes

Metrics are measured on content words that occur at least three times in the relevant corpus portion, reducing the effect of one-off words. The split is by sentence, not by vocabulary: words can occur in both splits, but each held-out sentence itself is excluded from the evaluation model's training. CBOW metrics measure whether the masked center word is ranked; Skip-gram context hit@1/hit@5 measure whether predicted context words include actual neighbors for a held-out center word. The app displays top-1 and top-5/hit@5 scores for both the in-sample training data and held-out test data. Training scores describe fit to data the model already saw and are not a generalization estimate; they may also be optimistic because prediction ranking uses corpus co-occurrence counts. Held-out scores are the more appropriate estimate for unseen sentences, but they remain specific to this corpus and split, not a measure of general language understanding. After evaluation, inference models use the complete bundled corpus and retain its full vocabulary. Word2Vec learns local word associations; it is not a generative language model, so results depend on corpus coverage.

## Tests

Run the focused backend tests with:

```powershell
python -m unittest discover -s tests -v
```
