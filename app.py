from pathlib import Path

import streamlit as st

from practice_examples import PRACTICE_EXAMPLES, SKIPGRAM_EXAMPLES
from word2vec_engine import (
    TrainingResult,
    predict_masked_word,
    predict_skipgram_context,
    train_word2vec,
    tokenize_corpus,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_CORPUS = ROOT / "sample_corpus.txt"
VECTOR_SIZE = 80
CONTEXT_WINDOW = 3
TRAINING_EPOCHS = 60
TEST_FRACTION = 0.2
MIN_WORD_COUNT = 1
TRAINING_CACHE_VERSION = 2


@st.cache_resource(show_spinner=False)
def get_bundled_model(
    corpus_text: str,
    architecture: str,
    cache_version: int,
) -> TrainingResult:
    return train_word2vec(
        corpus_text,
        architecture=architecture,
        vector_size=VECTOR_SIZE,
        window=CONTEXT_WINDOW,
        epochs=TRAINING_EPOCHS,
        test_fraction=TEST_FRACTION,
        min_count=MIN_WORD_COUNT,
    )


def load_practice_example() -> None:
    st.session_state["prediction_sentence"] = PRACTICE_EXAMPLES[
        st.session_state["practice_example"]
    ]
    st.session_state["last_prediction"] = None
    st.session_state["prediction_requested"] = True


def load_skipgram_example() -> None:
    st.session_state["skipgram_center_word"] = SKIPGRAM_EXAMPLES[
        st.session_state["skipgram_example"]
    ]
    st.session_state["last_prediction"] = None
    st.session_state["prediction_requested"] = True


def request_prediction() -> None:
    st.session_state["last_prediction"] = None
    st.session_state["prediction_requested"] = True


def set_workspace(workspace: str) -> None:
    st.session_state["workspace"] = workspace
    st.session_state["last_prediction"] = None
    st.session_state["prediction_requested"] = True


st.set_page_config(
    page_title="WordLab | CBOW & Skip-gram",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#17231f; --muted:#718079; --green:#157a61; --line:#e3eae5; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:#f6f8f5; color:var(--ink); }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink); }
    [data-testid="stSidebar"] { background:#eff4ef; border-right:1px solid var(--line); }
    .hero { padding:1.15rem 0 1.5rem; }
    .eyebrow { color:#157a61; font-size:.75rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
    .hero h1 { font-size:2.65rem; line-height:1.08; margin:.45rem 0 .5rem; }
    .subtle { color:#718079; font-size:1rem; }
    .panel { background:white; border:1px solid #e3eae5; border-radius:16px; padding:1.25rem 1.4rem; }
    .metric-card { background:white; border:1px solid #e3eae5; border-radius:14px; padding:1rem 1.15rem; min-height:100px; }
    .metric-label { color:#718079; font-size:.8rem; font-weight:600; }
    .metric-value { font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; margin-top:.3rem; }
    .tag { display:inline-block; background:#e6f3ee; color:#176c57; border-radius:99px; padding:.25rem .65rem; font-size:.78rem; font-weight:700; }
    .prediction-word { white-space:nowrap; font-weight:700; }
    div.stButton > button[kind="primary"] { background:#157a61; border:0; border-radius:10px; font-weight:700; }
    div.stButton > button[kind="primary"]:hover { background:#105f4b; }
    .stTextInput input { border-radius:10px; }
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        min-height:4.1rem;
        height:auto;
        width:100%;
        justify-content:flex-start;
        text-align:left;
        white-space:pre-line;
        line-height:1.45;
        padding:.8rem 1rem;
        border-radius:13px;
        font-weight:600;
        border:1px solid #dce6df;
        box-shadow:0 2px 5px rgba(23,35,31,.035);
        transition:transform .15s ease, box-shadow .15s ease, background .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        transform:translateY(-1px);
        box-shadow:0 6px 14px rgba(23,35,31,.08);
        border-color:#9dbdaf;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
        color:#fff;
        background:linear-gradient(125deg,#176b55,#20866a);
        border:1px solid #176b55;
    }
    .sidebar-kicker { margin:.75rem 0 .45rem; color:#809087; font-size:.68rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .sidebar-nav-title { color:#52645a; font-weight:700; }
    .sidebar-nav-hint { color:#8a9990; font-size:.76rem; font-weight:400; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "prediction_sentence" not in st.session_state:
    st.session_state.prediction_sentence = next(iter(PRACTICE_EXAMPLES.values()))
if "prediction_requested" not in st.session_state:
    st.session_state.prediction_requested = True
if "skipgram_center_word" not in st.session_state:
    st.session_state.skipgram_center_word = next(iter(SKIPGRAM_EXAMPLES.values()))
if "workspace" not in st.session_state:
    st.session_state.workspace = "Dataset"

with st.sidebar:
    st.markdown("## 🧠 WordLab")
    st.caption("Word embedding studio")
    st.divider()
    st.markdown('<div class="sidebar-kicker">Workspace</div>', unsafe_allow_html=True)
    for workspace, label, key in (
        ("Dataset", "🗂️  Dataset overview\nCorpus details and training data", "nav_dataset"),
        ("CBOW", "🧩  CBOW model\nContext predicts the center word", "nav_cbow"),
        ("Skip-gram", "🔎  Skip-gram model\nOne word predicts its context", "nav_skipgram"),
    ):
        st.button(
            label,
            key=key,
            type="primary" if st.session_state.workspace == workspace else "secondary",
            use_container_width=True,
            on_click=set_workspace,
            args=(workspace,),
        )

    sample_text = SAMPLE_CORPUS.read_text(encoding="utf-8")
    corpus_sentences = tokenize_corpus(sample_text)
    corpus_sentence_count = len(corpus_sentences)

architecture = (
    st.session_state.workspace
    if st.session_state.workspace in {"CBOW", "Skip-gram"}
    else None
)
training = None
if architecture is not None:
    training_key = architecture
    if st.session_state.get("training_key") != training_key:
        st.session_state.last_prediction = None
    try:
        with st.spinner(f"Training the built-in {architecture} model..."):
            training = get_bundled_model(
                sample_text, architecture, TRAINING_CACHE_VERSION
            )
        st.session_state.training_key = training_key
        st.session_state.training = training
    except ValueError as error:
        st.error(f"Could not train on the bundled corpus: {error}")

if st.session_state.workspace == "Dataset":
    total_tokens = sum(len(sentence) for sentence in corpus_sentences)
    unique_tokens = len({token for sentence in corpus_sentences for token in sentence})
    st.markdown(
        '<div class="hero"><div class="eyebrow">Project workspace · included data</div>'
        '<h1>Dataset overview</h1><div class="subtle">Explore the compact, bundled corpus used by both Word2Vec models. '
        'Choose CBOW or Skip-gram in the sidebar to open its prediction workspace; no upload is needed.</div></div>',
        unsafe_allow_html=True,
    )
    dataset_metrics = st.columns(3)
    for column, label, value in zip(
        dataset_metrics,
        ("Sentences", "Word tokens", "Unique words"),
        (corpus_sentence_count, total_tokens, unique_tokens),
        strict=True,
    ):
        with column:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value:,}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("### Model accuracy")
    with st.spinner("Training and evaluating both Word2Vec models..."):
        overview_training = {
            model_architecture: get_bundled_model(
                sample_text, model_architecture, TRAINING_CACHE_VERSION
            )
            for model_architecture in ("CBOW", "Skip-gram")
        }
    for model_architecture, metric_labels, evaluated_label in (
        ("CBOW", ("Top-1 accuracy", "Top-5 accuracy"), "Evaluated words"),
        ("Skip-gram", ("Context hit@1", "Context hit@5"), "Evaluated centers"),
    ):
        result = overview_training[model_architecture]
        st.markdown(f"**{model_architecture}**")
        st.markdown("**Training set — in-sample**")
        train_columns = st.columns(2)
        train_columns[0].metric(
            metric_labels[0], f"{result.training_metrics['top1_accuracy']:.1%}"
        )
        train_columns[1].metric(
            metric_labels[1], f"{result.training_metrics['top5_accuracy']:.1%}"
        )
        st.markdown("**Held-out test set**")
        test_columns = st.columns(3)
        test_columns[0].metric(
            metric_labels[0], f"{result.metrics['top1_accuracy']:.1%}"
        )
        test_columns[1].metric(
            metric_labels[1], f"{result.metrics['top5_accuracy']:.1%}"
        )
        test_columns[2].metric(
            evaluated_label, f"{result.metrics['evaluated_tokens']:,}"
        )
    st.caption(
        f"Training scores are in-sample, measured on sentences used to fit the full-corpus "
        f"prediction model. Held-out scores use a reproducible {TEST_FRACTION:.0%} sentence "
        "split and are the better estimate of performance on unseen sentences. CBOW ranks "
        "the missing center word; Skip-gram measures whether true neighbors appear in context predictions. "
        "Training scores may be optimistic because ranking uses co-occurrence counts from the same corpus."
    )
    st.write("")
    st.markdown("### About this dataset")
    st.markdown(
        '<div class="panel"><p><strong>Source:</strong> <em>Alice&#8217;s Adventures in Wonderland</em> by Lewis Carroll, '
        'Project Gutenberg eBook 11; bundled in <code>sample_corpus.txt</code> and capped at fewer than 10,000 '
        'tokenized words. <a href="https://www.gutenberg.org/ebooks/11">Project Gutenberg source</a>.</p>'
        '<p><strong>Topics:</strong> Alice, the White Rabbit, the rabbit-hole, and other characters and events '
        'from the opening chapters.</p>'
        '<p><strong>Preparation:</strong> sentence splitting and Gensim tokenization '
        '(lowercase, accents removed, tokens 1–30 characters); sentences shorter than 3 tokens '
        'and duplicate sentences are discarded.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    with st.expander("Preview the training corpus"):
        for sentence in corpus_sentences[:12]:
            st.markdown(f"- {' '.join(sentence)}")
        st.caption(f"Showing 12 of {corpus_sentence_count} unique tokenized sentences.")
    st.markdown("### Word2Vec implementation")
    st.markdown(
        f'<div class="panel"><strong>Library:</strong> Gensim Word2Vec; one worker and fixed seed (42)<br>'
        f'<strong>Embedding size:</strong> {VECTOR_SIZE} dimensions<br>'
        f'<strong>Context window:</strong> {CONTEXT_WINDOW} words on the left and {CONTEXT_WINDOW} on the right '
        '(7 positions including the center); fixed during training<br>'
        '<strong>CBOW:</strong> averages the context-word vectors<br>'
        '<strong>Skip-gram:</strong> learns center-word to context-word pairs<br>'
        f'<strong>Epochs:</strong> {TRAINING_EPOCHS} per fit; final vocabulary keeps words occurring at least {MIN_WORD_COUNT} time<br>'
        '<strong>Objective:</strong> 5 negative samples per positive pair; hierarchical softmax disabled<br>'
        '<strong>Frequent-word subsampling:</strong> disabled<br>'
        '<strong>Optimizer:</strong> Gensim Word2Vec built-in updates (no separately configured Adam or Adagrad)<br>'
        f'<strong>Evaluation:</strong> {1 - TEST_FRACTION:.0%} of sentences train the evaluation model; '
        f'{TEST_FRACTION:.0%} are held out and excluded from its training. After scoring, a separate prediction model '
        'is refit on the full corpus; this refit does not affect the held-out scores<br>'
        '<strong>Prediction ranking:</strong> combines learned vector scores with smoothed word co-occurrence counts</div>',
        unsafe_allow_html=True,
    )
elif architecture is not None and training is not None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">Trained on built-in dataset · {architecture}</div>'
        f'<h1>{"Context to center word" if architecture == "CBOW" else "One word to its context"}</h1>'
        f'<div class="subtle">The {architecture} model is trained automatically on {training.train_sentence_count} included sentences. '
        f'Vocabulary: {training.vocabulary_size:,} words.</div></div>',
        unsafe_allow_html=True,
    )
    if architecture == "CBOW":
        st.markdown("### Fill in a word")
        st.caption("Choose a related example or write a sentence. CBOW uses nearby words to rank the missing center word.")
        st.selectbox(
            "Choose a practice sentence",
            options=list(PRACTICE_EXAMPLES),
            key="practice_example",
            on_change=load_practice_example,
        )
        sentence = st.text_input(
            "Sentence with one blank",
            key="prediction_sentence",
            help="Use ___, [MASK], or {mask} for the missing center word.",
        )
        prediction_input = sentence
    else:
        st.markdown("### Predict context words")
        st.caption("Choose or enter one center word. Skip-gram predicts words that appeared around it in the training corpus.")
        st.selectbox(
            "Choose a center-word example",
            options=list(SKIPGRAM_EXAMPLES),
            key="skipgram_example",
            on_change=load_skipgram_example,
        )
        center_word = st.text_input(
            "Center word",
            key="skipgram_center_word",
            help="Skip-gram predicts context words around this one word.",
        )
        prediction_input = center_word
    topn = st.slider(
        "Context words to predict" if architecture == "Skip-gram" else "Suggestions to show",
        3,
        10,
        5 if architecture == "CBOW" else 8,
    )
    predict_clicked = st.button(
        "Predict missing word" if architecture == "CBOW" else "Predict surrounding words",
        type="primary",
    )
    if predict_clicked or st.session_state.prediction_requested:
        st.session_state.prediction_requested = False
        st.session_state.last_prediction = None
        try:
            if architecture == "CBOW":
                suggestions = predict_masked_word(training, prediction_input, topn=topn)
                prediction_title = prediction_input
            else:
                suggestions = predict_skipgram_context(
                    training, prediction_input, topn=topn
                )
                prediction_title = f"Words that appear around “{prediction_input.strip()}”"
            st.session_state.last_prediction = (prediction_title, suggestions)
        except ValueError as error:
            st.error(str(error))
    if st.session_state.last_prediction:
        prediction_title, suggestions = st.session_state.last_prediction
        st.markdown(
            "#### Predicted context words"
            if architecture == "Skip-gram"
            else "#### Missing-word predictions"
        )
        st.caption(prediction_title)
        for rank, item in enumerate(suggestions, start=1):
            col_word, col_bar, col_probability = st.columns([1.5, 3.5, 1])
            with col_word:
                st.markdown(
                    f'<div class="prediction-word">{rank}. {item["word"]}</div>',
                    unsafe_allow_html=True,
                )
            with col_bar:
                st.progress(min(item["probability"], 1.0))
            with col_probability:
                st.markdown(f"{item['probability']:.1%}")
        st.markdown("#### Held-out test performance")
        metric_labels = (
            ("Context hit@1", "Context hit@5", "Center words")
            if architecture == "Skip-gram"
            else ("Top-1 accuracy", "Top-5 accuracy", "Test words")
        )
        m1, m2, m3 = st.columns(3)
        m1.metric(metric_labels[0], f"{training.metrics['top1_accuracy']:.1%}")
        m2.metric(metric_labels[1], f"{training.metrics['top5_accuracy']:.1%}")
        m3.metric(metric_labels[2], f"{training.metrics['evaluated_tokens']:,}")
        st.markdown("#### Training set fit (in-sample)")
        train_m1, train_m2, train_m3 = st.columns(3)
        train_m1.metric(
            metric_labels[0], f"{training.training_metrics['top1_accuracy']:.1%}"
        )
        train_m2.metric(
            metric_labels[1], f"{training.training_metrics['top5_accuracy']:.1%}"
        )
        train_m3.metric(
            "Training words" if architecture == "CBOW" else "Training centers",
            f"{training.training_metrics['evaluated_tokens']:,}",
        )
    with st.expander("Model evaluation details"):
        st.markdown(
            f"**Held-out {architecture} test:** "
            f"{training.metrics['top1_accuracy']:.1%} hit@1 · "
            f"{training.metrics['top5_accuracy']:.1%} hit@5 · "
            f"{training.metrics['evaluated_tokens']:,} evaluated center words. "
            f"**Training-set in-sample:** {training.training_metrics['top1_accuracy']:.1%} "
            f"hit@1 · {training.training_metrics['top5_accuracy']:.1%} hit@5 · "
            f"{training.training_metrics['evaluated_tokens']:,} evaluated training words/centers."
        )
elif architecture is not None:
    st.info("This model could not be trained from the included dataset.")
