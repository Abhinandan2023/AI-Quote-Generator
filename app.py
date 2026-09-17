import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Quote Generator",
    page_icon="✨",
    layout="centered"
)


# -----------------------------
# Load Model & Tokenizer
# -----------------------------

@st.cache_resource
def load_resources():

    model = load_model("quote_lstm_model_v3.keras")

    with open("tokenizer_v3.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_resources()


# -----------------------------
# Model Configuration
# -----------------------------

VOCAB_SIZE = 40000
MAX_LEN = 60


# -----------------------------
# Quote Generation
# -----------------------------

def generate_quote(
    category,
    max_words=30,
    temperature=0.7,
    repetition_penalty=1.5,
    min_words=8
):

    sequence = tokenizer.texts_to_sequences(
        [f"<CAT> {category} <START>"]
    )[0]

    generated_tokens = []

    for _ in range(max_words):

        input_sequence = sequence[-(MAX_LEN - 1):]

        padded = pad_sequences(
            [input_sequence],
            maxlen=MAX_LEN - 1,
            padding="post"
        )

        predictions = model.predict(
            padded,
            verbose=0
        )

        last_position = len(input_sequence) - 1

        probabilities = predictions[
            0,
            last_position
        ].astype("float64")

        # Prevent OOV token
        oov_id = tokenizer.word_index.get("<OOV>")

        if oov_id is not None and oov_id < len(probabilities):
            probabilities[oov_id] = 0

        # Reduce repetition
        for token in set(generated_tokens[-5:]):

            if token < len(probabilities):
                probabilities[token] /= repetition_penalty

        # Temperature
        probabilities = np.log(
            probabilities + 1e-8
        ) / temperature

        probabilities = np.exp(probabilities)

        probabilities /= probabilities.sum()

        next_token = np.random.choice(
            len(probabilities),
            p=probabilities
        )

        next_word = tokenizer.index_word.get(
            next_token,
            ""
        )

        # Stop conditions
        if next_token == 0:
            break

        if next_word.lower() in [
            "<cat>",
            "<start>",
            "<oov>"
        ]:
            break

        generated_tokens.append(next_token)

        sequence.append(next_token)

        if len(generated_tokens) >= min_words:

            # Optional stopping condition
            if next_word.endswith((".", "!", "?")):
                break

    words = [
        tokenizer.index_word.get(token, "")
        for token in generated_tokens
    ]

    return " ".join(words)


# -----------------------------
# UI
# -----------------------------

st.title("✨ AI Quote Generator")

st.write(
    "Generate original quotes using a category-conditioned LSTM model."
)


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("⚙️ Generation Settings")

categories = [
    "love",
    "life",
    "wisdom",
    "happiness",
    "success",
    "inspiration",
    "friendship",
    "motivational",
    "knowledge",
    "hope"
]

category = st.sidebar.selectbox(
    "Choose a category",
    categories
)

max_words = st.sidebar.slider(
    "Maximum words",
    min_value=10,
    max_value=50,
    value=30
)

temperature = st.sidebar.slider(
    "Creativity",
    min_value=0.2,
    max_value=1.5,
    value=0.7,
    step=0.1
)

repetition_penalty = st.sidebar.slider(
    "Repetition penalty",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1
)


# -----------------------------
# Generate Button
# -----------------------------

if st.button(
    "✨ Generate Quote",
    use_container_width=True
):

    with st.spinner("Creating your quote..."):

        quote = generate_quote(
            category=category,
            max_words=max_words,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )

    st.subheader("Generated Quote")

    st.markdown(
        f"""
        > {quote}
        """
    )

    st.caption(
        f"Category: {category}"
    )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Powered by a category-conditioned LSTM neural network"
)