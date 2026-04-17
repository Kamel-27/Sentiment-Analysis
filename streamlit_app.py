import streamlit as st
import os
import re
import pickle
import numpy as np

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠")

# ── PREPROCESSING LOGIC (From your Flask app) ─────────────────
IDIOM_MAP = {
    "can't wait"          : "very excited",
    "cannot wait"         : "very excited",
    "can't get enough"    : "love it",
    "cannot get enough"   : "love it",
    "can't stop playing"  : "love playing",
    "cannot stop playing" : "love playing",
    "can't put it down"   : "very engaging",
    "not bad"             : "decent good",
    "not too bad"         : "decent good",
    "not terrible"        : "acceptable good",
    "not boring"          : "engaging fun",
    "not disappointed"    : "satisfied pleased",
    "didn't disappoint"   : "satisfied pleased",
    "doesn't disappoint"  : "satisfied pleased",
    "never disappoints"   : "always satisfies",
    "not a bad"           : "a decent",
    "not the worst"       : "acceptable",
    "nothing wrong"       : "everything fine",
    "no complaints"       : "fully satisfied",
    "no issues"           : "works perfectly",
    "no problems"         : "works perfectly",
    "not boring at all"   : "very entertaining",
    "not slow"            : "fast responsive",
    "not recommended"     : "dislike avoid bad",
    "too good to be true" : "suspicious unreliable",
    "good for nothing"    : "useless worthless",
    "pretty bad"          : "bad poor",
    "kind of bad"         : "bad poor",
    "sort of bad"         : "bad poor",
}


NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing",
    "nor", "dont", "don't", "doesn't", "doesnt", "didn't",
    "didnt", "can't", "cant", "cannot", "won't", "wont",
    "wouldn't", "wouldnt", "isn't", "isnt", "aren't", "arent",
    "wasn't", "wasnt", "weren't", "werent", "haven't", "havent",
    "hasn't", "hasnt", "hardly", "barely", "scarcely"
}

def handle_double_negation(tokens: list) -> list:
    result = []; i = 0
    while i < len(tokens):
        current = re.sub(r'[^\w]', '', tokens[i]).lower()
        if current in NEGATION_WORDS and i + 1 < len(tokens):
            found_double = False
            for j in range(i + 1, min(i + 4, len(tokens))):
                next_clean = re.sub(r'[^\w]', '', tokens[j]).lower()
                if next_clean in NEGATION_WORDS:
                    result.append("definitely"); i = j + 1
                    found_double = True; break
            if not found_double:
                result.append(tokens[i]); i += 1
        else:
            result.append(tokens[i]); i += 1
    return result

def tag_negation_scope(tokens: list) -> list:
    result = []; negating = False; neg_count = 0; MAX_SCOPE = 4
    for token in tokens:
        clean = re.sub(r'[^\w]', '', token).lower()
        if re.search(r'[.!?,;]', token):
            negating = False; neg_count = 0; result.append(token); continue
        if clean in NEGATION_WORDS:
            negating = True; neg_count = 0; result.append(token)
        elif negating:
            result.append(token + "_NEG"); neg_count += 1
            if neg_count >= MAX_SCOPE: negating = False; neg_count = 0
        else:
            result.append(token)
    return result

def preprocess_text(text: str) -> str:
    processed = text.lower().strip()
    for phrase, replacement in sorted(IDIOM_MAP.items(), key=lambda x: -len(x[0])):
        processed = processed.replace(phrase, replacement)
    tokens = processed.split()
    tokens = handle_double_negation(tokens)
    tokens = tag_negation_scope(tokens)
    return " ".join(tokens)

# ── MODEL LOADING (Cached for performance) ───────────────────
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        return None, None
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle['pipeline'], bundle['label_encoder']

pipeline, label_encoder = load_model()

# ── UI LAYOUT ────────────────────────────────────────────────
st.title("🧠 Sentiment Analysis")

if pipeline is None:
    st.error("Model file `model.pkl` not found in the directory!")
    st.stop()

text_input = st.text_area("Review Text", placeholder="Type your review here...", height=150)
analyse_btn = st.button("Analyse", type="primary")

if analyse_btn:
    text = text_input.strip()
    if not text:
        st.warning("Please enter some text.")
    else:
        # INFERENCE
        processed_text = preprocess_text(text)
        label_encoded = pipeline.predict([processed_text])[0]
        probas = pipeline.predict_proba([processed_text])[0]
        label = label_encoder.inverse_transform([label_encoded])[0]

        # RESULTS
        st.divider()
        if label.lower() == "positive":
            st.success(f"😄 Sentiment: **{label.upper()}**")
        elif label.lower() == "negative":
            st.error(f"😞 Sentiment: **{label.upper()}**")
        else:
            st.info(f"😐 Sentiment: **{label.upper()}**")

        st.write(f"**Confidence:** {float(np.max(probas)):.2%}")
        
        # Breakdown
        all_probas = {cls: prob for cls, prob in zip(label_encoder.classes_, probas)}
        for cls, prob in sorted(all_probas.items(), key=lambda x: -x[1]):
            st.write(f"- {cls}: {prob:.2%}")
            st.progress(float(prob))