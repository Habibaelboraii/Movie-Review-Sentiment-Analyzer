from pathlib import Path
import pickle

import streamlit as st
from tensorflow.keras.models import load_model

from sentiment import clean_text, predict_sentiment


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "bilstm_model.keras"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer.pkl"

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
)

toggle_space, theme_control = st.columns([5, 1])
with theme_control:
    dark_mode = st.toggle("Dark mode", value=True, key="dark_mode")


@st.cache_resource
def load_assets():
    missing = [
        path.name for path in (MODEL_PATH, TOKENIZER_PATH) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Required model files were not found: " + ", ".join(missing))

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)
    model = load_model(MODEL_PATH, compile=False)
    return model, tokenizer


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink: #f2f3ff; --muted: #a6acc4; color-scheme: dark; }
    .stApp { color: #eef0ff; background: radial-gradient(ellipse at 12% 0%, rgba(91,75,180,.24) 0, transparent 38%),
                           radial-gradient(ellipse at 100% 20%, rgba(26,140,126,.12) 0, transparent 32%), #090b14; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #eef0ff; }
    [data-testid="stAppViewContainer"] { background: transparent; }
    .block-container { max-width: 1020px; padding-top: 2.8rem; padding-bottom: 3rem; }
    header[data-testid="stHeader"] { background: transparent; }
    .hero { position: relative; overflow: hidden; padding: 3.1rem 3rem 2.8rem;
            border: 1px solid rgba(255,255,255,.24); border-radius: 30px;
            background: linear-gradient(118deg, #10172f 0%, #28255e 48%, #664ee5 100%);
            color: white; margin-bottom: 1.8rem; box-shadow: 0 28px 75px rgba(45,42,103,.27); }
    .hero:before { content: ''; position: absolute; width: 360px; height: 360px; right: -85px; top: -185px;
                   border-radius: 50%; background: rgba(156,140,255,.34); filter: blur(2px);
                   box-shadow: 0 0 90px 35px rgba(143,123,255,.24); }
    .hero:after { content: '✦'; position: absolute; right: 10%; bottom: -90px; font-size: 250px;
                  line-height: 1; color: rgba(255,255,255,.09); transform: rotate(-12deg); }
    .hero-content { position: relative; z-index: 1; }
    .eyebrow { display: inline-flex; align-items: center; gap: .5rem; color: #e5e1ff;
               font-size: .72rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase;
               padding: .48rem .8rem; margin-bottom: 1.15rem; border-radius: 999px;
               border: 1px solid rgba(255,255,255,.2); background: rgba(255,255,255,.1);
               backdrop-filter: blur(10px); }
    .hero h1 { font-family: 'Manrope', sans-serif; color: white; font-size: clamp(2rem, 4vw, 3.15rem);
               letter-spacing: -.045em; line-height: 1.12; margin: 0 0 .8rem; }
    .hero p { max-width: 650px; color: #e1e4ff; font-size: 1.05rem; line-height: 1.7; margin: 0; }
    .hero-tags { display: flex; flex-wrap: wrap; gap: .65rem; margin-top: 1.55rem; }
    .hero-tag { padding: .48rem .78rem; color: #f3f2ff; font-size: .78rem; font-weight: 600;
                border-radius: 999px; background: rgba(255,255,255,.11); border: 1px solid rgba(255,255,255,.13); }
    .section-title { font-family: 'Manrope', sans-serif; font-size: 1.25rem; font-weight: 800;
                     color: var(--ink); margin: .35rem 0 .2rem; }
    .section-title { color: #f0efff; }
    .section-copy { color: var(--muted); margin: 0 0 1rem; }
    div[data-testid="stExpander"] { background: rgba(20,23,38,.86); border: 1px solid #292d43; border-radius: 16px; }
    div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary p { color: #e5e7f5; }
    div[data-testid="stForm"] { padding: 1.35rem 1.5rem 1rem; background: linear-gradient(145deg, rgba(23,26,42,.97), rgba(15,17,29,.97));
                                  border: 1px solid #292d43; border-radius: 22px;
                                  box-shadow: 0 18px 48px rgba(0,0,0,.28); }
    div[data-testid="stTextArea"] label, div[data-testid="stTextArea"] label p { color: #e8e9f6; }
    div[data-testid="stTextArea"] textarea { border-radius: 14px; border-color: #343951; background: #0d101b; color: #f4f5ff; font-size: 1rem; }
    div[data-testid="stTextArea"] textarea::placeholder { color: #777f9c; }
    div[data-testid="stTextArea"] textarea:focus { border-color: #8275ee; box-shadow: 0 0 0 3px rgba(109,92,232,.2); }
    div[data-testid="stCode"] { background: #101321; border: 1px solid #2a2e43; border-radius: 12px; }
    div[data-testid="stCode"] code { color: #d7d9ea; }
    div.stFormSubmitButton > button { min-height: 3rem; border-radius: 12px; font-weight: 700;
                                      border: 0; background: linear-gradient(100deg, #6557dc, #8575f4);
                                      box-shadow: 0 8px 18px rgba(101,87,220,.22); }
    div.stFormSubmitButton > button:hover { border: 0; background: linear-gradient(100deg, #5446c9, #7464e6); }
    .result-card { position: relative; overflow: hidden; padding: 1.4rem 1.6rem; border-radius: 20px;
                   margin: 1.35rem 0 .9rem; border: 1px solid; box-shadow: 0 12px 30px rgba(34,42,75,.08); }
    .result-card:after { position: absolute; right: 1.35rem; top: -.72rem; font-size: 5rem; opacity: .11; }
    .result-positive { background: linear-gradient(110deg, #10251f, #12362d); border-color: #246449; color: #8cf0bd; }
    .result-positive:after { content: '✦'; }
    .result-negative { background: linear-gradient(110deg, #2b1721, #381a29); border-color: #713048; color: #ff9bb4; }
    .result-negative:after { content: '◉'; }
    .result-short { background: linear-gradient(110deg, #292315, #352b17); border-color: #665329; color: #f5d889; }
    .result-short:after { content: '?'; }
    .result-label { font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .72; }
    .result-value { font-family: 'Manrope', sans-serif; font-size: 1.65rem; font-weight: 800; margin-top: .25rem; }
    div[data-testid="stMetric"] { padding: .85rem 1rem; background: #151827; border: 1px solid #2b3046;
                                   border-radius: 15px; box-shadow: 0 8px 24px rgba(0,0,0,.2); }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p { color: #a9afc8; }
    div[data-testid="stMetricValue"] { color: #f1f2ff; }
    div[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #6557dc, #29bda0); }
    div[data-testid="stAlert"] { background: #171a29; border: 1px solid #30354d; color: #e6e8f4; }
    .footer-note { color: #747c96; font-size: .83rem; text-align: center; padding-top: 1.5rem; }
    @media (max-width: 640px) {
      .block-container { padding: 1.2rem 1rem 2rem; }
      .hero { padding: 2rem 1.45rem; border-radius: 23px; }
      .hero:after { right: -5%; font-size: 175px; }
      .hero p { font-size: .96rem; }
      div[data-testid="stForm"] { padding: 1rem .9rem .8rem; }
      .result-value { font-size: 1.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not dark_mode:
    st.markdown(
        """
        <style>
        .stApp { color: #172033; background: radial-gradient(ellipse at 12% 0%, #e8eaff 0, transparent 38%),
                               radial-gradient(ellipse at 100% 20%, #dff7f1 0, transparent 32%), #f7f8fc; }
        html, body, [class*="css"] { color: #172033; }
        .section-title { color: #172033; }
        .section-copy { color: #64748b; }
        div[data-testid="stExpander"] { background: rgba(255,255,255,.9); border-color: #e7e9f2; }
        div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary p { color: #172033; }
        div[data-testid="stForm"] { background: rgba(255,255,255,.92); border-color: #e7e9f2;
                                      box-shadow: 0 12px 35px rgba(34,42,75,.07); }
        div[data-testid="stTextArea"] label, div[data-testid="stTextArea"] label p { color: #263149; }
        div[data-testid="stTextArea"] textarea { background: #fbfcff; border-color: #dfe3ee; color: #172033; }
        div[data-testid="stTextArea"] textarea::placeholder { color: #8a93a8; }
        div[data-testid="stCode"] { background: #f5f6fb; border-color: #e2e5ef; }
        div[data-testid="stCode"] code { color: #263149; }
        div[data-testid="stMetric"] { background: #fff; border-color: #e7e9f2;
                                       box-shadow: 0 6px 18px rgba(34,42,75,.04); }
        div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p { color: #64748b; }
        div[data-testid="stMetricValue"] { color: #172033; }
        div[data-testid="stAlert"] { background: #fff; border-color: #e2e5ef; color: #263149; }
        .footer-note { color: #8791a4; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <div class="hero-content">
        <div class="eyebrow">✦ &nbsp; AI-POWERED REVIEW INSIGHTS</div>
        <h1>Movie Review<br>Sentiment Analyzer</h1>
        <p>Every review has a feeling. Reveal yours with a fast.</p>
        <div class="hero-tags"><span class="hero-tag">✧ &nbsp; Positive or negative</span><span class="hero-tag">⚡ &nbsp; Instant analysis</span><span class="hero-tag">◈ &nbsp; BiLSTM powered</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Try an example</div><div class="section-copy">Choose a sample to see the kind of reviews the model can analyze.</div>', unsafe_allow_html=True)
with st.expander("Example reviews"):
    st.code("I love this movie. The acting was brilliant and the story was wonderful.")
    st.code("This movie was boring. The plot was weak and the acting was terrible.")

with st.form("review_form"):
    review = st.text_area(
        "Enter your movie review",
        placeholder="Example: I love this movie because it is amazing and entertaining.",
        height=170,
        max_chars=10000,
        help="Write your review in English. Longer reviews with clear sentiment usually produce more reliable predictions.",
    )
    submitted = st.form_submit_button("Analyze review", type="primary", use_container_width=True)

if submitted:
    if not review.strip():
        st.warning("Please enter a review before submitting.")
    else:
        try:
            with st.spinner("Analyzing your review..."):
                model, tokenizer = load_assets()
                cleaned = clean_text(review)
                if cleaned:
                    known_words = sum(
                        1 for word in cleaned.split()
                        if word in tokenizer.word_index
                        and tokenizer.word_index[word] < tokenizer.num_words
                    )
                else:
                    known_words = 0
                sentiment, positive_score = predict_sentiment(review, model, tokenizer)

            short_review = len(cleaned.split()) < 5
            if short_review:
                model_direction = "positive" if sentiment == "positive" else "negative"
                st.markdown(
                    f'<div class="result-card result-short"><div class="result-label">Limited context ? short review</div><div class="result-value">Model currently leans {model_direction}</div></div>',
                    unsafe_allow_html=True,
                )
            elif sentiment == "positive":
                st.markdown('<div class="result-card result-positive"><div class="result-label">Analysis complete</div><div class="result-value">? Positive sentiment</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-card result-negative"><div class="result-label">Analysis complete</div><div class="result-value">? Negative sentiment</div></div>', unsafe_allow_html=True)

            negative_score = 1.0 - positive_score
            col1, col2 = st.columns(2)
            col1.metric("Positive score", f"{positive_score:.1%}")
            col2.metric("Negative score", f"{negative_score:.1%}")
            st.progress(max(0.0, min(1.0, positive_score)))

            if short_review:
                st.info("This model was trained on longer movie reviews. Add more detail for a more reliable sentiment estimate.")
            if known_words == 0:
                st.warning("The tokenizer did not recognize any words in this review. Try using common English words and check the spelling.")

            with st.expander("View processed text"):
                st.write(cleaned if cleaned else "No text remained after preprocessing.")
                st.caption(f"Recognized vocabulary words: {known_words}")
        except FileNotFoundError as error:
            st.error(str(error))
        except ValueError as error:
            st.warning(str(error))
        except Exception as error:
            st.error(f"An unexpected error occurred while analyzing the review: {error}")

st.markdown('<div class="footer-note">Made for exploring movie reviews · Predictions are model estimates and may not always reflect the review accurately.</div>', unsafe_allow_html=True)
