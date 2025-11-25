import streamlit as st
import pandas as pd
import numpy as np

# ---------- page config ----------
st.set_page_config(
    page_title="Vyber - The Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------- data ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/movies_sample.csv")

MOVIES = load_data()

# simple mood → genre mapping for demo purposes
MOOD_TO_GENRES = {
    "chill": ["comedy", "family", "indie"],
    "upbeat": ["adventure", "music", "comedy"],
    "romantic": ["romance", "drama", "music"],
    "tense": ["thriller", "crime", "sci-fi"]
}

def mood_match_score(mood: str, genres_text: str) -> float:
    """Very simple rule-based mood matching."""
    g = (genres_text or "").lower()
    base = 0.3  # base score
    for kw in MOOD_TO_GENRES.get(mood, []):
        if kw in g:
            base += 0.35
    return min(base, 1.0)

def recommend(mood: str, query_text: str, max_runtime: int, topn: int) -> pd.DataFrame:
    """Small demo recommender. Later you can replace this with your teammate's model API."""
    df = MOVIES.copy()

    # query (vibe text) match boost
    if query_text:
        q = query_text.lower().strip()
        df["q_match"] = (
            df["title"].str.lower().str.contains(q).fillna(False).astype(int) * 0.2
            + df["plot"].str.lower().str.contains(q).fillna(False).astype(int) * 0.2
        )
    else:
        df["q_match"] = 0.0

    # mood score
    df["mood_score"] = df["genres"].apply(lambda x: mood_match_score(mood, x))

    # freshness bonus (newer movies a bit higher)
    min_y, max_y = int(df["year"].min()), int(df["year"].max())
    span = max(1, max_y - min_y)
    df["freshness"] = (df["year"] - min_y) / span * 0.15

    # diversity jitter
    rng = np.random.default_rng(123)
    df["diversity_bonus"] = rng.random(len(df)) * 0.05

    # runtime constraint
    df = df[df["runtime"] <= max_runtime]

    # final score (weights easy to explain in presentation)
    df["score"] = (
        0.5 * df["mood_score"]
        + 0.25 * df["q_match"]
        + 0.15 * df["freshness"]
        + 0.10 * df["diversity_bonus"]
    )

    return df.sort_values("score", ascending=False).head(topn).reset_index(drop=True)

# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    default_topn = st.slider("Default Top-N", 5, 20, 10)
    default_runtime = st.selectbox("Default Max Runtime (min)", [90, 100, 120, 180], index=2)
    st.caption("Demo uses a small mock dataset.\nLater you can connect real model APIs here.")

# ---------- header ----------
st.title("🎬 Vyber - The Movie Recommender (UI Demo)")
st.caption("Select your mood and vibe, and get transparent, mood-aligned movie suggestions.")

# ---------- controls ----------
col1, col2, col3 = st.columns([1.2, 1.8, 1.2])
with col1:
    mood = st.select_slider("Mood of the day", options=["chill", "upbeat", "romantic", "tense"])
with col2:
    query_text = st.text_input(
        "Optional vibe text",
        placeholder="e.g., short & funny, festival, detective, comfort"
    )
with col3:
    max_runtime = st.selectbox(
        "Max runtime (minutes)",
        [90, 100, 120, 180],
        index=2
    )

go = st.button("✨ Recommend")
st.divider()

# ---------- results ----------
if go:
    with st.spinner("Finding great options for your mood..."):
        recs = recommend(mood, query_text, max_runtime, topn=default_topn)

    st.subheader(f"Top {len(recs)} picks for **{mood}** mood")
    for _, r in recs.iterrows():
        with st.container():
            c1, c2 = st.columns([3, 7])
            with c1:
                # poster placeholder – later you can show real posters
                st.markdown(
                    """
                    <div style='width:100%;height:160px;border-radius:12px;
                                border:1px dashed #bbb;display:flex;
                                align-items:center;justify-content:center;
                                background:rgba(0,0,0,0.03);'>
                        <span style='opacity:0.7'>Poster</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f"**{r['title']}** ({int(r['year'])})  •  _{r['genres']}_  •  {int(r['runtime'])} min\n\n"
                    f"{r['plot']}\n\n"
                    f"**Why this?** Because you're feeling **{mood}** and this matches your vibe (genres & runtime)."
                )
                fb1, fb2, fb3 = st.columns(3)
                with fb1:
                    if st.button("👍 Like", key=f"like_{r['id']}"):
                        st.toast(f"Liked {r['title']} (event logging stub)", icon="👍")
                with fb2:
                    if st.button("➕ Save", key=f"save_{r['id']}"):
                        st.toast(f"Saved {r['title']} (event logging stub)", icon="⭐")
                with fb3:
                    if st.button("👎 Skip", key=f"skip_{r['id']}"):
                        st.toast(f"Skipped {r['title']} (event logging stub)", icon="👎")

    st.caption("Next step: send these events to a backend `/events` endpoint and use RL to adjust ranking.")
else:
    st.info("Set your mood and click **Recommend** to see suggestions.")

st.divider()
st.markdown(
    "**About this UI**  \n"
    "- Built with Streamlit as a front-end demo for the AI movie recommender.  \n"
    "- This front-end can later call your team’s real ML APIs for recommendations."
)