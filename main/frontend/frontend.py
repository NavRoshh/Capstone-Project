import sys
from pathlib import Path
import os

CURRENT_FILE = Path(__file__).resolve()
MAIN_DIR = CURRENT_FILE.parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from backend.ai.emotion_detection import load_movies, detect_mood, recommend, surprise_me

from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import base64
from pathlib import Path


def set_background(image_name: str):
    """
    Set a blurred movie-poster collage as the app background,
    with a dark overlay for accessibility (good contrast).
    """
    bg_path = Path(__file__).parent / "bg_vyber.jpg"

    if not bg_path.exists():
        # Fail gracefully if the image is missing
        return

    with open(bg_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
# ---------- BG CSS ----------
    st.markdown(
        f"""
        <style>
        /* Page background */
        .stApp {{
            background-image: linear-gradient(
                rgba(0, 0, 0, 0.70),
                rgba(0, 0, 0, 0.70)
            ), url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}

        /* Main block container – keep it slightly transparent but solid enough for text */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.35);
            border-radius: 14px;
            padding: 1.2rem 2rem;
        }}

        /* Ensure default text is high-contrast */
        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: #f5f5f5 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
# ---------- Header CSS ----------
st.markdown("""
<style>
.header-container {
    display: flex;
    align-items: center;
    gap: 20px;  /* spacing between logo and text */
    margin-bottom: 20px;
}

.header-logo img {
    width: 140px;
}

.header-title {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.header-title h1 {
    margin: 0;
    padding: 0;
    font-size: 38px;
    font-weight: bold;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Vyber – Movies that match your vibe",
    page_icon="🎬",
    layout="wide"
)
set_background("bg_vyber.jpg")

# ---------- LOAD DATA (for stats / debug only) ----------
@st.cache_data
def get_movies_df() -> pd.DataFrame:
    """
    Get the full movies dataframe from the recommender utilities.
    The utility internally loads loaded_movies_df.csv + ratings etc.
    """
    try:
        return load_movies()
    except Exception as e:
        st.error(f"Failed to load movies from utils: {e}")
        return pd.DataFrame()


MOVIES_DF = get_movies_df()


# ---------- MOOD LABELS FOR UI ----------
MOOD_LABELS = {
    "happy": "😄 Happy",
    "sad": "😢 Sad",
    "romantic": "🥰 Romantic",
    "action": "🔥 Action",
    "scary": "👻 Scary",
    "fantasy": "✨ Fantasy",
}

MOOD_ORDER = ["happy", "sad", "romantic", "action", "scary", "fantasy"]


def mood_code_to_label(mood: str) -> str:
    mood = (mood or "").lower()
    return MOOD_LABELS.get(mood, mood.capitalize())


def mood_label_to_code(label: str) -> str:
    for code, txt in MOOD_LABELS.items():
        if txt == label:
            return code
    # fallback: strip emoji if user somehow passes raw label
    return label.split()[-1].lower()


# ---------- LOGO + HEADER ----------
logo_path = os.path.join(MAIN_DIR, "frontend", "vyber main final transparent.png")
logo_col, title_col = st.columns([1, 2])

with logo_col:
    if os.path.exists(logo_path):
        st.image(logo_path, width=350)
    else:
        st.markdown("### 🎬 Vyber")

with title_col:
    st.markdown(
        """
        <div class="header-container">
        <h1> Vyber - Movies that match your vibe </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- SIDEBAR CONTROLS ----------
#with st.sidebar:
 #st.header("⚙️ Settings")

top_n = st.slider("Number of recommendations", min_value=3, max_value=15, value=6)
show_raw = st.checkbox("Show raw recommendation table (for debugging)", value=False)

#st.markdown("---")
#st.caption(
 #       "This UI directly uses the Vyber recommender utilities:\n"
  #      "- `detect_mood(text)` → infer mood from text\n"
    #    "- `recommend(mood, ...)` → ranked list of movies\n"
     #   "- `surprise_me(mood, ...)` → one offbeat pick"
    #)


# ---------- MAIN INPUT AREA ----------
st.subheader("1. Tell us your vibe")

mode = st.radio(
    "How do you want to set your mood?",
    options=["🧠 Let AI detect from my text", "🎛️ Pick mood manually"],
    horizontal=True,
)

col_left, col_right = st.columns(2)

detected_mood_code = None

with col_left:
    if mode.startswith("🧠"):
        user_text = st.text_area(
            "Describe how you're feeling today (1–2 sentences)",
            placeholder="Example: I'm exhausted but I want something light and funny to relax with.",
            height=120,
        )
        detect_btn = st.button("🔍 Detect my mood")
        if detect_btn and user_text.strip():
            with st.spinner("Analyzing your text and detecting mood..."):
                try:
                    detected_mood_code = detect_mood(user_text)
                    st.success(f"Detected mood: **{mood_code_to_label(detected_mood_code)}**")
                except Exception as e:
                    st.error(f"Could not detect mood: {e}")
    else:
        user_text = st.text_area(
            "Optional: What kind of movie are you in the mood for?",
            placeholder="Example: short feel-good romance, minimal violence, comfort watch.",
            height=120,
        )

with col_right:
    st.markdown("#### Or choose a mood directly")

    default_label = MOOD_LABELS["happy"]
    if detected_mood_code:
        default_label = MOOD_LABELS.get(detected_mood_code, default_label)

    mood_label = st.select_slider(
        "Select mood",
        options=[MOOD_LABELS[m] for m in MOOD_ORDER],
        value=default_label,
    )
    chosen_mood_code = mood_label_to_code(mood_label)

st.markdown("---")

# ---------- ACTION BUTTONS ----------
col_rec, col_surprise = st.columns([2, 1])
with col_rec:
    rec_btn = st.button("✨ Get recommendations", use_container_width=True)
with col_surprise:
    surprise_btn = st.button("🎲 Surprise me", use_container_width=True)


# ---------- RECOMMENDATIONS DISPLAY ----------
def render_movie_card(movie: Dict[str, Any], rank: int):
    """
    movie is a dict produced by vyber_emotion_utils.recommend() / surprise_me(),
    expected keys: title, genres (list), mood, avg_rating, vibe_cluster, explanation.
    """
    title = movie.get("title", "Unknown title")
    genres = movie.get("genres", [])
    mood_code = movie.get("mood", "")
    avg_rating = movie.get("avg_rating", None)
    vibe_cluster = movie.get("vibe_cluster", None)
    explanation = movie.get("explanation", "")

    # small badges
    badges = []
    if isinstance(genres, (list, tuple)):
        badges.append(" | ".join(genres[:3]))
    if avg_rating is not None:
        badges.append(f"⭐ {avg_rating:.1f}/5")
    if vibe_cluster is not None:
        badges.append(f"Vibe cluster #{vibe_cluster}")

    badge_text = " • ".join(badges)

    with st.container():
        st.markdown(
            f"### #{rank} – {title}\n"
            f"<span style='color:#ffaa00;'>{badge_text}</span>",
            unsafe_allow_html=True,
        )
        st.write(explanation)
        st.markdown("---")


recommendations_df = None

if rec_btn:
    st.subheader("2. Your recommendations")

    with st.spinner("Matching your mood to movies..."):
        try:
            raw_recs: List[Dict[str, Any]] = recommend(
                mood=chosen_mood_code,
                top_n=top_n,
                user_text=user_text if isinstance(user_text, str) else None,
            )
            if not raw_recs:
                st.warning("No recommendations were returned. Try a different mood or check the backend.")
            else:
                for i, m in enumerate(raw_recs, start=1):
                    render_movie_card(m, rank=i)

                recommendations_df = pd.DataFrame(raw_recs)
        except Exception as e:
            st.error(f"Recommendation failed: {e}")

if surprise_btn:
    st.subheader("🎲 Surprise pick")

    with st.spinner("Picking something a little different but still on-vibe..."):
        try:
            surprise_movie = surprise_me(
                mood=chosen_mood_code,
                user_text=user_text if isinstance(user_text, str) else None,
            )
            render_movie_card(surprise_movie, rank=1)
        except Exception as e:
            st.error(f"Surprise-me failed: {e}")

# ---------- DEBUG / RAW TABLE ----------
if show_raw and recommendations_df is not None:
    st.markdown("### Debug – raw recommendation data")
    st.dataframe(recommendations_df)
