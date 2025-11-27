# Vyber – MVP Requirements (Business Analyst Draft)

## 1) Goal
Provide **mood-based movie recommendations** that feel relevant to a user's current vibe using a simple plug-in (API + UI).

## 2) MVP Scope
- **Inputs**: 
  - Mood button *(happy, sad, calm, excited, neutral)*, or
  - Free‑text mood input (e.g., "I feel down today")
- **Output**:
  - 5 recommended movies (title, genre, poster_url if available, short explanation)
- **Core features**:
  - Mood detection (map text → one of 5 moods)
  - Mood → Genre mapping (see CSV file)
  - Content-based ranking (genres/keywords)
  - Simple feedback (👍/👎)

## 3) Non-Goals (MVP)
- No user accounts or auth
- No advanced RL or personalization history
- No production-scale infra (demo only)

## 4) Data Sources (Demo)
- MovieLens (movies + ratings)
- TMDb metadata (posters/genres if available)

## 5) API Endpoints (draft)
- `POST /mood_input` → `{ "text": "I am happy" }` → `{ "mood": "happy" }`
- `GET  /recommendations?mood=happy` → `[ { "title": "...", "genre": "...", "poster_url": "...", "explanation": "..." }, ... ]`
- `POST /feedback` → `{ "movie_id": 123, "mood": "happy", "feedback": "up|down" }`

## 6) Acceptance Criteria
- Given any of the 5 moods, API returns **≥5** movies within **2s** on a laptop.
- Explanations reference the selected mood and a relevant genre.
- No crashes on unknown text (fallback to "neutral").
- Feedback request is accepted (200 OK).

## 7) Files
- `/docs/requirements.md` (this file)
- `/docs/test_plan.md`
- `/data/mood_genre_mapping.csv` (already prepared)
