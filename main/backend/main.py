"""
Usage (dev):
    pip install -r requirements.txt
    uvicorn Vyber_FastAPI_Backend_Main:app --reload
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField, SQLModel, Session, create_engine, select
import joblib
import os
import hashlib
import secrets

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vyber.db")
MODEL_PATH = os.getenv("MODEL_PATH", "./model.joblib")
ACCESS_TOKEN_EXPIRE_SECONDS = 60 * 60 * 24  # 1 day

# Database models
class Movie(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    tmdb_id: Optional[int] = SQLField(index=True)
    title: str
    description: Optional[str] = None
    genres: Optional[str] = None  # comma-separated
    vector_id: Optional[int] = None  # for embedding store

class User(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    username: str = SQLField(index=True)
    hashed_password: str
    is_active: bool = True

class Feedback(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: Optional[int] = SQLField(index=True)
    movie_id: Optional[int] = SQLField(index=True)
    rating: Optional[int] = None
    comment: Optional[str] = None

# Pydantic / API schemas
class MovieOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    genres: Optional[str]

class RecommendRequest(BaseModel):
    mood: str = Field(..., example="happy")
    limit: int = 10

class Token(BaseModel):
    access_token: str
    token_type: str

# App / DB init
app = FastAPI(title="Vyber — Movies that match your vibe")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    create_db_and_tables()
    load_model_background()
    seed_demo_data()
    yield  # Shutdown logic can go here

app = FastAPI(title="Vyber — Movies that match your vibe", lifespan=lifespan)

# Simple in-memory auth (demo)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
_fake_tokens = {}  # token -> username

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.exec(select(User).where(User.username == username)).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(username: str) -> str:
    token = secrets.token_urlsafe(32)
    _fake_tokens[token] = username
    return token

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    username = _fake_tokens.get(token)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == username)).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

# ML Model stub & loader
_ml_pipeline = None

def load_model():
    global _ml_pipeline
    try:
        if os.path.exists(MODEL_PATH):
            _ml_pipeline = joblib.load(MODEL_PATH)
            print("Model loaded from", MODEL_PATH)
        else:
            _ml_pipeline = None
            print("No model found at", MODEL_PATH, "— recommendation endpoint will use heuristics.")
    except Exception as e:
        print("Failed to load model:", e)
        _ml_pipeline = None

def load_model_background():
    # call on startup
    try:
        load_model()
    except Exception:
        pass

# Demo seeding
def seed_demo_data():
    with Session(engine) as session:
        exists = session.exec(select(Movie)).first()
        if exists:
            return
        demo = [
            Movie(title="The Grand Adventure", description="An uplifting journey.", genres="Adventure,Drama"),
            Movie(title="Quiet Nights", description="A calming, introspective film.", genres="Drama,Romance"),
            Movie(title="Laugh Riot", description="High energy comedy.", genres="Comedy"),
        ]
        session.add_all(demo)
        # create demo user
        demo_user = User(username="demo", hashed_password=hash_password("demo123"))
        session.add(demo_user)
        session.commit()

# Helper: simple heuristic recommender
def heuristic_recommend(mood: str, limit: int = 10) -> List[MovieOut]:
    """Very simple mood->genre mapping"""
    mood_map = {
        "happy": ["Comedy", "Family"],
        "sad": ["Drama", "Romance"],
        "calm": ["Drama", "Documentary"],
        "excited": ["Action", "Adventure"],
    }
    preferred = mood_map.get(mood.lower(), ["Drama"])
    with Session(engine) as session:
        stmt = select(Movie)
        movies = session.exec(stmt).all()
        # filter by genres if possible
        filtered = [m for m in movies if any(g.strip() in (m.genres or "") for g in preferred)]
        chosen = filtered[:limit] if filtered else movies[:limit]
        return [MovieOut(id=m.id or -1, title=m.title, description=m.description, genres=m.genres) for m in chosen]

# Routes
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/auth/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        user = authenticate_user(session, form_data.username, form_data.password)
        if not user:
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        token = create_access_token(user.username)
        return {"access_token": token, "token_type": "bearer"}

@app.get("/movies", response_model=List[MovieOut])
def list_movies(q: Optional[str] = None, limit: int = 50):
    with Session(engine) as session:
        stmt = select(Movie)
        if q:
            stmt = stmt.where(Movie.title.contains(q))
        movies = session.exec(stmt).all()
        return [MovieOut(id=m.id or -1, title=m.title, description=m.description, genres=m.genres) for m in movies[:limit]]

@app.post("/recommend", response_model=List[MovieOut])
def recommend(req: RecommendRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    # If ML pipeline available, use it. Else fallback to heuristic.
    if _ml_pipeline:
        try:
            # Example: the pipeline expects a dict with 'mood' key
            ids = _ml_pipeline.predict([req.mood])  # this is a stub — replace with real method
            # Convert ids to MovieOut
            with Session(engine) as session:
                movies = session.exec(select(Movie).where(Movie.id.in_(ids[:req.limit]))).all()
                return [MovieOut(id=m.id or -1, title=m.title, description=m.description, genres=m.genres) for m in movies]
        except Exception as e:
            print("Model predict failed:", e)
    # Heuristic fallback
    return heuristic_recommend(req.mood, req.limit)

@app.post("/feedback")
def feedback(movie_id: int, rating: Optional[int] = None, comment: Optional[str] = None, user: User = Depends(get_current_user)):
    with Session(engine) as session:
        fb = Feedback(user_id=user.id, movie_id=movie_id, rating=rating, comment=comment)
        session.add(fb)
        session.commit()
    return {"status": "saved"}

# Admin endpoint to add movies (protected)
@app.post("/admin/movies", status_code=201)
def add_movie(movie: MovieOut, current_user: User = Depends(get_current_user)):
    # In production check user.is_admin
    with Session(engine) as session:
        m = Movie(title=movie.title, description=movie.description, genres=movie.genres)
        session.add(m)
        session.commit()
        session.refresh(m)
    return {"id": m.id}

# Utilities: export database or run one-off tasks
@app.post("/admin/reload_model")
def reload_model(current_user: User = Depends(get_current_user)):
    load_model()
    return {"status": "model reloaded"}

# If run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Vyber_FastAPI_Backend_Main:app", host="0.0.0.0", port=8000, reload=True)
