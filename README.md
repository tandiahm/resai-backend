# ResAI Backend (FastAPI)

## What it does
Provides API endpoints for resume analysis, optimization, cover letters, interview prep, market positioning, skill plans, and job search.

## Run locally
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

## Required env vars
- `GOOGLE_API_KEY`
- `TAVILY_API_KEY` (needed for job search endpoint)

## Deploy (Render)
1. Create a new Web Service from this repo/folder.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Set env vars from `.env.example`.
5. Add your frontend domain in `ALLOWED_ORIGINS`.
