"""
ResAI Backend – FastAPI application.
LLM provider: Anthropic Claude (via llm_provider.py)
"""

import hashlib
import logging
import os
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pdf2image
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.llm_provider import (
    generate_text,
    generate_with_image,
    get_model_name,
    image_to_claude_block,
)

# ---------------------------------------------------------------------------
# Environment & config
# ---------------------------------------------------------------------------

load_dotenv()

# Fail fast if the required key is missing – llm_provider will raise at first
# call, but an early check here gives a clearer startup error.
if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY is required")

MAX_RESUME_SIZE_MB = int(os.getenv("MAX_RESUME_SIZE_MB", "10"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "2"))
ANALYSIS_CACHE_TTL_SECONDS = int(os.getenv("ANALYSIS_CACHE_TTL_SECONDS", "86400"))
ANALYSIS_CACHE_MAX_ENTRIES = int(os.getenv("ANALYSIS_CACHE_MAX_ENTRIES", "500"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resai-backend")

# ---------------------------------------------------------------------------
# In-memory analysis cache
# ---------------------------------------------------------------------------

analysis_cache: dict = {}


def prune_analysis_cache() -> None:
    now = time.time()
    expired = [k for k, v in analysis_cache.items() if now - v["ts"] > ANALYSIS_CACHE_TTL_SECONDS]
    for key in expired:
        analysis_cache.pop(key, None)
    if len(analysis_cache) > ANALYSIS_CACHE_MAX_ENTRIES:
        oldest = sorted(analysis_cache.items(), key=lambda item: item[1]["ts"])[
            : len(analysis_cache) - ANALYSIS_CACHE_MAX_ENTRIES
        ]
        for key, _ in oldest:
            analysis_cache.pop(key, None)


def get_cached_analysis(cache_key: str) -> str | None:
    prune_analysis_cache()
    hit = analysis_cache.get(cache_key)
    return hit["analysis"] if hit else None


def set_cached_analysis(cache_key: str, analysis: str) -> None:
    prune_analysis_cache()
    analysis_cache[cache_key] = {"analysis": analysis, "ts": time.time()}


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    match_percentage: float
    analysis: str


class TextResponse(BaseModel):
    content: str


class OptimizeResponse(BaseModel):
    suggestions: str
    keywords: List[str]


class JobSearchResponse(BaseModel):
    results_markdown: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ResAI Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    logger.error(
        "Unhandled error request_id=%s path=%s method=%s error=%s\n%s",
        request_id,
        request.url.path,
        request.method,
        str(exc),
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error (request_id={request_id})."},
    )


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_bytes: bytes):
    poppler_path = os.getenv("POPPLER_PATH") or None
    kwargs = {"poppler_path": poppler_path} if poppler_path else {}
    try:
        return pdf2image.convert_from_bytes(pdf_bytes, **kwargs)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to parse PDF. Ensure Poppler is installed and set POPPLER_PATH if needed. "
                f"Original error: {exc}"
            ),
        ) from exc


def validate_resume_upload(filename: str, pdf_bytes: bytes) -> None:
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")
    max_bytes = MAX_RESUME_SIZE_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF is too large. Max allowed size is {MAX_RESUME_SIZE_MB}MB.",
        )


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def parse_percentage(text: str) -> float:
    """
    Extract the first percentage value from model output.
    Robust: tries explicit XX% pattern first, then looks for numbers near
    keywords like 'match', 'score', 'fit' so a slightly varied response
    doesn't silently return 0.
    """
    # Primary: look for explicit percent sign
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        try:
            return min(float(match.group(1)), 100.0)
        except ValueError:
            pass

    # Fallback: look for a number near match-related keywords
    snippet_match = re.search(
        r"(?:match|score|fit|compatibility)[^\d]{0,20}(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if snippet_match:
        try:
            return min(float(snippet_match.group(1)), 100.0)
        except ValueError:
            pass

    return 0.0


def extract_keywords(job_description: str) -> List[str]:
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "you", "your", "our",
        "are", "will", "have", "has", "had", "not", "but", "all", "any", "job",
        "role", "work", "team", "ability", "experience", "years", "required",
        "preferred", "skills", "skill", "using", "their", "they", "can", "should",
        "into", "about", "who", "what", "where", "when", "how", "more", "must",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{2,}", job_description.lower())
    freq: dict = {}
    for word in words:
        if word in stop_words:
            continue
        freq[word] = freq.get(word, 0) + 1
    ordered = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [word for word, _ in ordered[:15]]


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def extract_resume_text(images) -> str:
    """OCR up to MAX_OCR_PAGES pages using Claude vision."""
    text_parts = []
    for image in images[:MAX_OCR_PAGES]:
        try:
            extracted = generate_with_image(
                prompt="Extract all text from this resume image, preserving line breaks and formatting.",
                image=image,
                context="resume OCR",
            )
            if extracted:
                text_parts.append(extracted)
        except HTTPException:
            continue
    return "\n".join(text_parts)


def run_prompt_with_resume(prompt: str, image, job_description: str) -> str:
    """Send a prompt + first-page image + job description to Claude."""
    return generate_with_image(
        prompt=prompt,
        image=image,
        extra_text=job_description,
        context="resume prompt",
    )


# ---------------------------------------------------------------------------
# Tavily job search
# ---------------------------------------------------------------------------

def tavily_job_search(resume_text: str, job_description: str, count: int = 5) -> str:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Job search is unavailable: TAVILY_API_KEY is not configured."

    # Extract skills and job title in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        skills_fut = pool.submit(
            generate_text,
            "Extract the top 10 professional skills from this resume as a comma-separated list:\n"
            f"{resume_text}",
            "resume skill extraction",
        )
        title_fut = pool.submit(
            generate_text,
            "Extract the exact job title from this job description. Return only the title, nothing else.\n"
            f"{job_description}",
            "job title extraction",
        )
        resume_skills = skills_fut.result().strip()
        job_title = title_fut.result().strip().split("\n")[0]

    search_query = f'"{job_title}" jobs {" ".join(resume_skills.split(",")[:3])} hiring now'

    payload = {
        "api_key": tavily_api_key,
        "query": search_query,
        "search_depth": "basic",
        "include_domains": ["linkedin.com", "indeed.com", "glassdoor.com", "monster.com"],
        "max_results": count,
        "include_raw_content": False,
    }

    try:
        response = requests.post("https://api.tavily.com/search", json=payload, timeout=30)
    except requests.RequestException:
        return "Job search service is temporarily unavailable. Please retry shortly."

    if response.status_code != 200:
        return f"Tavily API error {response.status_code}: {response.text}"

    results = response.json().get("results", [])
    if not results:
        return "No matching jobs found."

    lines = [f"## Job Search Results for: {job_title}\n"]
    for idx, result in enumerate(results, start=1):
        title = result.get("title", "Untitled Job")
        link = result.get("url", "#")
        snippet = result.get("content", "No description available")
        lines.append(f"### {idx}. [{title}]({link})")
        lines.append(f"{snippet[:300]}...\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": get_model_name(), "provider": "anthropic"}


@app.post("/api/resume/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    pdf_bytes = await resume.read()
    validate_resume_upload(resume.filename or "", pdf_bytes)

    cache_key = hashlib.sha256(
        pdf_bytes + b"||" + job_description.strip().encode("utf-8")
    ).hexdigest()
    cached = get_cached_analysis(cache_key)
    if cached:
        return AnalyzeResponse(match_percentage=parse_percentage(cached), analysis=cached)

    images = pdf_to_images(pdf_bytes)
    if not images:
        raise HTTPException(status_code=400, detail="No pages found in uploaded PDF.")

    prompt = (
        "You are an experienced technical recruiter. Review this resume against the job description. "
        "Structure your response with these sections:\n"
        "OVERVIEW\nSTRENGTHS\nGAPS\nMATCH PERCENTAGE\nRECOMMENDATION\n\n"
        "Start the MATCH PERCENTAGE section with a line like: '82%' (just the number and percent sign)."
    )
    analysis = run_prompt_with_resume(prompt, images[0], job_description)
    set_cached_analysis(cache_key, analysis)
    return AnalyzeResponse(match_percentage=parse_percentage(analysis), analysis=analysis)


@app.post("/api/resume/optimize", response_model=OptimizeResponse)
async def optimize_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    pdf_bytes = await resume.read()
    validate_resume_upload(resume.filename or "", pdf_bytes)
    images = pdf_to_images(pdf_bytes)

    prompt = (
        "You are a career coach. Provide 5 specific, actionable resume improvements "
        "based on this resume and job description. Use concise bullet points."
    )
    suggestions = run_prompt_with_resume(prompt, images[0], job_description)
    keywords = extract_keywords(job_description)
    return OptimizeResponse(suggestions=suggestions, keywords=keywords)


@app.post("/api/cover-letter", response_model=TextResponse)
async def cover_letter(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    company_name: str = Form(default="Company"),
    hiring_manager: str = Form(default="Hiring Manager"),
    focus_areas: str = Form(default="balanced"),
):
    pdf_bytes = await resume.read()
    validate_resume_upload(resume.filename or "", pdf_bytes)
    images = pdf_to_images(pdf_bytes)

    prompt = (
        f"Write a tailored 300-400 word cover letter. Company: {company_name}. "
        f"Hiring Manager: {hiring_manager}. Focus areas: {focus_areas}. "
        "Use professional format and make concrete connections to the job description."
    )
    content = run_prompt_with_resume(prompt, images[0], job_description)
    return TextResponse(content=content)


@app.post("/api/interview-prep", response_model=TextResponse)
async def interview_prep(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    pdf_bytes = await resume.read()
    validate_resume_upload(resume.filename or "", pdf_bytes)
    images = pdf_to_images(pdf_bytes)

    prompt = (
        "Generate an interview prep guide with:\n"
        "- 5 technical questions (with answer strategy notes)\n"
        "- 3 behavioral questions (with answer strategy notes)\n"
        "- 2 gap-risk questions the interviewer may probe (with how to address them)\n"
        "Base everything on the resume and job description provided."
    )
    content = run_prompt_with_resume(prompt, images[0], job_description)
    return TextResponse(content=content)


@app.post("/api/market-position", response_model=TextResponse)
async def market_position(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    pdf_bytes = await resume.read()
    validate_resume_upload(resume.filename or "", pdf_bytes)
    images = pdf_to_images(pdf_bytes)

    prompt = (
        "Create an ideal candidate profile for this role, compare this resume against it, "
        "and estimate the candidate's competitive positioning in today's job market for this role."
    )
    content = run_prompt_with_resume(prompt, images[0], job_description)
    return TextResponse(content=content)


@app.post("/api/skill-plan", response_model=TextResponse)
async def skill_plan(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    pdf_bytes = await resume.read()
    validate_resume_upload(resume.filename or "", pdf_bytes)
    images = pdf_to_images(pdf_bytes)

    prompt = (
        "Create a focused 3-month skill development plan:\n"
        "- Top skills to close gaps for this role\n"
        "- Recommended resources (courses, projects, certifications)\n"
        "- Weekly timeline\n"
        "- Resume update strategy after each milestone"
    )
    content = run_prompt_with_resume(prompt, images[0], job_description)
    return TextResponse(content=content)


@app.post("/api/jobs/search", response_model=JobSearchResponse)
async def jobs_search(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    count: int = Form(default=5),
):
    pdf_bytes = await resume.read()
    validate_resume_upload(resume.filename or "", pdf_bytes)
    images = pdf_to_images(pdf_bytes)
    resume_text = extract_resume_text(images)
    results = tavily_job_search(resume_text=resume_text, job_description=job_description, count=count)
    return JobSearchResponse(results_markdown=results)
