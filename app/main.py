import base64
import io
import logging
import os
import re
import traceback
import uuid
from typing import List

import google.generativeai as genai
import pdf2image
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is required")

genai.configure(api_key=api_key)

_SELECTED_MODEL_NAME = None
_SELECTED_MODEL = None
MAX_RESUME_SIZE_MB = int(os.getenv("MAX_RESUME_SIZE_MB", "10"))
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "2"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resai-backend")


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


app = FastAPI(title="ResAI Backend", version="1.0.0")

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
        content={
            "detail": f"Internal server error (request_id={request_id}).",
        },
    )


def choose_model_name(preferred_env_var: str = "GOOGLE_MODEL") -> str:
    global _SELECTED_MODEL_NAME
    if _SELECTED_MODEL_NAME:
        return _SELECTED_MODEL_NAME

    preferred = os.getenv(preferred_env_var)

    try:
        models = genai.list_models()
        names = [getattr(m, "name", str(m)) for m in models]
    except Exception:
        names = []

    def bad_variant(name: str) -> bool:
        lowered = name.lower()
        return "preview" in lowered or "audio" in lowered

    def latest_by_family(family: str):
        best_name = None
        best_score = -9999.0
        for name in names:
            lowered = name.lower()
            if family not in lowered:
                continue
            match = re.search(rf"{re.escape(family)}[-_/]?(\d+(?:\.\d+)?)", lowered)
            version = float(match.group(1)) if match else 0.0
            score = version + (0 if not bad_variant(name) else -1000)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name:
            return best_name
        for name in names:
            if family in name.lower() and not bad_variant(name):
                return name
        return None

    if preferred:
        pref = preferred.strip().lower()
        if pref == "latest":
            for family in ["gemini", "text-bison", "bison"]:
                candidate = latest_by_family(family)
                if candidate:
                    _SELECTED_MODEL_NAME = candidate
                    return _SELECTED_MODEL_NAME
        elif pref.endswith("-latest"):
            family = pref[:-7]
            candidate = latest_by_family(family)
            if candidate:
                _SELECTED_MODEL_NAME = candidate
                return _SELECTED_MODEL_NAME
        else:
            for name in names:
                if pref in name.lower() and not bad_variant(name):
                    _SELECTED_MODEL_NAME = name
                    return _SELECTED_MODEL_NAME
            for name in names:
                if pref in name.lower():
                    _SELECTED_MODEL_NAME = name
                    return _SELECTED_MODEL_NAME

    for token in ["gemini-2.5", "gemini-1.5", "gemini", "text-bison", "bison"]:
        for name in names:
            if token in name.lower() and not bad_variant(name):
                _SELECTED_MODEL_NAME = name
                return _SELECTED_MODEL_NAME

    _SELECTED_MODEL_NAME = names[0] if names else "models/text-bison-001"
    return _SELECTED_MODEL_NAME


def get_model():
    global _SELECTED_MODEL, _SELECTED_MODEL_NAME
    if _SELECTED_MODEL is not None:
        return _SELECTED_MODEL

    tried = set()
    for _ in range(2):
        name = choose_model_name()
        if name in tried:
            break
        tried.add(name)
        try:
            _SELECTED_MODEL = genai.GenerativeModel(name)
            return _SELECTED_MODEL
        except Exception as exc:
            err = str(exc).lower()
            if any(token in err for token in ["404", "not found", "unsupported", "not supported"]):
                _SELECTED_MODEL = None
                _SELECTED_MODEL_NAME = None
                continue
            raise

    _SELECTED_MODEL = genai.GenerativeModel("models/text-bison-001")
    return _SELECTED_MODEL


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


def validate_resume_upload(filename: str, pdf_bytes: bytes):
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")
    max_bytes = MAX_RESUME_SIZE_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF is too large. Max allowed size is {MAX_RESUME_SIZE_MB}MB.",
        )


def image_to_part(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(img_byte_arr.getvalue()).decode(),
    }


def safe_generate_content(payload, context: str) -> str:
    try:
        model = get_model()
        response = model.generate_content(payload)
        return (response.text or "").strip()
    except HTTPException:
        raise
    except Exception as exc:
        err = str(exc).lower()
        if any(token in err for token in ["quota", "rate limit", "resource exhausted", "429"]):
            raise HTTPException(
                status_code=429,
                detail=f"Model is rate-limited for {context}. Please retry in a minute.",
            ) from exc
        if any(token in err for token in ["deadline", "timed out", "timeout"]):
            raise HTTPException(
                status_code=504,
                detail=f"Model request timed out for {context}. Please retry.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"Model request failed for {context}.",
        ) from exc


def extract_resume_text(images) -> str:
    text = []
    for image in images[:MAX_OCR_PAGES]:
        try:
            prompt = "Extract all text from this image, preserving line breaks and formatting as much as possible."
            extracted = safe_generate_content([prompt, image_to_part(image)], context="resume OCR")
            if extracted:
                text.append(extracted)
        except HTTPException:
            continue
    return "\n".join(text)


def run_prompt_with_resume(prompt: str, first_page_part: dict, job_description: str) -> str:
    return safe_generate_content([prompt, first_page_part, job_description], context="resume prompt")


def parse_percentage(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)%", text)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def extract_keywords(job_description: str) -> List[str]:
    prompt = (
        "Extract top 15 important keywords from this job description. "
        "Return only a comma-separated list.\n\n"
        f"{job_description}"
    )
    try:
        response = safe_generate_content(prompt, context="keyword extraction")
    except HTTPException:
        return []
    return [item.strip() for item in response.split(",") if item.strip()]


def tavily_job_search(resume_text: str, job_description: str, count: int = 5) -> str:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "TAVILY_API_KEY is missing"

    skills_response = safe_generate_content(
        "Extract top 10 professional skills from this resume as a comma-separated list:\n"
        f"{resume_text}",
        context="resume skill extraction",
    )
    resume_skills = skills_response.strip()

    title_response = safe_generate_content(
        "Extract the exact job title from this job description. Return only the title.\n"
        f"{job_description}",
        context="job title extraction",
    )
    job_title = title_response.strip().split("\n")[0]

    search_query = f'"{job_title}" jobs {" ".join(resume_skills.split(",")[:3])} hiring now'

    payload = {
        "api_key": tavily_api_key,
        "query": search_query,
        "search_depth": "advanced",
        "include_domains": ["linkedin.com", "indeed.com", "glassdoor.com", "monster.com"],
        "max_results": count,
        "include_raw_content": True,
    }
    try:
        response = requests.post("https://api.tavily.com/search", json=payload, timeout=60)
    except requests.RequestException:
        return "Job search service is temporarily unavailable. Please retry shortly."
    if response.status_code != 200:
        return f"Tavily API error {response.status_code}: {response.text}"

    results = response.json().get("results", [])
    if not results:
        return "No jobs found"

    lines = ["## Personalized Job Search Results\n"]
    for idx, result in enumerate(results, start=1):
        title = result.get("title", "Untitled Job")
        link = result.get("url", "#")
        snippet = result.get("raw_content", "No description available")

        relevance_prompt = (
            "Analyze relevance of this job for the candidate. Return: score (0-100%), key matching skills, fit notes.\n"
            f"Candidate Skills: {resume_skills}\n"
            f"Job Title: {title}\n"
            f"Job Description: {snippet}"
        )
        try:
            relevance = safe_generate_content(relevance_prompt, context="job relevance analysis")
        except HTTPException:
            relevance = "Relevance analysis unavailable for this result."

        lines.append(f"### {idx}. {title}")
        lines.append(f"Link: [{link}]({link})")
        lines.append(f"Description: {snippet}")
        lines.append(f"Relevance:\n{relevance}\n")

    return "\n".join(lines)


@app.get("/health")
def health():
    return {"status": "ok", "model": choose_model_name()}


@app.post("/api/resume/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    pdf_bytes = await resume.read()
    validate_resume_upload(resume.filename or "", pdf_bytes)
    images = pdf_to_images(pdf_bytes)
    if not images:
        raise HTTPException(status_code=400, detail="No pages found in uploaded PDF")

    first_page_part = image_to_part(images[0])
    prompt = (
        "You are an experienced technical recruiter. Review this resume against the job description. "
        "Return sections: OVERVIEW, STRENGTHS, GAPS, MATCH PERCENTAGE, RECOMMENDATION. "
        "Start with percentage line as 'XX%'."
    )
    analysis = run_prompt_with_resume(prompt, first_page_part, job_description)
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
        "You are a career coach. Provide 5 specific actionable resume improvements based on this resume and job description. "
        "Use concise bullet points."
    )
    suggestions = run_prompt_with_resume(prompt, image_to_part(images[0]), job_description)
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
        f"Hiring Manager: {hiring_manager}. Focus: {focus_areas}. "
        "Use professional format and concrete relevance to the job description."
    )
    content = run_prompt_with_resume(prompt, image_to_part(images[0]), job_description)
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
        "Generate interview prep with 5 technical questions, 3 behavioral questions, and 2 gap-risk questions. "
        "For each, provide answer strategy notes."
    )
    content = run_prompt_with_resume(prompt, image_to_part(images[0]), job_description)
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
        "Create ideal candidate profile, compare this resume against it, and estimate competitive positioning."
    )
    content = run_prompt_with_resume(prompt, image_to_part(images[0]), job_description)
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
        "Create a 3-month skill plan: top skills to close gaps, resources, weekly timeline, and resume update strategy."
    )
    content = run_prompt_with_resume(prompt, image_to_part(images[0]), job_description)
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
