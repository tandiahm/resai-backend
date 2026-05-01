"""
llm_provider.py – Anthropic Claude adapter for ResAI Backend.

All LLM logic in main.py should call this module instead of calling the SDK directly.
"""

import base64
import io
import logging
import os
import time

import anthropic
from fastapi import HTTPException

logger = logging.getLogger("resai-backend")

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Model name helper
# ---------------------------------------------------------------------------

def get_model_name() -> str:
    """Return the configured model name, defaulting to claude-sonnet-4-5."""
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20251001")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def image_to_jpeg_bytes(image) -> bytes:
    """Convert a PIL Image to raw JPEG bytes (provider-neutral)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def jpeg_bytes_to_claude_block(jpeg_bytes: bytes) -> dict:
    """
    Wrap raw JPEG bytes in the Anthropic image content block format.

    Shape:
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "<base64-string>",
            },
        }
    """
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpeg_bytes).decode("utf-8"),
        },
    }


def image_to_claude_block(image) -> dict:
    """Convert a PIL Image directly to a Claude image content block."""
    return jpeg_bytes_to_claude_block(image_to_jpeg_bytes(image))


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------

_RETRY_ATTEMPTS = int(os.getenv("MODEL_RETRY_ATTEMPTS", "3"))
_RETRY_BASE_SECONDS = float(os.getenv("MODEL_RETRY_BASE_SECONDS", "1.0"))
_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "2048"))


def _extract_text(response: anthropic.types.Message) -> str:
    """
    Pull plain text out of an Anthropic Message response.

    The content field is a list of blocks; we join all text blocks.
    """
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts).strip()


def generate_text(prompt: str, context: str = "generate_text") -> str:
    """
    Call Claude with a plain text prompt.
    Retries on rate-limit / timeout errors with exponential back-off.
    """
    content = [{"type": "text", "text": prompt}]
    return _call_with_retry(content, context)


def generate_with_image(
    prompt: str,
    image,  # PIL Image OR pre-built Claude image block dict
    extra_text: str = "",
    context: str = "generate_with_image",
) -> str:
    """
    Call Claude with an image + text prompt.

    `image` can be:
      - a PIL Image object  → converted automatically
      - a dict with "type": "image"  → used as-is (already a Claude block)
    """
    if isinstance(image, dict) and image.get("type") == "image":
        image_block = image
    else:
        image_block = image_to_claude_block(image)

    content: list = [image_block, {"type": "text", "text": prompt}]
    if extra_text:
        content.append({"type": "text", "text": extra_text})

    return _call_with_retry(content, context)


def _call_with_retry(content: list, context: str) -> str:
    client = get_client()
    model = get_model_name()
    last_exc: Exception | None = None

    for attempt in range(_RETRY_ATTEMPTS):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=_MAX_TOKENS,
                temperature=0,
                messages=[{"role": "user", "content": content}],
            )
            return _extract_text(response)

        except HTTPException:
            raise

        except anthropic.RateLimitError as exc:
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                delay = _RETRY_BASE_SECONDS * (2 ** attempt)
                logger.warning(
                    "Rate limited context=%s attempt=%s/%s delay=%.1fs",
                    context, attempt + 1, _RETRY_ATTEMPTS, delay,
                )
                time.sleep(delay)
                continue
            raise HTTPException(
                status_code=429,
                detail=f"Claude is rate-limited for {context}. Please retry in a moment.",
            ) from exc

        except anthropic.APITimeoutError as exc:
            last_exc = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                delay = _RETRY_BASE_SECONDS * (2 ** attempt)
                logger.warning(
                    "Timeout context=%s attempt=%s/%s delay=%.1fs",
                    context, attempt + 1, _RETRY_ATTEMPTS, delay,
                )
                time.sleep(delay)
                continue
            raise HTTPException(
                status_code=504,
                detail=f"Claude request timed out for {context}. Please retry.",
            ) from exc

        except anthropic.APIError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Claude API error for {context}: {exc}",
            ) from exc

    # Exhausted retries
    raise HTTPException(
        status_code=502,
        detail=f"Claude request failed for {context} after {_RETRY_ATTEMPTS} attempts.",
    )
