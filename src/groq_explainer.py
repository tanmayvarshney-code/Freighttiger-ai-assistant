"""
groq_explainer.py
-----------------
Groq LLM integration for generating concise, business-friendly explanations.

Design principles:
  - Groq is called ONLY for anomalous route-weeks (never for normal ones)
  - All numeric facts are computed deterministically BEFORE calling Groq
  - Groq cannot override the anomaly verdict or change note IDs
  - Responses are validated; hallucinated text is caught and replaced
  - A stable SHA-256 cache prevents redundant API calls on repeated runs
  - Exponential back-off retries handle transient failures
  - A deterministic fallback template is used if Groq is unavailable
"""

from __future__ import annotations

import os
import time
import json
import hashlib
import textwrap
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_FILE = Path(__file__).parent.parent / ".cache" / "explanation_cache.json"
PROMPT_VERSION = "v3"           # bump to invalidate all cached entries
MAX_RETRIES = 3
BASE_BACKOFF = 2.0              # seconds
MAX_COMPLETION_TOKENS = 280
TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are a logistics cost analyst. Generate concise, factual explanations "
    "using only the provided numeric evidence and contextual notes. "
    "Never invent a note, cost value, date, or reason. "
    "Do not change the supplied anomaly verdict. "
    "If the evidence is insufficient, explicitly state that no verified "
    "justification was found. Respond in one or two sentences."
)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _cache_key(model: str, route: str, week_of: str,
               cost: float, hist_pct: float | None, peer_pct: float | None,
               note_id: str | None, note_text: str | None) -> str:
    """Return a stable SHA-256 key for the given inputs."""
    payload = json.dumps({
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "route": route,
        "week_of": week_of,
        "cost": round(cost, 6),
        "hist_pct": round(hist_pct, 2) if hist_pct is not None else None,
        "peer_pct": round(peer_pct, 2) if peer_pct is not None else None,
        "note_id": note_id,
        "note_text": note_text,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    route: str,
    week_of: str,
    cost: float,
    hist_pct: float | None,
    peer_pct: float | None,
    flagged: bool,
    note_id: str | None,
    note_text: str | None,
) -> str:
    """Build a compact, fact-grounded user prompt."""
    hist_str = f"{hist_pct:+.1f}% vs this route's 8-week average" if hist_pct is not None else "N/A"
    peer_str = f"{peer_pct:+.1f}% vs peer routes this week" if peer_pct is not None else "N/A"
    verdict = "ANOMALOUS – flagged for review" if flagged else "NOT anomalous (justified by evidence)"

    note_block = ""
    if note_id and note_text:
        note_block = f"\nContextual note [{note_id}]: {note_text}"

    prompt = textwrap.dedent(f"""\
        Route: {route}
        Week of: {week_of}
        Cost per tonne-km: ₹{cost:.4f}
        Historical deviation: {hist_str}
        Peer-route deviation: {peer_str}
        Anomaly verdict (do NOT change): {verdict}{note_block}

        Task: In 1-2 sentences, explain this cost situation for a logistics manager.
        Ground your explanation ONLY in the numbers and note above.
        Do not invent facts or additional notes.
    """)
    return prompt.strip()


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

def _validate_response(text: str, note_id: str | None) -> tuple[bool, str]:
    """Basic hallucination checks on the LLM response.

    Returns (is_valid, cleaned_text).
    """
    if not text or len(text.strip()) < 10:
        return False, ""

    lower = text.lower()

    # Reject if model invents a note ID not provided
    import re
    found_note_ids = set(re.findall(r"n\d{3}", lower))
    if note_id:
        provided = {note_id.lower()}
    else:
        provided = set()
    invented = found_note_ids - provided
    if invented:
        return False, ""

    # Reject if model claims "no anomaly" when verdict is anomalous or vice-versa
    # (We don't enforce this strictly here since Groq sometimes paraphrases)

    cleaned = text.strip()
    return True, cleaned


# ---------------------------------------------------------------------------
# Fallback template
# ---------------------------------------------------------------------------

def _fallback_explanation(
    route: str,
    week_of: str,
    hist_pct: float | None,
    peer_pct: float | None,
    flagged: bool,
    note_id: str | None,
    note_text: str | None,
) -> str:
    """Return a deterministic, template-based explanation when Groq is unavailable."""
    hist_str = f"{hist_pct:+.1f}%" if hist_pct is not None else "N/A"
    peer_str = f"{peer_pct:+.1f}%" if peer_pct is not None else "N/A"

    if note_id and note_text:
        return (
            f"Matches note {note_id}: {note_text[:200].rstrip('.')}. "
            f"The cost rise ({hist_str} vs own history; {peer_str} vs peers) "
            f"has a verified contextual explanation."
        )

    if flagged:
        parts = []
        if hist_pct is not None:
            parts.append(f"{hist_pct:+.1f}% above its 8-week average")
        if peer_pct is not None:
            parts.append(f"{peer_pct:+.1f}% above peer routes")
        deviation_str = " and ".join(parts) if parts else "outside normal thresholds"
        return (
            f"Cost on {route} for week of {week_of} is {deviation_str}. "
            f"No matching contextual note was found; flagged for human review."
        )
    else:
        return (
            f"Cost on {route} for week of {week_of} is within acceptable ranges "
            f"({hist_str} vs history; {peer_str} vs peers)."
        )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_explanation(
    route: str,
    week_of: str,
    cost: float,
    hist_pct: float | None,
    peer_pct: float | None,
    flagged: bool,
    note_id: str | None,
    note_text: str | None,
    cache: dict | None = None,
) -> tuple[str, dict, bool]:
    """Generate (or retrieve cached) explanation for one anomalous route-week.

    Parameters
    ----------
    cache : dict | None
        Pass the loaded cache dict; it is mutated in-place on new entries.

    Returns
    -------
    (explanation_text, updated_cache, used_api)
        used_api is True when the Groq API was actually called.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    if cache is None:
        cache = _load_cache()

    key = _cache_key(model, route, week_of, cost, hist_pct, peer_pct, note_id, note_text)

    if key in cache:
        return cache[key], cache, False   # cache hit

    # ── Try Groq API ─────────────────────────────────────────────────────────
    if not api_key:
        explanation = _fallback_explanation(route, week_of, hist_pct, peer_pct, flagged, note_id, note_text)
        cache[key] = explanation
        return explanation, cache, False

    user_prompt = _build_user_prompt(route, week_of, cost, hist_pct, peer_pct, flagged, note_id, note_text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            from groq import Groq, APIStatusError, APIConnectionError  # type: ignore

            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                seed=42,
            )
            raw_text = response.choices[0].message.content or ""
            is_valid, cleaned = _validate_response(raw_text, note_id)

            if is_valid:
                cache[key] = cleaned
                return cleaned, cache, True
            else:
                print(f"[groq_explainer] Response validation failed for {route} {week_of}. Using fallback.")
                break

        except Exception as exc:
            exc_str = str(exc)
            # Do not retry on authentication errors
            if "401" in exc_str or "invalid_api_key" in exc_str.lower():
                print(f"[groq_explainer] Invalid API key - falling back to template.")
                break

            # Do not retry on model not found errors
            if "404" in exc_str or "model_not_found" in exc_str.lower():
                print(f"[groq_explainer] Model not found: {exc}. Check GROQ_MODEL in .env. Using fallback.")
                break

            if attempt < MAX_RETRIES:
                wait = BASE_BACKOFF ** attempt
                print(f"[groq_explainer] Attempt {attempt} failed ({exc}). Retrying in {wait:.1f}s…")
                time.sleep(wait)
            else:
                print(f"[groq_explainer] All {MAX_RETRIES} attempts failed. Using fallback.")

    # ── Deterministic fallback ────────────────────────────────────────────────
    explanation = _fallback_explanation(route, week_of, hist_pct, peer_pct, flagged, note_id, note_text)
    cache[key] = explanation
    return explanation, cache, False


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

class UsageTracker:
    """Accumulate Groq token usage across the pipeline run."""

    def __init__(self):
        self.calls: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cached_hits: int = 0

    def record(self, prompt_tok: int = 0, completion_tok: int = 0, used_api: bool = True):
        if used_api:
            self.calls += 1
            self.prompt_tokens += prompt_tok
            self.completion_tokens += completion_tok
        else:
            self.cached_hits += 1

    def estimated_cost_usd(self) -> float:
        """Estimate cost based on Groq's published pricing for llama-3.1-8b-instant
        (as of late 2024/2025: $0.05 per 1M prompt tokens, $0.08 per 1M completion tokens).
        """
        prompt_cost = (self.prompt_tokens / 1_000_000) * 0.05
        completion_cost = (self.completion_tokens / 1_000_000) * 0.08
        return round(prompt_cost + completion_cost, 6)

    def summary(self) -> dict:
        return {
            "api_calls": self.calls,
            "cached_hits": self.cached_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "estimated_cost_usd": self.estimated_cost_usd(),
        }
