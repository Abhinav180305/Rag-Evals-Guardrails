"""Lightweight, deterministic guardrails (swap in NeMo Guardrails / Guardrails AI later if needed)."""
import re
from . import config as C

INJECTION = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard .{0,30}(instructions|system prompt|rules)",
    r"(reveal|show|print|repeat) .{0,40}(system prompt|hidden instructions|internal configuration)",
    r"you are now (dan|an? unrestricted)",
    r"developer mode|jailbreak",
]
PII = {
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "AADHAAR": r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)",
    "CARD": r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)",
    "PHONE": r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\d{10}(?!\d)",
}
MAX_QUERY_CHARS = 1000

def redact_pii(text):
    flags = []
    for name, pat in PII.items():
        if re.search(pat, text):
            flags.append(name)
            text = re.sub(pat, f"[{name}]", text)
    return text, flags

def check_input(query):
    """Returns (ok, cleaned_query, flags)."""
    if len(query) > MAX_QUERY_CHARS:
        return False, query, ["TOO_LONG"]
    if any(re.search(p, query, re.I) for p in INJECTION):
        return False, query, ["PROMPT_INJECTION"]
    cleaned, pii = redact_pii(query)
    return True, cleaned, [f"PII_IN_{p}" for p in pii]

def check_output(answer):
    cleaned, pii = redact_pii(answer)
    flags = [f"PII_OUT_{p}" for p in pii]
    if not cleaned.strip():
        return C.REFUSAL, ["EMPTY_OUTPUT"]
    return cleaned, flags
