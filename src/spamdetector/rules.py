from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

URL_HOST_RE = re.compile(r"(?:https?://|www\.)\s*([^\s/]+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@([a-z0-9.-]+\.[a-z]{2,})", re.IGNORECASE)
DOMAIN_RE = re.compile(r"\b([a-z0-9.-]+\.[a-z]{2,})\b", re.IGNORECASE)
TRAILING_PUNCT = """'\"),.;:!?]"""


def normalize_domain(domain: str) -> str:
    domain = domain.strip().lower().strip(TRAILING_PUNCT)
    if domain.startswith("www."):
        domain = domain[4:]
    if ":" in domain:
        domain = domain.split(":", 1)[0]
    return domain


def extract_domains(text: str) -> Set[str]:
    domains = set()

    for host in URL_HOST_RE.findall(text):
        normalized = normalize_domain(host)
        if normalized:
            domains.add(normalized)

    for host in EMAIL_RE.findall(text):
        normalized = normalize_domain(host)
        if normalized:
            domains.add(normalized)

    for host in DOMAIN_RE.findall(text):
        normalized = normalize_domain(host)
        if normalized:
            domains.add(normalized)

    return domains


def load_whitelist(path: Optional[str]) -> Set[str]:
    if not path:
        return set()

    file_path = Path(path)
    if not file_path.exists():
        return set()

    domains = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        domains.add(normalize_domain(line))
    return domains


def is_whitelisted(text: str, whitelist: Iterable[str]) -> bool:
    allowlist = {normalize_domain(domain) for domain in whitelist if domain}
    if not allowlist:
        return False

    domains = extract_domains(text)
    for domain in domains:
        for allowed in allowlist:
            if domain == allowed or domain.endswith(f".{allowed}"):
                return True
    return False


def apply_rules(
    label: str,
    confidence: Optional[float],
    text: str,
    whitelist: Iterable[str],
    threshold: Optional[float],
    low_confidence_label: str,
) -> Tuple[str, Optional[float], Optional[str]]:
    if is_whitelisted(text, whitelist):
        return "ham", confidence, "whitelist"

    if confidence is not None and threshold is not None and confidence < threshold:
        return low_confidence_label, confidence, "low_confidence"

    return label, confidence, None
