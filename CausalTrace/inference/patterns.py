"""
Pattern definitions for data flow inference.

Contains regex patterns and detection logic for:
- Credentials (API keys, tokens, passwords)
- URLs and domains
- Prompt injection indicators
- DOM element references
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DataItemType(Enum):
    """Types of data items that can flow between actions."""
    CREDENTIAL = "credential"
    URL = "url"
    DOM_ELEMENT = "dom_element"
    USER_INPUT = "user_input"
    EXTRACTED_TEXT = "extracted_text"
    CODE_SNIPPET = "code_snippet"


# =============================================================================
# Credential Patterns
# =============================================================================

CREDENTIAL_PATTERNS: Dict[str, str] = {
    # GitLab specific (16+ chars to catch shorter test tokens)
    "gitlab_deploy_token": r"gldt-[A-Za-z0-9_-]{10,}",
    "gitlab_access_token": r"glpat-[A-Za-z0-9_-]{10,}",

    # Generic tokens
    "bearer_token": r"Bearer\s+([A-Za-z0-9._-]{20,})",
    "api_key": r"(?:api[_-]?key|apikey)[=:]\s*[\"']?([A-Za-z0-9_-]{16,})[\"']?",
    "access_token": r"(?:access[_-]?token)[=:]\s*[\"']?([A-Za-z0-9_.-]{20,})[\"']?",

    # SSH keys
    "ssh_public_key": r"ssh-(?:rsa|ed25519|ecdsa)\s+[A-Za-z0-9+/=]{40,}",

    # Passwords (careful - high false positive rate)
    "password_field": r"(?:password|passwd|pwd)\s*[:=]\s*[\"']([^\"']{4,})[\"']",

    # Generic secrets
    "secret_key": r"(?:secret[_-]?key|private[_-]?key)[=:]\s*[\"']?([A-Za-z0-9_-]{16,})[\"']?",
}


# =============================================================================
# URL Patterns
# =============================================================================

URL_PATTERNS: Dict[str, str] = {
    "full_url": r"https?://[^\s<>\"']+",
    "domain_only": r"(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(?:/[^\s]*)?",
}

# Known attacker domain patterns (from WASP configs)
ATTACKER_DOMAIN_PATTERNS: List[str] = [
    r"attacker\.com",
    r"evil\.com",
    r"malicious\.",
    r"foo\.com",
    r"exfil[.-]",
    r"data-harvest",
    r"steal[.-]",
]


# =============================================================================
# Injection Patterns
# =============================================================================

INJECTION_PATTERNS: List[str] = [
    r"ATTENTION",
    r"NEW\s+(?:PRIORITY\s+)?INSTRUCTION",
    r"OBJECTIVE[:\s]",
    r"IGNORE\s+PREVIOUS",
    r"SYSTEM\s+UPDATE",
    r"IMPORTANT\s*:",
    r"URGENT\s*:",
    r"YOU\s+MUST",
    r"immediately|urgently",
]


# =============================================================================
# DOM Element Patterns
# =============================================================================

DOM_PATTERNS: Dict[str, str] = {
    "element_id": r"\[(\d+)\]",  # Browser element IDs like [15]
    "css_selector": r"(?:button|input|link|a|div|span)\#([a-zA-Z0-9_-]+)",
    "xpath": r"//[a-zA-Z]+\[@[a-zA-Z]+=[\"'][^\"']+[\"']\]",
}


# =============================================================================
# Detection Functions
# =============================================================================

@dataclass
class PatternMatch:
    """Result of a pattern match."""
    pattern_name: str
    value: str
    start: int
    end: int
    item_type: DataItemType
    metadata: Dict[str, Any]


def extract_credentials(text: str) -> List[PatternMatch]:
    """Extract all credentials from text."""
    matches = []
    for name, pattern in CREDENTIAL_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group(1) if match.groups() else match.group(0)
            matches.append(PatternMatch(
                pattern_name=name,
                value=value,
                start=match.start(),
                end=match.end(),
                item_type=DataItemType.CREDENTIAL,
                metadata={"credential_type": name},
            ))
    return matches


def extract_urls(text: str) -> List[PatternMatch]:
    """Extract all URLs from text."""
    matches = []
    for match in re.finditer(URL_PATTERNS["full_url"], text):
        url = match.group(0)
        is_attacker = is_attacker_url(url)
        matches.append(PatternMatch(
            pattern_name="url",
            value=url,
            start=match.start(),
            end=match.end(),
            item_type=DataItemType.URL,
            metadata={"is_attacker": is_attacker},
        ))
    return matches


def extract_dom_elements(text: str) -> List[PatternMatch]:
    """Extract DOM element references from text."""
    matches = []
    for name, pattern in DOM_PATTERNS.items():
        for match in re.finditer(pattern, text):
            value = match.group(1) if match.groups() else match.group(0)
            matches.append(PatternMatch(
                pattern_name=name,
                value=value,
                start=match.start(),
                end=match.end(),
                item_type=DataItemType.DOM_ELEMENT,
                metadata={"element_type": name},
            ))
    return matches


def is_attacker_url(url: str) -> bool:
    """Check if URL belongs to attacker domain."""
    url_lower = url.lower()
    for pattern in ATTACKER_DOMAIN_PATTERNS:
        if re.search(pattern, url_lower, re.IGNORECASE):
            return True
    return False


def detect_injection_content(text: str) -> Tuple[bool, List[str]]:
    """
    Detect if text contains prompt injection patterns.

    Returns:
        (has_injection, matched_patterns)
    """
    text_upper = text.upper()
    matched = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_upper, re.IGNORECASE):
            matched.append(pattern)
    return len(matched) > 0, matched


def extract_text_segments(
    text: str,
    min_length: int = 10,
    max_length: int = 500,
    max_segments: int = 10
) -> List[str]:
    """
    Extract meaningful text segments from page content.

    Focuses on segments that might contain data references.
    """
    segments = []

    # Split into lines
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if len(line) < min_length or len(line) > max_length:
            continue

        # Look for data-like content indicators (expanded list)
        data_indicators = [
            ':', '=', 'token', 'key', 'id', 'value', 'email', 'user',
            'account', 'balance', 'amount', 'transfer', 'password', 'secret',
            'credential', 'code', 'number', 'phone', 'address', 'name',
            '$', '#', '@',  # Common data prefixes
        ]
        if any(ind in line.lower() for ind in data_indicators):
            segments.append(line)
            if len(segments) >= max_segments:
                break

    return segments


# Pattern to extract alphanumeric identifiers (account numbers, IDs, etc.)
IDENTIFIER_PATTERN = r'\b([A-Z]{2,4}[0-9]{4,12})\b'  # e.g., ACC789456, ID12345678


def extract_identifiers(text: str) -> List[str]:
    """
    Extract alphanumeric identifiers that look like account numbers, IDs, etc.
    """
    return re.findall(IDENTIFIER_PATTERN, text)


def compute_string_similarity(s1: str, s2: str) -> float:
    """
    Compute simple similarity between two strings.

    Returns value between 0.0 and 1.0.
    """
    if not s1 or not s2:
        return 0.0

    s1_lower = s1.lower()
    s2_lower = s2.lower()

    # Exact match
    if s1_lower == s2_lower:
        return 1.0

    # Containment
    if s1_lower in s2_lower or s2_lower in s1_lower:
        shorter = min(len(s1), len(s2))
        longer = max(len(s1), len(s2))
        return shorter / longer

    # Token overlap
    tokens1 = set(s1_lower.split())
    tokens2 = set(s2_lower.split())
    if tokens1 and tokens2:
        overlap = len(tokens1 & tokens2)
        total = len(tokens1 | tokens2)
        return overlap / total if total > 0 else 0.0

    return 0.0
