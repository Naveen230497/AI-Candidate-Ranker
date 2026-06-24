"""
Education Scorer Module
=======================
Scores a candidate's education against role requirements.
Uses the highest-scoring education entry when multiple are present.
Self-taught candidates (no entries) receive a baseline 0.3.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Degree-level mapping
# ---------------------------------------------------------------------------
DEGREE_SCORES: dict[str, float] = {
    "phd": 1.0,
    "doctorate": 1.0,
    "ph.d": 1.0,
    "ph.d.": 1.0,
    "m.tech": 0.8,
    "mtech": 0.8,
    "m.s.": 0.8,
    "m.s": 0.8,
    "ms": 0.8,
    "m.sc": 0.8,
    "msc": 0.8,
    "masters": 0.8,
    "master": 0.8,
    "master's": 0.8,
    "mba": 0.8,
    "m.e.": 0.8,
    "me": 0.8,
    "b.tech": 0.6,
    "btech": 0.6,
    "b.e.": 0.6,
    "be": 0.6,
    "b.s.": 0.6,
    "bs": 0.6,
    "b.sc": 0.6,
    "bsc": 0.6,
    "bachelors": 0.6,
    "bachelor": 0.6,
    "bachelor's": 0.6,
    "bca": 0.6,
    "diploma": 0.3,
}

# Institution tier scores
TIER_SCORES: dict[str, float] = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
}
DEFAULT_TIER_SCORE = 0.4  # unknown / unranked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return text.strip().lower()


def _field_match_score(field_of_study: str, preferred_fields: list[str]) -> float:
    """
    Return 0-1 indicating how well *field_of_study* matches any preferred
    field.  Exact match → 1.0, partial/substring → 0.6, none → 0.0.
    """
    if not field_of_study:
        return 0.0

    f = _norm(field_of_study)

    for pf in preferred_fields:
        pf_low = _norm(pf)
        if f == pf_low:
            return 1.0

    # Partial / substring match
    for pf in preferred_fields:
        pf_low = _norm(pf)
        if pf_low in f or f in pf_low:
            return 0.6

    # Token-overlap heuristic
    f_tokens = set(f.split())
    for pf in preferred_fields:
        pf_tokens = set(_norm(pf).split())
        overlap = f_tokens & pf_tokens
        if overlap:
            return 0.4

    return 0.0


def _tier_score(tier: str) -> float:
    """Map tier string to numeric score."""
    if not tier:
        return DEFAULT_TIER_SCORE
    t = _norm(tier).replace(" ", "_")
    return TIER_SCORES.get(t, DEFAULT_TIER_SCORE)


def _degree_score(degree: str) -> float:
    """Map degree string to numeric score."""
    if not degree:
        return 0.4  # unknown
    d = _norm(degree)

    # Direct lookup
    if d in DEGREE_SCORES:
        return DEGREE_SCORES[d]

    # Substring search (e.g. "Master of Science" contains "master")
    for key, val in sorted(DEGREE_SCORES.items(), key=lambda x: -len(x[0])):
        if key in d:
            return val

    return 0.4  # unknown degree type


def _parse_grade(grade_str: str | None) -> float:
    """
    Parse a grade string into a normalised 0-1 bonus.
    Handles GPA (0-10 or 0-4) and percentage (0-100).
    """
    if not grade_str:
        return 0.0

    # Extract first number from the string
    match = re.search(r"(\d+\.?\d*)", str(grade_str))
    if not match:
        return 0.0

    value = float(match.group(1))

    if value <= 4.0:
        # GPA on 4.0 scale
        return min(value / 4.0, 1.0)
    elif value <= 10.0:
        # GPA on 10.0 scale
        return min(value / 10.0, 1.0)
    elif value <= 100.0:
        # Percentage
        return min(value / 100.0, 1.0)
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Score a single education entry
# ---------------------------------------------------------------------------

def _score_single_education(edu: dict, preferred_fields: list[str]) -> float:
    """Return a weighted score for one education entry."""
    field = _field_match_score(
        edu.get("field_of_study", ""),
        preferred_fields,
    )
    tier = _tier_score(edu.get("institution_tier", edu.get("tier", "")))
    degree = _degree_score(edu.get("degree", ""))
    grade = _parse_grade(edu.get("grade", edu.get("gpa", edu.get("percentage", None))))

    return (
        field  * 0.4
        + tier   * 0.3
        + degree * 0.2
        + grade  * 0.1
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_education(candidate: dict, job_requirements: dict) -> float:
    """
    Returns an education score between 0.0 and 1.0.

    Uses the highest-scoring education entry. If no education entries exist,
    returns 0.3 (don't penalise self-taught engineers too hard).
    """
    preferred_fields: list[str] = job_requirements.get("preferred_fields", [])
    education_entries: list[dict] = candidate.get("education", [])

    if not education_entries:
        return 0.3

    best = max(
        _score_single_education(edu, preferred_fields)
        for edu in education_entries
    )

    return max(0.0, min(best, 1.0))
