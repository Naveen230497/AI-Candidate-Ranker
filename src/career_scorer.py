"""
Career Scorer Module
====================
Scores career alignment for the Senior AI Engineer role.
Detects title-chasers, consulting-only careers, and pure researchers
that the JD explicitly disqualifies.
"""

from __future__ import annotations

from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Titles that are completely unrelated — score ≈ 0
# ---------------------------------------------------------------------------
DISQUALIFYING_TITLES: list[str] = [
    "marketing manager", "graphic designer", "accountant",
    "civil engineer", "hr manager", "sales executive",
    "customer support", "content writer", "mechanical engineer",
]

# ---------------------------------------------------------------------------
# Equivalent-title clusters for fuzzy matching
# ---------------------------------------------------------------------------
TITLE_EQUIVALENTS: dict[str, list[str]] = {
    "ai_engineer": [
        "ai engineer", "artificial intelligence engineer",
        "senior ai engineer", "staff ai engineer",
        "lead ai engineer", "principal ai engineer",
    ],
    "ml_engineer": [
        "ml engineer", "machine learning engineer",
        "senior ml engineer", "staff ml engineer",
        "lead ml engineer", "principal ml engineer",
        "applied ml engineer",
    ],
    "data_scientist": [
        "data scientist", "senior data scientist",
        "lead data scientist", "staff data scientist",
        "applied scientist",
    ],
    "research_engineer": [
        "research engineer", "ml research engineer",
        "ai research engineer", "research scientist",
    ],
    "nlp_engineer": [
        "nlp engineer", "natural language processing engineer",
        "computational linguist",
    ],
    "search_engineer": [
        "search engineer", "ranking engineer",
        "information retrieval engineer", "relevance engineer",
    ],
    "backend_engineer": [
        "backend engineer", "software engineer",
        "senior software engineer", "staff software engineer",
        "platform engineer",
    ],
}

# Seniority progression (lower index = more junior)
SENIORITY_LADDER: list[str] = [
    "intern", "trainee", "associate", "junior",
    "mid", "",  # empty = no prefix = mid-level
    "senior", "staff", "lead", "principal", "director", "vp",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return text.strip().lower()


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _best_title_score(title: str, target_keywords: list[str]) -> float:
    """
    Return the best match score (0-1) of *title* against target keywords
    using fuzzy matching AND the equivalence clusters.
    """
    t = _norm(title)

    # Hard disqualification
    for dq in DISQUALIFYING_TITLES:
        if _fuzzy_ratio(t, dq) > 0.80:
            return 0.05

    best = 0.0

    # Direct fuzzy match against supplied keywords
    for kw in target_keywords:
        best = max(best, _fuzzy_ratio(t, _norm(kw)))

    # Equivalence cluster match
    for _cluster, variants in TITLE_EQUIVALENTS.items():
        for v in variants:
            ratio = _fuzzy_ratio(t, v)
            if ratio > best:
                best = ratio

    return min(best, 1.0)


def _seniority_index(title: str) -> int:
    """Extract a seniority index from a title string."""
    t = _norm(title)
    for idx, level in enumerate(SENIORITY_LADDER):
        if level and level in t:
            return idx
    return 5  # default to mid-level


def _is_consulting_firm(company: str, consulting_firms: list[str]) -> bool:
    c = _norm(company)
    return any(_norm(cf) in c or c in _norm(cf) for cf in consulting_firms)


# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------

def _title_alignment(candidate: dict, job_requirements: dict) -> float:
    """Weight 0.35 — current title matters most, past titles contribute."""
    target_kws: list[str] = job_requirements.get("title_keywords", [])
    experiences = candidate.get("experience", [])
    if not experiences:
        return 0.1

    # Current role = first entry (most recent)
    current = experiences[0]
    current_score = _best_title_score(current.get("title", ""), target_kws)

    # Past roles average
    past_scores = [
        _best_title_score(exp.get("title", ""), target_kws)
        for exp in experiences[1:]
    ]
    past_avg = sum(past_scores) / len(past_scores) if past_scores else 0.0

    # Current title = 70 %, past = 30 %
    return current_score * 0.7 + past_avg * 0.3


def _career_progression(candidate: dict) -> float:
    """Weight 0.20 — titles should show growth, healthy tenures."""
    experiences = candidate.get("experience", [])
    if len(experiences) < 2:
        return 0.5  # not enough data

    # Check seniority growth
    indices = [_seniority_index(exp.get("title", "")) for exp in experiences]
    # Reverse so chronological (oldest first)
    indices_chrono = list(reversed(indices))
    growth_steps = sum(
        1 for i in range(1, len(indices_chrono))
        if indices_chrono[i] >= indices_chrono[i - 1]
    )
    progression_ratio = growth_steps / max(len(indices_chrono) - 1, 1)

    # Duration health
    durations = [exp.get("duration_months", 24) for exp in experiences]
    healthy = sum(1 for d in durations if 18 <= d <= 48)
    duration_score = healthy / max(len(durations), 1)

    # Average duration for job-hopper check
    avg_dur = sum(durations) / max(len(durations), 1)
    hopper_penalty = 1.0
    if avg_dur < 12:
        hopper_penalty = 0.5

    return min((progression_ratio * 0.6 + duration_score * 0.4) * hopper_penalty, 1.0)


def _industry_relevance(candidate: dict, job_requirements: dict) -> float:
    """Weight 0.20 — tech/AI industries score higher."""
    preferred: list[str] = job_requirements.get("preferred_industries", [])
    experiences = candidate.get("experience", [])
    if not experiences:
        return 0.2

    matched = 0
    for exp in experiences:
        industry = exp.get("industry", exp.get("company_industry", ""))
        if not industry:
            continue
        i_low = _norm(industry)
        for pref in preferred:
            if _norm(pref) in i_low or i_low in _norm(pref):
                matched += 1
                break

    return min(matched / max(len(experiences), 1), 1.0)


def _company_quality(candidate: dict, job_requirements: dict) -> float:
    """
    Weight 0.15 — product companies beat consulting.
    Pure consulting career → 0.2 multiplier.
    """
    consulting_firms: list[str] = job_requirements.get("consulting_firms", [])
    experiences = candidate.get("experience", [])
    if not experiences:
        return 0.3

    consulting_count = 0
    scores: list[float] = []

    for exp in experiences:
        company = exp.get("company", exp.get("company_name", ""))
        size = exp.get("company_size", 0)

        if _is_consulting_firm(company, consulting_firms):
            consulting_count += 1
            scores.append(0.3)
        else:
            # Rough heuristic: larger product companies score higher
            if isinstance(size, (int, float)) and size > 0:
                if size >= 5000:
                    scores.append(1.0)
                elif size >= 1000:
                    scores.append(0.85)
                elif size >= 200:
                    scores.append(0.7)
                else:
                    scores.append(0.55)
            else:
                scores.append(0.5)

    base = sum(scores) / max(len(scores), 1)

    # Pure consulting penalty
    if consulting_count == len(experiences) and len(experiences) > 0:
        base *= 0.2

    return min(base, 1.0)


def _tenure_stability(candidate: dict) -> float:
    """
    Weight 0.10 — average tenure.
    2-4 years ideal, <1 year = hopper, >6 years in one role = slight ding.
    """
    experiences = candidate.get("experience", [])
    if not experiences:
        return 0.4

    durations = [exp.get("duration_months", 24) for exp in experiences]
    avg_months = sum(durations) / max(len(durations), 1)
    avg_years = avg_months / 12.0

    if 2.0 <= avg_years <= 4.0:
        score = 1.0
    elif avg_years < 1.0:
        score = 0.3  # job hopper
    elif avg_years < 2.0:
        score = 0.6
    elif avg_years <= 6.0:
        score = 0.8
    else:
        score = 0.65  # stagnant

    # Extra penalty if ANY single role > 6 years
    if any(d > 72 for d in durations):
        score *= 0.9

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_career(candidate: dict, job_requirements: dict) -> float:
    """
    Returns a composite career-alignment score between 0.0 and 1.0.

    Weights:
        title_alignment   0.35
        career_progression 0.20
        industry_relevance 0.20
        company_quality    0.15
        tenure_stability   0.10
    """
    ta = _title_alignment(candidate, job_requirements)
    cp = _career_progression(candidate)
    ir = _industry_relevance(candidate, job_requirements)
    cq = _company_quality(candidate, job_requirements)
    ts = _tenure_stability(candidate)

    score = (
        ta * 0.35
        + cp * 0.20
        + ir * 0.20
        + cq * 0.15
        + ts * 0.10
    )

    return max(0.0, min(score, 1.0))
