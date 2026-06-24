"""
Skills Scorer Module
====================
Scores how well a candidate's skills match the job requirements.
Includes honeypot detection for keyword-stuffed profiles with inflated skills
but zero endorsements.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical skill groups straight from the JD
# ---------------------------------------------------------------------------
MUST_HAVE_GROUPS: dict[str, list[str]] = {
    "embeddings_retrieval": [
        "sentence-transformers", "OpenAI embeddings", "BGE", "E5",
        "embeddings", "retrieval", "semantic search",
    ],
    "vector_databases": [
        "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
        "Elasticsearch", "FAISS", "vector database", "vector search",
    ],
    "python": ["Python"],
    "ranking_evaluation": [
        "NDCG", "MRR", "MAP", "A/B testing", "ranking",
        "evaluation", "information retrieval",
    ],
    "nlp_ml": [
        "NLP", "natural language processing", "machine learning",
        "deep learning", "PyTorch", "TensorFlow",
    ],
}

NICE_TO_HAVE: list[str] = [
    "LoRA", "QLoRA", "PEFT", "fine-tuning", "Fine-tuning LLMs",
    "XGBoost", "learning-to-rank", "LTR",
    "distributed systems", "inference optimization",
    "Spark", "Airflow", "data engineering",
    "Kubernetes", "Docker", "MLOps",
    "RAG", "LangChain", "LLM",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Lower-case, strip whitespace for comparison."""
    return name.strip().lower()


def _skill_names_set(candidate: dict) -> set[str]:
    """Return the set of normalised skill names a candidate lists."""
    return {_normalize(s.get("name", "")) for s in candidate.get("skills", [])}


def _skill_lookup(candidate: dict) -> dict[str, dict]:
    """Map normalised skill name → skill dict for quick access."""
    return {
        _normalize(s.get("name", "")): s
        for s in candidate.get("skills", [])
    }


def _matches_any(needle: str, haystack: list[str]) -> bool:
    """Check if *needle* matches any term in *haystack* (case-insensitive, substring)."""
    n = _normalize(needle)
    for h in haystack:
        h_low = _normalize(h)
        if n == h_low or n in h_low or h_low in n:
            return True
    return False


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

def _is_honeypot(candidate: dict) -> bool:
    """
    Detect keyword-stuffed profiles.
    Rule: 8+ skills at 'expert' or 'advanced' but total endorsements < 5.
    """
    skills = candidate.get("skills", [])
    high_prof_count = sum(
        1 for s in skills
        if _normalize(s.get("proficiency", "")) in {"expert", "advanced"}
    )
    total_endorsements = sum(s.get("endorsements", 0) for s in skills)
    return high_prof_count >= 8 and total_endorsements < 5


HONEYPOT_PENALTY = 0.3


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

def _exact_match_ratio(candidate: dict, job_requirements: dict) -> float:
    """
    Fraction of required skills (must-have + nice-to-have) the candidate
    actually lists.  Must-haves are counted per *group* (hitting any keyword
    in a group counts the group as matched), so a candidate doesn't need
    every synonym.
    """
    must_have_list: list[str] = job_requirements.get("must_have_skills", [])
    nice_to_have_list: list[str] = job_requirements.get("nice_to_have_skills", [])
    cand_names = _skill_names_set(candidate)

    # --- Must-have groups ---------------------------------------------------
    groups_matched = 0
    total_groups = len(MUST_HAVE_GROUPS)
    for _group_name, keywords in MUST_HAVE_GROUPS.items():
        for kw in keywords:
            if any(_matches_any(kw, [cn]) for cn in cand_names):
                groups_matched += 1
                break
    # Also allow raw strings from job_requirements to count
    extra_must = [
        m for m in must_have_list
        if not any(_matches_any(m, kws) for kws in MUST_HAVE_GROUPS.values())
    ]
    extra_matched = sum(
        1 for m in extra_must
        if any(_matches_any(m, [cn]) for cn in cand_names)
    )
    must_total = total_groups + len(extra_must)
    must_matched = groups_matched + extra_matched

    # --- Nice-to-have -------------------------------------------------------
    nice_matched = sum(
        1 for nh in nice_to_have_list
        if any(_matches_any(nh, [cn]) for cn in cand_names)
    )
    nice_total = max(len(nice_to_have_list), 1)

    # Must-haves worth 70 %, nice-to-haves 30 %
    ratio = 0.7 * (must_matched / max(must_total, 1)) + 0.3 * (nice_matched / nice_total)
    return min(ratio, 1.0)


def _proficiency_weighted(candidate: dict, job_requirements: dict) -> float:
    """
    Average proficiency level of matched skills, normalised to 0-1.
    Credibility: duration_months > 24 → ×1.1
    """
    prof_map = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.6, "beginner": 0.3}
    all_required = set(
        _normalize(s) for s in
        job_requirements.get("must_have_skills", []) + job_requirements.get("nice_to_have_skills", [])
    )
    # Also flatten canonical groups
    for kws in MUST_HAVE_GROUPS.values():
        all_required.update(_normalize(k) for k in kws)
    for nh in NICE_TO_HAVE:
        all_required.add(_normalize(nh))

    lookup = _skill_lookup(candidate)
    scores: list[float] = []
    for req in all_required:
        if req in lookup:
            skill = lookup[req]
            base = prof_map.get(_normalize(skill.get("proficiency", "")), 0.4)
            if skill.get("duration_months", 0) > 24:
                base *= 1.1
            scores.append(min(base, 1.0))

    return sum(scores) / max(len(all_required), 1)


def _assessment_verified(candidate: dict) -> float:
    """
    Use Redrob assessment scores as verified ground truth.
    Returns average of available assessment scores normalised to 0-1.
    """
    signals = candidate.get("redrob_signals", {})
    assessments: dict = signals.get("skill_assessment_scores", {})
    if not assessments:
        return 0.0
    values = [v for v in assessments.values() if isinstance(v, (int, float)) and v >= 0]
    if not values:
        return 0.0
    # Assume scores are 0-100
    avg = sum(values) / len(values)
    return min(avg / 100.0, 1.0)


def _endorsement_trust(candidate: dict) -> float:
    """
    Aggregate endorsement-based trust.
    Endorsements > 10 → trust multiplier 1.2
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    total_end = sum(s.get("endorsements", 0) for s in skills)
    if total_end == 0:
        return 0.0

    # Normalise: assume 50 total endorsements is "great"
    base = min(total_end / 50.0, 1.0)
    if total_end > 10:
        base *= 1.2
    return min(base, 1.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_skills(candidate: dict, job_requirements: dict) -> float:
    """
    Returns a composite skills score between 0.0 and 1.0.

    Formula:
        score = exact_match * 0.4
              + proficiency  * 0.3
              + assessment   * 0.2
              + endorsement  * 0.1

    A honeypot penalty (×0.3) is applied when the candidate lists 8+ expert/
    advanced skills but has fewer than 5 total endorsements.
    """
    em = _exact_match_ratio(candidate, job_requirements)
    pw = _proficiency_weighted(candidate, job_requirements)
    av = _assessment_verified(candidate)
    et = _endorsement_trust(candidate)

    score = em * 0.4 + pw * 0.3 + av * 0.2 + et * 0.1

    if _is_honeypot(candidate):
        score *= HONEYPOT_PENALTY

    return max(0.0, min(score, 1.0))
