"""Hard Filter.

Quick pre-filter to reduce ~100K candidates to a manageable pool.
The filter is intentionally LOOSE — the scoring stages handle fine-grained
ranking. We only eliminate candidates who are clearly unqualified.
"""

from typing import List


def should_keep(candidate: dict, job_requirements: dict) -> bool:
    """Quick filter. Returns True if the candidate should be KEPT.

    A candidate passes if ALL of the following hold:

    1. **Experience** — within the JD range with a ±3 year tolerance
       (e.g. 2–12 years for a 5–9 range). Missing experience data passes.
    2. **Profile completeness** — ``redrob_signals.profile_completeness_score``
       is at least 30. Missing data passes.
    3. **Skill overlap** — at least one skill from the combined must-have or
       nice-to-have lists matches (case-insensitive partial match).
    4. **Location** — country is India OR ``willing_to_relocate`` is True.

    Args:
        candidate: A candidate profile dictionary.
        job_requirements: Structured job requirements from ``jd_parser``.

    Returns:
        ``True`` if the candidate passes all checks, ``False`` otherwise.
    """

    # ── 1. Experience check (loose: ±3 years tolerance) ────────────────────
    min_exp = job_requirements.get('min_experience_years', 0) - 3.0
    max_exp = job_requirements.get('max_experience_years', 100) + 3.0

    total_exp = candidate.get('total_experience_years')
    if total_exp is None:
        # Also try nested profile
        total_exp = candidate.get('profile', {}).get('total_experience_years')

    if total_exp is not None:
        try:
            total_exp = float(total_exp)
            if total_exp < min_exp or total_exp > max_exp:
                return False
        except (TypeError, ValueError):
            pass  # Cannot parse — let the candidate through

    # ── 2. Profile completeness (>= 30) ───────────────────────────────────
    redrob = candidate.get('redrob_signals', {})
    completeness = redrob.get('profile_completeness_score')
    if completeness is not None:
        try:
            if float(completeness) < 30:
                return False
        except (TypeError, ValueError):
            pass

    # ── 3. At least one relevant skill (case-insensitive partial match) ───
    all_required = set()
    for skill in job_requirements.get('must_have_skills', []):
        all_required.add(skill.lower())
    for skill in job_requirements.get('nice_to_have_skills', []):
        all_required.add(skill.lower())

    candidate_skills: List[str] = []
    for s in candidate.get('skills', []):
        name = s.get('name', '') if isinstance(s, dict) else str(s)
        if name:
            candidate_skills.append(name.lower())

    # Also pull skills from headline/summary for broader matching
    profile = candidate.get('profile', {})
    headline = (profile.get('headline', '') or '').lower()
    summary = (profile.get('summary', '') or '').lower()

    has_skill_match = False
    for req_skill in all_required:
        # Check explicit skill list
        for cand_skill in candidate_skills:
            if req_skill in cand_skill or cand_skill in req_skill:
                has_skill_match = True
                break
        if has_skill_match:
            break
        # Check headline / summary
        if req_skill in headline or req_skill in summary:
            has_skill_match = True
            break

    if not has_skill_match:
        return False

    # ── 4. Location — India OR willing to relocate ─────────────────────────
    location = candidate.get('location', {})
    if isinstance(location, dict):
        country = (location.get('country', '') or '').strip().lower()
    else:
        country = ''

    willing = candidate.get('willing_to_relocate', False)
    if not willing:
        willing = candidate.get('profile', {}).get('willing_to_relocate', False)

    if country and country != 'india' and not willing:
        return False

    return True


def apply_hard_filter(candidates: list, job_requirements: dict) -> list:
    """Filter a list of candidates using :func:`should_keep`.

    Args:
        candidates: List of candidate profile dictionaries.
        job_requirements: Structured job requirements from ``jd_parser``.

    Returns:
        A list containing only the candidates that passed the hard filter.
    """
    return [c for c in candidates if should_keep(c, job_requirements)]
