"""
Signals Scorer Module
=====================
Scores behavioural signals from the Redrob platform.
These universal signals (response time, interview completion, GitHub activity,
notice period, recency, etc.) differentiate truly available and engaged
candidates from passive or ghost profiles.
"""

from __future__ import annotations

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    """Coerce a value to float, returning *default* on failure or sentinel -1."""
    if value is None:
        return default
    try:
        v = float(value)
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(value, hi))


# ---------------------------------------------------------------------------
# Per-signal scoring functions
# ---------------------------------------------------------------------------

def _recruiter_response_rate(signals: dict) -> float:
    """Direct 0-1 value."""
    return _clamp(_safe_float(signals.get("recruiter_response_rate"), 0.0))


def _avg_response_time(signals: dict) -> float:
    """Inverse normalise: faster is better."""
    hours = _safe_float(signals.get("avg_response_time_hours"), 72.0)
    if hours < 0:
        hours = 72.0  # sentinel
    if hours <= 24:
        return 1.0
    elif hours <= 72:
        return 0.7
    elif hours <= 168:
        return 0.4
    else:
        return 0.1


def _interview_completion(signals: dict) -> float:
    """Direct 0-1 value."""
    return _clamp(_safe_float(signals.get("interview_completion_rate"), 0.0))


def _offer_acceptance(signals: dict) -> float:
    """Direct 0-1. Sentinel -1 → treat as 0.5 (neutral)."""
    raw = signals.get("offer_acceptance_rate")
    if raw is None or (isinstance(raw, (int, float)) and raw < 0):
        return 0.5
    return _clamp(float(raw))


def _github_activity(signals: dict) -> float:
    """Normalise 0-100 → 0-1. Sentinel -1 → 0.3."""
    raw = signals.get("github_activity_score")
    if raw is None or (isinstance(raw, (int, float)) and raw < 0):
        return 0.3
    return _clamp(float(raw) / 100.0)


def _notice_period(signals: dict) -> float:
    """Shorter notice is preferred. Sub-30 days = ideal."""
    days = _safe_float(signals.get("notice_period_days"), 60.0)
    if days < 0:
        days = 60.0
    if days <= 30:
        return 1.0
    elif days <= 60:
        return 0.7
    elif days <= 90:
        return 0.4
    else:
        return 0.2


def _profile_completeness(signals: dict) -> float:
    """Normalise 0-100 → 0-1."""
    return _clamp(_safe_float(signals.get("profile_completeness_score"), 0.0) / 100.0)


def _open_to_work(signals: dict) -> float:
    """Boolean flag."""
    flag = signals.get("open_to_work_flag")
    if flag is True or (isinstance(flag, str) and flag.lower() == "true"):
        return 1.0
    return 0.3


def _saved_by_recruiters(signals: dict) -> float:
    """Normalised recruiter-save count in last 30 days."""
    count = int(_safe_float(signals.get("saved_by_recruiters_30d"), 0))
    if count <= 0:
        return 0.0
    elif count <= 5:
        return 0.5
    elif count <= 15:
        return 0.8
    else:
        return 1.0


def _recency(signals: dict) -> float:
    """
    Score based on last_active_date.
    Within 30 days = 1.0, 30-90 = 0.6, 90-180 = 0.3, >180 = 0.1.
    """
    last_active = signals.get("last_active_date")
    if not last_active:
        return 0.3  # unknown → neutral-ish

    try:
        if isinstance(last_active, str):
            # Try ISO format first, then common formats
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    dt = datetime.strptime(last_active, fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.3
        elif isinstance(last_active, datetime):
            dt = last_active
        else:
            return 0.3

        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta_days = (now - dt).days

        if delta_days <= 30:
            return 1.0
        elif delta_days <= 90:
            return 0.6
        elif delta_days <= 180:
            return 0.3
        else:
            return 0.1
    except Exception:
        return 0.3


# ---------------------------------------------------------------------------
# Verification bonus
# ---------------------------------------------------------------------------

def _verification_multiplier(signals: dict) -> float:
    """
    If email AND phone AND LinkedIn are all verified, award a 1.05× bonus.
    """
    email_ok = signals.get("verified_email", False) is True
    phone_ok = signals.get("verified_phone", False) is True
    linkedin_ok = signals.get("linkedin_connected", False) is True

    if email_ok and phone_ok and linkedin_ok:
        return 1.05
    return 1.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_signals(candidate: dict) -> float:
    """
    Returns a behavioural-signals score between 0.0 and 1.0.

    Components and weights:
        recruiter_response_rate   0.20
        avg_response_time_hours   0.10
        interview_completion_rate 0.15
        offer_acceptance_rate     0.10
        github_activity_score     0.10
        notice_period_days        0.10
        profile_completeness      0.10
        open_to_work_flag         0.05
        saved_by_recruiters_30d   0.05
        recency                   0.05

    Verification bonus: ×1.05 if all three verifications present (capped at 1.0).
    """
    signals: dict = candidate.get("redrob_signals", {})

    score = (
        _recruiter_response_rate(signals) * 0.20
        + _avg_response_time(signals)       * 0.10
        + _interview_completion(signals)    * 0.15
        + _offer_acceptance(signals)        * 0.10
        + _github_activity(signals)         * 0.10
        + _notice_period(signals)           * 0.10
        + _profile_completeness(signals)    * 0.10
        + _open_to_work(signals)            * 0.05
        + _saved_by_recruiters(signals)     * 0.05
        + _recency(signals)                 * 0.05
    )

    score *= _verification_multiplier(signals)

    return _clamp(score, 0.0, 1.0)
