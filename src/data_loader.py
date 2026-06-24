"""Data Loader.

Loads candidates from JSONL files efficiently using streaming and batching.
Also provides utilities for building candidate text blobs for semantic matching.
"""

import json
from typing import Generator, List


def load_candidates_stream(jsonl_path: str) -> Generator[dict, None, None]:
    """Stream candidates one at a time from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file containing candidate records.

    Yields:
        A dictionary representing a single candidate profile.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates_batch(
    jsonl_path: str, batch_size: int = 1000
) -> Generator[List[dict], None, None]:
    """Stream candidates in batches from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file containing candidate records.
        batch_size: Number of candidates per batch. Defaults to 1000.

    Yields:
        A list of candidate dictionaries (one batch at a time).
    """
    batch: List[dict] = []
    for candidate in load_candidates_stream(jsonl_path):
        batch.append(candidate)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_candidates(jsonl_path: str) -> int:
    """Count total candidates in the JSONL file.

    Args:
        jsonl_path: Path to the JSONL file containing candidate records.

    Returns:
        The number of non-empty lines (i.e. candidate records) in the file.
    """
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def build_candidate_text(candidate: dict) -> str:
    """Build a single text blob from a candidate's profile for semantic matching.

    Uses Information Distillation: Prioritizes the current job, top endorsed skills,
    and limits the output to ~250 tokens (approx 1000 characters) to prevent
    silent truncation by sentence-transformer models.

    Args:
        candidate: A dictionary representing a candidate profile.

    Returns:
        A highly concentrated, space-joined string of profile text.
    """
    parts: List[str] = []
    profile = candidate.get('profile', {})
    
    # 1. Current Title is highest priority
    title = profile.get('current_title', '')
    if title:
        parts.append(f"Title: {title}.")

    # 2. Most recent/relevant job (max 1 to save space)
    history = candidate.get('career_history', [])
    if history:
        latest_job = history[0]
        desc = latest_job.get('description', '')
        # Truncate description to ~300 chars
        if len(desc) > 300:
            desc = desc[:297] + '...'
        parts.append(f"Experience: {latest_job.get('title', '')} at {latest_job.get('company', '')}. {desc}")

    # 3. Top Skills (Sort by endorsements, take top 15)
    skills = candidate.get('skills', [])
    if skills:
        sorted_skills = sorted(skills, key=lambda x: x.get('endorsements', 0), reverse=True)
        top_skills = [s.get('name', '') for s in sorted_skills[:15]]
        parts.append('Skills: ' + ', '.join(top_skills) + '.')

    # 4. Education (Degree and Field)
    edu = candidate.get('education', [])
    if edu:
        top_edu = edu[0]
        parts.append(f"Education: {top_edu.get('degree', '')} in {top_edu.get('field_of_study', '')}.")

    # 5. Add summary ONLY if we have space left
    summary = profile.get('summary', '')
    if summary:
        parts.append(f"Summary: {summary[:200]}")

    # Join and strictly limit to 1200 characters (approx 250 tokens for MiniLM)
    full_text = ' '.join(p for p in parts if p)
    return full_text[:1200]


def load_sample_candidates(json_path: str) -> List[dict]:
    """Load sample candidates from a JSON array file (for testing).

    Args:
        json_path: Path to a JSON file containing a list of candidate dicts.

    Returns:
        A list of candidate dictionaries.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
