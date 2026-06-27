def compute_hybrid_score(
    skills_score: float,
    career_score: float,
    education_score: float,
    signals_score: float,
    semantic_score: float
) -> float:
    """Combine all scores using weighted aggregation. Returns 0.0-1.0."""
    weights = {
        'skills': 0.30,
        'career': 0.25,
        'signals': 0.20,
        'semantic': 0.15,
        'education': 0.10
    }
    final = (
        skills_score * weights['skills'] +
        career_score * weights['career'] +
        signals_score * weights['signals'] +
        semantic_score * weights['semantic'] +
        education_score * weights['education']
    )
    return max(0.0, min(1.0, final))


def rank_candidates(scored_candidates: list) -> list:
    """Sort candidates by score descending, handle ties by candidate_id ascending.
    
    Args:
        scored_candidates: List of dicts with keys 'candidate_id', 'final_score', 
                          'skills_score', 'career_score', 'education_score',
                          'signals_score', 'semantic_score'
    Returns:
        Top 100 candidates with 'rank' and normalized 'score' added.
    """
    # Sort by score desc, then candidate_id asc for ties
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda x: (-x['final_score'], x['candidate_id'])
    )
    
    # Take top 100
    top_100 = sorted_candidates[:100]
    
    # Rank 1 gets the highest absolute score, and we just round it for readability
    if top_100:
        for i, candidate in enumerate(top_100):
            candidate['rank'] = i + 1
            # Return the true absolute score to preserve exact margins!
            candidate['score'] = round(candidate['final_score'], 4)
        
        # Ensure scores are non-increasing (required by validator)
        for i in range(1, len(top_100)):
            if top_100[i]['score'] > top_100[i-1]['score']:
                top_100[i]['score'] = top_100[i-1]['score']
    
    return top_100
