def generate_reasoning(candidate: dict, scores: dict) -> str:
    """Generate a one-line reasoning string for a ranked candidate.
    
    Args:
        candidate: The full candidate dict
        scores: Dict with 'skills_score', 'career_score', 'education_score',
                'signals_score', 'semantic_score', 'final_score'
    Returns:
        A concise reasoning string
    """
    profile = candidate.get('profile', {})
    current_title = profile.get('current_title', 'Unknown')
    yoe = profile.get('years_of_experience', 0)
    
    # Count matching skills
    skills = candidate.get('skills', [])
    ai_skills = [s for s in skills if s.get('proficiency') in ('advanced', 'expert')]
    num_strong_skills = len(ai_skills)
    
    # Get response rate
    signals = candidate.get('redrob_signals', {})
    response_rate = signals.get('recruiter_response_rate', 0)
    
    # Get top scoring dimension
    score_names = {
        'skills_score': 'skill match',
        'career_score': 'career fit', 
        'education_score': 'education',
        'signals_score': 'engagement signals',
        'semantic_score': 'semantic alignment'
    }
    top_dimension = max(
        [(k, v) for k, v in scores.items() if k in score_names],
        key=lambda x: x[1]
    )
    
    # Build reasoning
    parts = [
        f"{current_title} with {yoe:.1f} yrs",
        f"{num_strong_skills} advanced/expert skills",
        f"response rate {response_rate:.2f}",
        f"strongest: {score_names.get(top_dimension[0], 'overall')}"
    ]
    
    reasoning = '; '.join(parts)
    
    # Truncate to fit CSV nicely (max ~200 chars)
    if len(reasoning) > 200:
        reasoning = reasoning[:197] + '...'
    
    return reasoning
