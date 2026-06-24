def score_location(candidate: dict, job_requirements: dict) -> float:
    """Score location fit. Returns 0.0-1.0.
    
    Location scoring:
    - Pune/Noida: 1.0 (exact match)
    - Other preferred cities (Hyderabad, Mumbai, Delhi, Bangalore): 0.85
    - Other India locations: 0.7
    - Outside India but willing to relocate: 0.5
    - Outside India, not willing to relocate: 0.2
    """
    profile = candidate.get('profile', {})
    location = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    signals = candidate.get('redrob_signals', {})
    willing_to_relocate = signals.get('willing_to_relocate', False)
    
    preferred_locations = [loc.lower() for loc in job_requirements.get('preferred_locations', [])]
    
    # Check for Pune/Noida (top priority)
    top_cities = ['pune', 'noida']
    if any(city in location for city in top_cities):
        return 1.0
    
    # Check other preferred Indian cities
    other_preferred = ['hyderabad', 'mumbai', 'delhi', 'ncr', 'bengaluru', 'bangalore', 'gurgaon', 'gurugram']
    if any(city in location for city in other_preferred):
        return 0.85
    
    # Check if in India
    if 'india' in country or any(loc in location for loc in preferred_locations):
        return 0.7
    
    # Outside India
    if willing_to_relocate:
        return 0.5
    
    return 0.2
