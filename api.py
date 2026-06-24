from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import csv
import json
import os

app = FastAPI(title="AI Candidate Ranker API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store our top 100 candidates
TOP_CANDIDATES = []
CANDIDATE_MAP = {}

def load_data():
    """Load the Top 100 candidates from submission.csv and cross-reference with candidates.jsonl"""
    global TOP_CANDIDATES, CANDIDATE_MAP
    
    base_dir = os.path.dirname(__file__)
    submission_path = os.path.join(base_dir, 'submission.csv')
    candidates_path = os.path.join(base_dir, 'data', 'candidates.jsonl')
    
    if not os.path.exists(submission_path):
        print("Warning: submission.csv not found. Have you run the ranking pipeline?")
        return
        
    # 1. Read ranked IDs and scores
    ranked_info = {}
    with open(submission_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranked_info[row['candidate_id']] = {
                'rank': int(row['rank']),
                'score': float(row['score']),
                'reasoning': row['reasoning']
            }
            
    # 2. Extract full profiles for only those 100 IDs from the 100K JSONL
    # (Doing this in one pass is very fast)
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand.get('candidate_id')
            if cid in ranked_info:
                # Merge the rank info into the candidate object
                cand.update(ranked_info[cid])
                CANDIDATE_MAP[cid] = cand
                
    # 3. Sort by rank
    TOP_CANDIDATES = sorted(CANDIDATE_MAP.values(), key=lambda x: x['rank'])
    print(f"Loaded {len(TOP_CANDIDATES)} top candidates for the API.")

# Load data on startup
load_data()

@app.get("/api/candidates")
def get_candidates():
    """Returns the list of top candidates with their core info"""
    if not TOP_CANDIDATES:
        raise HTTPException(status_code=404, detail="No candidates found. Ensure submission.csv exists.")
    return TOP_CANDIDATES

@app.get("/api/candidates/{candidate_id}")
def get_candidate_detail(candidate_id: str):
    """Returns deep details for a specific candidate"""
    if candidate_id not in CANDIDATE_MAP:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return CANDIDATE_MAP[candidate_id]

from collections import Counter, defaultdict

@app.get("/api/analytics")
def get_analytics():
    """Returns macro analytics for the Top 100 talent pool"""
    skill_counts = defaultdict(int)
    location_counts = defaultdict(int)
    company_counts = defaultdict(int)
    
    for c in list(CANDIDATE_MAP.values())[:100]:
        for s in c.get('skills', []):
            if s.get('proficiency') in ['expert', 'advanced']:
                skill_counts[s.get('name')] += 1
                
        loc = c.get('profile', {}).get('location', 'Remote')
        if not loc: loc = 'Remote'
        location_counts[loc] += 1
        
        for job in c.get('career_history', []):
            comp = job.get('company')
            if comp: company_counts[comp] += 1

    top_skills = [{"name": k, "count": v} for k, v in sorted(skill_counts.items(), key=lambda item: item[1], reverse=True)[:15]]
    top_locations = [{"name": k, "count": v} for k, v in sorted(location_counts.items(), key=lambda item: item[1], reverse=True)[:10]]
    top_companies = [{"name": k, "count": v} for k, v in sorted(company_counts.items(), key=lambda item: item[1], reverse=True)[:10]]
    
    # ROI Calculation
    # Assuming 100,000 resumes processed
    total_resumes = 100000
    minutes_per_resume = 2
    total_hours = (total_resumes * minutes_per_resume) / 60
    cost_per_hour = 50
    money_saved = total_hours * cost_per_hour
    
    roi = {
        "resumes_processed": total_resumes,
        "hours_saved": int(total_hours),
        "money_saved": int(money_saved),
        "time_to_run_ai_seconds": 315 # approx 5 mins
    }
    
    return {
        "top_skills": top_skills,
        "top_locations": top_locations,
        "top_companies": top_companies,
        "roi": roi,
        "total_candidates": len(TOP_CANDIDATES)
    }

@app.get("/api/candidates/{candidate_id}/email")
def generate_email_draft(candidate_id: str):
    """Generates a personalized outreach email based on candidate data"""
    if candidate_id not in CANDIDATE_MAP:
        raise HTTPException(status_code=404, detail="Candidate not found")
        
    c = CANDIDATE_MAP[candidate_id]
    profile = c.get('profile', {})
    history = c.get('career_history', [])
    skills = c.get('skills', [])
    
    recent_job = history[0] if history else {}
    company = recent_job.get('company', 'your current company')
    top_skill = skills[0].get('name', 'AI') if skills else 'AI'
    
    email_body = f"""Subject: Founding AI Engineer Role @ Redrob AI - Impressed by your work at {company}

Hi {profile.get('name', 'there')},

My name is Naveen from the founding team at Redrob AI. We are building the next generation of AI-powered recruitment intelligence. 

I was looking at your background and was incredibly impressed by your tenure at {company}. We are specifically looking for a Senior AI Engineer who has deep expertise in {top_skill}, and your trajectory perfectly aligns with the intelligence layer we are building from the ground up.

Would you be open to a quick 15-minute chat this week to discuss a Founding Engineer role with us?

Best,
Naveen
Redrob AI
"""
    return {"email": email_body}

from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

@app.post("/api/candidates/{candidate_id}/chat")
def ask_ai_chat(candidate_id: str, request: ChatRequest):
    """Simulates a RAG chatbot over the candidate's profile"""
    if candidate_id not in CANDIDATE_MAP:
        raise HTTPException(status_code=404, detail="Candidate not found")
        
    c = CANDIDATE_MAP[candidate_id]
    q = request.question.lower()
    
    # Simple simulated RAG logic
    answer = ""
    
    if "skill" in q or "know" in q or "experience with" in q:
        target_skill = q.replace("does he know", "").replace("do they know", "").replace("experience with", "").replace("skills in", "").strip().replace("?", "")
        found = False
        for s in c.get('skills', []):
            if target_skill in s.get('name', '').lower() or s.get('name', '').lower() in target_skill:
                answer = f"Yes, they have experience with {s.get('name')}. Their proficiency is marked as '{s.get('proficiency')}' with {s.get('endorsements')} endorsements."
                found = True
                break
        if not found:
            answer = f"I could not find explicit mention of '{target_skill}' in their top skills. Their primary skills are " + ", ".join([s.get('name') for s in c.get('skills', [])[:3]]) + "."
            
    elif "work" in q or "company" in q or "where" in q:
        history = c.get('career_history', [])
        if history:
            answer = f"They most recently worked at {history[0].get('company')} as a {history[0].get('title')}."
        else:
            answer = "I do not have career history data for this candidate."
            
    elif "education" in q or "degree" in q or "university" in q or "college" in q:
        edu = c.get('education', [])
        if edu:
            answer = f"They studied at {edu[0].get('institution')} and earned a {edu[0].get('degree')} in {edu[0].get('field_of_study')}."
        else:
            answer = "I do not have education data for this candidate."
            
    else:
        answer = f"Based on their profile, they have {c.get('profile', {}).get('years_of_experience', 0)} years of experience and are currently a {c.get('profile', {}).get('current_title', 'professional')}. How else can I help?"
        
    import time
    time.sleep(0.5) # Simulate AI thinking time
    
    return {"answer": answer}

@app.get("/api/candidates/{candidate_id}/interview-questions")
def generate_interview_questions(candidate_id: str):
    """Generates targeted technical interview questions based on candidate profile"""
    if candidate_id not in CANDIDATE_MAP:
        raise HTTPException(status_code=404, detail="Candidate not found")
        
    c = CANDIDATE_MAP[candidate_id]
    skills = [s.get('name', '').lower() for s in c.get('skills', [])]
    history = c.get('career_history', [])
    
    must_haves = ['sentence-transformers', 'embeddings', 'semantic search', 'vector database', 'pinecone', 'python', 'ranking', 'nlp', 'pytorch']
    
    missing = [m for m in must_haves if not any(m in s for s in skills)]
    strengths = [s.get('name') for s in c.get('skills', []) if s.get('proficiency') in ['expert', 'advanced']]
    
    questions = []
    
    # Q1: Based on a strength
    if strengths:
        q1 = f"I see you have deep expertise in {strengths[0]}. Can you walk me through the most complex architecture you've built using it?"
    else:
        q1 = "Can you walk me through the architecture of the most complex machine learning pipeline you've built?"
    questions.append(q1)
    
    # Q2: Based on a missing skill (Gap probe)
    if missing:
        q2 = f"This role requires heavy usage of {missing[0]}. While it's not explicitly on your profile, how would you approach learning and integrating it into an existing stack?"
    else:
        q2 = "You meet all our core requirements perfectly. How do you stay updated with the latest state-of-the-art developments in AI?"
    questions.append(q2)
    
    # Q3: Based on career history
    if history:
        q3 = f"During your time at {history[0].get('company')}, what was the biggest technical bottleneck you faced when deploying models, and how did you resolve it?"
    else:
        q3 = "What is the biggest technical bottleneck you've faced when deploying machine learning models to production?"
    questions.append(q3)
        
    return {"questions": questions}

if __name__ == "__main__":
    import uvicorn
    # Start server
    print("Starting API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
