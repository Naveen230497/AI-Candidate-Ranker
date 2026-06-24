# 🤖 AI Candidate Ranker — Intelligent Candidate Discovery & Ranking

> **INDIA RUNS Data & AI Challenge** — Redrob / Hack2Skill Hackathon 2026

An AI-powered system that ranks 100,000 candidates for a Senior AI Engineer role — not by keyword matching, but by truly **understanding** who fits the role through semantic analysis, career trajectory evaluation, and behavioral signal intelligence.

## 🏗️ Architecture

```
Job Description ──→ JD Parser ──→ Structured Requirements
                                        │
100K Candidates ──→ Data Loader ──→ Hard Filter ──→ Multi-Dimensional Scoring
                                                          │
                                                    ┌─────┴─────┐
                                                    │  5 Pillars │
                                                    ├────────────┤
                                                    │ Skills     │ 30%
                                                    │ Career     │ 25%
                                                    │ Signals    │ 20%
                                                    │ Semantic   │ 15%
                                                    │ Education  │ 10%
                                                    └─────┬──────┘
                                                          │
                                                    Hybrid Ranker
                                                          │
                                                    Top 100 + Reasoning
                                                          │
                                                    submission.csv
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- ~2GB RAM for sentence-transformers model

### Installation
```bash
pip install -r requirements.txt
```

### Run (Full Dataset — 100K candidates)
```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

### Run (Sample — for testing)
```bash
python rank.py --candidates ./data/sample_candidates.json --out ./test_submission.csv --sample
```

### Validate Submission
```bash
python validate_submission.py submission.csv
```

## 🧠 Approach

### Why Not Keyword Matching?
The JD explicitly warns: *"The 'right answer' is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."*

A candidate listing 15 AI buzzwords but working as a "Marketing Manager" is NOT a fit. Our system catches these honeypots.

### The 5 Scoring Pillars

1. **Skills Scorer (30%)** — Matches skills against JD requirements with honeypot detection. Candidates with inflated skills (expert proficiency, 0 endorsements, no assessments) get penalized.

2. **Career Scorer (25%)** — Evaluates title alignment, career progression, industry relevance, and company quality. Detects disqualifiers: title-chasers, consulting-only careers, non-technical backgrounds.

3. **Behavioral Signals (20%)** — Uses Redrob platform signals: recruiter response rate, interview completion rate, GitHub activity, notice period, and recency. A perfect-on-paper candidate who hasn't logged in for 6 months is not actually available.

4. **Semantic Similarity (15%)** — Uses `all-MiniLM-L6-v2` sentence embeddings to compute cosine similarity between the JD and each candidate's career narrative. Catches candidates who describe relevant work without using exact keywords.

5. **Education Scorer (10%)** — Field of study match, institution tier, degree level. Intentionally low weight because great engineers can be self-taught.

### Honeypot Detection
- Candidates with 8+ "expert/advanced" skills but <5 total endorsements → keyword stuffer penalty
- Skill assessment scores from Redrob platform used as verified ground truth
- Title vs. skills mismatch detection (Marketing Manager with ML skills = trap)

## 📁 Project Structure

```
ai-candidate-ranker/
├── rank.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── submission_metadata.yaml    # Hackathon metadata
├── validate_submission.py      # Official submission validator
├── data/
│   ├── candidates.jsonl        # Full 100K dataset (487MB)
│   ├── sample_candidates.json  # Sample for testing
│   ├── job_description.docx    # The job description
│   └── candidate_schema.json   # Schema reference
└── src/
    ├── __init__.py
    ├── jd_parser.py            # Job description parser
    ├── data_loader.py          # JSONL streaming loader
    ├── hard_filter.py          # Quick pre-filter
    ├── skills_scorer.py        # Skills matching + honeypot detection
    ├── career_scorer.py        # Career trajectory scorer
    ├── education_scorer.py     # Education fit scorer
    ├── signals_scorer.py       # Behavioral signals scorer
    ├── semantic_scorer.py      # Sentence embedding similarity
    ├── location_scorer.py      # Geographic fit scorer
    ├── hybrid_ranker.py        # Weighted score combiner
    └── reasoning_generator.py  # Human-readable explanations
```

## ⚙️ Compute Constraints

- **No GPU required** — runs entirely on CPU
- **No network calls** during ranking
- **Runtime:** ~5 minutes for 100K candidates on 16GB RAM
- **Model:** `all-MiniLM-L6-v2` (80MB, runs on CPU)

## 📊 Output Format

```csv
candidate_id,rank,score,reasoning
CAND_0004989,1,0.9920,ML Engineer with 6.4 yrs; 7 advanced/expert skills; response rate 0.88; strongest: career fit
...
```

## 🛠️ AI Tools Used

- **Gemini / Claude** — Architecture discussion, code review
- **No AI/LLM calls during ranking** — All scoring is deterministic

## 📝 License

MIT License — Built for the INDIA RUNS Hackathon 2026.
