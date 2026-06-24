#!/usr/bin/env python3
"""
AI Candidate Ranker — Intelligent Candidate Discovery & Ranking System
=====================================================================
Main entry point that orchestrates the full ranking pipeline.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./data/sample_candidates.json --out ./test_submission.csv --sample

For the INDIA RUNS Data & AI Challenge by Redrob / Hack2Skill.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from jd_parser import parse_job_description
from data_loader import (
    load_candidates_stream,
    load_candidates_batch,
    load_sample_candidates,
    build_candidate_text,
    count_candidates,
)
from hard_filter import should_keep
from skills_scorer import score_skills
from career_scorer import score_career
from education_scorer import score_education
from signals_scorer import score_signals
from semantic_scorer import SemanticScorer
from hybrid_ranker import compute_hybrid_score, rank_candidates
from reasoning_generator import generate_reasoning
from location_scorer import score_location


def run_pipeline(candidates_path: str, output_path: str, is_sample: bool = False):
    """Execute the full ranking pipeline using a Two-Stage Retrieval architecture."""
    
    start_time = time.time()
    print("=" * 70)
    print("  AI Candidate Ranker - Intelligent Discovery & Ranking System")
    print("=" * 70)
    
    # ─── Step 1: Parse Job Description ────────────────────────────────────
    print("\n[1/7] Parsing job description...")
    jd_path = os.path.join(os.path.dirname(__file__), 'data', 'job_description.docx')
    job_req = parse_job_description(jd_path if os.path.exists(jd_path) else None)
    print(f"  -> Role: {job_req['title']}")
    print(f"  -> Experience range: {job_req['min_experience_years']}-{job_req['max_experience_years']} years")
    
    # ─── Step 2: Load and Filter Candidates ───────────────────────────────
    print("\n[2/7] Loading and filtering candidates...")
    
    if is_sample:
        all_candidates = load_sample_candidates(candidates_path)
        print(f"  -> Loaded {len(all_candidates)} sample candidates.")
        filtered_candidates = [c for c in all_candidates if should_keep(c, job_req)]
    else:
        print("  -> Counting candidates...")
        total = count_candidates(candidates_path)
        print(f"  -> Total candidates in dataset: {total:,}")
        
        filtered_candidates = []
        for candidate in tqdm(
            load_candidates_stream(candidates_path),
            total=total,
            desc="  -> Filtering",
            unit=" candidates"
        ):
            if should_keep(candidate, job_req):
                filtered_candidates.append(candidate)
    
    print(f"  -> Candidates after hard filter: {len(filtered_candidates):,}")
    
    # ─── Step 3: STAGE 1 - Fast Deterministic Scoring ──────────────────────
    print(f"\n[3/7] STAGE 1: Fast deterministic scoring on {len(filtered_candidates):,} candidates...")
    
    stage1_candidates = []
    for candidate in tqdm(filtered_candidates, desc="  -> Scoring", unit=" candidates"):
        skills_s = score_skills(candidate, job_req)
        career_s = score_career(candidate, job_req)
        education_s = score_education(candidate, job_req)
        signals_s = score_signals(candidate)
        location_s = score_location(candidate, job_req)
        
        # Intermediate Stage 1 score (without semantic embedding)
        stage1_score = (
            skills_s * 0.35 +
            career_s * 0.30 +
            signals_s * 0.20 +
            education_s * 0.10 +
            location_s * 0.05
        )
        
        stage1_candidates.append({
            'candidate_id': candidate['candidate_id'],
            'candidate': candidate,
            'stage1_score': stage1_score,
            'skills_score': skills_s,
            'career_score': career_s,
            'education_score': education_s,
            'signals_score': signals_s,
            'location_score': location_s,
        })
    
    # ─── Step 4: Top-K Selection for Re-ranking ────────────────────────────
    # Select top 2000 candidates to pass to the heavy AI model
    RE_RANK_LIMIT = 2000 if not is_sample else len(filtered_candidates)
    print(f"\n[4/7] Filtering top {RE_RANK_LIMIT:,} candidates for AI Re-ranking...")
    
    stage1_candidates.sort(key=lambda x: x['stage1_score'], reverse=True)
    top_k_candidates = stage1_candidates[:RE_RANK_LIMIT]
    
    # ─── Step 5: STAGE 2 - Semantic Re-ranking (Heavy AI) ──────────────────
    print(f"\n[5/7] STAGE 2: Semantic embedding re-ranking...")
    
    semantic_scorer = SemanticScorer(model_name='all-MiniLM-L6-v2')
    jd_text = job_req.get('jd_full_text', job_req['jd_summary'])
    semantic_scorer.set_job_description(jd_text)
    
    candidate_texts = [build_candidate_text(c['candidate']) for c in top_k_candidates]
    
    print(f"  -> Computing embeddings for {len(candidate_texts)} candidates...")
    batch_size = 256
    semantic_scores = []
    for i in tqdm(range(0, len(candidate_texts), batch_size), desc="  -> Embedding", unit=" batches"):
        batch_texts = candidate_texts[i:i + batch_size]
        batch_scores = semantic_scorer.score_batch(batch_texts)
        semantic_scores.extend(batch_scores)
    
    # Compute Final Hybrid Score
    for idx, entry in enumerate(top_k_candidates):
        entry['semantic_score'] = semantic_scores[idx]
        
        # Apply the final official weights
        final_score = (
            entry['skills_score'] * 0.30 +
            entry['career_score'] * 0.25 +
            entry['signals_score'] * 0.20 +
            entry['semantic_score'] * 0.15 +
            entry['education_score'] * 0.07 +
            entry['location_score'] * 0.03
        )
        entry['final_score'] = max(0.0, min(1.0, final_score))
        
    # ─── Step 6: Final Ranking & Reasoning ─────────────────────────────────
    print(f"\n[6/7] Finalizing rankings and reasoning...")
    top_100 = rank_candidates(top_k_candidates)
    
    for entry in top_100:
        scores_dict = {
            'skills_score': entry['skills_score'],
            'career_score': entry['career_score'],
            'education_score': entry['education_score'],
            'signals_score': entry['signals_score'],
            'semantic_score': entry['semantic_score'],
            'final_score': entry['final_score'],
        }
        entry['reasoning'] = generate_reasoning(entry['candidate'], scores_dict)
        
    print(f"  -> #1 score: {top_100[0]['score']:.4f} | #100 score: {top_100[-1]['score']:.4f}")
    
    # ─── Step 7: Write Output CSV ─────────────────────────────────────────
    print(f"\n[7/7] Writing submission to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for entry in top_100:
            writer.writerow([
                entry['candidate_id'],
                entry['rank'],
                f"{entry['score']:.4f}",
                entry['reasoning'],
            ])
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {elapsed:.1f} seconds!")
    print(f"  Output: {output_path}")
    print(f"  Top 5 candidates:")
    for entry in top_100[:5]:
        profile = entry['candidate']['profile']
        print(f"    #{entry['rank']}: {entry['candidate_id']} - "
              f"{profile.get('current_title', '?')} | "
              f"{profile.get('years_of_experience', 0):.1f} yrs | "
              f"Score: {entry['score']:.4f}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description='AI Candidate Ranker - Intelligent Candidate Discovery & Ranking System.'
    )
    parser.add_argument(
        '--candidates', '-c',
        required=True,
        help='Path to candidates JSONL file.'
    )
    parser.add_argument(
        '--out', '-o',
        default='./submission.csv',
        help='Output CSV file path.'
    )
    parser.add_argument(
        '--sample', '-s',
        action='store_true',
        help='Use sample mode (load from JSON array).'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.candidates):
        print(f"ERROR: Candidates file not found: {args.candidates}")
        sys.exit(1)
    
    run_pipeline(args.candidates, args.out, is_sample=args.sample)


if __name__ == '__main__':
    main()
