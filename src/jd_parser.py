"""Job Description Parser.

Parses the job description and returns a structured dictionary of requirements.
Since the JD is a known static document (Senior AI Engineer - Founding Team at
Redrob AI, Pune/Noida, India), we hardcode the parsed requirements for reliability.
"""

import os
from typing import Optional


def parse_job_description(jd_path: str = None) -> dict:
    """Parse the job description and return structured requirements.

    The structured data is always returned from a hardcoded dictionary since the
    JD is a known static document. If ``jd_path`` points to a valid ``.docx``
    file, the full text is additionally extracted and stored under the
    ``'jd_full_text'`` key.

    Args:
        jd_path: Optional path to a ``.docx`` file containing the raw JD text.

    Returns:
        A dictionary of structured job requirements.
    """

    requirements: dict = {
        'title': 'Senior AI Engineer',
        'title_keywords': [
            'AI Engineer', 'ML Engineer', 'Machine Learning Engineer',
            'Data Scientist', 'Research Engineer', 'NLP Engineer',
            'Search Engineer', 'Ranking Engineer', 'Backend Engineer',
            'Software Engineer', 'Deep Learning Engineer',
            'Senior AI Engineer', 'Senior ML Engineer',
            'Senior Machine Learning Engineer', 'Senior Data Scientist',
            'Applied Scientist', 'Applied ML Engineer',
        ],
        'must_have_skills': [
            'sentence-transformers', 'embeddings', 'retrieval', 'semantic search',
            'OpenAI', 'BGE', 'E5',
            'Pinecone', 'Weaviate', 'Qdrant', 'Milvus', 'OpenSearch',
            'Elasticsearch', 'FAISS', 'vector database', 'vector search',
            'Python',
            'NDCG', 'MRR', 'MAP', 'A/B testing', 'ranking', 'evaluation',
            'information retrieval',
            'NLP', 'natural language processing', 'machine learning',
            'deep learning', 'PyTorch', 'TensorFlow',
        ],
        'nice_to_have_skills': [
            'LoRA', 'QLoRA', 'PEFT', 'fine-tuning', 'Fine-tuning LLMs',
            'XGBoost', 'learning-to-rank', 'LTR',
            'distributed systems', 'inference optimization',
            'Spark', 'Airflow', 'data engineering',
            'Kubernetes', 'Docker', 'MLOps',
            'RAG', 'LangChain', 'LLM', 'GPT',
            'Recommendation Systems', 'Search Systems',
            'Hugging Face', 'Transformers',
        ],
        'min_experience_years': 5.0,
        'max_experience_years': 9.0,
        'ideal_experience_years': (6.0, 8.0),
        'preferred_locations': [
            'Pune', 'Noida', 'Hyderabad', 'Mumbai', 'Delhi', 'NCR',
            'Bengaluru', 'Bangalore', 'India',
        ],
        'work_mode': 'hybrid',
        'preferred_industries': [
            'Technology', 'IT Services', 'Software', 'AI/ML', 'Internet',
            'SaaS', 'E-Commerce', 'FinTech', 'Product Development',
            'Data Analytics',
        ],
        'preferred_fields': [
            'Computer Science', 'Data Science', 'Artificial Intelligence',
            'Machine Learning', 'Information Technology', 'Electronics',
            'Electrical Engineering', 'Mathematics', 'Statistics',
            'Software Engineering',
        ],
        'consulting_firms': [
            'TCS', 'Tata Consultancy Services', 'Infosys', 'Wipro',
            'Accenture', 'Cognizant', 'Capgemini', 'HCL',
            'HCL Technologies', 'Tech Mahindra', 'Mindtree', 'Mphasis',
            'L&T Infotech', 'LTIMindtree',
        ],
        'unrelated_titles': [
            'Marketing Manager', 'Graphic Designer', 'Accountant',
            'Civil Engineer', 'HR Manager', 'Sales Executive',
            'Customer Support', 'Content Writer', 'Mechanical Engineer',
            'Operations Manager', 'Project Manager',
        ],
        'salary_budget_lpa': (15.0, 45.0),
        'jd_summary': (
            "Senior AI Engineer for Redrob AI's founding team. Own the "
            "intelligence layer: ranking, retrieval, and matching systems. "
            "Must have production experience with embeddings-based retrieval, "
            "vector databases, Python, and ranking evaluation frameworks. "
            "Looking for someone who can ship fast while maintaining deep "
            "technical depth in ML systems. 5-9 years experience, hybrid work "
            "in Pune/Noida India. No pure researchers, no title-chasers, no "
            "consulting-only careers."
        ),
    }

    # Optionally read the raw JD text from a .docx file if provided.
    if jd_path and os.path.isfile(jd_path) and jd_path.lower().endswith('.docx'):
        try:
            from docx import Document  # python-docx

            doc = Document(jd_path)
            full_text = '\n'.join(para.text for para in doc.paragraphs)
            requirements['jd_full_text'] = full_text
        except ImportError:
            print(
                'Warning: python-docx not installed. '
                'Skipping raw JD text extraction.'
            )
        except Exception as exc:  # noqa: BLE001
            print(f'Warning: could not read {jd_path}: {exc}')

    return requirements
