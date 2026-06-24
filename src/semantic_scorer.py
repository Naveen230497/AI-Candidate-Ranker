"""Semantic Scorer.

Computes semantic similarity between the job description and candidate
profiles using sentence-transformers embeddings and cosine similarity.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticScorer:
    """Scores candidate text against a job description using embeddings.

    Uses a sentence-transformer model to encode both the JD and candidate
    profiles into dense vectors, then computes cosine similarity.  Scores
    are shifted from the ``[-1, 1]`` cosine range into ``[0, 1]``.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        """Initialize with a sentence-transformer model.

        Args:
            model_name: Name or path of the sentence-transformer model to
                load.  Defaults to ``'all-MiniLM-L6-v2'``.
        """
        print(f'Loading semantic model: {model_name}...')
        self.model = SentenceTransformer(model_name)
        self.jd_embedding: np.ndarray | None = None

    def set_job_description(self, jd_text: str) -> None:
        """Encode the job description text and cache the embedding.

        Args:
            jd_text: The full text (or summary) of the job description.
        """
        self.jd_embedding = self.model.encode(
            jd_text, normalize_embeddings=True
        )

    def score_candidate(self, candidate_text: str) -> float:
        """Score a single candidate's text against the JD.

        Args:
            candidate_text: A text blob representing the candidate profile
                (see :func:`data_loader.build_candidate_text`).

        Returns:
            A float in ``[0.0, 1.0]`` representing semantic similarity.

        Raises:
            ValueError: If :meth:`set_job_description` has not been called.
        """
        if self.jd_embedding is None:
            raise ValueError('Call set_job_description first')

        candidate_embedding = self.model.encode(
            candidate_text, normalize_embeddings=True
        )
        # Cosine similarity (vectors are already L2-normalised)
        similarity = float(np.dot(self.jd_embedding, candidate_embedding))
        # Shift from [-1, 1] to [0, 1]
        score = (similarity + 1) / 2
        return max(0.0, min(1.0, score))

    def score_batch(self, candidate_texts: List[str]) -> List[float]:
        """Score a batch of candidate texts against the JD.

        Encoding in batch is significantly more efficient than calling
        :meth:`score_candidate` one by one.

        Args:
            candidate_texts: A list of text blobs, one per candidate.

        Returns:
            A list of floats in ``[0.0, 1.0]``, one per candidate.

        Raises:
            ValueError: If :meth:`set_job_description` has not been called.
        """
        if self.jd_embedding is None:
            raise ValueError('Call set_job_description first')

        embeddings = self.model.encode(
            candidate_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        similarities = np.dot(embeddings, self.jd_embedding)
        scores = (similarities + 1) / 2
        return [max(0.0, min(1.0, float(s))) for s in scores]
