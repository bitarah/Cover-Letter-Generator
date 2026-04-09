"""
Skill Matcher - Matches CV skills against job requirements with semantic similarity.
"""

from typing import List, Dict
import numpy as np
from langchain_openai import OpenAIEmbeddings

from ..models import SkillMatch
from .skill_extractor import normalize_skills


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def calculate_skill_match(cv_skills: List[str], job_skills: List[str], similarity_threshold: float = 0.8) -> SkillMatch:
    """
    Compare CV skills against job requirements.

    Uses exact matching and semantic similarity to identify:
    - Matched skills (present in both)
    - Missing skills (in job but not in CV)
    - Match percentage

    Args:
        cv_skills: List of skills from CV
        job_skills: List of required skills from job description
        similarity_threshold: Minimum similarity score for semantic matching (default: 0.8)

    Returns:
        SkillMatch object with analysis results
    """
    # Normalize both lists
    cv_skills_norm = normalize_skills(cv_skills)
    job_skills_norm = normalize_skills(job_skills)

    matched_skills = []
    missing_skills = []
    semantic_matches = {}

    # Initialize embeddings for semantic matching
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    # Track which job skills have been matched
    job_skills_matched = set()

    # First pass: exact matches
    for job_skill in job_skills_norm:
        if job_skill in cv_skills_norm:
            matched_skills.append(job_skill)
            job_skills_matched.add(job_skill)

    # Second pass: semantic matching for unmatched job skills
    # Get unmatched job skills
    unmatched_job_skills = [skill for skill in job_skills_norm if skill not in job_skills_matched]

    if unmatched_job_skills and cv_skills_norm:
        # Batch embed all CV skills and unmatched job skills (much faster than one-by-one)
        all_skills_to_embed = cv_skills_norm + unmatched_job_skills
        try:
            embeddings_batch = embeddings.embed_documents(all_skills_to_embed)

            # Split embeddings back into CV and job skill embeddings
            cv_embeddings = embeddings_batch[:len(cv_skills_norm)]
            job_embeddings = embeddings_batch[len(cv_skills_norm):]

            # Now compare using pre-computed embeddings
            for idx, job_skill in enumerate(unmatched_job_skills):
                job_embedding = job_embeddings[idx]
                best_match = None
                best_similarity = 0.0

                for cv_idx, cv_skill in enumerate(cv_skills_norm):
                    cv_embedding = cv_embeddings[cv_idx]
                    similarity = calculate_cosine_similarity(cv_embedding, job_embedding)

                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_match = cv_skill

                if best_match:
                    matched_skills.append(job_skill)
                    semantic_matches[best_match] = job_skill
                    job_skills_matched.add(job_skill)
        except Exception as e:
            print(f"Warning: Batch embedding failed, skipping semantic matching: {e}")

    # Identify missing skills
    for job_skill in job_skills_norm:
        if job_skill not in job_skills_matched:
            missing_skills.append(job_skill)

    # Calculate match percentage
    if len(job_skills_norm) > 0:
        match_percentage = (len(matched_skills) / len(job_skills_norm)) * 100
    else:
        match_percentage = 0.0

    # Create SkillMatch object
    skill_match = SkillMatch(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        match_percentage=round(match_percentage, 1),
        cv_skills_count=len(cv_skills_norm),
        job_skills_count=len(job_skills_norm),
        semantic_matches=semantic_matches
    )

    return skill_match


def format_skill_match_report(skill_match: SkillMatch) -> str:
    """
    Format skill match results as a readable report.

    Args:
        skill_match: SkillMatch object

    Returns:
        Formatted string report
    """
    report = f"Skill Match Analysis\n"
    report += "=" * 50 + "\n\n"

    report += f"Match Percentage: {skill_match.match_percentage}%\n"
    report += f"CV Skills: {skill_match.cv_skills_count} | Job Skills: {skill_match.job_skills_count}\n\n"

    if skill_match.matched_skills:
        report += f"Matched Skills ({len(skill_match.matched_skills)}):\n"
        for skill in skill_match.matched_skills[:10]:  # Limit to top 10
            report += f"  ✓ {skill}\n"
        if len(skill_match.matched_skills) > 10:
            report += f"  ... and {len(skill_match.matched_skills) - 10} more\n"
        report += "\n"

    if skill_match.semantic_matches:
        report += f"Semantic Matches ({len(skill_match.semantic_matches)}):\n"
        for cv_skill, job_skill in list(skill_match.semantic_matches.items())[:5]:
            report += f"  ~ {cv_skill} ≈ {job_skill}\n"
        report += "\n"

    if skill_match.missing_skills:
        report += f"Missing Skills ({len(skill_match.missing_skills)}):\n"
        for skill in skill_match.missing_skills[:10]:
            report += f"  ✗ {skill}\n"
        if len(skill_match.missing_skills) > 10:
            report += f"  ... and {len(skill_match.missing_skills) - 10} more\n"

    return report
