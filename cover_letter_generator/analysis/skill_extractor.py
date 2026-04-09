"""
Skill Extractor - Extracts skills from text using LLM and normalization.
"""

from typing import List
from langchain_openai import ChatOpenAI


def extract_skills_with_llm(text: str) -> List[str]:
    """
    Use GPT to extract skills from text.

    Args:
        text: Text to extract skills from

    Returns:
        List of extracted skills
    """
    try:
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

        prompt = f"""Extract all technical skills, soft skills, tools, technologies, and competencies from the following text.
Return only a comma-separated list of skills, nothing else.

Text:
{text[:3000]}

Skills (comma-separated):"""

        response = llm.invoke(prompt)
        skills_text = response.content.strip()

        # Parse comma-separated skills
        skills = [s.strip() for s in skills_text.split(',') if s.strip()]

        return skills
    except Exception as e:
        print(f"Warning: Could not extract skills with LLM: {e}")
        return []


def normalize_skills(skills: List[str]) -> List[str]:
    """
    Normalize skill names for better matching.

    Args:
        skills: List of raw skills

    Returns:
        List of normalized skills
    """
    normalized = []

    # Common synonyms mapping
    synonyms = {
        'js': 'javascript',
        'ts': 'typescript',
        'py': 'python',
        'ml': 'machine learning',
        'ai': 'artificial intelligence',
        'dl': 'deep learning',
        'k8s': 'kubernetes',
        'react.js': 'react',
        'node.js': 'node',
        'vue.js': 'vue',
    }

    for skill in skills:
        # Convert to lowercase for consistent comparison
        skill_lower = skill.lower().strip()

        # Apply synonym mapping
        if skill_lower in synonyms:
            skill_lower = synonyms[skill_lower]

        # Remove common prefixes/suffixes
        skill_lower = skill_lower.replace('experience with', '').replace('experience in', '').strip()

        if skill_lower and skill_lower not in normalized:
            normalized.append(skill_lower)

    return normalized


def extract_and_normalize_skills(text: str) -> List[str]:
    """
    Extract and normalize skills from text.

    Args:
        text: Text to extract skills from

    Returns:
        List of normalized skills
    """
    skills = extract_skills_with_llm(text)
    return normalize_skills(skills)
