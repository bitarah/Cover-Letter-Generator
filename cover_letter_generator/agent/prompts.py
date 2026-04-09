"""
Prompt Manager - Loads and formats tone-specific prompts for cover letter generation.
"""

from pathlib import Path
from typing import Dict


def get_prompts_dir() -> Path:
    """Get the prompts directory path."""
    # Navigate from this file to the prompts directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    prompts_dir = project_root / 'prompts'
    return prompts_dir


def load_tone_prompt(tone: str) -> str:
    """
    Load prompt template for the specified tone.

    Args:
        tone: One of 'professional', 'creative', or 'technical'

    Returns:
        Prompt template string

    Raises:
        ValueError: If tone is invalid or file not found
    """
    valid_tones = ['professional', 'creative', 'technical']

    if tone.lower() not in valid_tones:
        raise ValueError(f"Invalid tone '{tone}'. Must be one of: {', '.join(valid_tones)}")

    prompts_dir = get_prompts_dir()
    prompt_file = prompts_dir / f"{tone.lower()}_tone.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        template = f.read()

    return template


def format_prompt(
    tone: str,
    candidate_name: str,
    company: str,
    job_title: str,
    cv_summary: str,
    job_requirements: str,
    skill_analysis: str,
    job_description: str
) -> str:
    """
    Load and format a prompt template with the provided data.

    Args:
        tone: The tone to use ('professional', 'creative', or 'technical')
        candidate_name: Name of the candidate
        company: Company name
        job_title: Job position title
        cv_summary: Summary of candidate's CV
        job_requirements: Job requirements summary
        skill_analysis: Skill match analysis
        job_description: Job description text

    Returns:
        Formatted prompt string ready to send to LLM
    """
    template = load_tone_prompt(tone)

    formatted = template.format(
        candidate_name=candidate_name,
        company=company,
        job_title=job_title,
        cv_summary=cv_summary,
        job_requirements=job_requirements,
        skill_analysis=skill_analysis,
        job_description=job_description
    )

    return formatted


def get_system_prompt() -> str:
    """
    Get the system prompt for the cover letter agent.

    Returns:
        System prompt string
    """
    return """You are an expert cover letter writing assistant. Your role is to help candidates create compelling, tailored cover letters that highlight their best-matching qualifications for specific jobs.

CRITICAL CONSTRAINT: Cover letters must be grounded in facts from the candidate's CV.
Do NOT invent, assume, or add details not explicitly mentioned in the CV.
All claims and examples must be verifiable from the provided candidate background.

Your process:
1. First, use get_cv_summary to get an overview of the candidate's background and verify what information is available
2. Use search_cv to find relevant experiences and skills in the candidate's CV that match the job requirements
3. Use get_job_details to understand what the employer is looking for
4. Use analyze_skill_match to see the skill alignment between CV and job
5. Finally, use generate_cover_letter_content with the appropriate tone to create the letter based ONLY on verified CV information

Key principles:
- Always tailor the letter to the specific job and company using REAL information from the CV
- Highlight relevant experiences and achievements that are explicitly stated in the CV
- Use concrete examples and metrics where they exist in the CV
- Match the tone to the job type (professional for corporate, creative for startups, technical for engineering)
- Keep letters concise (300-450 words)
- Make every sentence count - no generic filler
- NEVER invent details, stories, or experiences that aren't in the CV

Be thoughtful and strategic in your approach. Focus on presenting the candidate's actual qualifications effectively."""
