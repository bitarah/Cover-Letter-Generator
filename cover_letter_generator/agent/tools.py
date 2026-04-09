"""
Agent Tools - Custom tools for the cover letter generation agent.
"""

from typing import Annotated, List
from langchain_core.tools import tool, InjectedToolArg, StructuredTool
from functools import partial

from ..models import CVData, JobDescription, SkillMatch
from ..analysis.skill_matcher import format_skill_match_report


def create_bound_tools(cv_data: CVData, job_data: JobDescription, skill_match: SkillMatch, vector_store) -> List[StructuredTool]:
    """
    Create tools with context bound to them.

    Args:
        cv_data: Structured CV data
        job_data: Structured job description
        skill_match: Skill matching results
        vector_store: FAISS vector store

    Returns:
        List of tools with context bound
    """
    from ..vector_store.cv_vectorstore import create_cv_retriever_tool

    # Create the retriever tool
    retriever_tool = create_cv_retriever_tool(vector_store)

    # Create bound versions of context-dependent tools
    bound_skill_match = StructuredTool.from_function(
        func=lambda: _analyze_skill_match_impl(skill_match),
        name="analyze_skill_match",
        description="""Analyze the match between CV skills and job requirements.

Returns a detailed report of matched skills, missing skills, and match percentage.
Use this tool to understand how well the candidate's skills align with the job."""
    )

    bound_job_details = StructuredTool.from_function(
        func=lambda: _get_job_details_impl(job_data),
        name="get_job_details",
        description="""Get details about the job position including title, company, requirements, and responsibilities.

Use this tool to understand what the job entails and what the employer is looking for."""
    )

    bound_cv_summary = StructuredTool.from_function(
        func=lambda: _get_cv_summary_impl(cv_data),
        name="get_cv_summary",
        description="""Get a summary of the candidate's CV including personal info, experience, education, and skills.

Use this tool to get an overview of the candidate's background."""
    )

    bound_generate_letter = StructuredTool.from_function(
        func=lambda tone: _generate_cover_letter_content_impl(tone, cv_data, job_data, skill_match),
        name="generate_cover_letter_content",
        description="""Generate a cover letter with the specified tone.

Args:
    tone: The tone for the cover letter - must be one of: "professional", "creative", or "technical"

Use this tool after you have researched the CV and job requirements.
The tool will generate a complete, tailored cover letter.""",
        args_schema=None  # Will infer from function signature
    )

    return [retriever_tool, bound_skill_match, bound_job_details, bound_cv_summary, bound_generate_letter]


def _analyze_skill_match_impl(skill_match: SkillMatch) -> str:
    """Implementation of skill match analysis."""
    if not skill_match:
        return "Error: Skill match data not available in context."

    report = format_skill_match_report(skill_match)
    return report


def _get_job_details_impl(job_data: JobDescription) -> str:
    """Implementation of job details retrieval."""
    if not job_data:
        return "Error: Job description data not available in context."

    details = f"Job Title: {job_data.title}\n"
    details += f"Company: {job_data.company}\n"

    if job_data.location:
        details += f"Location: {job_data.location}\n"

    details += f"\nDescription:\n{job_data.description}\n"

    if job_data.responsibilities:
        details += f"\nKey Responsibilities:\n"
        for resp in job_data.responsibilities[:5]:
            details += f"  - {resp}\n"

    if job_data.requirements:
        details += f"\nRequirements:\n"
        for req in job_data.requirements[:5]:
            details += f"  - {req}\n"

    if job_data.required_skills:
        details += f"\nRequired Skills: {', '.join(job_data.required_skills[:10])}\n"

    return details


def _get_cv_summary_impl(cv_data: CVData) -> str:
    """Implementation of CV summary retrieval."""
    if not cv_data:
        return "Error: CV data not available in context."

    summary = f"Candidate: {cv_data.personal_info.name or 'Not specified'}\n"

    if cv_data.personal_info.email:
        summary += f"Email: {cv_data.personal_info.email}\n"

    if cv_data.summary:
        summary += f"\nProfessional Summary:\n{cv_data.summary}\n"

    summary += f"\nExperience ({len(cv_data.experience)} positions):\n"
    for exp in cv_data.experience[:3]:
        summary += f"  - {exp.role} at {exp.company}"
        if exp.start_date:
            summary += f" ({exp.start_date} - {exp.end_date or 'Present'})"
        summary += "\n"

    summary += f"\nEducation ({len(cv_data.education)} entries):\n"
    for edu in cv_data.education:
        summary += f"  - {edu.degree} from {edu.institution}\n"

    if cv_data.skills:
        summary += f"\nSkills: {', '.join(cv_data.skills[:15])}\n"
        if len(cv_data.skills) > 15:
            summary += f"  ... and {len(cv_data.skills) - 15} more\n"

    if cv_data.projects:
        summary += f"\nProjects: {len(cv_data.projects)}\n"

    return summary


def _generate_cover_letter_content_impl(tone: str, cv_data: CVData, job_data: JobDescription, skill_match: SkillMatch) -> str:
    """Implementation of cover letter content generation."""
    from langchain_openai import ChatOpenAI
    from .prompts import load_tone_prompt

    # Validate tone
    valid_tones = ['professional', 'creative', 'technical']
    if tone.lower() not in valid_tones:
        return f"Error: Invalid tone '{tone}'. Must be one of: {', '.join(valid_tones)}"

    if not cv_data or not job_data:
        return "Error: CV or job data not available in context."

    # Load tone-specific prompt
    try:
        prompt_template = load_tone_prompt(tone.lower())
    except Exception as e:
        return f"Error loading prompt template: {e}"

    # Prepare data for prompt
    cv_summary = f"{cv_data.personal_info.name or 'The candidate'} has {len(cv_data.experience)} years of experience "
    cv_summary += f"with skills including {', '.join(cv_data.skills[:10])}. "

    if cv_data.experience:
        cv_summary += f"Most recent role: {cv_data.experience[0].role} at {cv_data.experience[0].company}. "

    job_requirements = ', '.join(job_data.requirements[:5]) if job_data.requirements else "See job description"

    skill_analysis = f"Skill match: {skill_match.match_percentage}% " if skill_match else ""
    skill_analysis += f"Matched skills: {', '.join(skill_match.matched_skills[:5])}" if skill_match and skill_match.matched_skills else ""

    # Format the complete prompt
    full_prompt = prompt_template.format(
        candidate_name=cv_data.personal_info.name or "I",
        company=job_data.company,
        job_title=job_data.title,
        cv_summary=cv_summary,
        job_requirements=job_requirements,
        skill_analysis=skill_analysis,
        job_description=job_data.description[:500]  # Limit to avoid token issues
    )

    # Generate cover letter
    llm = ChatOpenAI(model='gpt-4o', temperature=0.3)
    response = llm.invoke(full_prompt)

    return response.content


@tool
def analyze_skill_match(config: Annotated[dict, InjectedToolArg]) -> str:
    """
    Analyze the match between CV skills and job requirements.

    Returns a detailed report of matched skills, missing skills, and match percentage.
    Use this tool to understand how well the candidate's skills align with the job.
    """
    context = config.get('context', {})
    skill_match: SkillMatch = context.get('skill_match')

    if not skill_match:
        return "Error: Skill match data not available in context."

    report = format_skill_match_report(skill_match)
    return report


@tool
def get_job_details(config: Annotated[dict, InjectedToolArg]) -> str:
    """
    Get details about the job position including title, company, requirements, and responsibilities.

    Use this tool to understand what the job entails and what the employer is looking for.
    """
    context = config.get('context', {})
    job_data: JobDescription = context.get('job_data')

    if not job_data:
        return "Error: Job description data not available in context."

    details = f"Job Title: {job_data.title}\n"
    details += f"Company: {job_data.company}\n"

    if job_data.location:
        details += f"Location: {job_data.location}\n"

    details += f"\nDescription:\n{job_data.description}\n"

    if job_data.responsibilities:
        details += f"\nKey Responsibilities:\n"
        for resp in job_data.responsibilities[:5]:
            details += f"  - {resp}\n"

    if job_data.requirements:
        details += f"\nRequirements:\n"
        for req in job_data.requirements[:5]:
            details += f"  - {req}\n"

    if job_data.required_skills:
        details += f"\nRequired Skills: {', '.join(job_data.required_skills[:10])}\n"

    return details


@tool
def get_cv_summary(config: Annotated[dict, InjectedToolArg]) -> str:
    """
    Get a summary of the candidate's CV including personal info, experience, education, and skills.

    Use this tool to get an overview of the candidate's background.
    """
    context = config.get('context', {})
    cv_data: CVData = context.get('cv_data')

    if not cv_data:
        return "Error: CV data not available in context."

    summary = f"Candidate: {cv_data.personal_info.name or 'Not specified'}\n"

    if cv_data.personal_info.email:
        summary += f"Email: {cv_data.personal_info.email}\n"

    if cv_data.summary:
        summary += f"\nProfessional Summary:\n{cv_data.summary}\n"

    summary += f"\nExperience ({len(cv_data.experience)} positions):\n"
    for exp in cv_data.experience[:3]:
        summary += f"  - {exp.role} at {exp.company}"
        if exp.start_date:
            summary += f" ({exp.start_date} - {exp.end_date or 'Present'})"
        summary += "\n"

    summary += f"\nEducation ({len(cv_data.education)} entries):\n"
    for edu in cv_data.education:
        summary += f"  - {edu.degree} from {edu.institution}\n"

    if cv_data.skills:
        summary += f"\nSkills: {', '.join(cv_data.skills[:15])}\n"
        if len(cv_data.skills) > 15:
            summary += f"  ... and {len(cv_data.skills) - 15} more\n"

    if cv_data.projects:
        summary += f"\nProjects: {len(cv_data.projects)}\n"

    return summary


@tool
def generate_cover_letter_content(
    tone: str,
    config: Annotated[dict, InjectedToolArg]
) -> str:
    """
    Generate a cover letter with the specified tone.

    Args:
        tone: The tone for the cover letter - must be one of: "professional", "creative", or "technical"

    Use this tool after you have researched the CV and job requirements.
    The tool will generate a complete, tailored cover letter.
    """
    from langchain_openai import ChatOpenAI
    from .prompts import load_tone_prompt

    # Validate tone
    valid_tones = ['professional', 'creative', 'technical']
    if tone.lower() not in valid_tones:
        return f"Error: Invalid tone '{tone}'. Must be one of: {', '.join(valid_tones)}"

    context = config.get('context', {})
    cv_data: CVData = context.get('cv_data')
    job_data: JobDescription = context.get('job_data')
    skill_match: SkillMatch = context.get('skill_match')

    if not cv_data or not job_data:
        return "Error: CV or job data not available in context."

    # Load tone-specific prompt
    try:
        prompt_template = load_tone_prompt(tone.lower())
    except Exception as e:
        return f"Error loading prompt template: {e}"

    # Prepare data for prompt
    cv_summary = f"{cv_data.personal_info.name or 'The candidate'} has {len(cv_data.experience)} years of experience "
    cv_summary += f"with skills including {', '.join(cv_data.skills[:10])}. "

    if cv_data.experience:
        cv_summary += f"Most recent role: {cv_data.experience[0].role} at {cv_data.experience[0].company}. "

    job_requirements = ', '.join(job_data.requirements[:5]) if job_data.requirements else "See job description"

    skill_analysis = f"Skill match: {skill_match.match_percentage}% " if skill_match else ""
    skill_analysis += f"Matched skills: {', '.join(skill_match.matched_skills[:5])}" if skill_match and skill_match.matched_skills else ""

    # Format the complete prompt
    full_prompt = prompt_template.format(
        candidate_name=cv_data.personal_info.name or "I",
        company=job_data.company,
        job_title=job_data.title,
        cv_summary=cv_summary,
        job_requirements=job_requirements,
        skill_analysis=skill_analysis,
        job_description=job_data.description[:500]  # Limit to avoid token issues
    )

    # Generate cover letter
    llm = ChatOpenAI(model='gpt-4o', temperature=0.3)
    response = llm.invoke(full_prompt)

    return response.content
