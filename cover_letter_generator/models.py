"""
Pydantic models for the cover letter generator.

Defines data structures for CV, job descriptions, skill matching, and generated cover letters.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PersonalInfo(BaseModel):
    """Personal information from CV."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None


class Experience(BaseModel):
    """Work experience entry."""
    company: str
    role: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str
    bullets: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)


class Education(BaseModel):
    """Education entry."""
    institution: str
    degree: str
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    description: Optional[str] = None


class Project(BaseModel):
    """Project entry."""
    name: str
    description: str
    technologies: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)


class CVData(BaseModel):
    """Complete structured CV data."""
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    summary: Optional[str] = None
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_text: str = ""  # Full CV text for fallback


class JobDescription(BaseModel):
    """Parsed job description."""
    title: str
    company: str
    location: Optional[str] = None
    description: str
    requirements: List[str] = Field(default_factory=list)
    required_skills: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    raw_text: str = ""  # Full job description text


class SkillMatch(BaseModel):
    """Skill matching analysis result."""
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    match_percentage: float = 0.0
    cv_skills_count: int = 0
    job_skills_count: int = 0
    semantic_matches: Dict[str, str] = Field(default_factory=dict)  # CV skill -> Job skill mapping


class CoverLetter(BaseModel):
    """Generated cover letter."""
    content: str
    tone: str  # "professional", "creative", or "technical"
    timestamp: datetime = Field(default_factory=datetime.now)
    job_title: str
    company: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """Agent session state for interactive refinement."""
    cv_data: Optional[CVData] = None
    job_data: Optional[JobDescription] = None
    skill_match: Optional[SkillMatch] = None
    generated_letters: List[CoverLetter] = Field(default_factory=list)
    current_letter: Optional[str] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
