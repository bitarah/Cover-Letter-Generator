"""
Validation utilities for input files and data.
"""

from pathlib import Path
from docx import Document
from ..models import CVData


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class DocumentParsingError(Exception):
    """Raised when document parsing fails."""
    pass


def validate_cv_file(cv_path: str) -> None:
    """
    Validate CV file exists and is readable.

    Args:
        cv_path: Path to CV file

    Raises:
        ValidationError: If file is invalid
        DocumentParsingError: If file cannot be parsed
    """
    path = Path(cv_path)

    # Check if file exists
    if not path.exists():
        raise ValidationError(f"CV file not found: {cv_path}")

    # Check if file has .docx extension
    if path.suffix.lower() != '.docx':
        raise ValidationError(f"CV must be .docx format, got: {path.suffix}")

    # Try to open document
    try:
        doc = Document(cv_path)
        # Check if document has content
        if not doc.paragraphs:
            raise DocumentParsingError("CV document is empty")
    except Exception as e:
        raise DocumentParsingError(f"Cannot read CV file: {e}")


def validate_job_file(job_path: str) -> None:
    """
    Validate job description file exists and has content.

    Args:
        job_path: Path to job description file

    Raises:
        ValidationError: If file is invalid
    """
    path = Path(job_path)

    # Check if file exists
    if not path.exists():
        raise ValidationError(f"Job description file not found: {job_path}")

    # Check if file has .txt extension
    if path.suffix.lower() != '.txt':
        raise ValidationError(f"Job description must be .txt format, got: {path.suffix}")

    # Check if file has content
    try:
        content = path.read_text(encoding='utf-8').strip()
        if len(content) < 50:
            raise ValidationError(
                "Job description is too short (minimum 50 characters). "
                f"Found: {len(content)} characters"
            )
    except UnicodeDecodeError:
        raise ValidationError("Job description file must be UTF-8 encoded text")
    except Exception as e:
        raise ValidationError(f"Cannot read job description file: {e}")


def validate_cv_data(cv_data: CVData) -> None:
    """
    Validate parsed CV has minimum required content.

    Args:
        cv_data: Parsed CV data

    Raises:
        ValidationError: If CV data is insufficient
    """
    errors = []

    # Check for experience
    if not cv_data.experience or len(cv_data.experience) == 0:
        errors.append("CV must contain at least one work experience entry")

    # Check for skills
    if not cv_data.skills or len(cv_data.skills) == 0:
        errors.append("CV must contain at least one skill")

    # Check for education
    if not cv_data.education or len(cv_data.education) == 0:
        errors.append("CV should contain at least one education entry (warning)")

    # Check for any content
    if not cv_data.raw_text or len(cv_data.raw_text) < 100:
        errors.append("CV appears to be too short or empty")

    if errors:
        raise ValidationError("CV validation failed:\n  - " + "\n  - ".join(errors))


def validate_tone(tone: str) -> None:
    """
    Validate tone is one of the allowed values.

    Args:
        tone: Tone to validate

    Raises:
        ValidationError: If tone is invalid
    """
    from ..config import TONES

    if tone.lower() not in TONES:
        raise ValidationError(
            f"Invalid tone '{tone}'. Must be one of: {', '.join(TONES)}"
        )


def validate_openai_api_key() -> None:
    """
    Validate OpenAI API key is set.

    Raises:
        ValidationError: If API key is not set
    """
    import os

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValidationError(
            "OpenAI API key not found. "
            "Please set OPENAI_API_KEY environment variable in your .env file"
        )

    if not api_key.startswith("sk-"):
        raise ValidationError(
            "Invalid OpenAI API key format. "
            "API key should start with 'sk-'"
        )
