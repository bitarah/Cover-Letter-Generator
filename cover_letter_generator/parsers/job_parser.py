"""
Job Description Parser - Extracts and structures information from job description text files.
"""

import re
from pathlib import Path
from typing import List, Dict
from langchain_openai import ChatOpenAI

from ..models import JobDescription


class JobParser:
    """Parser for extracting structured data from job descriptions."""

    SECTION_KEYWORDS = {
        'requirements': ['requirements', 'required qualifications', 'must have', 'you have'],
        'qualifications': ['qualifications', 'preferred qualifications', 'nice to have', 'preferred'],
        'responsibilities': ['responsibilities', 'what you will do', 'your role', 'duties', 'you will'],
        'about': ['about us', 'about the company', 'about', 'who we are'],
    }

    def __init__(self, txt_path: str):
        """Initialize parser with path to text file."""
        self.txt_path = Path(txt_path)
        self.text = ""
        self.lines = []

    def read_file(self) -> str:
        """Read job description text file."""
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.lines = [line.strip() for line in self.text.split('\n') if line.strip()]
        return self.text

    def extract_job_metadata(self) -> Dict[str, str]:
        """Extract job title and company from first few lines."""
        metadata = {
            'title': '',
            'company': '',
            'location': ''
        }

        # Try first 5 lines for title and company
        for i, line in enumerate(self.lines[:5]):
            # Usually title is in first few lines and is relatively short
            if not metadata['title'] and len(line.split()) <= 10:
                metadata['title'] = line
            # Company often contains keywords or is line 2
            elif not metadata['company'] and i < 3:
                if any(keyword in line.lower() for keyword in ['inc', 'llc', 'company', 'corp', 'ltd']):
                    metadata['company'] = line
                elif i == 1:  # Default to second line if no company found
                    metadata['company'] = line

            # Location detection
            if 'location' in line.lower() or 'remote' in line.lower():
                metadata['location'] = line

        # If still no company, use a placeholder
        if not metadata['company']:
            metadata['company'] = 'the company'

        if not metadata['title']:
            metadata['title'] = self.lines[0] if self.lines else 'Position'

        return metadata

    def identify_sections(self) -> Dict[str, List[str]]:
        """Identify sections in job description."""
        sections = {}
        current_section = 'description'
        current_content = []

        for line in self.lines:
            section_found = False
            line_lower = line.lower().strip()

            # Check if this line is a section header
            for section_name, keywords in self.SECTION_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in line_lower and len(line.split()) <= 8:
                        # Save previous section
                        if current_content:
                            sections[current_section] = current_content
                        # Start new section
                        current_section = section_name
                        current_content = []
                        section_found = True
                        break
                if section_found:
                    break

            if not section_found:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = current_content

        return sections

    def parse_list_items(self, content: List[str]) -> List[str]:
        """Parse bullet point lists from content."""
        items = []

        for line in content:
            # Skip if line is too short or looks like a header
            if len(line) < 10 or line.endswith(':'):
                continue

            # Remove bullet points and numbering
            cleaned = re.sub(r'^[\•\-\–\·\*\d]+[\.\)]*\s*', '', line).strip()

            if cleaned:
                items.append(cleaned)

        return items

    def extract_skills_with_llm(self, text: str) -> List[str]:
        """Use GPT to extract required skills from text."""
        try:
            llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

            prompt = f"""Extract all required skills, technologies, and qualifications from the following job description.
Return only a comma-separated list of skills, nothing else.

Job Description:
{text[:2000]}  # Limit to avoid token limits

Skills (comma-separated):"""

            response = llm.invoke(prompt)
            skills_text = response.content.strip()

            # Parse comma-separated skills
            skills = [s.strip() for s in skills_text.split(',') if s.strip()]

            return skills
        except Exception as e:
            print(f"Warning: Could not extract skills with LLM: {e}")
            # Fallback to simple keyword extraction
            return self._extract_skills_simple(text)

    def _extract_skills_simple(self, text: str) -> List[str]:
        """Simple fallback skill extraction using common patterns."""
        skills = []

        # Common skill patterns
        skill_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|Ruby|Go|Rust|Swift|Kotlin)\b',
            r'\b(React|Angular|Vue|Node\.js|Django|Flask|Spring|\.NET)\b',
            r'\b(AWS|Azure|GCP|Docker|Kubernetes|CI/CD|Git)\b',
            r'\b(SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Redis)\b',
            r'\b(Machine Learning|AI|Deep Learning|NLP|Computer Vision)\b',
        ]

        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)

        return list(set(skills))  # Remove duplicates

    def parse_job_description(self) -> JobDescription:
        """Main parsing function - returns structured job description."""
        # Read file
        self.read_file()

        # Extract metadata
        metadata = self.extract_job_metadata()

        # Identify sections
        sections = self.identify_sections()

        # Parse requirements
        requirements = []
        if 'requirements' in sections:
            requirements = self.parse_list_items(sections['requirements'])

        # Parse qualifications
        qualifications = []
        if 'qualifications' in sections:
            qualifications = self.parse_list_items(sections['qualifications'])

        # Parse responsibilities
        responsibilities = []
        if 'responsibilities' in sections:
            responsibilities = self.parse_list_items(sections['responsibilities'])

        # Extract description (usually early content)
        description = ' '.join(sections.get('description', [])[:3])  # First 3 paragraphs

        # Extract skills using LLM
        required_skills = self.extract_skills_with_llm(self.text)

        # Create JobDescription object
        job_data = JobDescription(
            title=metadata['title'],
            company=metadata['company'],
            location=metadata.get('location'),
            description=description,
            requirements=requirements,
            required_skills=required_skills,
            qualifications=qualifications,
            responsibilities=responsibilities,
            raw_text=self.text
        )

        return job_data


def parse_job_description(txt_path: str) -> JobDescription:
    """
    Parse a job description text file and return structured data.

    Args:
        txt_path: Path to the job description text file

    Returns:
        JobDescription object with structured information
    """
    parser = JobParser(txt_path)
    return parser.parse_job_description()
