"""
Configuration settings for the cover letter generator.
"""

import os
from pathlib import Path


# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CV_DIR = DATA_DIR / "cv"
JOBS_DIR = DATA_DIR / "jobs"
OUTPUT_DIR = BASE_DIR / "output"
COVER_LETTERS_DIR = OUTPUT_DIR / "cover_letters"
PROMPTS_DIR = BASE_DIR / "prompts"

# OpenAI settings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
EMBEDDING_MODEL = "text-embedding-3-large"

# FAISS settings
FAISS_K_RESULTS = 5  # Number of results to retrieve

# Skill matching settings
SKILL_SIMILARITY_THRESHOLD = 0.8  # Minimum similarity for semantic matching

# Cover letter settings
TONES = ["professional", "creative", "technical"]
DEFAULT_TONE = "professional"
MIN_LETTER_LENGTH = 250  # Minimum words
MAX_LETTER_LENGTH = 500  # Maximum words

# Agent settings
MAX_AGENT_ITERATIONS = 10
STREAM_MODE = "values"

# Ensure directories exist
CV_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
COVER_LETTERS_DIR.mkdir(parents=True, exist_ok=True)
