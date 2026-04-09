"""Parsers for CV and job description documents."""

from .cv_parser import parse_cv, CVParser
from .job_parser import parse_job_description, JobParser

__all__ = ['parse_cv', 'CVParser', 'parse_job_description', 'JobParser']
