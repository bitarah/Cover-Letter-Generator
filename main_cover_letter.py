#!/usr/bin/env python3
"""
Main Cover Letter Generator CLI Application

Generates personalized cover letters from CV and job descriptions using AI agents.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cover_letter_generator.parsers.cv_parser import parse_cv
from cover_letter_generator.parsers.job_parser import parse_job_description
from cover_letter_generator.vector_store.cv_vectorstore import create_cv_embeddings
from cover_letter_generator.analysis.skill_matcher import calculate_skill_match
from cover_letter_generator.agent.cover_letter_agent import create_cover_letter_agent, generate_with_agent
from cover_letter_generator.models import CoverLetter
from interactive_refinement import interactive_refinement_loop


def print_banner():
    """Print application banner."""
    print("=" * 70)
    print(" " * 15 + "AI-Powered Cover Letter Generator")
    print("=" * 70)
    print()


def get_file_path(prompt: str, extension: str) -> str:
    """
    Get and validate file path from user.

    Args:
        prompt: Prompt to display to user
        extension: Expected file extension

    Returns:
        Validated file path
    """
    while True:
        path = input(prompt).strip()

        # Remove quotes if user wrapped path in quotes
        path = path.strip('"').strip("'")

        if not path:
            print("Error: Path cannot be empty")
            continue

        path_obj = Path(path)

        if not path_obj.exists():
            print(f"Error: File not found: {path}")
            continue

        if path_obj.suffix.lower() != extension:
            print(f"Error: Expected {extension} file, got {path_obj.suffix}")
            continue

        return str(path_obj)


def save_cover_letter(letter: str, tone: str, job_title: str, company: str, output_dir: Path):
    """
    Save cover letter to file.

    Args:
        letter: Cover letter content
        tone: Tone used
        job_title: Job title
        company: Company name
        output_dir: Output directory path
    """
    # Create safe filename
    safe_company = "".join(c for c in company if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_company = safe_company.replace(' ', '_')

    filename = f"{tone}_{safe_company}_{job_title.replace(' ', '_')}.txt"
    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(letter)

    return filepath


def main():
    """Main application entry point."""
    print_banner()

    # Step 1: Get input files
    print("[Step 1/7] Input Files")
    print("-" * 70)

    cv_path = get_file_path("Enter path to your CV (.docx): ", ".docx")
    job_path = get_file_path("Enter path to job description (.txt): ", ".txt")

    print()

    # Step 2: Parse CV
    print("[Step 2/7] Parsing CV...")
    print("-" * 70)

    try:
        cv_data = parse_cv(cv_path)
        print(f"✓ Successfully parsed CV")
        print(f"  - Name: {cv_data.personal_info.name or 'Not found'}")
        print(f"  - Experience entries: {len(cv_data.experience)}")
        print(f"  - Education entries: {len(cv_data.education)}")
        print(f"  - Skills: {len(cv_data.skills)}")
        print(f"  - Projects: {len(cv_data.projects)}")
    except Exception as e:
        print(f"✗ Error parsing CV: {e}")
        return 1

    print()

    # Step 3: Parse job description
    print("[Step 3/7] Parsing Job Description...")
    print("-" * 70)

    try:
        job_data = parse_job_description(job_path)
        print(f"✓ Successfully parsed job description")
        print(f"  - Title: {job_data.title}")
        print(f"  - Company: {job_data.company}")
        print(f"  - Requirements: {len(job_data.requirements)}")
        print(f"  - Required skills: {len(job_data.required_skills)}")
    except Exception as e:
        print(f"✗ Error parsing job description: {e}")
        return 1

    print()

    # Step 4: Create vector store
    print("[Step 4/7] Creating CV Embeddings...")
    print("-" * 70)

    try:
        vector_store = create_cv_embeddings(cv_data)
        print(f"✓ Created vector store with {vector_store.index.ntotal} CV sections")
    except Exception as e:
        print(f"✗ Error creating vector store: {e}")
        return 1

    print()

    # Step 5: Analyze skills
    print("[Step 5/7] Analyzing Skill Match...")
    print("-" * 70)

    try:
        skill_match = calculate_skill_match(cv_data.skills, job_data.required_skills)
        print(f"✓ Skill Match: {skill_match.match_percentage}%")
        print(f"  - CV Skills: {skill_match.cv_skills_count}")
        print(f"  - Job Skills: {skill_match.job_skills_count}")
        print(f"  - Matched: {len(skill_match.matched_skills)}")
        print(f"  - Missing: {len(skill_match.missing_skills)}")

        if skill_match.matched_skills:
            print(f"\n  Top Matched Skills:")
            for skill in skill_match.matched_skills[:5]:
                print(f"    ✓ {skill}")

        if skill_match.missing_skills:
            print(f"\n  Missing Skills:")
            for skill in skill_match.missing_skills[:5]:
                print(f"    ✗ {skill}")
    except Exception as e:
        print(f"✗ Error analyzing skills: {e}")
        return 1

    print()

    # Step 6: Create agent
    print("[Step 6/7] Initializing AI Agent...")
    print("-" * 70)

    try:
        agent = create_cover_letter_agent(cv_data, job_data, skill_match, vector_store)
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        return 1

    print()

    # Step 7: Generate cover letters
    print("[Step 7/7] Generating Cover Letters...")
    print("-" * 70)

    tones = ['professional', 'creative', 'technical']
    generated_letters = {}

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('output/cover_letters') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    for tone in tones:
        print(f"\nGenerating {tone.upper()} version...")

        try:
            letter = generate_with_agent(
                agent,
                tone,
                cv_data,
                job_data,
                skill_match,
                vector_store,
                thread_id=f"thread-{tone}"
            )

            generated_letters[tone] = letter

            # Save to file
            filepath = save_cover_letter(letter, tone, job_data.title, job_data.company, output_dir)
            print(f"✓ Saved to: {filepath}")

        except Exception as e:
            print(f"✗ Error generating {tone} letter: {e}")

    print()
    print("=" * 70)
    print(" " * 20 + "Generation Complete!")
    print("=" * 70)
    print(f"\nAll cover letters saved to: {output_dir}")
    print()

    # Step 8: Interactive refinement
    if generated_letters:
        print("Would you like to refine any of the generated letters?")
        print("\nAvailable versions:")
        for i, tone in enumerate(tones, 1):
            if tone in generated_letters:
                print(f"  {i}. {tone.capitalize()}")

        print("  0. Skip refinement\n")

        try:
            choice = int(input("Select a version (0-3): "))

            if 1 <= choice <= 3:
                selected_tone = tones[choice - 1]
                if selected_tone in generated_letters:
                    print(f"\n{'='*70}")
                    print(f" Refining {selected_tone.upper()} Version")
                    print(f"{'='*70}\n")
                    print(generated_letters[selected_tone])
                    print(f"\n{'='*70}\n")

                    # Run interactive refinement
                    interactive_refinement_loop(
                        agent,
                        cv_data,
                        job_data,
                        skill_match,
                        vector_store,
                        thread_id=f"thread-{selected_tone}-refine"
                    )

                    print("\n✓ Refinement complete!")

        except (ValueError, KeyboardInterrupt):
            print("\nSkipping refinement.")

    print("\nThank you for using the AI-Powered Cover Letter Generator!")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
