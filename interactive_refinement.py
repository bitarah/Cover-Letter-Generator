"""
Interactive Refinement - Chat interface for refining generated cover letters.
"""

from cover_letter_generator.models import CVData, JobDescription, SkillMatch
from cover_letter_generator.agent.cover_letter_agent import stream_agent_response


def interactive_refinement_loop(
    agent,
    cv_data: CVData,
    job_data: JobDescription,
    skill_match: SkillMatch,
    vector_store,
    thread_id: str = "refinement"
):
    """
    Interactive chat loop for refining cover letters.

    Args:
        agent: LangGraph agent instance
        cv_data: Structured CV data
        job_data: Structured job description
        skill_match: Skill matching results
        vector_store: FAISS vector store
        thread_id: Thread ID for conversation memory
    """
    print("=" * 70)
    print(" " * 15 + "Interactive Refinement Mode")
    print("=" * 70)
    print()
    print("Chat with the AI agent to refine your cover letter.")
    print("You can ask to:")
    print("  - Make it more concise or detailed")
    print("  - Add or emphasize specific skills or experiences")
    print("  - Change the tone or style")
    print("  - Fix any issues or update specific sections")
    print()
    print("Commands:")
    print("  'done', 'exit', or 'quit' - Finish refinement")
    print("  'show' - Show the current version")
    print()
    print("=" * 70)
    print()

    current_letter = None

    while True:
        try:
            user_input = input("\n\033[1mYou:\033[0m ").strip()

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ['done', 'exit', 'quit', 'q']:
                print("\n✓ Exiting refinement mode...")
                break

            # Handle show command
            if user_input.lower() == 'show':
                if current_letter:
                    print(f"\n\033[1mCurrent Letter:\033[0m\n")
                    print(current_letter)
                else:
                    print("\nNo letter to show yet. Start by requesting a refinement.")
                continue

            # Stream agent response
            print("\n\033[1mAssistant:\033[0m ", end='', flush=True)

            response_text = ""
            last_content = ""

            for content in stream_agent_response(
                agent,
                user_input,
                cv_data,
                job_data,
                skill_match,
                vector_store,
                thread_id
            ):
                # Only print new content (not the full accumulated content)
                if content != last_content:
                    new_text = content[len(last_content):]
                    print(new_text, end='', flush=True)
                    response_text = content
                    last_content = content

            print()  # New line after streaming

            # Update current letter if response contains a full letter
            if len(response_text) > 200:  # Heuristic: full letters are longer
                current_letter = response_text

        except KeyboardInterrupt:
            print("\n\n✓ Refinement interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n\n✗ Error: {e}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    print("This module should be imported and used with the main application.")
    print("Run: python main_cover_letter.py")
