"""
Cover Letter Agent - LangGraph ReAct agent for generating cover letters.
"""

import os
from typing import Dict, Any
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ..models import CVData, JobDescription, SkillMatch
from ..vector_store.cv_vectorstore import create_cv_retriever_tool
from .tools import analyze_skill_match, get_job_details, get_cv_summary, generate_cover_letter_content
from .prompts import get_system_prompt

# Enable verbose mode by default
VERBOSE_MODE = os.getenv("AGENT_VERBOSE", "true").lower() == "true"


def create_cover_letter_agent(cv_data: CVData, job_data: JobDescription, skill_match: SkillMatch, vector_store):
    """
    Create a LangGraph agent for cover letter generation.

    Args:
        cv_data: Structured CV data
        job_data: Structured job description
        skill_match: Skill matching analysis results
        vector_store: FAISS vector store for CV search

    Returns:
        LangGraph ReAct agent
    """
    # Create LLM with verbose mode enabled
    # Lower temperature for more factual, grounded responses (avoid hallucinations)
    llm = ChatOpenAI(model='gpt-4o', temperature=0.3, verbose=True)

    # Import tool creation functions
    from .tools import create_bound_tools

    # Create tools with context bound to them
    tools = create_bound_tools(cv_data, job_data, skill_match, vector_store)

    # Create system prompt
    system_prompt = get_system_prompt()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])

    # Create checkpointer for conversation memory
    checkpointer = InMemorySaver()

    # Create agent
    agent = create_react_agent(
        llm,
        tools,
        checkpointer=checkpointer,
        prompt=prompt
    )

    return agent


def generate_with_agent(
    agent,
    tone: str,
    cv_data: CVData,
    job_data: JobDescription,
    skill_match: SkillMatch,
    vector_store,
    thread_id: str = "default"
) -> str:
    """
    Generate a cover letter using the agent.

    Args:
        agent: LangGraph agent instance
        tone: Tone for the cover letter ('professional', 'creative', or 'technical')
        cv_data: Structured CV data
        job_data: Structured job description
        skill_match: Skill matching results
        vector_store: FAISS vector store
        thread_id: Thread ID for conversation memory

    Returns:
        Generated cover letter content
    """
    # Configuration for the agent (no need to pass context since tools are bound)
    config = {
        "configurable": {"thread_id": thread_id}
    }

    # Create the user message
    user_message = f"""Please generate a {tone} cover letter for the {job_data.title} position at {job_data.company}.

First, analyze the candidate's background and the job requirements, then create a tailored cover letter."""

    if VERBOSE_MODE:
        print(f"\n{'='*70}")
        print(f"[AGENT VERBOSE MODE]")
        print(f"{'='*70}")
        print(f"User Message: {user_message}")
        print(f"Thread ID: {thread_id}")
        print(f"{'='*70}\n")

    # Invoke agent with stream to see intermediate steps
    try:
        if VERBOSE_MODE:
            print(f"\n[Agent] Starting generation process...\n")

        for event in agent.stream(
            {"messages": [("user", user_message)]},
            config=config,
            stream_mode="updates"
        ):
            if VERBOSE_MODE:
                # Print agent thoughts and tool calls
                if "agent" in event:
                    agent_msg = event["agent"]["messages"][-1]
                    if hasattr(agent_msg, 'content') and agent_msg.content:
                        print(f"\n[Agent Thought]: {agent_msg.content}\n")
                    if hasattr(agent_msg, 'tool_calls') and agent_msg.tool_calls:
                        for tool_call in agent_msg.tool_calls:
                            print(f"[Tool Call]: {tool_call['name']}")
                            if 'args' in tool_call:
                                print(f"  Args: {tool_call['args']}")

                if "tools" in event:
                    tool_msg = event["tools"]["messages"][-1]
                    if hasattr(tool_msg, 'content'):
                        content_preview = tool_msg.content[:200] if len(tool_msg.content) > 200 else tool_msg.content
                        print(f"[Tool Response]: {content_preview}...")
                        print()

        # Get final response
        response = agent.invoke(
            {"messages": [("user", user_message)]},
            config=config
        )

        # Extract the final response
        if response and 'messages' in response:
            final_message = response['messages'][-1]
            if hasattr(final_message, 'content'):
                if VERBOSE_MODE:
                    print(f"\n{'='*70}")
                    print(f"[AGENT] Generation complete ({len(final_message.content)} chars)")
                    print(f"{'='*70}\n")
                return final_message.content

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()

    return "Error: Could not generate cover letter"


def stream_agent_response(
    agent,
    user_message: str,
    cv_data: CVData,
    job_data: JobDescription,
    skill_match: SkillMatch,
    vector_store,
    thread_id: str = "default"
):
    """
    Stream agent responses for interactive refinement.

    Args:
        agent: LangGraph agent instance
        user_message: User's message
        cv_data: Structured CV data
        job_data: Structured job description
        skill_match: Skill matching results
        vector_store: FAISS vector store
        thread_id: Thread ID for conversation memory

    Yields:
        Streamed response chunks
    """
    # Configuration (no need to pass context since tools are bound)
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Stream response with updates mode to see intermediate steps
    try:
        accumulated_content = ""

        for event in agent.stream(
            {"messages": [("user", user_message)]},
            config=config,
            stream_mode="updates"
        ):
            if VERBOSE_MODE:
                # Print agent thoughts and tool calls
                if "agent" in event:
                    agent_msg = event["agent"]["messages"][-1]
                    if hasattr(agent_msg, 'content') and agent_msg.content:
                        print(f"\n\n[Agent Thought]: {agent_msg.content}\n")
                    if hasattr(agent_msg, 'tool_calls') and agent_msg.tool_calls:
                        for tool_call in agent_msg.tool_calls:
                            print(f"\n[Calling Tool]: {tool_call['name']}")
                            if 'args' in tool_call:
                                print(f"  Args: {tool_call['args']}\n")

                if "tools" in event:
                    tool_msg = event["tools"]["messages"][-1]
                    if hasattr(tool_msg, 'content'):
                        content_preview = tool_msg.content[:200] if len(tool_msg.content) > 200 else tool_msg.content
                        print(f"[Tool Response]: {content_preview}...\n\n")

            # Extract and yield final content
            if "agent" in event:
                messages = event["agent"].get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        # Only yield if it's actual content (not tool calls)
                        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                            accumulated_content = last_message.content

        # Yield the final accumulated content
        if accumulated_content:
            yield accumulated_content

    except Exception as e:
        # If streaming fails, fall back to invoke
        print(f"\n\n[ERROR] Streaming failed: {e}\nFalling back to invoke...\n")
        import traceback
        traceback.print_exc()

        response = agent.invoke(
            {"messages": [("user", user_message)]},
            config=config
        )
        if response and 'messages' in response:
            final_message = response['messages'][-1]
            if hasattr(final_message, 'content'):
                yield final_message.content
