"""
CV Vector Store - Creates FAISS vector store for semantic search over CV content.
"""

from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_core.documents import Document

from ..models import CVData


def create_cv_documents(cv_data: CVData) -> List[Document]:
    """
    Convert CV data into documents for vector store.

    Each experience, project, and section becomes a separate document
    for more granular retrieval.
    """
    documents = []

    # Add summary if available
    if cv_data.summary:
        documents.append(Document(
            page_content=cv_data.summary,
            metadata={'type': 'summary'}
        ))

    # Add each experience entry
    for i, exp in enumerate(cv_data.experience):
        content = f"{exp.role} at {exp.company}\n"
        if exp.start_date:
            content += f"Duration: {exp.start_date} - {exp.end_date or 'Present'}\n"
        content += f"{exp.description}\n"
        if exp.bullets:
            content += "Key achievements:\n" + "\n".join(f"- {bullet}" for bullet in exp.bullets)

        documents.append(Document(
            page_content=content,
            metadata={
                'type': 'experience',
                'company': exp.company,
                'role': exp.role,
                'index': i
            }
        ))

    # Add each education entry
    for i, edu in enumerate(cv_data.education):
        content = f"{edu.degree}"
        if edu.field:
            content += f" in {edu.field}"
        content += f" from {edu.institution}\n"
        if edu.start_date:
            content += f"Years: {edu.start_date} - {edu.end_date or 'Present'}\n"
        if edu.description:
            content += f"{edu.description}"

        documents.append(Document(
            page_content=content,
            metadata={
                'type': 'education',
                'institution': edu.institution,
                'degree': edu.degree,
                'index': i
            }
        ))

    # Add each project
    for i, project in enumerate(cv_data.projects):
        content = f"Project: {project.name}\n{project.description}\n"
        if project.technologies:
            content += f"Technologies: {', '.join(project.technologies)}\n"
        if project.highlights:
            content += "Highlights:\n" + "\n".join(f"- {h}" for h in project.highlights)

        documents.append(Document(
            page_content=content,
            metadata={
                'type': 'project',
                'name': project.name,
                'index': i
            }
        ))

    # Add skills as a single document
    if cv_data.skills:
        content = f"Skills: {', '.join(cv_data.skills)}"
        documents.append(Document(
            page_content=content,
            metadata={'type': 'skills'}
        ))

    # Add certifications if available
    if cv_data.certifications:
        content = f"Certifications: {', '.join(cv_data.certifications)}"
        documents.append(Document(
            page_content=content,
            metadata={'type': 'certifications'}
        ))

    return documents


def create_cv_embeddings(cv_data: CVData) -> FAISS:
    """
    Create FAISS vector store from CV data.

    Args:
        cv_data: Structured CV data

    Returns:
        FAISS vector store with CV embeddings
    """
    # Convert CV to documents
    documents = create_cv_documents(cv_data)

    # Create embeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    # Create FAISS vector store
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vector_store


def create_cv_retriever_tool(vector_store: FAISS):
    """
    Create a retriever tool for the agent to search the CV.

    Args:
        vector_store: FAISS vector store with CV embeddings

    Returns:
        LangChain retriever tool
    """
    retriever = vector_store.as_retriever(
        search_kwargs={'k': 5}  # Return top 5 most relevant sections
    )

    tool = create_retriever_tool(
        retriever,
        name='search_cv',
        description='''Search the candidate's CV for relevant information.
        Use this to find specific experiences, skills, projects, or education that match the job requirements.
        Input should be a search query about what you're looking for (e.g., "Python experience", "leadership skills", "machine learning projects").
        '''
    )

    return tool
