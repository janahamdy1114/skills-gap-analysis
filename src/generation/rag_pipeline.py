

from src.retrieval.retriever import (
    load_collection,
    load_retriever_model,
    semantic_search,
)

from src.generation.llm_client import generate_response



def retrieve_job_context(target_role, top_k=5):
    """Retrieve relevant job description chunks for a target role."""

    collection = load_collection()
    model = load_retriever_model()

    results = semantic_search(
        query=f"What skills and qualifications are required for a {target_role}?",
        collection=collection,
        model=model,
        top_k=top_k,
    )

    return results


def build_analysis_prompt(user_skills, target_role, retrieved_chunks):
    """Build a prompt for analyzing the user's skills gap."""

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    prompt = f"""
    You are a professional career advisor and skills gap analyst.

    Target Job Role:
    {target_role}

    User's Current Skills:
    {", ".join(user_skills)}

    Retrieved Job Market Context:
    {context}

    Your tasks:
    1. Identify the skills required for the target job role.
    2. Compare the required skills with the user's current skills.
    3. Identify the missing skills (skills gap).
    4. Recommend which skills the user should learn.
    5. Suggest a learning roadmap in priority order.

    Important:
    - Base your analysis on the retrieved context.
    - Do not invent job requirements.
    - Clearly distinguish existing skills from missing skills.
    - If the retrieved context is insufficient, mention that.

    Provide a clear and structured analysis.
    """

    return prompt





def analyze_skills_gap(user_skills, target_role, top_k=5):
    """Analyze the gap between user's skills and target job requirements."""

    # Step 1: Retrieve relevant job context
    retrieved_chunks = retrieve_job_context(
        target_role=target_role,
        top_k=top_k,
    )

    # Step 2: Build the analysis prompt
    prompt = build_analysis_prompt(
        user_skills=user_skills,
        target_role=target_role,
        retrieved_chunks=retrieved_chunks,
    )

    # Step 3: Generate the analysis using Gemini
    analysis = generate_response(prompt)

    return analysis