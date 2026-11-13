from src.generation.rag_pipeline import analyze_skills_gap


def main():
    user_skills = ["Python", "SQL", "Pandas"]

    target_role = "Data Analyst"

    analysis = analyze_skills_gap(
        user_skills=user_skills,
        target_role=target_role,
        top_k=5,
    )

    print("\nSkills Gap Analysis:")
    print(analysis)


if __name__ == "__main__":
    main()
    