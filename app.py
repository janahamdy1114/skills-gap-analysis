import streamlit as st

from src.generation.rag_pipeline import analyze_skills_gap


# Page Configuration
st.set_page_config(
    page_title="Skills Gap Intelligence",
    page_icon="🎯",
    layout="wide",
)


# Main Title
st.title("🎯 Skills Gap Intelligence")

st.write(
    "Analyze your current skills and discover what you need "
    "to learn for your target career."
)


# User Inputs
st.header("Enter Your Information")

user_skills_input = st.text_input(
    "Your Current Skills",
    placeholder="Example: Python, SQL, Pandas",
)

target_role = st.text_input(
    "Target Job Role",
    placeholder="Example: Data Analyst",
)



# Analyze Button
if st.button("Analyze My Skills", type="primary"):

    if not user_skills_input.strip() or not target_role.strip():
        st.warning("Please enter your current skills and target job role.")

    else:
        user_skills = [
            skill.strip()
            for skill in user_skills_input.split(",")
            if skill.strip()
        ]

        with st.spinner("Analyzing your skills... Please wait."):

            try:
                analysis = analyze_skills_gap(
                    user_skills=user_skills,
                    target_role=target_role.strip(),
                    top_k=5,
                )

                st.success("Analysis completed successfully!")

                st.header("Your Skills Gap Analysis")
                st.markdown(analysis)

            except Exception as e:
                error_message = str(e)

                if "503" in error_message or "UNAVAILABLE" in error_message:
                    st.error(
                        "The AI service is temporarily unavailable. "
                        "Please try again later."
                    )

                else:
                    st.error(
                        "Something went wrong. Please try again."
                    )



# Sidebar
with st.sidebar:
    st.title("About the Project")

    st.write(
        "Skills Gap Intelligence uses RAG and Generative AI "
        "to analyze the gap between your current skills "
        "and your target job role."
    )
    st.divider()

    st.subheader("How It Works")

    st.write("1. Enter your current skills.")
    st.write("2. Select your target job role.")
    st.write("3. Get your personalized skills gap analysis.")

    st.divider()

    st.subheader("Technologies")

    st.write("- Python")
    st.write("- Streamlit")
    st.write("- ChromaDB")
    st.write("- Sentence Transformers")
    st.write("- Gemini API")