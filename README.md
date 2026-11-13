# Skills Gap Intelligence — RAG System

## Project Overview

Skills Gap Intelligence is a Retrieval-Augmented Generation (RAG) system designed to analyze the gap between a user's current skills and the requirements of a target AI, Machine Learning, or Data-related job role.

The system processes job descriptions, splits them into chunks, generates vector embeddings, and stores them in ChromaDB. It retrieves relevant job information using semantic search and uses a Large Language Model (LLM) to generate personalized skills gap analysis and learning recommendations.

## Technology Stack

- **Python** — Core programming language.
- **uv** — Dependency and environment management.
- **Sentence Transformers** — Text embeddings.
- **ChromaDB** — Vector database.
- **LangChain Text Splitters** — Document chunking.
- **Pandas and NumPy** — Data processing.
- **Google Gemini API** — LLM integration and response generation.
- **Streamlit** — Web application interface.

## Project Structure

```text
skills-gap-analysis/
├── app.py
├── data/
│   ├── processed/
│   ├── raw/
│   └── chroma_db/
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── check_chunks.py
│   ├── embeddings/
│   │   └── embedding_model.py
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── evaluation/
│   │   └── retrieval_evaluation.py
│   └── generation/
│       ├── __init__.py
│       ├── llm_client.py
│       └── rag_pipeline.py
├── tests/
│   ├── __init__.py
│   ├── test_gemini.py
│   ├── test_rag_pipeline.py
│   └── test_retrieval.py
├── .env
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/MN294/skills-gap-analysis.git
cd skills-gap-analysis
```

### 2. Install Dependencies

Install `uv`, then run:

```bash
uv sync
```

### 3. Configure Gemini API Key

Create a `.env` file in the project root directory.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

The API key is loaded using `python-dotenv`.

**Important:** Never commit your actual API key to GitHub.

### 4. Prepare the Dataset

Ensure the required dataset is available at:

```text
data/processed/selected_30_jobs.csv
```

## Running the Pipeline

### 1. Generate Document Chunks

```bash
uv run python -m src.chunk
```

This processes job descriptions and saves the generated chunks.

### 2. Build the Vector Database

```bash
uv run python -m src.retrieval.vector_store
```

This generates embeddings using `sentence-transformers/all-MiniLM-L6-v2` and stores them in a persistent ChromaDB collection.

### 3. Perform Retrieval

The retrieval implementation is available in:

```text
src/retrieval/retriever.py
```

Run:

```bash
uv run python -m src.retrieval.retriever
```

The retriever supports semantic search over stored job description chunks.

## LLM Integration and Generation Pipeline

The generation module is responsible for connecting the retrieval system with the Gemini API.

### 1. LLM Client

File:

```text
src/generation/llm_client.py
```

Responsibilities:

- Load the Gemini API key from environment variables.
- Initialize the Gemini client.
- Send prompts to the LLM.
- Return the generated response.

### 2. Prompt Engineering

The system builds a structured prompt containing:

- Target job role.
- User's current skills.
- Retrieved job description context.
- Skills gap analysis instructions.
- Learning roadmap requirements.

The prompt instructs the LLM to base its analysis on retrieved context and avoid inventing job requirements.

### 3. RAG Generation Pipeline

File:

```text
src/generation/rag_pipeline.py
```

The pipeline performs three main steps:

1. Retrieve relevant job description chunks from ChromaDB.
2. Build a structured analysis prompt using the retrieved context and user skills.
3. Send the prompt to Gemini and return the generated analysis.

### 4. Skills Gap Analysis

The system compares the user's current skills with the skills required by the target job role.

The generated analysis includes:

- Required job skills.
- Existing user skills.
- Missing skills.
- Recommended skills to learn.
- A prioritized learning roadmap.

## Streamlit Web Application

The project includes a Streamlit interface in:

```text
app.py
```

Run the application using:

```bash
uv run streamlit run app.py
```

The web application allows users to:

1. Enter their current skills.
2. Enter their target job role.
3. Request a personalized skills gap analysis.
4. View the generated recommendations.

The application also includes input validation, a loading indicator, and error handling for unavailable AI services.

## Testing

The project includes the following test files inside the `tests/` directory.

### 1. Retrieval Test

```bash
uv run python -m tests.test_retrieval
```

Tests semantic retrieval from the ChromaDB collection.

### 2. Gemini Integration Test

```bash
uv run python -m tests.test_gemini
```

Tests communication with the Gemini API.

### 3. RAG Pipeline Test

```bash
uv run python -m tests.test_rag_pipeline
```

Tests the complete retrieval and generation workflow.

### 4. Syntax Check

```bash
uv run python -m compileall src app.py
uv run python -m compileall tests
```

Checks Python files for syntax errors.

### Testing Notes

The retrieval and syntax checks were executed successfully.

The Streamlit interface was tested for input validation and error handling.

Gemini generation may temporarily return a `503 UNAVAILABLE` response during periods of high service demand. This is an external API availability issue and does not necessarily indicate a problem in the application code.

## Evaluation

The retrieval evaluation implementation and manually labeled queries are located in:

```text
src/evaluation/
data/processed/evaluation_queries.json
```

Run the retrieval evaluation using:

```bash
uv run python -m src.evaluation.retrieval_evaluation
```

## Dataset Source

**Dataset:** Thousand ML Jobs in USA

**Source:** https://www.kaggle.com/datasets/ivankmk/thousand-ml-jobs-in-usa

Check the dataset's license and redistribution conditions before sharing copies of the data.

## Notes

- The ChromaDB database is generated locally and is not included in the repository.
- The Sentence Transformers model is downloaded automatically when first needed.
- A valid Gemini API key is required for LLM generation.
- Run commands from the project root directory.
- Keep `.env` and all secret credentials out of version control.
