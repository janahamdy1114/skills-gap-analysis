# Skills Gap Intelligence — RAG System

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for analyzing skills and requirements in AI and Machine Learning job descriptions.

The system processes job descriptions, splits them into chunks, generates vector embeddings, stores them in ChromaDB, and retrieves relevant information through semantic search and metadata filtering.

## Technology Stack

- Python
- uv — dependency and environment management
- Sentence Transformers — text embeddings
- ChromaDB — vector database
- LangChain Text Splitters — document chunking
- NumPy and Pandas — data processing

## Project Structure
```
skills_gap/
├── data/
│   ├── processed/
│   │   ├── selected_30_jobs.csv
│   │   ├── chunks.jsonl
│   │   └── evaluation_queries.json  
│   └── raw/
├── src/
│   ├── __init__.py
│   ├── ingest.py
│   ├── chunk.py
│   ├── check_chunks.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_model.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   └── evaluation/
│       ├── __init__.py
│       └── retrieval_evaluation.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

### 2. Install dependencies

Make sure to install uv first, then run:

```bash
uv sync
```

### 3. Prepare the data

Ensure the required dataset is available at:

```text
data/processed/selected_30_jobs.csv
```


## Running the Pipeline

### 1. Generate document chunks

```bash
uv run python -m src.chunk
```

This processes the job descriptions and saves the generated chunks to:

```text
data/processed/chunks.jsonl
```

### 2. Build the vector database

```bash
uv run python -m src.retrieval.vector_store
```

This generates embeddings using `all-MiniLM-L6-v2` and stores them in a persistent ChromaDB collection. The ChromaDB folder should be generated automatically in data folder after this step.

### 3. Perform retrieval

Use the functions in:

```text
src/retrieval/retriever.py
```
by running
```
 uv run python -m src.retrieval.retriever
```
The retriever supports semantic search and metadata-filtered retrieval.

## Evaluation

The retrieval evaluation implementation and manually labeled queries are located in:

```text
src/evaluation/
data/processed/evaluation_queries.json
```

See the evaluation module for the available metrics and execution instructions.
You can check retrieval evaluation per query by running:
```
uv run python -m src.evaluation.retrieval_evaluation
```

## Dataset Source

Dataset: Thousand ML Jobs in USA  
Source: https://www.kaggle.com/datasets/ivankmk/thousand-ml-jobs-in-usa

Check the dataset's license and redistribution conditions before sharing copies of the data.

## Notes

- The ChromaDB database is generated locally and is not included in the repository.
- The Sentence Transformers model is downloaded automatically when first needed.
- Run commands from the project root directory.
