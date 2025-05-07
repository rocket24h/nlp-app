# Wikipedia Knowledge Graph Assistant

This project is designed to fetch Wikipedia pages, extract their content, and build a knowledge graph using large language models (LLMs) and Neo4j. The system enables users to query the knowledge graph in natural language, visualize entity relationships, and summarize content.

---

## Features

- Automatically retrieves and stores content from selected Wikipedia pages.
- Builds a knowledge graph using **LlamaIndex** and **Neo4j**.
- Integrates with **Google Gemini (GoogleGenAI)** for triplet extraction and question answering.
- Embeds content using **HuggingFace Sentence Transformers**.
- Provides a user-friendly interface for querying and visualizing the knowledge graph.

---

## Getting Started

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/rocket24h/nlp-app.git
cd nlp-app
```

---

### 2. Set Up the Environment

You can set up the environment using either `conda` or `pip`. The Python version must be 3.11.

#### Using Conda

Create a new environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate nlp_app
```

#### Using Pip

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

The application requires several environment variables to function correctly. These variables are defined in the `.env` file. To simplify the setup process, run the `init_script.py` script, which will guide you through creating the `.env` file interactively.

Run the following command:

```bash
python init_script.py
```

The script will prompt you to enter the necessary values for the environment variables. Below is a description of the required variables:

#### Environment Variables

| Variable Name       | Description                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| `GOOGLE_API_KEY`    | Your Google API key for accessing Google Gemini services.                            |
| `DOCS_PATH`         | Directory to store downloaded Wikipedia documents.                                   |
| `PERSIST_PATH`      | Directory to store the local graph database.                                         |
| `EMBEDDING_MODEL`   | HuggingFace embedding model to use (e.g., `sentence-transformers/all-MiniLM-L6-v2`). |
| `LLM_MODEL`         | LLM model to use (e.g., `gemini-2.0-flash`).                                         |
| `LLM_TEMPERATURE`   | Temperature setting for the LLM responses.                                           |
| `LLM_MAX_TOKENS`    | Maximum number of tokens for LLM responses.                                          |
| `SEARCH_DEPTH`      | Depth of traversal for graph queries.                                                |
| `CHAT_HISTORY_PATH` | Path to store chat history.                                                          |
| `NEO4J_URL`         | URL of the Neo4j database (e.g., `bolt://localhost:7687`).                           |
| `NEO4J_USERNAME`    | Username for the Neo4j database.                                                     |
| `NEO4J_PASSWORD`    | Password for the Neo4j database.                                                     |
| `NEO4J_DATABASE`    | Name of the Neo4j database.                                                          |

NOTE: For NEO4J environment variables, create a free instance of Neo4J Aura Database.
Fill in the variables based on the Database instance you created.

---

### 4. Run the Application

Once the `.env` file is created, you can start the application using Streamlit:

```bash
streamlit run app.py
```

This will launch the application in your default web browser.
