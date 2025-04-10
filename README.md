# 🧠 Wikipedia Knowledge Graph Assistant

This project fetches Wikipedia pages, extracts their content, and builds a knowledge graph using LLMs and Neo4j. It is capable of answering natural language queries using information stored in the graph, visualizing entity relationships, and summarizing content.

---

## 🚀 Features

- Automatically retrieves and stores content from selected Wikipedia pages.
- Uses **LlamaIndex** with **Neo4j** as a graph store to build a **Knowledge Graph (KG)**.
- Integrates with **Google Gemini (GoogleGenAI)** for triplet extraction and question answering.
- Embeds content using **HuggingFace Sentence Transformers**.
- Displays progress and logging with `tqdm` and Python `logging`.

---

## 📁 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Set up your environment

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root based on `.env.template`:

```
# Google Gemini / LLM Settings
LLM_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1500

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Neo4j Graph Database
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Directory to store downloaded Wikipedia docs
DOCS_PATH=./wiki_docs
```

> ✅ You must have access to Google Gemini API and a running Neo4j database.

---

## 🛠 Usage

### 1. Select the Wikipedia pages

Edit the `pages_to_get` list inside `main.py` to choose the pages to fetch:

```python
pages_to_get = [
    "Left 4 Dead",
    "Machine learning",
    "Python (programming language)",
]
```

### 2. Run the script

```bash
python main.py
```

- Summaries of the pages will be saved to `DOCS_PATH`.
- The script will generate triplets and upsert them into your Neo4j graph.
- Finally, it will perform a sample query and print the result.

---

## 🔍 Query Example

A sample query at the end of `main.py` is:

```python
"Who are the survivors in Left 4 Dead?"
```

You can modify this to ask questions about any of the fetched pages using entities in your knowledge graph.

---

## 📈 Visualizing the Graph

Once data is loaded into Neo4j, you can visualize and explore the triplets using [Neo4j Browser](https://neo4j.com/developer/neo4j-browser/).

Log into the Neo4j dashboard and run:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

---

## 📦 Dependencies

- `llama-index`
- `neo4j`
- `google-generativeai`
- `wikipedia`
- `tqdm`
- `python-dotenv`
- `huggingface_hub` (for embeddings)

Make sure all necessary packages are listed in your `requirements.txt`.

---

## ✍️ Notes

- The system uses a custom prompt template to generate triplets.
- The LLM retry mechanism helps avoid quota issues by backing off exponentially.
- You can customize the traversal depth and response strategy in `query_engine`.

---

## 📜 License

MIT License — feel free to use, share, and contribute!

---

## 🙋‍♂️ Contributing

Pull requests and issues are welcome! If you find bugs or want to suggest enhancements, feel free to open one.
