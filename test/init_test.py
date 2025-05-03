# ONLY RUN ONCE
# This file is used to test the initiation of the KGIndex and the loading of Wikipedia documents.

import os.path as path
# import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys
import os
import wikipedia
from tqdm import tqdm

# Adds the parent directory (Finals/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()


TEST_TOPICS = [
    "Python (programming language)",
    "Artificial Intelligence",
    "Machine Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Deep Learning",
    "Neural Networks",
    "Data Science",
    "Big Data",
    "Cloud Computing",
]


def load_llm():
    model = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),

    # Load the LLM configuration from environment variables or a config file
    llm_config = {
        "model": os.environ.get("LLM_MODEL", "gemini-2.0-flash"),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", 0.7)),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", 1500)),
    }
    llm = GoogleGenAI(
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
    )
    embedding_model = HuggingFaceEmbedding(
        model_name=model[0]
    )
    return llm, llm_config, embedding_model


def fetch_wikipedia_article(topic):
    try:
        page = wikipedia.page(topic)
        return page.content
    except wikipedia.DisambiguationError as e:
        print(f"Topic '{topic}' is ambiguous. Options: {e.options}")
    except wikipedia.PageError:
        print(f"Topic '{topic}' not found on Wikipedia.")
    except Exception as e:
        print(f"Failed to fetch Wikipedia article: {e}")
    return None


def main():
    from modules.kg_index import KGIndex
    # Load the LLM and embedding model
    llm, llm_config, embedding_model = load_llm()

    # Configuration for KGIndex
    config = {
        "persist_dir": os.environ.get("PERSIST_DIR", "./graph_store"),
        "llm": llm,
        "embedding_model": embedding_model,
    }
    print("Configuration loaded.")
    # Initialize KGIndex
    # kg = KGIndex(config)
    # print("KGIndex initialized.")
    article_list = []
    for topic in tqdm(TEST_TOPICS):
        print(f"Fetching article for topic: {topic}")
        article = fetch_wikipedia_article(topic)
        if article:
            article_list.append(article)
            print(f"Article fetched for topic: {topic}")
            with open(os.path.join(os.environ.get("DOCS_PATH", "wiki_docs"), f"{topic}.txt"), "w", encoding="utf-8") as f:
                f.write(article)
        else:
            print(f"Failed to fetch article for topic: {topic}")

    # kg.add_documents_from_texts(article_list)
    # print("Documents added to KGIndex.")


if __name__ == "__main__":
    main()
