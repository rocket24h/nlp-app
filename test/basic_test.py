
import os.path as path
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys
import os
import wikipedia

# Adds the parent directory (Finals/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


load_dotenv()


def load_llm():
    model = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),

    # Load the LLM configuration from environment variables or a config file
    llm_config = {
        "model": os.environ.get("LLM_MODEL", "geminmi-2.0-flash"),
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


def main():
    from modules.kg_index import KGIndex
    # Load the LLM and embedding model
    llm, llm_config, embedding_model = load_llm()

    # Configuration for KGIndex
    config = {
        "persist_dir": os.environ.get("PERSIST_DIR", "./graph_store"),
        "llm": llm,
        "embedding_model": embedding_model,
        "search_depth": int(os.environ.get("SEARCH_DEPTH", 2)),
    }

    # Initialize KGIndex
    kg = KGIndex(config)

    # Path to the directory containing Wikipedia documents
    wiki_docs_path = "./wiki_docs"

    # Check if the directory exists
    if not os.path.exists(wiki_docs_path):
        print(
            f"Directory '{wiki_docs_path}' does not exist. Please ensure the documents are stored there.")
        return

    # Add documents from the directory to the knowledge graph
    # print("Adding documents from the 'wiki_docs' directory...")
    # kg.add_documents_from_directory(wiki_docs_path)

    # Prompt the user for a query
    print("\nKnowledge Graph is ready. You can now query it.")
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Exiting the program.")
            break
        else:
            # Run the query and display the result
            print("\nQuerying the Knowledge Graph...")
            response = kg.query("What is Nlp")
            print("\nQuery Result:")
            # print(response)

            # # Testing the graph fetch
            # graph = kg.get_graph_on_query(user_query)

            # if graph:
            #     from pyvis.network import Network

            #     net = Network(height="800px", width="100%", directed=True)
            #     for n, d in graph.nodes(data=True):
            #         net.add_node(n, label=d.get("label", str(n)))

            #     for u, v, d in graph.edges(data=True):
            #         net.add_edge(u, v, label=d.get("label", ""))

            #     net.show("query_graph.html")
            # else:
            #     print("No graph data available for the query.")


if __name__ == "__main__":
    main()
