import time
import os
import sys
import json
import ast
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import wikipedia
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import (
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    PromptTemplate
)
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter


load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Test wikipedia pages
pages_to_get = [
    "Left 4 Dead",
    # "Python (programming language)",
    # "Machine learning",
    # "Deep learning",
]

wikipedia.set_lang("en")  # Set the language to English
wikipedia.set_rate_limiting(True)  # Enable rate limiting

system_prompt = (
    "You are an expert assistant, capable of thoroughly analyzing, understanding as well as deducing \
    the general meaning and relationships of entities in a text passage. \
    \
    You will be provided with a context passage, consisting of knowledge graph nodes and their relationships\
    (triplets). Your task is to answer the query based on the context information. \
    Only use the context information to answer the query."
)

custom_prompt_template = PromptTemplate(
    template=(
        f"(system_prompt)\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, answer the query:\n"
        "Query: {query_str}\n"
        "Answer: "
    )
)


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


def add_kg_index(index, docs, llm, max_retries=5, backoff_factor=2):
    """
    Add nodes and triplets to the KG index using LLM.

    Args:
        index: The KG index to update.
        docs: List of documents to add to the index.
        llm: The LLM to use for generating triplets.
        max_retries: Maximum number of retries for LLM calls.
        backoff_factor: Factor by which to increase the wait time after each retry.
    """

    node_parser = SentenceSplitter()
    nodes = node_parser.get_nodes_from_documents(docs, show_progress=True)

    for node in nodes:
        try:
            # Extract text from the node
            context = node.get_text()

            # Modify query here
            query = f"You are a knowledge graph generator, capable of recognizing named entities in a text passage. \
                    Generate a list of triplets containing entity relationships, each structured as follows \
                    (subject, predicate, object). The entities extracted should not be  \
                    Your reply should only contain the list of triplets, and no more supplementary words. \
                    Here is an example of a standard response: \
                    [('Left 4 Dead', 'is a', 'video game'), ('Left 4 Dead', 'is developed by', 'Valve Corporation')]. \
                    \
                    The text passage is: \
                    -------------------------\
                        {context}\
                    -------------------------"

            # Generate triplets using the LLM
            response = query_with_retry(
                llm, None, query, max_retries, backoff_factor, is_llm=True)
            for triplet in ast.literal_eval(response):
                # Add triplet to the index
                index.upsert_triplet_and_node(triplet, node)
        except Exception as e:
            logging.error(f"Error fetching LLM response: {e}")
            continue

    return index


def get_kg_index(docs: list = None, max_retries=5, backoff_factor=2):
    graph_store = Neo4jGraphStore(
        url=os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password"),
    )

    storage_context = StorageContext.from_defaults(
        graph_store=graph_store)

    # Load index from Neo4j database if it exists
    try:
        index = KnowledgeGraphIndex.from_documents(
            [],
            storage_context=storage_context,
        )
    except Exception as e:
        logging.warning(f"Index not found in Neo4j: {e}")
        return None

    if docs:
        index = add_kg_index(
            index, docs, Settings.llm, max_retries, backoff_factor)

    return index


def get_summary(content, llm):
    # This function would typically call the LLM to generate a summary
    # For now, we will just return the content as is for demonstration purposes
    return content


def query_with_retry(llm, query_engine, query, max_retries=5, backoff_factor=2, is_llm=False):
    """
    Query the LLM with retry logic for quota exceed errors.

    Args:
        llm: The LLM to use for the query.
        query_engine: The query engine to use.
        query: The query string.
        max_retries: Maximum number of retries.
        backoff_factor: Factor by which to increase the wait time after each retry.

    Returns:
        The response from the query engine.
    """
    retries = 0
    wait_time = 5  # Initial wait time in seconds

    while retries < max_retries:
        try:
            if is_llm:
                # If using LLM, call the LLM directly
                response = llm.complete(
                    query,
                    temperature=llm_config["temperature"],
                    max_tokens=llm_config["max_tokens"],
                )
            else:
                # Otherwise, use the query engine to get the response
                response = query_engine.query(query)
            return response
        except Exception as e:
            if "resource_exhausted" in str(e).lower():
                logging.warning(
                    f"Quota exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                wait_time *= backoff_factor
            else:
                logging.error(f"An error occurred: {e}")
                raise e

    raise RuntimeError("Max retries exceeded. Unable to complete the query.")


if __name__ == "__main__":
    llm, llm_config, embedding_model = load_llm()
    Settings.llm = llm
    Settings.embed_model = embedding_model
    Settings.chunk_size = 512

    docs_to_read = []
    with logging_redirect_tqdm():
        for page in tqdm(pages_to_get):
            try:
                # Fetch the summary of the page
                summary = wikipedia.summary(page, sentences=1)

                # Create a directory for the page if it doesn't exist
                dir_name = os.environ.get("DOCS_PATH")
                os.makedirs(dir_name, exist_ok=True)

                # Write the summary to a text file in the directory
                with open(os.path.join(dir_name, f"{page}.txt"), "w+", encoding="utf-8") as f:
                    f.write(wikipedia.page(page).content)

                # Get page content
                docs_to_read.append(
                    os.path.join(dir_name, f"{page}.txt")
                )

                logging.info(
                    f"Saved contents of '{page}' in '{dir_name}/{page}.txt'")
            except Exception as e:
                logging.error(f"Error fetching page '{page}': {e}")

    # wiki_docs = SimpleDirectoryReader(
    #     input_files=docs_to_read).load_data()

    # index = get_kg_index(wiki_docs)
    index = get_kg_index()

    query_engine = index.as_query_engine(
        include_text=True,
        graph_traversal_depth=3,
        response_mode="tree_summarize",
    )
    response = query_with_retry(
        llm, query_engine, "Who are the survivors in Left 4 Dead?", max_retries=5, backoff_factor=2)
    print("Query response:", response)
    index.get_networkx_graph()
