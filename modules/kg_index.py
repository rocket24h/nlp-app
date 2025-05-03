from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (
    SimpleDirectoryReader,
    PropertyGraphIndex,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from dotenv import load_dotenv

import neo4j
import networkx as nx
from networkx import DiGraph
import os
import logging
from typing import Optional
import nest_asyncio
import time
import re
import json


nest_asyncio.apply()


load_dotenv()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KGIndex:
    def __init__(self, config: dict, graph: Optional[nx.DiGraph] = None):

        self.graph = graph or nx.DiGraph()

        self.llm = config.get("llm", GoogleGenAI(
            model="gemini-2.0-flash", temperature=0.2, max_output_tokens=256
        ))

        self.embedding_model = config.get(
            "embedding_model", HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2"))

        # Setup Neo4j instance
        if not self.verify_neo4j_connection():
            raise Exception(
                "Failed to connect to Neo4j. Please check your connection settings.")

        self.graph_store = Neo4jPropertyGraphStore(
            url=os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
        )

        self.llm_extractors = None

        # Setup PropertyGraphIndex for the knowledge graph
        self.index = PropertyGraphIndex.from_existing(
            llm=self.llm,
            embed_model=self.embedding_model,
            property_graph_store=self.graph_store,
        )

        self.retriever = self.index.as_retriever(
            include_text=True,
        )

        self.query_engine = self.index.as_query_engine(
            # response_mode="tree_summarize",
            verbose=True,
            llm=self.llm,
        )

        self.chat_message = []

        # Load chat history if it exists
        self.load_chat_history()
        custom_history = []

        if len(self.chat_message) > 0:

            for i in range(len(self.chat_message)):
                if self.chat_message[i]["role"] == "user":
                    custom_history.append(
                        ChatMessage(role=MessageRole.USER, content=self.chat_message[i]["content"]))
                if self.chat_message[i]["role"] == "assistant":
                    custom_history.append(
                        ChatMessage(role=MessageRole.ASSISTANT, content=self.chat_message[i]["content"]))

        self.chat_engine = self.index.as_chat_engine(
            verbose=True,
            llm=self.llm,
            chat_history=custom_history,
        )

        self.search_depth = config.get("search_depth", 2)

    def verify_neo4j_connection(self):
        try:
            with neo4j.GraphDatabase.driver(
                    os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
                    auth=(os.environ.get("NEO4J_USERNAME", "neo4j"),
                          os.environ.get("NEO4J_PASSWORD", "password")),
            ) as driver:
                with driver.session() as session:
                    session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully.")

            # Check if the database has
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
        return True

    def add_documents_from_directory(self, dir_path: str):
        reader = SimpleDirectoryReader(dir_path)
        documents = reader.load_data()
        self._add_documents(documents)

    def add_documents_from_texts(self, texts: list[str]):
        documents = [Document(text=t) for t in texts]
        self._add_documents(documents)

    def _add_documents(self, documents: list[Document], retry_limit=5, retry_delay=10):
        node_parser = SentenceSplitter()
        nodes = node_parser.get_nodes_from_documents(documents)

        for i, node in enumerate(nodes):
            attempt = 0
            while attempt < retry_limit:
                try:
                    self.index.insert_nodes([node])
                    logger.info(
                        f"Inserted node {i + 1}/{len(nodes)} : {node.get_content()[:100]}")
                    break
                except Exception as e:
                    if "resource_exhausted" in str(e).lower():
                        logger.warning(
                            f"Rate limit hit on node {i}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        attempt += 1
                    else:
                        logger.error(f"Failed to insert node {i}")
                        break

    def query(self, user_input: str, include_knowledge=True):
        # context = self.retriever.retrieve(user_input)
        # if include_knowledge:
        #     return "\n".join([r.get_content() for r in context])

        response = self.query_engine.query(user_input)
        return response

    def chat(self, user_input: str):
        response = self.chat_engine.chat(user_input)
        self.chat_message.append(
            {"role": "user", "content": user_input})
        self.chat_message.append(
            {"role": "assistant", "content": response.response})
        return response.response

    def stream_chat(self, user_input: str):
        response = self.chat_engine.stream_chat(user_input)
        self.chat_message.append({"user": user_input, "bot": response})
        for chunk in response.response_gen:
            yield chunk

    def save_chat_history(self):
        file_path = os.environ.get("CHAT_HISTORY_PATH", "./chat_history.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.chat_message, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def load_chat_history(self):
        file_path = os.environ.get("CHAT_HISTORY_PATH", "./chat_history.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.chat_message = json.load(f)
            return self.chat_message
        else:
            logger.warning(f"Chat history file not found: {file_path}")
            self.chat_message = []
            return self.chat_message

    def delete_chat_history(self):
        file_path = os.environ.get("CHAT_HISTORY_PATH", "./chat_history.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            self.chat_message = []
            logger.info("Chat history deleted.")
        else:
            logger.warning(f"Chat history file not found: {file_path}")

    def get_graph_on_query(self, query):
        driver = neo4j.GraphDatabase.driver(
            os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USERNAME", "neo4j"),
                  os.environ.get("NEO4J_PASSWORD", "password")),
        )

        G = DiGraph()

        context = self.retriever.retrieve(query)
        if not context:
            return "No relevant information found in the knowledge graph."

        context_full = [r.get_content() for r in context]
        context_full = "\n".join(context_full)

        pattern = r"(.+?)\s*->\s*(.+?)\s*->\s*(.+)"

        node_ids = set()
        for line in context_full.splitlines():
            match = re.match(pattern, line.strip())
            if match:
                source = match.group(1).strip()
                target = match.group(3).strip()
                node_ids.add(source)
                node_ids.add(target)

        node_ids = list(node_ids)
        with driver.session() as session:
            cypher_query = f"""
            MATCH (n:__Entity__)
            WHERE n.name IN {node_ids}
            MATCH path = (n)-[*1..{self.search_depth}]-(neighbor)
            RETURN DISTINCT nodes(path) AS nodes, relationships(path) AS rels;
            """

            result = session.run(cypher_query, node_ids=node_ids)
            # print(f"Query result: {list(result)}")
            for record in result:
                nodes = record["nodes"]
                rels = record["rels"]

                for node in nodes:
                    node_id = str(node.id)
                    node_label = node.get("name", node_id)
                    G.add_node(node_id, label=node_label)

                for rel in rels:
                    start = str(rel.start_node.id)
                    end = str(rel.end_node.id)
                    rel_type = rel.type
                    G.add_edge(start, end, label=rel_type)

        driver.close()

        return G

    def summarize(self, doc_content: str) -> str:
        response = self.llm.complete(f"Summarize the following document's content: {doc_content}")
        return response.text
    

def load_KG_from_config() -> KGIndex:
    model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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
        model_name=model
    )

    # Configuration for KGIndex
    config = {
        "persist_dir": os.environ.get("PERSIST_DIR", "./graph_store"),
        "llm": llm,
        "embedding_model": embedding_model,
    }
    print("Configuration loaded.")

    kg_instance = KGIndex(config=config)
    return kg_instance