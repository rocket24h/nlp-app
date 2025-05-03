import streamlit as st
from modules import (
    KGIndex,
    WikipediaFetcher,
)
import os
from dotenv import load_dotenv
import logging
import networkx as nx
from networkx import DiGraph
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import threading
import torch
import time

# Manually prevent error message from PyTorch
torch.classes.__path__ = []

# TODO: Something to load personal config


def load_llm():
    model = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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
        model_name=model
    )
    return llm, llm_config, embedding_model


llm, llm_config, embedding_model = load_llm()

config = {
    "persist_dir": os.environ.get("PERSIST_DIR", "./graph_store"),
    "llm": llm,
    "embedding_model": embedding_model,
    "search_depth": int(os.environ.get("SEARCH_DEPTH", 2)),
}

kg_index = KGIndex(config)


def save_chat_history_periodically():
    while True:
        try:
            kg_index.save_chat_history()
        except Exception as e:
            logging.error(f"Error saving chat history: {e}")
        time.sleep(30)


# Start the daemon thread
chat_history_thread = threading.Thread(
    target=save_chat_history_periodically, args=(), daemon=True
)
chat_history_thread.start()


def render_chat_box(kg_index):
    st.markdown("### Chat Box")

    # Load chat history once
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = kg_index.load_chat_history()

    # Render chat history using st.chat_message (native)
    for chat_message in st.session_state.chat_history:
        with st.chat_message("user" if chat_message["role"] == "user" else "assistant"):
            st.markdown(chat_message["content"])

    # Chat input
    user_input = st.chat_input("Ask something...")
    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response
        with st.spinner("Thinking..."):
            response = kg_index.chat(user_input)

        # Show assistant message
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save to session and backend
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response})


# =======================
# Streamlit App
# =======================
# Set page config
st.set_page_config(
    layout="wide",
    page_title="Knowledge Graph Chatbot",
    page_icon=":robot_face:",
)

# --- Sidebar layout ---
with st.sidebar:
    st.markdown("### Toggle Menu")
    toggle_option = st.checkbox("Enable something")  # Example toggle
    st.markdown("### Search Bar")
    search_query = st.text_input("Search")

# --- Main content layout ---
col1, col2 = st.columns([2, 2], gap="medium")

# --- Center: Chat Box ---
with col1:
    render_chat_box(kg_index)

# --- Right: Graph Visualization ---
with col2:
    st.markdown("### Graph Visualization")
    st.info("Graph will be rendered here.")

    # Placeholder for a future graph (e.g. PyVis, Plotly, etc.)
    st.empty()
