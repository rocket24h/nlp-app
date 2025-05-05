import streamlit as st
from modules import WikipediaFetcher, load_KG_from_config
import os
import sys
from dotenv import load_dotenv, set_key, find_dotenv
import logging
import torch
import threading
import time
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import matplotlib.colors as mcolors
import networkx as nx
from streamlit_chat import message as st_message
import json
from streamlit_option_menu import option_menu  # NEW

# Prevent PyTorch warning
torch.classes.__path__ = []

if not find_dotenv():
    raise FileNotFoundError(
        "No .env file found. Please create one by running the init script.")

load_dotenv()

wiki_fetcher = WikipediaFetcher()
kg_index = load_KG_from_config()


def save_chat_history_periodically():
    while True:
        try:
            kg_index.save_chat_history()
        except Exception as e:
            logging.error(f"Error saving chat history: {e}")
        time.sleep(10)


chat_history_thread = threading.Thread(
    target=save_chat_history_periodically, daemon=True
)
chat_history_thread.start()

st.set_page_config(layout="centered",
                   page_title="Knowledge Graph Assistant", page_icon="🤖")

with st.sidebar:
    tab = option_menu(
        menu_title="Navigation",
        options=["Chat", "Documents", "Settings"],
        icons=["chat-dots", "file-earmark-text", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"background-color": "#262730"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {
                "color": "white",
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#404144",
            },
            "nav-link-selected": {
                "background-color": "#ff4b4b",
                "color": "#262730",
                "font-weight": "bold"
            },
        }
    )

# Persistent states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = kg_index.load_chat_history()

if "graph_html" not in st.session_state:
    st.session_state.graph_html = None

# === CHAT TAB ===
if tab == "Chat":
    st.markdown("### 💬 Chat Box")

    for chat_message in st.session_state.chat_history:
        with st.chat_message("user" if chat_message["role"] == "user" else "assistant"):
            st.markdown(chat_message["content"])

    user_input = st.chat_input("Ask something...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = kg_index.chat(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response})

        G = kg_index.get_graph_on_query(user_input)
        if isinstance(G, str):
            st.session_state.graph_html = None
            st.warning(G)
        else:
            components_list = list(nx.connected_components(G.to_undirected()))
            color_palette = list(mcolors.TABLEAU_COLORS.values()) + \
                list(mcolors.CSS4_COLORS.values())

            node_colors = {}
            for idx, component in enumerate(components_list):
                color = color_palette[idx % len(color_palette)]
                for node in component:
                    node_colors[node] = color

            net = Network(height="600px", width="100%",
                          directed=True, bgcolor="#0e1117", font_color="white")

            for node, data in G.nodes(data=True):
                net.add_node(
                    node,
                    label=data.get("label", str(node)),
                    color=node_colors.get(node, "#cccccc"),
                    font={"face": "Courier New", "size": 18, "color": "white"},
                    shape="dot",
                    size=20,
                )

            for source, target, data in G.edges(data=True):
                net.add_edge(
                    source,
                    target,
                    label=data.get("label", ""),
                    color="rgba(200,200,200,0.6)",
                    arrows="to",
                    physics=True,
                    smooth={"type": "dynamic"}
                )

            net.set_options("""{
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "springLength": 150
                    },
                    "solver": "forceAtlas2Based"
                }
            }""")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                net.save_graph(tmp_file.name)
                HtmlFile = open(tmp_file.name, 'r', encoding='utf-8')
                st.session_state.graph_html = HtmlFile.read()

    if st.session_state.graph_html:
        with st.expander("🧠 Show Knowledge Graph from Query", expanded=False):
            components.html(st.session_state.graph_html,
                            height=600, scrolling=True)

# === DOCUMENT TAB ===
elif tab == "Documents":
    st.markdown("### 📄 Wikipedia Article Management")

    query = st.text_input("🔍 Search Wikipedia Articles", key="wiki_search")
    n_results = st.slider("Number of results", 1, 10, 5)

    if query:
        with st.spinner("Searching Wikipedia..."):
            try:
                results = wiki_fetcher.search_articles(query, k=n_results)
                st.session_state.wiki_search_results = results
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.wiki_search_results = []

    selected_titles = st.session_state.get("selected_wiki_titles", [])

    if "wiki_search_results" in st.session_state:
        for title in st.session_state.wiki_search_results:
            if st.button(f"➕ {title}"):
                if title not in selected_titles:
                    selected_titles.append(title)
                    st.session_state.selected_wiki_titles = selected_titles

    if selected_titles:
        st.markdown("### ✅ Selected Articles")
        for title in selected_titles:
            st.markdown(f"- {title}")

        if st.button("📥 Load Articles into Knowledge Graph"):
            with st.spinner("Fetching articles and processing..."):
                contents = []
                for title in selected_titles:
                    try:
                        content = wiki_fetcher.fetch_article(title)
                        contents.append(content)
                    except Exception as e:
                        st.error(f"Error loading {title}: {e}")
                if None in contents:
                    error_msg = "The following articles could not be loaded:"
                    for i in range(len(contents)):
                        if contents[i] is None:
                            error_msg += f"\n- {selected_titles[i]}"
                    st.error(error_msg)

                else:
                    kg_index.add_documents_from_texts(
                        contents)
                    st.success("Articles loaded successfully!")

    st.markdown("### 📂 Wikipedia Knowledge Base")
    docs_path = os.getenv("DOCS_PATH")
    if docs_path and os.path.isdir(docs_path):
        text_files = [f for f in os.listdir(
            docs_path) if f.endswith(".txt")]

        for file in text_files:
            file_path = os.path.join(docs_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            with st.expander(f"📄 {file}"):
                if st.button(f"Summarize {file}"):
                    with st.spinner("Summarizing..."):
                        # Replace this with actual summary logic
                        try:
                            summary = kg_index.summarize(content)
                        except Exception as e:
                            st.error(f"Error summarizing {file}: {e}")
                            summary = None
                        if summary:
                            st.success("Summary:")
                            st.info(summary)

    else:
        st.warning(
            "📁 No local documents found or DOCS_PATH not set correctly.")


# === SETTINGS TAB ===
elif tab == "Settings":
    st.markdown("### ⚙️ Settings")

    api_key = st.text_input("API Key", type="password", )
    model = st.selectbox(
        "Choose model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.0-flash-lite"], index=0)
    temperature = st.slider("Response temperature", 0.0, 2.0, 0.7)
    embedding_model = st.selectbox(
        "Choose embedding model", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"], index=0)
    neo4j_url = st.text_input("Neo4j URL", "bolt://localhost:7687")
    neo4j_user = st.text_input("Neo4j Username", "neo4j")
    neo4j_password = st.text_input("Neo4j Password", type="password")
    neo4j_database = st.text_input("Neo4j Database", "neo4j")
    wiki_docs_path = st.text_input("Wikipedia documents path", "./wiki_docs")

    if st.button("Save Settings"):
        st.session_state.api_key = api_key
        st.session_state.model = model
        st.session_state.temperature = temperature
        # Save settings to .env file
        env_file = find_dotenv()
        if not env_file:
            env_file = ".env"
        if api_key is not None and api_key != "":
            set_key(env_file, "GOOGLE_API_KEY", api_key)
        if model is not None:
            set_key(env_file, "LLM_MODEL", model)
            kg_index.llm = GoogleGenAI(
                model=model, temperature=temperature, api_key=api_key)
        if temperature is not None:
            set_key(env_file, "LLM_TEMPERATURE", str(temperature))
        if embedding_model is not None:
            set_key(env_file, "EMBEDDING_MODEL", embedding_model)
            kg_index.embedding = HuggingFaceEmbedding(
                model_name=embedding_model)
        if neo4j_url is not None:
            set_key(env_file, "NEO4J_URL", neo4j_url)
        if neo4j_url is not None and neo4j_url != "":
            set_key(env_file, "NEO4J_URL", neo4j_url)
        if neo4j_user is not None and neo4j_user != "":
            set_key(env_file, "NEO4J_USER", neo4j_user)
        if neo4j_password is not None and neo4j_password != "":
            set_key(env_file, "NEO4J_PASSWORD", neo4j_password)
        if neo4j_database is not None and neo4j_database != "":
            set_key(env_file, "NEO4J_DATABASE", neo4j_database)
        if wiki_docs_path is not None and wiki_docs_path != "":
            set_key(env_file, "WIKI_DOCS_PATH", wiki_docs_path)

        # If any of the Neo4j or LLM settings were changed, notify the user
        # that the Knowledge Graph Index will be reloaded
        if neo4j_url or neo4j_user or neo4j_password or neo4j_database or model:
            st.warning(
                "Neo4j settings updated. Reloading Knowledge Graph Index...")
            no_error = True
            try:
                kg_index_tmp = load_KG_from_config()
            except Exception as e:
                st.error(f"Error reloading Knowledge Graph Index: {e}")
                st.warning(
                    "Failed to reload Knowledge Graph Index. Please check your settings.")
                no_error = False

            if no_error:
                kg_index = kg_index_tmp
                st.success("Knowledge Graph Index reloaded successfully!")

        st.success("Settings saved successfully!")
