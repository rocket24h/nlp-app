import streamlit as st

# Set page config
st.set_page_config(
    layout="wide",
    page_title="KG Chat UI",
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
    st.markdown("### Chat Box")
    with st.container(height=500, border=True):
        st.chat_message("assistant").write("Hello! Ask me something.")
        user_input = st.chat_input("Ask a question...")

# --- Right: Graph Visualization ---
with col2:
    st.markdown("### Graph Visualization")
    st.info("Graph will be rendered here.")

    # Placeholder for a future graph (e.g. PyVis, Plotly, etc.)
    st.empty()
