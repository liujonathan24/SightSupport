# ai_console.py

import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="AI Console",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Setup ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {"Session 1": []}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Session 1"

# --- Sidebar ---
st.sidebar.title("⚙️ Configurations")

# Model settings (example fields)
model = st.sidebar.selectbox("Model", ["GPT-4", "GPT-3.5", "Custom"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# History Tabs
st.sidebar.markdown("---")
st.sidebar.subheader("💬 History")

# Select current session
session_choice = st.sidebar.radio(
    "Choose a session:",
    list(st.session_state.conversations.keys()),
    index=list(st.session_state.conversations.keys()).index(st.session_state.current_session),
)

if session_choice != st.session_state.current_session:
    st.session_state.current_session = session_choice

# Add new session
if st.sidebar.button("➕ New Session"):
    new_name = f"Session {len(st.session_state.conversations) + 1}"
    st.session_state.conversations[new_name] = []
    st.session_state.current_session = new_name

# --- Main Console ---
st.title("🖥️ AI Console")

# Display conversation history
for role, message in st.session_state.conversations[st.session_state.current_session]:
    with st.chat_message(role):
        st.markdown(message)

# Input box
if query := st.chat_input("Type your query here..."):
    # Append user query
    st.session_state.conversations[st.session_state.current_session].append(("user", query))

    # --- Placeholder for AI response logic ---
    # In production, call your LLM API here
    response = f"(Simulated {model} response at temp={temperature}): {query[::-1]}"

    # Append response
    st.session_state.conversations[st.session_state.current_session].append(("assistant", response))

    # Display immediately
    with st.chat_message("assistant"):
        st.markdown(response)
