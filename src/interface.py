import subprocess
import os
import glob
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Transcript History",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- State Setup ---
if "processes" not in st.session_state:
    st.session_state.processes = []

# --- Transcript folder ---
TRANSCRIPTS_FOLDER = "/src/transcripts"  # PUT FULL PATH TO TRANSCRIPTS FOLDER HERE
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)

# --- Load all transcript files ---
files = sorted(
    glob.glob(os.path.join(TRANSCRIPTS_FOLDER, "*.txt")),
    key=os.path.getmtime,
    reverse=True
)

transcripts = {}
for f in files:
    name = os.path.basename(f)
    with open(f, "r") as file:
        transcripts[name] = file.read()

# --- Ensure at least one transcript ---
if not transcripts:
    default_name = "Transcript_1.txt"
    transcripts[default_name] = ""
    file_path = os.path.join(TRANSCRIPTS_FOLDER, default_name)
    with open(file_path, "w") as f:
        f.write("")

# --- CSS for big buttons ---
st.markdown("""
<style>
div.stButton > button {
    font-size: 22px;
    padding: 15px 50px;
    margin: 10px 10px 10px 0;
    border-radius: 12px;
}
.run-btn {
    background-color: #4CAF50;
    color: white;
}
.run-btn:hover {
    background-color: #45a049;
}
.stop-btn {
    background-color: #f44336;
    color: white;
}
.stop-btn:hover {
    background-color: #da190b;
}
</style>
""", unsafe_allow_html=True)

st.subheader("Run the HUD")  # <-- New title above buttons

# --- Top Buttons ---
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Run Scripts", key="run"):
        script_path = os.path.abspath("/src/hud/HUD.py") # PUT FULL PATH TO HUD.PY HERE
        script1 = subprocess.Popen(["python", script_path])
        st.session_state.processes.append(script1)
        st.success("HUD.py started!")

with col2:
    if st.button("⏹ Stop Scripts", key="stop"):
        for p in st.session_state.processes:
            p.terminate()
        st.session_state.processes.clear()
        st.warning("HUD.py stopped.")

# --- Sidebar ---
st.sidebar.title("💬 History")
current_transcript = st.sidebar.radio(
    "Choose a transcript:",
    list(transcripts.keys()),
    index=0
)

# --- Main Console ---
st.title("💬 Transcript Console")
st.subheader(f"Viewing: {current_transcript}")

# Display transcript in a scrollable text area (bounded box)
st.text_area(
    "Transcript Content",
    value=transcripts[current_transcript],
    height=400,
    max_chars=None,
    key=current_transcript,
    disabled=True
)
