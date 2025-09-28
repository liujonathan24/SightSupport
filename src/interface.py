# interface.py

import subprocess
import os
import glob
import streamlit as st
import sys
import threading
import queue
from src.live_transcribe import start_transcription, stop_transcription
from streamlit_autorefresh import st_autorefresh

__counter = st_autorefresh(interval=2000, key="log_refresher")

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

if "results" not in st.session_state:
    st.session_state.results = []

if "out_q" not in st.session_state:
    st.session_state.out_q = queue.Queue()

# --- Transcript folder ---
TRANSCRIPTS_FOLDER = os.path.abspath("transcripts")  # <-- use absolute path
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

st.subheader("Run the HUD")

# --- Top Buttons ---
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Run Scripts", key="run"):
        script_path = os.path.abspath(
            "C:/Users/ayush/OneDrive/Documents/Snapdragon/SightSupport/src/video_processing/lms_inference.py"
        )

        hud_script_path = os.path.abspath(
            "C:/Users/ayush/OneDrive/Documents/Snapdragon/SightSupport/src/hud.py"
        )
        
        script2 = subprocess.Popen([sys.executable, hud_script_path])
        st.session_state.processes.append(script2)
        st.success("HUD.py started!")

        print("[spawn] script_path exists:", os.path.exists(script_path))
        project_root = os.path.abspath("C:/Users/ayush/OneDrive/Documents/Snapdragon/SightSupport")
        print("[spawn] project_root exists:", os.path.exists(project_root))
        print("[spawn] launching with cwd:", project_root)

        start_transcription()

        script1 = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=project_root,  # ensure imports resolve
        )

        print("[spawn] child pid:", script1.pid)
        st.session_state.processes.append(script1)
        st.success("lms_inference.py started!")

        # --- Read child's stdout in background ---
        def _reader(proc, q):
            print("[spawn] reader thread started")
            first_line = False
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                if not first_line:
                    # print("[child] first line received")
                    first_line = True
                q.put(line.strip())
                # print("[child] [STDOUT]", line.strip())
            proc.stdout.close()

        threading.Thread(
            target=_reader,
            args=(script1, st.session_state.out_q),
            daemon=True,
        ).start()

        # --- Also read stderr for errors ---
        def _reader_err(proc):
            for line in iter(proc.stderr.readline, ""):
                if not line:
                    break
                # print("[child] [STDERR]", line.strip())
            proc.stderr.close()

        threading.Thread(
            target=_reader_err,
            args=(script1,),
            daemon=True,
        ).start()

        # --- Watch for process exit ---
        def _watch_exit(proc):
            rc = proc.wait()
            print(f"[child] process exited with code {rc}")

        threading.Thread(
            target=_watch_exit,
            args=(script1,),
            daemon=True,
        ).start()

with col2:
    if st.button("⏹ Stop Scripts", key="stop"):
        for p in st.session_state.processes:
            p.terminate()
        stop_transcription()
        st.session_state.processes.clear()
        st.warning("lms_inference.py stopped.")

# --- Drain queue and update results ---
while not st.session_state.out_q.empty():
    line = st.session_state.out_q.get_nowait()
    if "::RESULT::" in line:
        result = line.split("::RESULT::", 1)[1].strip()
        sentiment = "positiv" in result.lower().split()[-1]
        print("[SEN]", sentiment)
        st.session_state.results.append(result)
        print("[UI] RESULT CAPTURED:", result)

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

st.text_area(
    "Transcript Content",
    value=transcripts[current_transcript],
    height=400,
    max_chars=None,
    key=current_transcript,
    disabled=True
)
