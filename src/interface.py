# interface.py
import subprocess
import os
import glob
import streamlit as st
import sys
import threading
import time
import queue
import requests
from src.live_transcribe import start_transcription, stop_transcription
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv

load_dotenv()
PUSH_BULLET_ACCESS_TOKEN = os.getenv("PUSH_BULLET_API")

# NEW: sync text client for context + Q&A
from src.RAG_assistant import TextContextClient

# __counter = st_autorefresh(interval=2000, key="log_refresher")
# --- State Setup (add these early) ---
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

# Call autorefresh ONLY when not streaming
from streamlit_autorefresh import st_autorefresh
if not st.session_state.is_streaming:
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
    st.session_state.results = []  # assistant outputs (also from ::RESULT::)

if "out_q" not in st.session_state:
    st.session_state.out_q = queue.Queue()

# NEW: Text QA state
if "text_client" not in st.session_state:
    st.session_state.text_client = TextContextClient()

if "ctx_path" not in st.session_state:
    st.session_state.ctx_path = r""  # e.g., r"C:\path\to\context.txt"

if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""

# --- Transcript folder ---
TRANSCRIPTS_FOLDER = os.path.abspath("transcripts")
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
    with open(f, "r", encoding="utf-8", errors="replace") as file:
        transcripts[name] = file.read()

# --- Ensure at least one transcript ---
if not transcripts:
    default_name = "Transcript_1.txt"
    transcripts[default_name] = ""
    file_path = os.path.join(TRANSCRIPTS_FOLDER, default_name)
    with open(file_path, "w", encoding="utf-8") as f:
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


st.markdown("""
<style>
:root{
  --asst-bg: var(--secondary-background-color, #f0f2f6);
  --asst-fg: var(--text-color, #0e1117);
  --asst-border: rgba(49,51,63,0.2);
    --asst-defj: var(--secondary-background-color, #262730);
}
div.assistant-title{
  font-weight:600;
  margin-bottom:6px;
}
div.assistant-box{
  height:280px;
  overflow:auto;
  padding:10px;
  border:1px solid var(--asst-fg) !important;
  border-radius:8px;
  background: var(--asst-defj) !important;   /* force gray background */
  color: var(--asst-bg) !important;         /* force readable text */
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  line-height:1.4;
  white-space: pre-wrap;    /* preserves newlines */
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
            first_line = False
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                if not first_line:
                    first_line = True
                q.put(line.strip())
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

# --- Drain queue and update results (capture ::RESULT:: from child) ---
while not st.session_state.out_q.empty():
    line = st.session_state.out_q.get_nowait()
    if "::RESULT::" in line:
        result = line.split("::RESULT::", 1)[1].strip()
        sentiment = not ("negativ" in result.lower().split()[-1])
        print("[SEN]", sentiment)
        st.session_state.results.append(result)
        print("[UI] RESULT CAPTURED:", result)

        payload = {
            "type": "note",
            "title": "haptic feedback",
            "body": " ",
            "device_iden": "ujy5wHHEZ6Osjyr2puFwfk"
        }

        full_url = "https://api.pushbullet.com/v2/pushes"
        headers = {
            "Content-Type": "application/json",
            "Access-Token": f"{PUSH_BULLET_ACCESS_TOKEN}"
        }

        if sentiment:
            response = requests.post(full_url, json=payload, headers=headers)
            print("haptic sent:", response.status_code)
        else:
            response = requests.post(full_url, json=payload, headers=headers)
            print("haptic sent:", response.status_code)
            time.sleep(1.5)
            response = requests.post(full_url, json=payload, headers=headers)
            print("haptic sent:", response.status_code)


# --- Sidebar ---
st.sidebar.title("💬 History")
current_transcript = st.sidebar.radio(
    "Choose a transcript:",
    list(transcripts.keys()),
    index=0
)

# --- Main Console (split view) ---
st.title("💬 Transcript Console")
st.subheader(f"Viewing: {current_transcript}")

left, right = st.columns(2)

with left:
    st.text_area(
        "Transcript Content",
        value=transcripts[current_transcript],
        height=400,
        max_chars=None,
        key=f"transcript_{current_transcript}",
        disabled=True
    )

with right:
    # Local placeholder created each run at a fixed position
    assistant_ph = st.empty()

    # Build text for this run:
    history_text = "\n\n".join(st.session_state.results) if st.session_state.results else ""
    # Use .get(...) so we never crash if state was cleared somehow
    live_buf = st.session_state.get("assistant_live", "")
    assistant_source = live_buf if live_buf else history_text

    def _render_assistant_box(text: str, streaming: bool = False):
        # Basic escape only; rely on CSS pre-wrap for newlines
        safe = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        title = "Assistant (streaming…)" if streaming else "Assistant"
        assistant_ph.markdown(
            f"""
            <div class="assistant-title">{title}</div>
            <div class="assistant-box">{safe if safe.strip() else "&nbsp;"}</div>
            """,
            unsafe_allow_html=True,
        )


    # Paint assistant box for this run (shows live buffer if streaming, else history)
    _render_assistant_box(assistant_source, streaming=bool(live_buf))

    # Prompt input (normal widget so it stays put)
    st.text_area(
        "Your question to the assistant",
        value=st.session_state.user_prompt,
        height=100,
        key="user_prompt",
        help="Type your query and click Ask."
    )

    # Single Ask button (streaming only)
    if st.button("Ask", key="ask_stream"):
        if not st.session_state.is_streaming and st.session_state.user_prompt.strip():
            st.session_state.is_streaming = True
            st.session_state.assistant_live = ""   # reset live buffer
            try:
                import time
                prefix = history_text + ("\n\n" if history_text and not history_text.endswith("\n\n") else "")

                acc = ""
                for tok in st.session_state.text_client.stream(
                    user_text=st.session_state.user_prompt,
                    context_text=transcripts[current_transcript],
                    temperature=0.2,
                ):
                    acc += tok
                    st.session_state.assistant_live = prefix + acc
                    _render_assistant_box(st.session_state.assistant_live, streaming=True)
                    time.sleep(0.01)

                # Commit final to history
                if acc.strip():
                    st.session_state.results.append(acc)

                # Clear live buffer; next rerun will render from history
                st.session_state.assistant_live = ""
                _render_assistant_box("\n\n".join(st.session_state.results), streaming=False)

            except Exception as e:
                st.session_state.assistant_live = ""
                _render_assistant_box(f"{history_text}\n\n[Error]\n{e}", streaming=False)
            finally:
                st.session_state.is_streaming = False



# """
# from src.video_processing.text_inference_sync import TextContextClient
# import sys

# client = TextContextClient()

# # One-time context load at first call:
# # (Alternatively: client.load_context_from_file("C:/path/context.txt"))
# ctx_path = r"C:\path\to\context.txt"

# # Non-streaming:
# answer = client.ask("Give me a 3-bullet summary.", context_path=ctx_path)
# print("FULL:", answer)

# # Streaming (print tokens as they arrive):
# print("STREAM:", end="", flush=True)
# for token in client.stream("Now answer in one paragraph.", context_path=ctx_path):
#     sys.stdout.write(token)
#     sys.stdout.flush()
# print()  # newline at end

# """