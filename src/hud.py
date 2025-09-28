import sys, os, re, glob
import keyboard
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QCheckBox, QVBoxLayout, QFrame, QRadioButton,
    QButtonGroup, QTextEdit, QShortcut
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QKeySequence

# RAG client
from src.RAG_assistant import TextContextClient


class HUD(QWidget):
    # Worker used to stream RAG tokens back to the UI safely
    class RAGWorker(QObject):
        chunk = pyqtSignal(str)
        done = pyqtSignal()
        error = pyqtSignal(str)

        def __init__(self, client: TextContextClient, prompt: str, context_text: str = "", temperature: float = 0.2):
            super().__init__()
            self.client = client
            self.prompt = prompt
            self.context_text = context_text
            self.temperature = temperature
            self._stop = False

        def stop(self):
            self._stop = True

        def run(self):
            try:
                if self.context_text is not None:
                    self.client.set_context(self.context_text)
                for piece in self.client.stream(
                    user_text=self.prompt,
                    context_text=None,
                    temperature=self.temperature,
                ):
                    if self._stop:
                        break
                    if piece:
                        self.chunk.emit(piece)
                self.done.emit()
            except Exception as e:
                self.error.emit(str(e))

    def __init__(self):
        super().__init__()

        # --- Basic sizes ---
        self.default_width = 650
        self.default_height = 200
        self.settings_height = 0
        self.setGeometry(100, 100, self.default_width, self.default_height)

        # --- Window flags ---
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # --- Fonts ---
        self.base_font = QFont("Poppins", 15) if QFont("Poppins").exactMatch() else QFont("Segoe UI", 15)

        # --- Main HUD label ---
        self.label = QLabel("", self)
        self.label.setFont(QFont(self.base_font.family(), 18))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)

        # --- Input box ---
        self.input_box = QTextEdit(self)
        self.input_box.setPlaceholderText("Type your question here... (press Ctrl+Enter to send)")
        self.input_box.setFont(QFont(self.base_font.family(), 15))
        self.input_box.setFixedHeight(60)
        self.input_box.textChanged.connect(self.auto_resize_input)

        QShortcut(QKeySequence("Ctrl+Return"), self.input_box, activated=self.store_question)
        QShortcut(QKeySequence("Ctrl+Enter"), self.input_box, activated=self.store_question)

        # --- Answer box (kept but hidden to avoid layout changes elsewhere) ---
        self.answer_box = QTextEdit(self)
        self.answer_box.setReadOnly(True)
        self.answer_box.setFont(QFont(self.base_font.family(), 14))
        self.answer_box.setFixedHeight(0)      # collapse
        self.answer_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.answer_box.hide()                  # hide so only bottom output shows answers

        # --- History log ---
        self.output_box = QTextEdit(self)
        self.output_box.setReadOnly(True)
        self.output_box.setFont(QFont(self.base_font.family(), 13))
        self.output_box.setFixedHeight(100)
        self.output_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Auto-resize the Q/A box to fit full content
        self.output_box.textChanged.connect(self.auto_resize_output)

        # --- Styles ---
        self.dark_style = (
            "color: white; background-color: rgba(40,40,40,220); "
            "padding: 18px; border-radius: 20px; border: 2px solid rgba(255,255,255,60);"
        )
        self.light_style = (
            "color: black; background-color: rgba(245,245,245,230); "
            "padding: 18px; border-radius: 20px; border: 2px solid rgba(0,0,0,40);"
        )

        # --- Settings frame ---
        self.settings_frame = QFrame(self)
        self.settings_frame.setGeometry(0, self.default_height, self.default_width, 0)
        self.settings_layout = QVBoxLayout(self.settings_frame)
        self.settings_layout.setContentsMargins(20, 20, 20, 20)
        self.settings_layout.setSpacing(12)

        # --- Settings state dict ---
        self.settings_state = {}
        options = [
            ("Enable Dark Mode", True),
            ("On for Zoom, Off for Google Meet", False),
            ("Enable Haptic Feedback", False),
            ("Enable Sounds", True),
            ("Enable Talking while Talking", False),
            ("Enable Looking at Phone", False),
            ("Enable Emotions", True),
        ]

        self.checkboxes = {}
        for text, default in options:
            cb = QCheckBox(text)
            cb.setFont(QFont(self.base_font.family(), 14))
            cb.setChecked(default)
            cb.stateChanged.connect(lambda state, t=text: self.update_setting(t, state))
            cb.setStyleSheet(self.checkbox_style())
            self.settings_layout.addWidget(cb)
            self.checkboxes[text] = cb
            self.settings_state[text] = default

        # --- Data Collection Mode Section ---
        self.settings_layout.addSpacing(15)
        self.mode_container = QFrame(self.settings_frame)
        self.mode_container_layout = QVBoxLayout(self.mode_container)
        self.mode_container_layout.setContentsMargins(10, 10, 10, 10)
        self.mode_container_layout.setSpacing(10)
        self.mode_container.setStyleSheet(
            "background-color: rgba(80,80,80,150); border-radius: 12px;"
        )
        self.settings_layout.addWidget(self.mode_container)

        self.mode_label = QLabel("Data Collection Mode:")
        self.mode_label.setFont(QFont(self.base_font.family(), 15, QFont.Bold))
        self.mode_container_layout.addWidget(self.mode_label)

        self.mode_group = QButtonGroup(self)
        self.radio_live = QRadioButton("Live (every 3 seconds)")
        self.radio_manual = QRadioButton("Manual (Alt+D start/stop)")
        self.radio_smart = QRadioButton("Smart Prompt")

        for rb in [self.radio_live, self.radio_manual, self.radio_smart]:
            rb.setFont(QFont(self.base_font.family(), 14))
            rb.setStyleSheet(self.radio_style())
            self.mode_container_layout.addWidget(rb)

        self.mode_group.addButton(self.radio_live, 0)
        self.mode_group.addButton(self.radio_manual, 1)
        self.mode_group.addButton(self.radio_smart, 2)

        self.radio_live.setChecked(True)
        self.settings_state["Data Collection Mode"] = "Live"

        self.radio_live.toggled.connect(lambda: self.set_data_mode("Live"))
        self.radio_manual.toggled.connect(lambda: self.set_data_mode("Manual"))
        self.radio_smart.toggled.connect(lambda: self.set_data_mode("Smart Prompt"))

        # --- Track last non-settings message ---
        self.last_message = "What would you like to ask me?"

        # Store questions
        self.questions = []

        # --- Animation ---
        self.animation = QPropertyAnimation(self.settings_frame, b"geometry")
        self.animation.setDuration(300)

        # --- Hotkeys ---
        keyboard.add_hotkey("alt+d", lambda: QTimer.singleShot(0, self.show_alt_d))
        keyboard.add_hotkey("alt+x", lambda: QTimer.singleShot(0, self.show_alt_x))
        keyboard.add_hotkey("alt+z", lambda: QTimer.singleShot(0, self.toggle_settings))
        keyboard.add_hotkey("alt+b", lambda: QTimer.singleShot(0, self.terminate_hud))

        # --- RAG wiring ---
        self.rag_client = TextContextClient()
        self._context_text = "Default context here."
        self._rag_thread = None
        self._rag_worker = None

        # --- Apply theme ---
        self.apply_theme()

        # --- Start on Alt+X page (default) ---
        self.show_alt_x()
        self.show()

    # --- Auto-resize input ---
    def auto_resize_input(self):
        doc_height = self.input_box.document().size().height()
        new_h = int(doc_height) + 26
        new_h = max(48, min(300, new_h))
        if new_h != self.input_box.height():
            self.input_box.setFixedHeight(new_h)
            self.reposition_boxes()

    # --- Auto-resize output (Q/A history) ---
    def auto_resize_output(self):
        doc_height = self.output_box.document().size().height()
        new_h = int(doc_height) + 26
        # Grow to fit entire content; keep a generous cap to avoid runaway window growth
        max_output_h = 1200
        new_h = max(80, min(max_output_h, new_h))
        if new_h != self.output_box.height():
            self.output_box.setFixedHeight(new_h)
            self.reposition_boxes()

    # --- Styles ---
    def checkbox_style(self):
        return (
            "QCheckBox { padding: 6px; color: white; }"
            "QCheckBox::indicator { width: 18px; height: 18px; border-radius: 5px; border: 2px solid rgba(255,255,255,120); background: rgba(0,0,0,0); }"
            "QCheckBox::indicator:checked { background-color: rgba(255,255,255,150); border: 2px solid rgba(255,255,255,200); }"
        )

    def radio_style(self):
        return (
            "QRadioButton { padding: 6px; color: white; }"
            "QRadioButton::indicator { width: 18px; height: 18px; border-radius: 9px; border: 2px solid rgba(255,255,255,120); background: rgba(0,0,0,0); }"
            "QRadioButton::indicator:checked { background-color: rgba(255,255,255,180); border: 2px solid rgba(255,255,255,200); }"
        )

    # --- Settings and theme ---
    def compute_settings_height(self):
        total_height = 0
        for i in range(self.settings_layout.count()):
            widget = self.settings_layout.itemAt(i).widget()
            if widget:
                total_height += widget.sizeHint().height() + self.settings_layout.spacing()
        total_height += self.settings_layout.contentsMargins().top() + self.settings_layout.contentsMargins().bottom()
        self.settings_height = total_height

    def apply_theme(self):
        dark = self.settings_state.get("Enable Dark Mode", True)
        if dark:
            self.label.setStyleSheet(self.dark_style)
            self.settings_frame.setStyleSheet(
                "background-color: rgba(40,40,40,220); border-radius: 20px; border: 2px solid rgba(255,255,255,60);"
            )
            cb_color = "white"
            container_bg = "rgba(80,80,80,150)"
            input_bg = "rgba(40,40,40,220)"
            input_fg = "white"
        else:
            self.label.setStyleSheet(self.light_style)
            self.settings_frame.setStyleSheet(
                "background-color: rgba(245,245,245,230); border-radius: 20px; border: 2px solid rgba(0,0,0,40);"
            )
            cb_color = "black"
            container_bg = "rgba(200,200,200,150)"
            input_bg = "rgba(245,245,245,230)"
            input_fg = "black"

        for cb in self.checkboxes.values():
            cb.setStyleSheet(self.checkbox_style().replace("white", cb_color))
        for rb in [self.radio_live, self.radio_manual, self.radio_smart]:
            rb.setStyleSheet(self.radio_style().replace("white", cb_color))
        self.mode_label.setStyleSheet(f"color: {cb_color}; font-size: 15px; font-weight: bold;")
        self.mode_container.setStyleSheet(f"background-color: {container_bg}; border-radius: 12px;")

        input_style = f"border-radius: 10px; padding: 8px; background-color: {input_bg}; color: {input_fg};"
        self.input_box.setStyleSheet(input_style)
        self.answer_box.setStyleSheet(input_style)
        self.output_box.setStyleSheet(input_style)

    # --- Settings updates ---
    def update_setting(self, setting_name, state):
        self.settings_state[setting_name] = (state == Qt.Checked)
        if setting_name == "Enable Dark Mode":
            self.apply_theme()

    def set_data_mode(self, mode_name):
        self.settings_state["Data Collection Mode"] = mode_name

    # --- Hotkey functions ---
    def show_alt_d(self):
        self.input_box.hide()
        self.answer_box.hide()
        self.output_box.hide()
        if self.settings_frame.height() > 0:
            self.animate_settings(False)
        self.show_message("Alt D was pressed!")

    def show_alt_x(self):
        if self.settings_frame.height() > 0:
            self.animate_settings(False)
        self.show_message("What would you like to ask me?")
        self.input_box.show()
        # self.answer_box.show()  # keep hidden to remove the top output
        self.output_box.show()
        self.input_box.raise_()
        # self.answer_box.raise_()
        self.output_box.raise_()
        self.input_box.setFocus()
        self.auto_resize_input()
        self.reposition_boxes()

    def toggle_settings(self):
        self.input_box.hide()
        self.answer_box.hide()
        self.output_box.hide()
        if self.settings_frame.height() == 0:
            self.compute_settings_height()
            self.show_message("Settings")
            self.animate_settings(True)
        else:
            self.animate_settings(False)
            self.show_message(self.last_message)

    # --- Animation ---
    def animate_settings(self, opening):
        label_height = self.label.height() + 30
        start_rect = QRect(0, label_height, self.default_width, self.settings_frame.height())
        end_height = self.settings_height if opening else 0
        end_rect = QRect(0, label_height, self.default_width, end_height)
        self.animation.stop()
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.start()
        total_height = label_height + (self.settings_height if opening else 0)
        self.setGeometry(100, 100, self.default_width, max(self.default_height, total_height))

    # --- Display message ---
    def show_message(self, message):
        self.label.setText(message)
        self.label.setFixedWidth(self.default_width)
        self.label.adjustSize()
        text_height = self.label.height() + 30
        self.settings_frame.setGeometry(0, text_height, self.default_width, self.settings_frame.height())
        self.reposition_boxes()
        self.setGeometry(100, 100, self.default_width, max(self.default_height, text_height + self.settings_frame.height()))
        if message != "Settings":
            self.last_message = message

    # --- Reposition boxes ---
    def reposition_boxes(self):
        label_height = self.label.height() + 36
        input_height = self.input_box.height()
        answer_height = self.answer_box.height()  # 0 because hidden/collapsed
        output_height = self.output_box.height()
        gap = 10
        left = 20
        width = self.default_width - (left * 2)

        self.input_box.setGeometry(left, label_height, width, input_height)
        self.answer_box.setGeometry(left, label_height + input_height + gap, width, answer_height)
        self.output_box.setGeometry(left, label_height + input_height + answer_height + 2*gap, width, output_height)

        needed_height = label_height + input_height + answer_height + output_height + 3*gap + 20
        self.setGeometry(100, 100, self.default_width, max(self.default_height, needed_height))

    # --- Store user question ---
    def store_question(self):
        question = self.input_box.toPlainText().strip()
        if question:
            self.questions.append(question)
            # Log Q:
            self.output_box.append(f'Q: {question}')
            # Prepare the A: line and stream into it
            self.output_box.append('A: ')
            self.auto_resize_output()
            self.input_box.clear()
            self.ask_rag(question)

    # --- Terminate ---
    def terminate_hud(self):
        QApplication.quit()

    # --- RAG helper ---
    def ask_rag(self, user_prompt: str):
        self.load_context_file()
        if self._rag_thread is not None:
            try:
                if self._rag_worker:
                    self._rag_worker.stop()
            except Exception:
                pass

        # (answer_box kept but unused)
        self.answer_box.clear()

        self._rag_thread = QThread()
        self._rag_worker = HUD.RAGWorker(
            client=self.rag_client,
            prompt=user_prompt,
            context_text=self._context_text,
            temperature=0.2,
        )
        self._rag_worker.moveToThread(self._rag_thread)
        self._rag_thread.started.connect(self._rag_worker.run)
        self._rag_worker.chunk.connect(self._on_rag_chunk)
        self._rag_worker.done.connect(self._on_rag_done)
        self._rag_worker.error.connect(self._on_rag_error)
        self._rag_worker.done.connect(self._rag_thread.quit)
        self._rag_worker.error.connect(self._rag_thread.quit)
        self._rag_thread.finished.connect(self._cleanup_rag_thread)
        self._rag_thread.start()

    def _on_rag_chunk(self, piece: str):
        # Stream directly into bottom output after the 'A: ' prefix
        cursor = self.output_box.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(piece)
        self.output_box.setTextCursor(cursor)
        self.auto_resize_output()

    def _on_rag_done(self):
        # Just move to a new line; don't duplicate the answer
        self.output_box.append("")
        self.auto_resize_output()

    def _on_rag_error(self, msg: str):
        self.output_box.append(f"\n[Error] {msg}")
        self.auto_resize_output()

    def _cleanup_rag_thread(self):
        try:
            if self._rag_worker:
                self._rag_worker.deleteLater()
        except Exception:
            pass
        self._rag_worker = None
        try:
            if self._rag_thread:
                self._rag_thread.deleteLater()
        except Exception:
            pass
        self._rag_thread = None

    def load_context_file(self):
        path = self.curr_transcript_path()

        try:
            with open(path, "r", encoding="utf-8") as f:
                self._context_text = f.read()
        except Exception as e:
            self.output_box.append(f"[Context load error] {e}")

        print("YAYAYYAYA", self._context_text)

    def curr_transcript_path(self, prefix="live_transcript", ext=".txt", folder="transcripts"):
        os.makedirs(folder, exist_ok=True)
        # Match live_transcript.txt or live_transcript_<n>.txt
        rx = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?{re.escape(ext)}$")
        max_n = -1
        for p in glob.glob(os.path.join(folder, f"{prefix}*{ext}")):
            name = os.path.basename(p)
            m = rx.match(name)
            if not m:
                continue
            if m.group(1) is None:
                max_n = max(max_n, 0)
            else:
                max_n = max(max_n, int(m.group(1)))
        next_n = (max_n + 1) if max_n >= 0 else 0
        return os.path.join(folder, f"{prefix}_{next_n - 1}{ext}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hud = HUD()
    hud.show()
    sys.exit(app.exec())
