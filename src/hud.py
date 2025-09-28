import sys
import keyboard
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QCheckBox, QVBoxLayout, QFrame, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect
from PyQt5.QtGui import QFont

class HUD(QWidget):
    def __init__(self):
        super().__init__()

        # Basic sizes
        self.default_width = 650
        self.default_height = 160
        self.settings_height = 0  # will compute dynamically
        self.setGeometry(100, 100, self.default_width, self.default_height)

        # Window flags
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Fonts
        self.base_font = QFont("Poppins", 15) if QFont("Poppins").exactMatch() else QFont("Segoe UI", 15)

        # Main label
        self.label = QLabel("", self)
        self.label.setFont(QFont(self.base_font.family(), 18))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)

        # Styles
        self.dark_style = (
            "color: white; "
            "background-color: rgba(40,40,40,220); "
            "padding: 18px; "
            "border-radius: 20px; "
            "border: 2px solid rgba(255,255,255,60);"
        )
        self.light_style = (
            "color: black; "
            "background-color: rgba(245,245,245,230); "
            "padding: 18px; "
            "border-radius: 20px; "
            "border: 2px solid rgba(0,0,0,40);"
        )

        # Settings frame
        self.settings_frame = QFrame(self)
        self.settings_frame.setGeometry(0, self.default_height, self.default_width, 0)
        self.settings_layout = QVBoxLayout(self.settings_frame)
        self.settings_layout.setContentsMargins(20, 20, 20, 20)
        self.settings_layout.setSpacing(12)

        # Settings state dict
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
            cb.setFont(QFont(self.base_font.family(), 14))  # slightly smaller
            cb.setChecked(default)
            cb.stateChanged.connect(lambda state, t=text: self.update_setting(t, state))
            cb.setStyleSheet(self.checkbox_style())
            self.settings_layout.addWidget(cb)
            self.checkboxes[text] = cb
            self.settings_state[text] = default

        # --- Data Collection Mode Section ---
        self.settings_layout.addSpacing(15)

        # Container for Data Collection Mode
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

        # Slightly smaller font
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

        self.apply_theme()

        # Track last non-settings message
        self.last_message = "What would you like to ask me?"

        # Animation setup
        self.animation = QPropertyAnimation(self.settings_frame, b"geometry")
        self.animation.setDuration(300)  # ms

        # Hotkeys
        keyboard.add_hotkey("alt+d", lambda: QTimer.singleShot(0, self.show_alt_d))
        keyboard.add_hotkey("alt+x", lambda: QTimer.singleShot(0, self.show_alt_x))
        keyboard.add_hotkey("alt+z", lambda: QTimer.singleShot(0, self.toggle_settings))

        self.show_message(self.last_message)
        self.show()

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
                f"background-color: rgba(40,40,40,220); border-radius: 20px; border: 2px solid rgba(255,255,255,60);"
            )
            cb_color = "white"
            container_bg = "rgba(80,80,80,150)"
        else:
            self.label.setStyleSheet(self.light_style)
            self.settings_frame.setStyleSheet(
                f"background-color: rgba(245,245,245,230); border-radius: 20px; border: 2px solid rgba(0,0,0,40);"
            )
            cb_color = "black"
            container_bg = "rgba(200,200,200,150)"

        for cb in self.checkboxes.values():
            cb.setStyleSheet(self.checkbox_style().replace("white", cb_color))
        for rb in [self.radio_live, self.radio_manual, self.radio_smart]:
            rb.setStyleSheet(self.radio_style().replace("white", cb_color))
        self.mode_label.setStyleSheet(f"color: {cb_color}; font-size: 15px; font-weight: bold;")
        self.mode_container.setStyleSheet(f"background-color: {container_bg}; border-radius: 12px;")

    def update_setting(self, setting_name, state):
        self.settings_state[setting_name] = (state == Qt.Checked)
        if setting_name == "Enable Dark Mode":
            self.apply_theme()

    def set_data_mode(self, mode_name):
        self.settings_state["Data Collection Mode"] = mode_name

    def show_alt_d(self):
        if self.settings_frame.height() > 0:
            self.animate_settings(False)
        self.show_message("Alt D was pressed!")

    def show_alt_x(self):
        if self.settings_frame.height() > 0:
            self.animate_settings(False)
        self.show_message("What would you like to ask me?")

    def toggle_settings(self):
        if self.settings_frame.height() == 0:
            self.compute_settings_height()
            self.show_message("Settings")
            self.animate_settings(True)
        else:
            self.animate_settings(False)
            self.show_message(self.last_message)

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

    def show_message(self, message):
        self.label.setText(message)
        self.label.setFixedWidth(self.default_width)
        self.label.adjustSize()
        text_height = self.label.height() + 30

        self.settings_frame.setGeometry(
            0, text_height, self.default_width, self.settings_frame.height()
        )

        self.setGeometry(
            100, 100, self.default_width, max(self.default_height, text_height + self.settings_frame.height())
        )

        if message != "Settings":
            self.last_message = message


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hud = HUD()
    hud.show()
    sys.exit(app.exec())
