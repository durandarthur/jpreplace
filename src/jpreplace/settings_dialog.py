"""Dialog for choosing the translation model: bundled PLaMo, any local GGUF
file, or an LLM server (OpenAI-compatible API)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .config import Settings
from .translate import DEFAULT_SYSTEM_PROMPT, api_base, describe_error, list_models

TRANSLATORS = [
    ("auto", "PLaMo 8B + 2B (auto: the 2B for large amounts of text)"),
    ("plamo-8b", "PLaMo 8B (best quality)"),
    ("plamo-2b", "PLaMo 2B (fastest, least VRAM)"),
    ("file", "Local model file (GGUF, run with llama.cpp)"),
    ("server", "LLM server (OpenAI-compatible API)"),
]
PROMPT_STYLES = [
    ("auto", "Automatic (PLaMo prompt for PLaMo models, else chat)"),
    ("chat", "Chat: an instruction for a general chat model"),
    ("plamo", "PLaMo Translate's own format"),
]
# Settings that need the translator restarted when they change.
_RELOAD_KEYS = (
    "translator", "model_file", "gpu_layers", "server_url", "server_model",
    "server_api_key", "prompt_style", "system_prompt", "parallel",
)


def _combo(items: list[tuple[str, str]], current: str) -> QComboBox:
    combo = QComboBox()
    for key, label in items:
        combo.addItem(label, key)
    combo.setCurrentIndex(max(0, combo.findData(current)))
    return combo


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.translator_changed = False
        self.setWindowTitle("jpreplace — translation model")
        self.setMinimumWidth(560)
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.translator = _combo(TRANSLATORS, settings.translator)
        form.addRow("Translator", self.translator)

        self.model_file = QLineEdit(settings.model_file)
        self.model_file.setPlaceholderText("Path to a .gguf model")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        file_row = QHBoxLayout()
        file_row.addWidget(self.model_file, 1)
        file_row.addWidget(browse)
        self.file_row = self._row(file_row)
        form.addRow("Model file", self.file_row)
        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(0, 999)
        self.gpu_layers.setValue(settings.gpu_layers)
        self.gpu_layers.setToolTip("Model layers to run on the GPU (999 = all). Lower it if the model does not fit in VRAM.")
        form.addRow("GPU layers", self.gpu_layers)

        self.server_url = QLineEdit(settings.server_url)
        self.server_url.setPlaceholderText("e.g. http://127.0.0.1:11434/v1 (Ollama), http://127.0.0.1:1234/v1 (LM Studio)")
        form.addRow("Server URL", self.server_url)
        self.server_model = QComboBox()
        self.server_model.setEditable(True)
        self.server_model.setCurrentText(settings.server_model)
        self.server_model.lineEdit().setPlaceholderText("(the server's first model)")
        fetch = QPushButton("List models")
        fetch.clicked.connect(self._fetch_models)
        model_row = QHBoxLayout()
        model_row.addWidget(self.server_model, 1)
        model_row.addWidget(fetch)
        self.model_row = self._row(model_row)
        form.addRow("Model", self.model_row)
        self.api_key = QLineEdit(settings.server_api_key)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("(none)")
        form.addRow("API key", self.api_key)

        self.prompt_style = _combo(PROMPT_STYLES, settings.prompt_style)
        form.addRow("Prompt", self.prompt_style)
        self.system_prompt = QPlainTextEdit(settings.system_prompt)
        self.system_prompt.setPlaceholderText(DEFAULT_SYSTEM_PROMPT)
        self.system_prompt.setFixedHeight(80)
        form.addRow("Instruction", self.system_prompt)
        self.parallel = QSpinBox()
        self.parallel.setRange(1, 64)
        self.parallel.setValue(settings.parallel)
        self.parallel.setToolTip("Translation requests sent at once. Lower it for a slow or rate-limited server.")
        form.addRow("Parallel requests", self.parallel)

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.message)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.form = form

        self.translator.currentIndexChanged.connect(self._update_rows)
        self.prompt_style.currentIndexChanged.connect(self._update_rows)
        self._update_rows()

    @staticmethod
    def _row(layout: QHBoxLayout) -> QWidget:
        layout.setContentsMargins(0, 0, 0, 0)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _update_rows(self) -> None:
        kind = self.translator.currentData()
        rows = {
            self.file_row: kind == "file",
            self.gpu_layers: kind == "file",
            self.server_url: kind == "server",
            self.model_row: kind == "server",
            self.api_key: kind == "server",
            self.prompt_style: kind in ("file", "server"),
            self.system_prompt: kind in ("file", "server") and self.prompt_style.currentData() != "plamo",
        }
        for widget, visible in rows.items():
            self.form.setRowVisible(widget, visible)
        self.adjustSize()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a GGUF model", self.model_file.text(), "GGUF models (*.gguf)")
        if path:
            self.model_file.setText(path)

    def _fetch_models(self) -> None:
        if not self.server_url.text().strip():
            self.message.setText("Enter the server URL first.")
            return
        url = api_base(self.server_url.text())
        try:
            models = list_models(url, self.api_key.text().strip(), timeout=5.0)
        except Exception as exc:
            self.message.setText(f"Could not list the models at {url}: {describe_error(exc)}")
            return
        current = self.server_model.currentText()
        self.server_model.clear()
        self.server_model.addItems(models)
        self.server_model.setCurrentText(current if current in models else (models[0] if models else ""))
        self.message.setText(f"{len(models)} model{'s' if len(models) != 1 else ''} at {url}")

    def accept(self) -> None:
        kind = self.translator.currentData()
        if kind == "file" and not self.model_file.text().strip():
            self.message.setText("Choose a model file.")
            return
        if kind == "server" and not self.server_url.text().strip():
            self.message.setText("Enter the server URL.")
            return
        s = self.settings
        before = {key: getattr(s, key) for key in _RELOAD_KEYS}
        s.translator = kind
        s.model_file = self.model_file.text().strip().strip('"')
        s.gpu_layers = self.gpu_layers.value()
        s.server_url = self.server_url.text().strip()
        s.server_model = self.server_model.currentText().strip()
        s.server_api_key = self.api_key.text().strip()
        s.prompt_style = self.prompt_style.currentData()
        s.system_prompt = self.system_prompt.toPlainText().strip()
        s.parallel = self.parallel.value()
        self.translator_changed = before != {key: getattr(s, key) for key in _RELOAD_KEYS}
        super().accept()
