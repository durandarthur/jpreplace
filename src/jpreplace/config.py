"""Paths, model catalogue and user settings."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _root() -> Path:
    env = os.environ.get("JPREPLACE_HOME")
    if env:
        return Path(env)
    # src/jpreplace/config.py -> project root
    return Path(__file__).resolve().parents[2]


ROOT = _root()
MODELS_DIR = ROOT / "models"
RUNTIME_DIR = ROOT / "runtime"
LLAMA_DIR = RUNTIME_DIR / "llama"
SETTINGS_PATH = ROOT / "settings.json"

LLAMA_BUILD = "b12428"
LLAMA_ASSETS = (
    f"llama-{LLAMA_BUILD}-bin-win-cuda-13.5-x64.zip",
    "cudart-llama-bin-win-cuda-13.5-x64.zip",
)
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{build}/{name}"


@dataclass(frozen=True)
class HFFile:
    repo: str
    filename: str

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.repo.split("/")[-1] / self.filename


# Translation models (GGUF, served by llama.cpp). PLaMo 3 Translate is a
# dedicated JA<->EN translation LLM; the 8B is the quality pick, the 2B the
# speed pick.
TRANSLATORS: dict[str, HFFile] = {
    "plamo-8b": HFFile("pfnet/plamo-3-translate-8b-GGUF", "plamo-3-translate-8b-Q4_K_M.gguf"),
    "plamo-2b": HFFile("pfnet/plamo-3-translate-2b-GGUF", "plamo-3-translate-2b-Q4_K_M.gguf"),
}

# OCR models (ONNX). Each entry is (detector, recognizer).
OCR_MODELS: dict[str, tuple[str, str]] = {
    "medium": ("PaddlePaddle/PP-OCRv6_medium_det_onnx", "PaddlePaddle/PP-OCRv6_medium_rec_onnx"),
    "small": ("PaddlePaddle/PP-OCRv6_small_det_onnx", "PaddlePaddle/PP-OCRv6_small_rec_onnx"),
}
OCR_FILES = ("inference.onnx", "inference.yml")


def ocr_dir(repo: str) -> Path:
    return MODELS_DIR / repo.split("/")[-1]


@dataclass
class Settings:
    # "plamo-8b" (best quality), "plamo-2b" (fastest, least VRAM), "auto" (both
    # loaded; the 2B takes over when a screen has too much text for the 8B),
    # "file" (any GGUF model, see model_file) or "server" (an LLM server you
    # run yourself or a hosted API, see server_url).
    translator: str = "auto"
    quality_max_chars: int = 1200  # auto: more untranslated characters than this -> 2B
    # translator "file": a .gguf model run with the bundled llama.cpp.
    model_file: str = ""
    gpu_layers: int = 999  # layers offloaded to the GPU; lower it if the model does not fit
    # translator "server": base URL of an OpenAI-compatible API, e.g.
    # http://127.0.0.1:11434/v1 (Ollama) or http://127.0.0.1:1234/v1 (LM Studio).
    server_url: str = ""
    server_model: str = ""  # "" = the first model the server lists
    server_api_key: str = ""
    server_extra: dict = field(default_factory=dict)  # extra JSON fields sent with every request
    # "plamo" (PLaMo Translate's native prompt), "chat" (an instruction for a
    # general chat model) or "auto" (plamo if the model's name says PLaMo).
    prompt_style: str = "auto"
    system_prompt: str = ""  # chat style; "" = the built-in instruction
    ocr: str = "medium"
    llama_port: int = 8765
    parallel: int = 16
    hotkey_translate: str = "ctrl+alt+t"
    hotkey_clear: str = "ctrl+alt+c"
    hotkey_live: str = "ctrl+alt+l"
    hotkey_fullscreen: str = "ctrl+alt+f"
    live: bool = False
    live_interval_ms: int = 400
    font_family: str = "Segoe UI"
    ocr_scale: float = 1.0  # 2.0 helps with very small text, at some speed cost
    # "auto" (DXGI desktop duplication, falling back to GDI), "dxgi" or "gdi".
    capture: str = "auto"
    geometry: list[int] = field(default_factory=lambda: [200, 200, 900, 600])

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self) -> None:
        SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def builtin_translators(self) -> list[str]:
        """The bundled PLaMo models this configuration needs."""
        if self.translator == "auto":
            return ["plamo-8b", "plamo-2b"]
        return [self.translator] if self.translator in TRANSLATORS else []

    def resolved_prompt_style(self, model_name: str) -> str:
        if self.prompt_style in ("plamo", "chat"):
            return self.prompt_style
        return "plamo" if "plamo" in model_name.lower() else "chat"
