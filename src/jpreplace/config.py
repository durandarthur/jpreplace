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
    translator: str = "plamo-8b"
    ocr: str = "medium"
    llama_port: int = 8765
    parallel: int = 8
    hotkey: str = "ctrl+shift+t"
    live: bool = False
    live_interval_ms: int = 400
    font_family: str = "Segoe UI"
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
