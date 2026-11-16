"""Download llama.cpp binaries and model weights (`jpreplace-setup`)."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Callable

import httpx
from huggingface_hub import hf_hub_download

from . import config


def llama_server_path() -> Path:
    return config.LLAMA_DIR / "llama-server.exe"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done * 100 // total}%", end="", flush=True)
    print()
    tmp.replace(dest)


Status = Callable[[str], None]


def ensure_llama(status: Status = print) -> Path:
    exe = llama_server_path()
    marker = config.LLAMA_DIR / f".{config.LLAMA_BUILD}"
    if exe.exists() and marker.exists():
        return exe
    for name in config.LLAMA_ASSETS:
        url = config.LLAMA_URL.format(build=config.LLAMA_BUILD, name=name)
        archive = config.RUNTIME_DIR / name
        status(f"Downloading {name}…")
        _download(url, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(config.LLAMA_DIR)
        archive.unlink()
    if not exe.exists():
        raise RuntimeError(f"llama-server.exe not found in {config.LLAMA_DIR} after extraction")
    marker.touch()
    return exe


def _hf(repo: str, filename: str, status: Status) -> Path:
    local_dir = config.MODELS_DIR / repo.split("/")[-1]
    target = local_dir / filename
    if target.exists():
        return target
    status(f"Downloading {filename} (first run only)…")
    return Path(hf_hub_download(repo, filename, local_dir=local_dir))


def ensure_translator(name: str, status: Status = print) -> Path:
    f = config.TRANSLATORS[name]
    return _hf(f.repo, f.filename, status)


def ensure_ocr(name: str, status: Status = print) -> tuple[Path, Path]:
    det, rec = config.OCR_MODELS[name]
    for repo in (det, rec):
        for filename in config.OCR_FILES:
            _hf(repo, filename, status)
    return config.ocr_dir(det), config.ocr_dir(rec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download everything jpreplace needs to run offline.")
    p.add_argument("--translator", nargs="*", default=None, choices=list(config.TRANSLATORS))
    p.add_argument("--ocr", nargs="*", default=None, choices=list(config.OCR_MODELS))
    args = p.parse_args(argv)
    settings = config.Settings.load()

    ensure_llama()
    for name in args.ocr or [settings.ocr]:
        ensure_ocr(name)
    for name in args.translator or [settings.translator]:
        ensure_translator(name)
    print("All assets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
