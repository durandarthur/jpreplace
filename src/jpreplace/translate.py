"""Japanese -> English translation with PLaMo 3 Translate on a local llama.cpp server."""

from __future__ import annotations

import atexit
import subprocess
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import httpx

_STOP = ["<|plamo:tag|>", "<|plamo:bos|>"]
_CTX_PER_SLOT = 2048
_MAX_CONTEXT_CHARS = 400


def build_prompt(text: str, context: str = "") -> str:
    """PFML3 prompt, the native format of PLaMo translation models."""
    parts = ["<|plamo:tag|>dataset<|plamo:msg|>translation"]
    if context:
        parts.append(f"<|plamo:tag|>context<|plamo:msg|>{context}")
    parts.append(f"<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>{text}")
    parts.append("<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>")
    return "".join(parts)


class LlamaServer:
    """Owns a llama-server child process."""

    def __init__(self, exe: Path, model: Path, port: int = 8765, parallel: int = 8) -> None:
        self.exe, self.model, self.port, self.parallel = exe, model, port, parallel
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.log_path = exe.parent.parent / "llama-server.log"

    def start(self, timeout: float = 120.0) -> None:
        if self.healthy():
            return  # reuse a server that is already running on this port
        cmd = [
            str(self.exe),
            "-m", str(self.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-ngl", "999",
            "-np", str(self.parallel),
            "-c", str(self.parallel * _CTX_PER_SLOT),
            "--no-webui",
        ]
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=self.exe.parent,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        atexit.register(self.stop)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            if self.healthy():
                return
            time.sleep(0.2)
        self.stop()
        raise TimeoutError(f"llama-server did not become ready; see {self.log_path}")

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


class Translator:
    """Translates many text blocks concurrently, with an LRU cache."""

    def __init__(self, url: str, parallel: int = 8, cache_size: int = 4096) -> None:
        self.url = url
        self.parallel = parallel
        self.client = httpx.Client(timeout=60.0, limits=httpx.Limits(max_connections=parallel * 2))
        self.pool = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="translate")
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.cache_size = cache_size
        self._lock = threading.Lock()

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.client.close()

    def cached(self, text: str) -> str | None:
        with self._lock:
            hit = self.cache.get(text)
            if hit is not None:
                self.cache.move_to_end(text)
            return hit

    def _remember(self, text: str, translation: str) -> None:
        with self._lock:
            self.cache[text] = translation
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

    def _request(self, text: str, context: str) -> str:
        max_tokens = min(768, 48 + len(text) * 3)
        r = self.client.post(
            f"{self.url}/completion",
            json={
                "prompt": build_prompt(text, context),
                "n_predict": max_tokens,
                "temperature": 0,
                "stop": _STOP,
                "cache_prompt": False,
            },
        )
        r.raise_for_status()
        return r.json()["content"].strip()

    def translate_one(self, text: str, context: str = "") -> str:
        hit = self.cached(text)
        if hit is not None:
            return hit
        out = self._request(text, context)
        self._remember(text, out)
        return out

    def translate(
        self,
        texts: list[str],
        use_context: bool = True,
        on_result: Callable[[int, str], None] | None = None,
    ) -> list[str]:
        """Translate `texts` (in reading order). Longest blocks are sent first
        so the slowest request starts earliest."""
        results: list[str | None] = [None] * len(texts)
        pending = []
        for i, text in enumerate(texts):
            hit = self.cached(text)
            if hit is not None:
                results[i] = hit
                if on_result:
                    on_result(i, hit)
            else:
                pending.append(i)
        pending.sort(key=lambda i: -len(texts[i]))

        def work(i: int) -> None:
            context = _neighbour_context(texts, i) if use_context else ""
            results[i] = self.translate_one(texts[i], context)
            if on_result:
                on_result(i, results[i])

        for future in [self.pool.submit(work, i) for i in pending]:
            future.result()
        return [r or "" for r in results]


def _neighbour_context(texts: list[str], i: int) -> str:
    """Nearby blocks on screen, so short labels are translated in context."""
    picked: list[tuple[int, str]] = []
    budget = _MAX_CONTEXT_CHARS
    for offset in range(1, len(texts)):
        for j in (i - offset, i + offset):
            if 0 <= j < len(texts) and budget > 0:
                snippet = texts[j][:budget]
                picked.append((j, snippet))
                budget -= len(snippet)
        if budget <= 0:
            break
    return "\n".join(t for _, t in sorted(picked))
