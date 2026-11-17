"""Japanese -> English translation with an LLM: PLaMo 3 Translate (or any GGUF
model) on a local llama.cpp server, or any OpenAI-compatible API."""

from __future__ import annotations

import atexit
import re
import socket
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

_STOP = ["<|plamo:tag|>", "<|plamo:bos|>"]
_CTX_PER_SLOT = 768  # the KV cache is shared, so this is an average, not a cap
_CHAT_CTX_PER_SLOT = 1024  # chat prompts also carry an instruction and a chat template
_MAX_UNIT_CHARS = 160  # hard upper bound on one request
_CHUNK_CHARS = 70  # long blocks are split into sentence groups of about this size
_LABEL_CHARS = 90  # units shorter than this get surrounding text as context
_CONTEXT_BUDGET = 1500  # total context characters per screen; context costs time
_SENTENCE_END = re.compile(r"(?<=[。！？!?])(?![。！？!?」』）)\]】])")
_THINKING = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_LABEL_PREFIX = re.compile(r"^(?:english|translation|text to translate|英語|英訳)\s*[:：]\s*", re.IGNORECASE)
_ERRORS = (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError)

DEFAULT_SYSTEM_PROMPT = (
    "You translate Japanese text read off a screen (game dialogue, manga, menus, web pages) "
    "into natural, fluent English. Reply with the English translation only: no explanations, "
    "notes, romanisation or quotation marks around it. Keep names and numbers. Translate a "
    "single word or button label as a short label."
)


def build_prompt(text: str, context: str = "") -> str:
    """PFML3 prompt, the native format of PLaMo translation models."""
    parts = ["<|plamo:tag|>dataset<|plamo:msg|>translation"]
    if context:
        parts.append(f"<|plamo:tag|>context<|plamo:msg|>{context}")
    parts.append(f"<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>{text}")
    parts.append("<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>")
    return "".join(parts)


def chat_messages(text: str, context: str = "", system_prompt: str = "") -> list[dict[str, str]]:
    """Prompt for a general-purpose chat model."""
    # Instruction first and the text last: small models otherwise tend to
    # translate the context instead of the text.
    if context:
        request = (
            "Translate the text into English. Other text from the same screen is given only to "
            f"help you pick the right meaning: do not translate it.\n\nOther text on the screen:\n"
            f"{context}\n\nText to translate:\n{text}"
        )
    else:
        request = f"Translate the text into English.\n\nText to translate:\n{text}"
    return [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": request},
    ]


def clean_output(text: str) -> str:
    """Strip what chat models put around a translation: reasoning, a label."""
    text = _THINKING.sub("", text).strip()
    return _LABEL_PREFIX.sub("", text).strip()


def api_base(url: str) -> str:
    """Normalise what a user may paste ("localhost:11434", ".../v1/chat/completions")
    to the base URL of an OpenAI-compatible API."""
    url = url.strip().rstrip("/")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url
    for suffix in ("/chat/completions", "/completions", "/models"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    if urlsplit(url).path in ("", "/"):
        url += "/v1"
    return url


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def list_models(base_url: str, api_key: str = "", timeout: float = 5.0) -> list[str]:
    """Model ids an OpenAI-compatible server offers."""
    r = httpx.get(f"{base_url}/models", headers=_headers(api_key), timeout=timeout)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and "id" in m]


def describe_error(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = " ".join(exc.response.text.split())
        return f"HTTP {exc.response.status_code} from {exc.request.url}" + (f": {body[:200]}" if body else "")
    if isinstance(exc, httpx.ConnectError):
        return f"cannot connect to {exc.request.url}"
    if isinstance(exc, (KeyError, IndexError, TypeError, ValueError)):
        return f"unexpected reply ({type(exc).__name__}: {exc})"
    return str(exc) or type(exc).__name__


class LlamaServer:
    """Owns a llama-server child process."""

    def __init__(
        self,
        exe: Path,
        model: Path,
        port: int = 8765,
        parallel: int = 8,
        gpu_layers: int = 999,
        chat: bool = False,
    ) -> None:
        self.exe, self.model, self.port, self.parallel = exe, model, port, parallel
        self.gpu_layers, self.chat = gpu_layers, chat
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.log_path = exe.parent.parent / f"llama-server-{model.stem}.log"

    def start(self, timeout: float = 300.0) -> None:
        self.port = _free_port(self.port)
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [
            str(self.exe),
            "-m", str(self.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-ngl", str(self.gpu_layers),
            "-np", str(self.parallel),
            "-c", str(self.parallel * (_CHAT_CTX_PER_SLOT if self.chat else _CTX_PER_SLOT)),
            # One shared KV cache: with per-slot caches llama.cpp cannot batch
            # requests of different lengths together, which halves throughput.
            "--kv-unified",
            "--cache-ram", "0",  # no host-RAM prompt cache; every prompt is new
            "--no-webui",
        ]
        if self.chat:
            cmd += ["--reasoning", "off"]  # thinking models: answer straight away
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=self.exe.parent,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        atexit.register(self.stop)
        if sys.platform == "win32":
            from . import win32

            # Never leave a multi-gigabyte model behind if we die unexpectedly.
            win32.kill_with_parent(int(self.proc._handle))
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


def _free_port(preferred: int) -> int:
    """`preferred` if nothing is listening there, otherwise any free port."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free local port for llama-server")


def split_sentences(text: str, limit: int = _CHUNK_CHARS) -> list[str]:
    """Split a long block into chunks of whole sentences. Generation speed per
    request is fixed, so many short requests in parallel finish far sooner
    than one long one."""
    if len(text) <= limit * 1.3:
        return [text]
    sentences: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        # A "sentence" far beyond the limit (no full stops at all) is cut at
        # commas, or anywhere as a last resort, to bound time and memory.
        while len(sentence) > _MAX_UNIT_CHARS:
            cut = sentence.rfind("、", limit // 2, _MAX_UNIT_CHARS) + 1 or _MAX_UNIT_CHARS
            sentences.append(sentence[:cut])
            sentence = sentence[cut:]
        if sentence:
            sentences.append(sentence)
    chunks: list[str] = []
    for sentence in sentences:
        if chunks and len(chunks[-1]) + len(sentence) <= limit:
            chunks[-1] += sentence
        else:
            chunks.append(sentence)
    return chunks


@dataclass
class Backend:
    """One translation server. `max_chars` is the largest amount of
    untranslated Japanese it can handle within the time budget (0 = no limit).

    `native` means a llama-server we started ourselves: `url` is its root, and
    PLaMo prompts go to its raw /completion endpoint. Otherwise `url` is the
    base of an OpenAI-compatible API. `style` is "plamo" or "chat"."""

    url: str
    parallel: int
    max_chars: int = 0
    style: str = "plamo"
    native: bool = True
    model: str = ""
    api_key: str = ""
    system_prompt: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class _Unit:
    block: int
    text: str
    context: str = ""
    result: str | None = None


class Translator:
    """Translates the text blocks of a screen concurrently, with an LRU cache.

    With several backends (best quality first) the best one that can finish
    the pending work within its budget is used."""

    def __init__(self, backends: list[Backend], cache_size: int = 8192) -> None:
        self.backends = backends
        workers = max(b.parallel for b in backends)
        self.client = httpx.Client(timeout=60.0, limits=httpx.Limits(max_connections=workers * 2))
        self.pools = [ThreadPoolExecutor(b.parallel, thread_name_prefix="translate") for b in backends]
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.cache_size = cache_size
        self._lock = threading.Lock()
        self.last_error: str | None = None  # from the latest translate() call

    def close(self) -> None:
        for pool in self.pools:
            pool.shutdown(wait=False, cancel_futures=True)
        self.client.close()

    def _lookup(self, text: str) -> str | None:
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

    def cached(self, text: str) -> str | None:
        """The translation of a block if every part of it is already known."""
        parts = [self._lookup(chunk) for chunk in split_sentences(text)]
        return None if None in parts else " ".join(parts)

    def _call(self, backend: Backend, text: str, context: str = "") -> str:
        """One translation request; raises on any failure."""
        limit = min(400, 48 + len(text) * 3)
        base = f"{backend.url}/v1" if backend.native else backend.url
        if backend.style == "chat":
            url = f"{base}/chat/completions"
            body: dict = {
                "messages": chat_messages(text, context, backend.system_prompt),
                "max_tokens": limit,
                "temperature": 0,
            }
        elif backend.native:
            url = f"{backend.url}/completion"
            body = {
                "prompt": build_prompt(text, context),
                "n_predict": limit,
                "temperature": 0,
                "stop": _STOP,
                "cache_prompt": False,
            }
        else:
            url = f"{base}/completions"
            body = {"prompt": build_prompt(text, context), "max_tokens": limit, "temperature": 0, "stop": _STOP}
        if backend.model:
            body["model"] = backend.model
        body.update(backend.extra)
        r = self.client.post(url, json=body, headers=_headers(backend.api_key))
        r.raise_for_status()
        data = r.json()
        if backend.style == "chat":
            content = data["choices"][0]["message"]["content"]
        elif backend.native:
            content = data["content"]
        else:
            content = data["choices"][0]["text"]
        return clean_output(content or "")

    def _request(self, backend: Backend, unit: _Unit) -> None:
        try:
            unit.result = self._call(backend, unit.text, unit.context)
        except _ERRORS as exc:
            unit.result = ""  # leave this piece untranslated rather than fail the screen
            self.last_error = describe_error(exc)
            return
        self._remember(unit.text, unit.result)

    def warm_up(self) -> None:
        """Run one tiny request through every backend so the first real
        translation does not pay for lazy initialisation, and so a backend
        that does not work is reported now rather than as missing text."""
        for backend in self.backends:
            try:
                self._call(backend, "準備完了")
            except _ERRORS as exc:
                raise RuntimeError(f"the translation model does not answer: {describe_error(exc)}") from exc

    def pick_backend(self, pending_chars: int) -> int:
        for i, backend in enumerate(self.backends):
            if not backend.max_chars or pending_chars <= backend.max_chars:
                return i
        return len(self.backends) - 1

    def translate(self, texts: list[str], use_context: bool = True) -> list[str]:
        """Translate `texts`, the blocks of one screen in reading order."""
        self.last_error = None
        units = [_Unit(i, chunk) for i, text in enumerate(texts) for chunk in split_sentences(text)]
        pending = []
        for unit in units:
            unit.result = self._lookup(unit.text)
            if unit.result is None:
                pending.append(unit)
        if pending:
            if use_context:
                _add_context(units, texts)
            choice = self.pick_backend(sum(len(u.text) for u in pending))
            backend, pool = self.backends[choice], self.pools[choice]
            # Longest first, so the slowest request starts earliest.
            pending.sort(key=lambda u: -len(u.text))
            for future in [pool.submit(self._request, backend, unit) for unit in pending]:
                future.result()
        out = [""] * len(texts)
        for unit in units:
            out[unit.block] = f"{out[unit.block]} {unit.result}".strip()
        return out


def _add_context(units: list[_Unit], texts: list[str]) -> None:
    """Give each untranslated unit some surrounding text to disambiguate it.

    A later sentence of a block gets the sentence before it. A short block (a
    label, a button) gets its neighbours on screen. Long self-contained
    text gets nothing: context is prompt tokens, and those cost time."""
    wants: list[tuple[_Unit, str]] = []
    for index, unit in enumerate(units):
        if unit.result is not None:
            continue
        previous = units[index - 1] if index and units[index - 1].block == unit.block else None
        if previous is not None:
            wants.append((unit, previous.text[-50:]))
            continue
        room = _LABEL_CHARS - len(unit.text)
        if room <= 0:
            continue
        picked: list[tuple[int, str]] = []
        for offset in range(1, len(texts)):
            for j in (unit.block - offset, unit.block + offset):
                if 0 <= j < len(texts) and room > 0:
                    snippet = texts[j][:room]
                    picked.append((j, snippet))
                    room -= len(snippet)
            if room <= 0:
                break
        wants.append((unit, "\n".join(t for _, t in sorted(picked))))
    total = sum(len(context) for _, context in wants)
    scale = min(1.0, _CONTEXT_BUDGET / total) if total else 1.0
    for unit, context in wants:
        keep = int(len(context) * scale)
        unit.context = context[:keep] if keep >= 8 else ""
