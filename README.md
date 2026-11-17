# jpreplace

A Google Lens-style overlay for Windows. Put the frame over anything on screen
that contains Japanese; the text is erased and re-drawn in English, in place.
Everything runs locally on your GPU.

- **OCR**: PP-OCRv6 (detection + recognition) on ONNX Runtime / CUDA.
- **Translation**: [PLaMo 3 Translate](https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF)
  (a dedicated JA↔EN translation LLM) served by llama.cpp, or
  [your own model](#using-another-translation-model): any GGUF file, or any
  OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp, a hosted API).
- **Overlay**: PySide6. The translated layer is click-through and hidden from
  screen capture, so the app can keep re-reading the original screen below it.

## Install

Requires Windows 10 2004+ / Windows 11, an NVIDIA GPU with a recent driver,
and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run jpreplace-setup     # downloads llama.cpp + models (about 8 GB), once
```

## Run

```
uv run jpreplace           # or double-click run.bat
```

Drag the title bar to move the frame and its edges to resize it. Then:

| Action | Button | Default hotkey (global) |
| --- | --- | --- |
| Translate what is inside the frame | Translate | `Ctrl+Alt+T` |
| Follow the screen and re-translate when it changes | Live | `Ctrl+Alt+L` |
| Show the original again | Clear | `Ctrl+Alt+C` |
| Attach the frame to the window under it | Bind | |
| Full screen: no frame, cover the bound window or the whole monitor | ⛶ | `Ctrl+Alt+F` |
| Choose the translation model | ⚙ | |

Clicks inside the frame go to the application underneath, so you can keep
scrolling or playing with the overlay on. In live mode, text that was already
translated is re-placed instantly from a cache; only new text is translated.
Translating a screen whose text has not changed leaves the overlay as it is.

**Bind**: press Bind and choose a window (a game, a browser):

- **Click a window…**: the window under the mouse is highlighted; click it to
  bind (the click does not reach the application). Right-click or Esc cancels.
- one of the **open windows** listed in the menu;
- or **Window under the frame (keep this area)** to cover only the part of a
  window that the frame is over now.

The frame jumps over the window's content and its border turns green. From
then on it moves and resizes with that window, and hides while the window is
minimised, on another virtual desktop or covered by another application. You
can still drag or resize the frame to cover only part of the window; it keeps
that part. Press Bind again to release it.

**Full screen** (for games and videos running full screen): the frame
disappears and the overlay covers the bound window, or, if nothing is bound,
the whole monitor the frame is on. Use the hotkeys; status messages appear
briefly in the top-right corner. Games in *borderless* or *windowed* full
screen work. DirectX 11/12 games in "full screen" usually do too, since
Windows normally runs them borderless behind the scenes. A game that takes the display
exclusively (some older DirectX 9 or OpenGL games) cannot have anything drawn
over it: switch it to borderless.

## Speed

Measured on an RTX 4080, default settings, end to end (capture → OCR →
translation → rendering):

| Content | Japanese text | Time |
| --- | --- | --- |
| Manga page, 3 speech bubbles | 55 characters | 0.2 s |
| Game dialogue + menu | 63 characters | 0.3 s |
| News article with navigation | 270 characters | 0.8 s |
| Full 1600×1000 Wikipedia page | 1,400 characters | 1.4 s |
| Full 1600×1000 page of dense prose | 1,900 characters | 1.5 s |

The first translation after starting or resizing the frame can take up to a
second longer. Reproduce with `uv run python scripts/translate_image.py <screenshots>`, and
check the real overlay window (including Bind and full screen) with
`uv run python scripts/selftest_overlay.py`.

## Settings

`settings.json` (created next to this file on first exit):

| Key | Default | Meaning |
| --- | --- | --- |
| `translator` | `"auto"` | `"plamo-8b"`: best quality. `"plamo-2b"`: fastest, least VRAM. `"auto"`: both loaded; the 8B is used unless a screen has more untranslated text than `quality_max_chars`, then the 2B keeps it fast. `"file"` / `"server"`: see below. |
| `quality_max_chars` | `1200` | The `auto` switch-over point, in Japanese characters. |
| `model_file`, `gpu_layers` | `""`, `999` | For `"file"`: the GGUF model, and how many of its layers go on the GPU. |
| `server_url`, `server_model`, `server_api_key` | `""` | For `"server"`. An empty model means the server's first one. |
| `server_extra` | `{}` | Extra JSON fields for every request to the server, e.g. `{"top_p": 0.9}`. |
| `prompt_style` | `"auto"` | `"plamo"`, `"chat"`, or `"auto"`: PLaMo's format if the model's name contains "plamo", else chat. |
| `system_prompt` | `""` | Replaces the built-in instruction for chat models. |
| `ocr_scale` | `1.0` | Set to `2.0` if very small text is misread (slower). |
| `parallel` | `16` | Concurrent translation requests. Lower it for a slow or rate-limited server. |
| `hotkey_translate`, `hotkey_clear`, `hotkey_live`, `hotkey_fullscreen` | see above | e.g. `"ctrl+shift+f9"`. |
| `live_interval_ms` | `400` | How often live mode looks at the screen. |
| `capture` | `"auto"` | `"dxgi"` (desktop duplication: sees full-screen games and hardware video), `"gdi"`, or `"auto"`: DXGI with GDI as the fallback. |
| `font_family` | `"Segoe UI"` | Font for the English text. |

VRAM use is about 9 GB in `auto`, 6 GB with only the 8B and 3 GB with only
the 2B. If you run a demanding game next to the overlay, pick a single model.

## Using another translation model

Open ⚙ in the title bar (or edit `settings.json`) and pick:

- **Local model file**: any GGUF model, run with the bundled llama.cpp on
  your GPU. General chat models (Qwen, Gemma, Llama, Mistral…) are given an
  instruction to translate; "thinking" is switched off so they answer at
  once. If the model does not fit in VRAM, lower *GPU layers*.
- **LLM server**: anything that speaks the OpenAI API, for example
  `http://127.0.0.1:11434/v1` (Ollama), `http://127.0.0.1:1234/v1`
  (LM Studio), a llama.cpp or vLLM server, or a hosted API with its key.
  *List models* shows what the server offers. A PLaMo model served this way
  is recognised by its name and gets its own prompt format.

The translator restarts with the new model when you press OK; the OCR stays
loaded. If the model cannot be reached or does not answer, the title bar
says why and you can fix it in ⚙. Translations are about as good as the
model: PLaMo Translate, made for this, remains the recommended choice;
small general models may translate menus and short labels less reliably.

## How it works

1. The frame's interior is captured with DXGI desktop duplication (`dxcam`),
   or GDI (`mss`) as a fallback. The overlay itself is excluded from capture
   with `SetWindowDisplayAffinity`.
2. PP-OCRv6 finds text lines and reads them; lines are grouped into blocks
   (paragraphs, labels, vertical speech bubbles). Blocks without Japanese are
   left alone.
3. Blocks are translated concurrently. Long blocks are split into sentences so
   that many short requests run in parallel instead of one long one; short
   labels get neighbouring text as context so that, say, ロード on a game menu
   becomes "Load" rather than "Road".
4. The original glyphs are removed and the English is drawn in the same
   colour, sized to fit, using free space next to the block when there is
   some. Plain backgrounds get a flat fill. Repeating patterns (hatching,
   manga screentone, stripes, tiled textures) are rebuilt by copying the
   pattern, in phase, from where it repeats next to the text, and busy ones
   get the text drawn with a halo, as in manga. Other images get a
   stroke-level background reconstruction.

## Limitations

- English usually needs more room than Japanese, so text in tight spots is
  drawn smaller than the original.
- The overlay cannot appear in screenshots or screen sharing (that is what
  lets it read the screen underneath).
- Games using exclusive full screen cannot be overlaid; use borderless mode.
- Stylised, hand-drawn or very small text may be misread by the OCR, and text
  sitting directly on dense, dark hatching (with no light outline) may not be
  detected at all.

## Licences of the downloaded models

PLaMo 3 Translate is under the PLaMo Community License: free for personal and
non-commercial use; commercial use needs registration with Preferred Networks.
PP-OCRv6 is Apache-2.0, llama.cpp is MIT.

## Development

```
uv run pytest                                   # unit tests (no GPU needed)
uv run python tests/make_samples.py             # synthetic test screens -> out/samples
uv run python scripts/translate_image.py out/samples/webpage.png
uv run python scripts/bench_translate.py out/samples/webpage.png --show
```
