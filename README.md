# jpreplace

A Google Lens-style overlay for Windows. Put the frame over anything on screen
that contains Japanese; the text is erased and re-drawn in English, in place.
Everything runs locally on your GPU.

- **OCR**: PP-OCRv6 (detection + recognition) on ONNX Runtime / CUDA.
- **Translation**: [PLaMo 3 Translate](https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF)
  (a dedicated JA↔EN translation LLM) served by llama.cpp.
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

Clicks inside the frame go to the application underneath, so you can keep
scrolling or playing with the overlay on. In live mode, text that was already
translated is re-placed instantly from a cache; only new text is translated.

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
check the real overlay window with `uv run python scripts/selftest_overlay.py`.

## Settings

`settings.json` (created next to this file on first exit):

| Key | Default | Meaning |
| --- | --- | --- |
| `translator` | `"auto"` | `"plamo-8b"`: best quality. `"plamo-2b"`: fastest, least VRAM. `"auto"`: both loaded; the 8B is used unless a screen has more untranslated text than `quality_max_chars`, then the 2B keeps it fast. |
| `quality_max_chars` | `1200` | The `auto` switch-over point, in Japanese characters. |
| `ocr_scale` | `1.0` | Set to `2.0` if very small text is misread (slower). |
| `parallel` | `16` | Concurrent translation requests. |
| `hotkey_translate`, `hotkey_clear`, `hotkey_live` | see above | e.g. `"ctrl+shift+f9"`. |
| `live_interval_ms` | `400` | How often live mode looks at the screen. |
| `font_family` | `"Segoe UI"` | Font for the English text. |

VRAM use is about 9 GB in `auto`, 6 GB with only the 8B and 3 GB with only
the 2B. If you run a demanding game next to the overlay, pick a single model.

## How it works

1. The frame's interior is captured (`mss`). The overlay itself is excluded
   from capture with `SetWindowDisplayAffinity`.
2. PP-OCRv6 finds text lines and reads them; lines are grouped into blocks
   (paragraphs, labels, vertical speech bubbles). Blocks without Japanese are
   left alone.
3. Blocks are translated concurrently. Long blocks are split into sentences so
   that many short requests run in parallel instead of one long one; short
   labels get neighbouring text as context so that, say, ロード on a game menu
   becomes "Load" rather than "Road".
4. The original glyphs are removed (flat fill on plain backgrounds, stroke-level
   background reconstruction on images) and the English is drawn in the same
   colour, sized to fit, using free space next to the block when there is some.

## Limitations

- English usually needs more room than Japanese, so text in tight spots is
  drawn smaller than the original.
- The overlay cannot appear in screenshots or screen sharing (that is what
  lets it read the screen underneath).
- Exclusive-fullscreen games cannot be overlaid; use borderless windowed mode.
- Stylised, hand-drawn or very small text may be misread by the OCR.

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
