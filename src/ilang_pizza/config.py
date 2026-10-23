"""Runtime settings, all overridable with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

# Where `ilang-pizza` looks for events.csv and rsvps/ when --data is not given.
DATA_DIR = Path(os.getenv("ILANG_PIZZA_DATA", "data"))

# Weather responses and other small caches.
CACHE_DIR = Path(os.getenv("ILANG_PIZZA_CACHE", "~/.cache/ilang-pizza")).expanduser()

# Local LLM served by Ollama. Gemma 4 E4B fits comfortably on a 16 GB Mac.
LLM_MODEL = os.getenv("ILANG_PIZZA_LLM", "gemma4:e4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Which TabPFN checkpoint to use: "auto" picks TabPFN-3.5 when a Prior Labs license token is
# available (TABPFN_TOKEN) and falls back to the openly licensed TabPFN v2 weights otherwise.
TABPFN_VERSION = os.getenv("ILANG_PIZZA_TABPFN", "auto")

# Fallback coordinates when an event has no lat/lon (Metro Manila).
METRO_MANILA = (14.5995, 120.9842)

# Organizers' current rule of thumb: "about half of the RSVPs will show up".
RULE_OF_THUMB_SHOW_RATE = float(os.getenv("ILANG_PIZZA_RULE_RATE", "0.5"))
