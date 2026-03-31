"""Regex patterns for model-emitted tool/script tags."""

from __future__ import annotations

import re

IDASCRIPT_PATTERN = re.compile(r"<idascript>(.*?)</idascript>", re.DOTALL)
IDATOOL_PATTERN = re.compile(
    r"<idatool\s+name=['\"](.*?)['\"]\s*>(.*?)</idatool>",
    re.DOTALL | re.IGNORECASE,
)
DELEGATE_PATTERN = re.compile(r"<delegate agent=['\"](.*?)['\"]>(.*?)</delegate>", re.DOTALL | re.IGNORECASE)
