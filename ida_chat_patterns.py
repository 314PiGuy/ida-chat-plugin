"""Regex patterns for model-emitted tool/script tags."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

IDASCRIPT_PATTERN = re.compile(r"<idascript>(.*?)</idascript>", re.DOTALL)
IDATOOL_PATTERN = re.compile(
    r"<idatool\b[^>]*>(.*?)</idatool>",
    re.DOTALL | re.IGNORECASE,
)
DELEGATE_PATTERN = re.compile(r"<delegate\b[^>]*>(.*?)</delegate>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class AgentTagBlock:
    """One parsed wrapper block in model output text."""

    tag: str
    attrs: str
    content: str
    start: int
    end: int
    recovered: bool


TAG_OPEN_PATTERN = re.compile(r"<\s*(idascript|idatool|delegate)\b([^>]*)>", re.IGNORECASE)


def _normalize_agent_text(text: str) -> str:
    """Normalize common malformed wrappers seen in model output."""
    text = text or ""
    text = re.sub(r"</?\s*parallel\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*legate\b", "<delegate", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*legate\s*>", "</delegate>", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*idatools\s*>", "</idatool>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*idatool([a-z0-9_./:-]+)\s*>", r"<idatool \1>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*delegate([a-z0-9_./:-]+)\s*>", r"<delegate \1>", text, flags=re.IGNORECASE)
    return text


def _extract_named_attr(attrs: str, key: str) -> str:
    pattern = rf"\b{re.escape(key)}\s*=\s*(?:'([^']+)'|\"([^\"]+)\"|([^\s>]+))"
    match = re.search(pattern, attrs, flags=re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1) or match.group(2) or match.group(3) or ""
    return value.strip().strip("\"'")


def _extract_loose_attr_token(attrs: str) -> str:
    for raw in attrs.strip().split():
        token = raw.strip().strip("\"'")
        if not token or token.startswith("/"):
            continue
        if "=" in token:
            continue
        return token
    return ""


def _normalize_wrapper_payload(payload: str) -> str:
    """Normalize wrapper payload text before lightweight validation."""
    text = (payload or "").strip()
    text = re.sub(r"^\s*<!\[CDATA\[", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\]\]>\s*$", "", text, flags=re.IGNORECASE)
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _looks_structured_payload(payload: str) -> bool:
    """Return True when payload starts like JSON/JSON-string content."""
    normalized = _normalize_wrapper_payload(payload)
    if not normalized:
        return False
    return normalized.startswith("{") or normalized.startswith("[") or normalized.startswith('"')


def _extract_leading_structured_payload(payload: str) -> str:
    """Extract first valid JSON value from payload, tolerating trailing spillover."""
    normalized = _normalize_wrapper_payload(payload)
    if not _looks_structured_payload(normalized):
        return ""

    try:
        _, end_index = json.JSONDecoder().raw_decode(normalized)
    except Exception:
        return ""

    return normalized[:end_index].strip()


def extract_opening_tag_target(opening_tag: str, tag_name: str, attr_name: str) -> str:
    """Get target from an opening tag, supporting quoted, unquoted, or loose token forms."""
    opening_tag = (opening_tag or "").strip()
    match = re.match(r"<\s*([a-z0-9_:-]+)\b([^>]*)>", opening_tag, flags=re.IGNORECASE)
    if not match:
        return ""
    found_tag = (match.group(1) or "").strip().lower()
    if found_tag != (tag_name or "").strip().lower():
        return ""
    attrs = match.group(2) or ""
    return _extract_named_attr(attrs, attr_name) or _extract_loose_attr_token(attrs)


def parse_agent_blocks(text: str) -> list[AgentTagBlock]:
    """Parse wrapper blocks and recover malformed/missing closing tags.

    Recovery behavior for an unclosed wrapper:
    - End it at the next known wrapper opening tag, if present.
    - Otherwise, end it at the end of text.
    """
    text = _normalize_agent_text(text)
    if not text:
        return []

    blocks: list[AgentTagBlock] = []
    cursor = 0
    lowered = text.lower()

    while True:
        match = TAG_OPEN_PATTERN.search(text, cursor)
        if not match:
            break

        tag = (match.group(1) or "").lower()
        attrs = (match.group(2) or "").strip()
        content_start = match.end()
        close_tag = f"</{tag}>"

        next_open_match = TAG_OPEN_PATTERN.search(text, content_start)
        close_index = lowered.find(close_tag, content_start)
        recovered = False
        if close_index < 0:
            recovered = True
            if next_open_match:
                close_index = next_open_match.start()
                block_end = close_index
            else:
                close_index = len(text)
                block_end = len(text)
        else:
            # If another known wrapper starts before the close tag, treat this
            # block as malformed and recover by splitting at the next wrapper.
            if next_open_match and next_open_match.start() < close_index:
                recovered = True
                close_index = next_open_match.start()
                block_end = close_index
            else:
                block_end = close_index + len(close_tag)

        blocks.append(
            AgentTagBlock(
                tag=tag,
                attrs=attrs,
                content=text[content_start:close_index],
                start=match.start(),
                end=block_end,
                recovered=recovered,
            )
        )

        cursor = block_end if block_end > match.start() else match.end()

    return blocks


def extract_idascripts(text: str) -> list[str]:
    """Extract idascript bodies from model text."""
    return [block.content.strip() for block in parse_agent_blocks(text) if block.tag == "idascript"]


def extract_idatool_calls(text: str) -> list[tuple[str, str]]:
    """Extract idatool calls as (tool_name, payload_text).

    Supports both canonical forms:
    - <idatool lookup_funcs>{...}</idatool>
    - <idatool name="lookup_funcs">{...}</idatool>
    """
    calls: list[tuple[str, str]] = []
    for block in parse_agent_blocks(text):
        if block.tag != "idatool":
            continue
        name = _extract_named_attr(block.attrs, "name") or _extract_loose_attr_token(block.attrs)
        name = name.strip().lower().replace("-", "_")
        if not name:
            continue

        payload_text = block.content.strip()
        leading_payload = _extract_leading_structured_payload(payload_text)
        if leading_payload:
            payload_text = leading_payload

        # Recovered wrappers are useful for multi-tool extraction, but if the
        # payload is plain prose then it is almost always malformed spillover.
        if block.recovered and not _looks_structured_payload(payload_text):
            continue

        calls.append((name, payload_text))
    return calls


def extract_delegate_calls(text: str) -> list[tuple[str, str]]:
    """Extract delegate calls as (agent_name, task_text)."""
    calls: list[tuple[str, str]] = []
    for block in parse_agent_blocks(text):
        if block.tag != "delegate":
            continue
        agent = _extract_named_attr(block.attrs, "agent") or _extract_loose_attr_token(block.attrs)
        agent = agent.strip()
        if not agent:
            continue
        calls.append((agent, block.content.strip()))
    return calls


def strip_agent_tags(text: str) -> str:
    """Remove parsed idascript/idatool/delegate wrappers from text."""
    normalized_text = _normalize_agent_text(text)
    blocks = parse_agent_blocks(normalized_text)
    if not blocks:
        return normalized_text.strip()

    chunks: list[str] = []
    cursor = 0
    for block in blocks:
        if block.start > cursor:
            chunks.append(normalized_text[cursor:block.start])
        cursor = max(cursor, block.end)
    if cursor < len(normalized_text):
        chunks.append(normalized_text[cursor:])
    return "".join(chunks).strip()
