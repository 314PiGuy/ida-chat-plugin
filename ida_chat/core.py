"""
IDA Chat Core - Shared foundation for CLI and Plugin.

This module contains the common Agent SDK integration, script execution,
and message processing used by both the CLI and IDA plugin.
"""

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path
from typing import Callable, Protocol, TYPE_CHECKING

import claude_code_transcripts
from ida_chat.logging_utils import logger
from ida_chat.tools.catalog import AVAILABLE_IDATOOLS
from ida_chat.tools.patterns import DELEGATE_PATTERN, IDASCRIPT_PATTERN, IDATOOL_PATTERN

if TYPE_CHECKING:
    from ida_chat.history import MessageHistory

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
)

from ida_chat.providers.config import (
    ProviderConfig,
    build_provider_env,
    normalize_provider,
    provider_label,
    resolve_base_url,
    resolve_model,
)


# Project directory for agent SDK (contains PROMPT.md, USAGE.md, API_REFERENCE.md)
PROJECT_DIR = Path(__file__).resolve().parent.parent / "project"

# Prompt file locations
PROMPT_FILE = PROJECT_DIR / "PROMPT.md"
IDA_UI_FILE = PROJECT_DIR / "IDA.md"
USAGE_FILE = PROJECT_DIR / "USAGE.md"
API_REFERENCE_FILE = PROJECT_DIR / "API_REFERENCE.md"

DEFAULT_BUILTIN_TOOLS = {
    "type": "preset",
    "preset": "claude_code",
}

FILE_ACCESS_TOOL_MATCHER = "Read|Write|Edit|MultiEdit|Glob|Grep"
PATH_KEY_HINTS = {
    "path",
    "file_path",
    "file",
    "old_path",
    "new_path",
    "directory",
    "cwd",
    "include",
}
PATH_LIST_KEY_HINTS = {
    "paths",
    "file_paths",
}

OPENAI_COMPAT_DEFAULT_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "nim": "https://integrate.api.nvidia.com/v1",
}



def get_model_context_length(model: str) -> int:
    """Best-effort model context window lookup."""
    if not model:
        return 8192

    model = model.lower()

    if any(x in model for x in ("gpt-4.1", "gpt-5", "o1", "o3", "o4", "grok-4")):
        return 200000
    if any(x in model for x in ("claude-3", "claude-sonnet-4", "claude-opus-4")):
        return 200000
    if any(x in model for x in ("gemini-1.5", "gemini-2", "kimi", "deepseek")):
        return 128000
    if "gpt-4" in model:
        return 128000
    if any(x in model for x in ("32k", "llama-3.1-70b", "llama3.1-70b")):
        return 32768
    if any(x in model for x in ("16k", "gpt-3.5")):
        return 16384

    return 8192


def _compact_api_reference_text(text: str) -> str:
    """Build a compact API reference using headings + short intro lines."""
    lines = text.splitlines()
    headings = [line for line in lines if line.lstrip().startswith("#")]
    intro = lines[:30]
    compact = "\n".join((intro + [""] + headings)[:180])
    if not compact.strip():
        return text[:4000]
    return compact


def _compact_markdown_text(text: str, max_lines: int = 220) -> str:
    """Compact markdown while preserving headings and an initial context slice."""
    lines = text.splitlines()
    headings = [line for line in lines if line.lstrip().startswith("#")]
    head = [line for line in lines[: max_lines // 2] if line.strip()]
    compact = "\n".join((head + [""] + headings)[:max_lines])
    return compact if compact.strip() else text[:5000]


def _load_system_prompt(model_name: str | None = None, compact_docs: bool = False) -> str:
    """Load the system prompt from PROMPT.md.

    If running inside IDA Pro (IDA_CHAT_INSIDE_IDA env var is set),
    also appends IDA.md which contains the user interaction API.
    """
    prompt = ""

    if PROMPT_FILE.exists():
        prompt = PROMPT_FILE.read_text(encoding="utf-8")
    else:
        logger.warning(f"PROMPT.md not found at {PROMPT_FILE}")
        prompt = "You have access to an open IDA database via the `db` variable. Use <idascript> tags for code."

    # Append IDA UI interaction API when running inside IDA
    if os.environ.get("IDA_CHAT_INSIDE_IDA") == "1":
        if IDA_UI_FILE.exists():
            logger.info("Running inside IDA - appending IDA.md to system prompt")
            ida_ui_text = IDA_UI_FILE.read_text(encoding="utf-8")
            if compact_docs:
                ida_ui_text = _compact_markdown_text(ida_ui_text, max_lines=160)
            prompt += "\n\n" + ida_ui_text
        else:
            logger.warning(f"IDA.md not found at {IDA_UI_FILE}")

    usage_text = USAGE_FILE.read_text(encoding="utf-8") if USAGE_FILE.exists() else ""
    api_text = API_REFERENCE_FILE.read_text(encoding="utf-8") if API_REFERENCE_FILE.exists() else ""

    if compact_docs:
        usage_text = _compact_markdown_text(usage_text, max_lines=180)
        api_text = _compact_api_reference_text(api_text)
    elif model_name:
        context_limit = get_model_context_length(model_name)
        full_prompt_tokens = (len(prompt) + len(usage_text) + len(api_text)) / 4
        # If static instructions alone consume over half the context, compact docs.
        if context_limit < full_prompt_tokens * 2:
            logger.warning(
                "Model context appears small for full docs (model=%s, limit=%s, estimated=%s). "
                "Using compact API reference.",
                model_name,
                context_limit,
                int(full_prompt_tokens),
            )
            api_text = _compact_api_reference_text(api_text)

    prompt += "\n\n" + usage_text
    prompt += "\n\n" + api_text
    return prompt


def _normalize_openai_compat_endpoint(base_url: str) -> str:
    """Normalize base URLs to a chat completions endpoint."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _extract_error_message(response_body: str) -> str:
    """Extract best-effort API error text from a JSON response body."""
    text = response_body.strip()
    if not text:
        return "Unknown error"

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            for key in ("message", "detail", "code"):
                value = error_obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(error_obj, str) and error_obj.strip():
            return error_obj.strip()

        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return text


def _extract_openai_compat_text(payload: dict) -> str:
    """Extract assistant text from OpenAI-compatible chat completion payloads."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Provider response did not include any choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("Provider response choice format is invalid")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Provider response did not include a message block")

    content = message.get("content")
    if isinstance(content, str):
        return content

    # Some providers can return structured content parts.
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    if content is None:
        return ""

    return str(content)


def _build_openai_compat_headers(provider_config: ProviderConfig) -> dict[str, str]:
    """Build HTTP headers for OpenAI-compatible endpoints."""
    provider_name = normalize_provider(provider_config.provider)
    api_key = (provider_config.api_key or "").strip()

    headers = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if provider_name == "openrouter":
        headers.setdefault("HTTP-Referer", "https://hex-rays.com")
        headers.setdefault("X-Title", "IDA Chat Plugin")

    return headers


def _resolve_openai_compat_base_url(provider_config: ProviderConfig) -> str:
    """Resolve base URL for OpenAI-compatible providers."""
    explicit_or_default = resolve_base_url(provider_config)
    if explicit_or_default:
        return explicit_or_default

    provider_name = normalize_provider(provider_config.provider)
    fallback = OPENAI_COMPAT_DEFAULT_BASE_URLS.get(provider_name)
    if fallback:
        return fallback

    raise RuntimeError(f"No OpenAI-compatible base URL found for provider '{provider_name}'")


def _query_openai_compat_sync(
    provider_config: ProviderConfig,
    messages: list[dict[str, str]],
    timeout_seconds: float = 120.0,
) -> str:
    """Send a synchronous chat completion request to OpenAI-compatible endpoints."""
    provider_name = normalize_provider(provider_config.provider)
    if provider_name == "claude":
        raise RuntimeError("OpenAI-compatible transport should not be used for Claude provider")

    model = resolve_model(provider_config)
    if not model:
        raise RuntimeError("No model is configured for the selected provider")

    endpoint = _normalize_openai_compat_endpoint(_resolve_openai_compat_base_url(provider_config))
    payload = {
        "model": model,
        "messages": messages,
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_openai_compat_headers(provider_config),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as http_error:
        body = http_error.read().decode("utf-8", errors="replace")
        detail = _extract_error_message(body)
        raise RuntimeError(
            f"{provider_label(provider_name)} API request failed ({http_error.code}): {detail}"
        ) from http_error
    except urllib.error.URLError as url_error:
        reason = str(url_error.reason).strip() or str(url_error)
        raise RuntimeError(
            f"{provider_label(provider_name)} request failed: {reason}"
        ) from url_error

    try:
        response_json = json.loads(raw_body)
    except json.JSONDecodeError as decode_error:
        raise RuntimeError(
            f"{provider_label(provider_name)} returned invalid JSON"
        ) from decode_error

    if isinstance(response_json, dict) and response_json.get("error"):
        raise RuntimeError(_extract_error_message(json.dumps(response_json)))

    if not isinstance(response_json, dict):
        raise RuntimeError("Unexpected provider response format")

    assistant_text = _extract_openai_compat_text(response_json).strip()
    if not assistant_text:
        raise RuntimeError("Provider returned an empty response")

    return assistant_text


def _extract_openai_compat_stream_delta(chunk_json: dict) -> tuple[str, bool]:
    """Extract text from one OpenAI-compatible streaming chunk.

    Returns (text, maybe_cumulative_fragment). The second value is True when
    providers send full message fragments instead of incremental deltas.
    """

    def _extract_text(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if not isinstance(item, dict):
                    continue
                for key in ("text", "content", "token"):
                    text = item.get(key)
                    if isinstance(text, str) and text:
                        parts.append(text)
                        break
            return "".join(parts)
        return ""

    choices = chunk_json.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            # OpenAI-style: choices[0].delta.content
            delta = first_choice.get("delta")
            if isinstance(delta, dict):
                for key in ("content", "text", "reasoning_content"):
                    text = _extract_text(delta.get(key))
                    if text:
                        return text, False

            # Some providers stream full message fragments.
            message = first_choice.get("message")
            if isinstance(message, dict):
                for key in ("content", "text", "reasoning_content"):
                    text = _extract_text(message.get(key))
                    if text:
                        return text, True

            text = _extract_text(first_choice.get("text"))
            if text:
                return text, True

    # Additional compatibility with providers that stream output_text directly.
    for key in ("output_text", "text", "content"):
        text = _extract_text(chunk_json.get(key))
        if text:
            return text, True

    return "", False


def _query_openai_compat_stream_sync(
    provider_config: ProviderConfig,
    messages: list[dict[str, str]],
    timeout_seconds: float = 180.0,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """Send a streaming chat completion request and return the merged text."""
    provider_name = normalize_provider(provider_config.provider)
    if provider_name == "claude":
        raise RuntimeError("Streaming OpenAI-compatible transport should not be used for Claude provider")

    model = resolve_model(provider_config)
    if not model:
        raise RuntimeError("No model is configured for the selected provider")

    endpoint = _normalize_openai_compat_endpoint(_resolve_openai_compat_base_url(provider_config))
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_openai_compat_headers(provider_config),
        method="POST",
    )

    parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_type = str(response.headers.get("Content-Type") or "").lower()
            if "text/event-stream" not in content_type:
                raw_body = response.read().decode("utf-8", errors="replace")
                response_json = json.loads(raw_body)
                if not isinstance(response_json, dict):
                    raise RuntimeError("Unexpected provider response format")
                if response_json.get("error"):
                    raise RuntimeError(_extract_error_message(json.dumps(response_json)))
                assistant_text = _extract_openai_compat_text(response_json).strip()
                if not assistant_text:
                    raise RuntimeError("Provider returned an empty streaming response")
                if on_chunk:
                    on_chunk(assistant_text)
                return assistant_text

            cumulative_fragment = ""
            event_data_lines: list[str] = []

            def _flush_event_data() -> bool:
                nonlocal cumulative_fragment
                if not event_data_lines:
                    return False

                data = "\n".join(event_data_lines).strip()
                event_data_lines.clear()
                if not data:
                    return False
                if data == "[DONE]":
                    return True

                try:
                    chunk_json = json.loads(data)
                except json.JSONDecodeError:
                    # Some providers can emit raw text fragments.
                    parts.append(data)
                    if on_chunk:
                        on_chunk(data)
                    return False

                if isinstance(chunk_json, dict) and chunk_json.get("error"):
                    raise RuntimeError(_extract_error_message(json.dumps(chunk_json)))

                chunk_text, maybe_cumulative = _extract_openai_compat_stream_delta(chunk_json)
                if not chunk_text:
                    return False

                if maybe_cumulative:
                    full_fragment = chunk_text
                    if cumulative_fragment and full_fragment.startswith(cumulative_fragment):
                        chunk_text = full_fragment[len(cumulative_fragment):]
                    cumulative_fragment = full_fragment

                if not chunk_text:
                    return False

                parts.append(chunk_text)
                if on_chunk:
                    on_chunk(chunk_text)
                return False

            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

                if not line:
                    if _flush_event_data():
                        break
                    continue

                if line.startswith("data:"):
                    event_data_lines.append(line[5:].strip())
                    continue

                if line.startswith("event:") or line.startswith(":"):
                    continue

                # Fallback for providers that stream JSON lines instead of SSE.
                if line.startswith("{"):
                    event_data_lines.append(line)

            if event_data_lines:
                _flush_event_data()

    except urllib.error.HTTPError as http_error:
        body = http_error.read().decode("utf-8", errors="replace")
        detail = _extract_error_message(body)
        raise RuntimeError(
            f"{provider_label(provider_name)} streaming request failed ({http_error.code}): {detail}"
        ) from http_error
    except urllib.error.URLError as url_error:
        reason = str(url_error.reason).strip() or str(url_error)
        raise RuntimeError(
            f"{provider_label(provider_name)} streaming request failed: {reason}"
        ) from url_error
    except json.JSONDecodeError as decode_error:
        raise RuntimeError(
            f"{provider_label(provider_name)} returned invalid streaming JSON"
        ) from decode_error

    merged = "".join(parts).strip()
    if not merged:
        raise RuntimeError("Provider returned an empty streaming response")
    return merged


async def _query_openai_compat(
    provider_config: ProviderConfig,
    messages: list[dict[str, str]],
    timeout_seconds: float = 120.0,
) -> str:
    """Run OpenAI-compatible request off the event loop thread."""
    return await asyncio.to_thread(
        _query_openai_compat_sync,
        provider_config,
        messages,
        timeout_seconds,
    )


async def _query_openai_compat_stream(
    provider_config: ProviderConfig,
    messages: list[dict[str, str]],
    timeout_seconds: float = 180.0,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """Run OpenAI-compatible streaming request off the event loop thread."""
    return await asyncio.to_thread(
        _query_openai_compat_stream_sync,
        provider_config,
        messages,
        timeout_seconds,
        on_chunk,
    )


def _iter_candidate_paths(tool_input: dict) -> list[str]:
    """Collect path-like values from tool input dictionaries."""
    candidates: list[str] = []

    for key, value in tool_input.items():
        key_lower = key.lower()
        if key_lower in PATH_KEY_HINTS and isinstance(value, str):
            candidates.append(value)
        elif key_lower in PATH_LIST_KEY_HINTS and isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    candidates.append(item)

    return candidates


def _normalize_candidate_path(raw_path: str) -> Path | None:
    """Convert a raw path-ish value to a resolved path for policy checks."""
    value = raw_path.strip()
    if not value:
        return None

    # Ignore non-file URLs (e.g., http/https) for file policy checks.
    if "://" in value and not value.startswith("file://"):
        return None
    if value.startswith("file://"):
        value = value[7:]

    # If this looks like a glob, validate its static prefix.
    wildcard_positions = [value.find(ch) for ch in ("*", "?", "[") if ch in value]
    if wildcard_positions:
        prefix = value[: min(wildcard_positions)]
        value = prefix if prefix else "."

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path.resolve()


async def _restrict_file_access(input_data, tool_use_id, context):
    """Hook to block file operations outside PROJECT_DIR."""
    if input_data['hook_event_name'] != 'PreToolUse':
        return {}

    tool_name = input_data.get('tool_name', 'unknown')
    tool_input = input_data['tool_input']
    if not isinstance(tool_input, dict):
        return {}

    candidate_paths = _iter_candidate_paths(tool_input)

    for file_path in candidate_paths:
        resolved = _normalize_candidate_path(file_path)
        if resolved is None:
            continue

        try:
            resolved.relative_to(PROJECT_DIR)
        except ValueError:
            logger.warning(
                "Blocked %s file access outside PROJECT_DIR: %s",
                tool_name,
                file_path,
            )
            return {
                'hookSpecificOutput': {
                    'hookEventName': input_data['hook_event_name'],
                    'permissionDecision': 'deny',
                    'permissionDecisionReason': f'File access restricted to project directory'
                }
            }

    return {}


def export_transcript(session_file: Path, output_path: Path) -> None:
    """Export a chat session to HTML files.

    Generates index.html and page-XXX.html files in the same directory as output_path.

    Args:
        session_file: Path to the JSONL session file.
        output_path: Path for the main output HTML file (index.html will be renamed to this).

    Raises:
        FileNotFoundError: If session_file doesn't exist.
        Exception: If HTML generation fails.
    """
    if not session_file.exists():
        raise FileNotFoundError(f"Session file not found: {session_file}")

    output_dir = output_path.parent

    # Generate into a temp directory, then copy all HTML files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        claude_code_transcripts.generate_html(session_file, tmp_path)

        # Copy index.html to the target path
        generated_html = tmp_path / "index.html"
        if generated_html.exists():
            shutil.copy2(generated_html, output_path)
        else:
            raise RuntimeError("HTML generation failed: index.html not created")

        # Copy all page-XXX.html files
        for page_file in tmp_path.glob("page-*.html"):
            shutil.copy2(page_file, output_dir / page_file.name)

    logger.info(f"Exported transcript to {output_path}")


def export_transcript_to_dir(session_file: Path, output_dir: Path) -> Path:
    """Export a chat session to a directory (with all assets).

    Args:
        session_file: Path to the JSONL session file.
        output_dir: Directory to generate HTML into.

    Returns:
        Path to the generated index.html.

    Raises:
        FileNotFoundError: If session_file doesn't exist.
    """
    if not session_file.exists():
        raise FileNotFoundError(f"Session file not found: {session_file}")

    claude_code_transcripts.generate_html(session_file, output_dir)
    logger.info(f"Exported transcript to {output_dir}")
    return output_dir / "index.html"


async def test_provider_connection(provider_config: ProviderConfig) -> tuple[bool, str]:
    """Test provider connectivity with a lightweight prompt.

    This is a lightweight test that doesn't require a database or full
    agent configuration. Used by the onboarding panel to verify setup.

    Returns:
        Tuple of (success, message):
        - On success: (True, provider response)
        - On failure: (False, error message)
    """
    provider_name = normalize_provider(provider_config.provider)
    logger.info("Testing provider connection: %s", provider_name)

    if provider_name != "claude":
        try:
            response_text = await _query_openai_compat(
                provider_config,
                [{"role": "user", "content": "Reply with one short sentence saying the connection is working."}],
                timeout_seconds=120.0,
            )
            logger.info("Provider connection test successful via OpenAI-compatible transport: %s", provider_name)
            return True, response_text
        except Exception as e:
            logger.error("OpenAI-compatible connection test failed for %s: %s", provider_name, e)
            return False, str(e)

    stderr_lines: list[str] = []

    def _collect_stderr(line: str) -> None:
        text = line.rstrip()
        if text:
            stderr_lines.append(text)

    options = ClaudeAgentOptions(
        cwd=str(PROJECT_DIR),
        permission_mode="bypassPermissions",
        allowed_tools=[],  # No tools needed for simple test
        env=build_provider_env(provider_config),
        model=resolve_model(provider_config),
        stderr=_collect_stderr,
    )

    client = ClaudeSDKClient(options=options)
    try:
        await client.connect()
        await client.query("Reply with one short sentence saying the connection is working")

        response_text = ""
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text

        await client.disconnect()
        logger.info("Provider connection test successful: %s", provider_name)
        return True, response_text.strip()

    except Exception as e:
        logger.error("Connection test failed for provider %s: %s", provider_name, e)
        err = str(e).strip() or "Unknown error"

        if stderr_lines:
            stderr_excerpt = "\n".join(stderr_lines[-12:])
            return False, f"{err}\n\nSDK stderr:\n{stderr_excerpt}"

        return False, err


class ChatCallback(Protocol):
    """Protocol for handling chat output events.

    Implementations of this protocol handle the presentation layer,
    whether that's terminal output (CLI) or Qt widgets (Plugin).
    """

    def on_metric(self, text: str) -> None:
        """Called to log a metric or event string."""
        ...

    def on_event(
        self,
        kind: str,
        title: str,
        details: str,
        duration_ms: float | None = None,
    ) -> None:
        """Called with structured event details for expandable logs."""
        ...

    def on_turn_start(self, turn: int, max_turns: int) -> None:
        """Called at the start of each agentic turn."""
        ...

    def on_thinking(self) -> None:
        """Called when the agent starts processing."""
        ...

    def on_thinking_done(self) -> None:
        """Called when the agent produces first output."""
        ...

    def on_tool_use(self, tool_name: str, details: str) -> None:
        """Called when the agent uses a tool (Read, Glob, Grep, Skill)."""
        ...

    def on_text(self, text: str) -> None:
        """Called when the agent outputs text (excluding idascript blocks)."""
        ...

    def on_script_code(self, code: str) -> None:
        """Called with the script code before execution."""
        ...

    def on_script_output(self, output: str) -> None:
        """Called with the output of an executed idascript."""
        ...

    def on_error(self, error: str) -> None:
        """Called when an error occurs."""
        ...

    def on_result(self, num_turns: int, cost: float | None) -> None:
        """Called when the agent finishes with stats."""
        ...


class IDAChatCore:
    """Shared chat backend for CLI and Plugin.

    Handles Agent SDK integration, message processing, and script execution.
    Implements an agentic loop that feeds script results back to the agent.
    Output is delegated to the callback for presentation.
    """

    def __init__(
        self,
        db,
        callback: ChatCallback,
        script_executor: Callable[[str], str] | None = None,
        verbose: bool = False,
        max_turns: int = 20,
        provider_config: ProviderConfig | None = None,
        history: "MessageHistory | None" = None,
    ):
        """Initialize the chat core.

        Args:
            db: An open ida_domain Database instance.
            callback: Handler for output events.
            script_executor: Optional custom script executor. If None, uses
                default direct execution. Plugin can inject a thread-safe
                executor that runs on the main thread.
            verbose: If True, report additional stats.
            max_turns: Maximum agentic turns before stopping (default 20).
            provider_config: Provider/backend configuration.
            history: Optional MessageHistory for persisting conversations.
        """
        self.db = db
        self.callback = callback
        self.verbose = verbose
        self.max_turns = max_turns
        self.provider_config = provider_config or ProviderConfig()
        self.history = history
        self.client: ClaudeSDKClient | None = None
        self._cancelled = False
        self._provider_name = normalize_provider(self.provider_config.provider)
        self._using_claude_sdk = self._provider_name == "claude"
        self._system_prompt = ""
        # Use injected executor or default to direct execution
        self._execute_script = script_executor or self._default_execute_script
        self._uses_custom_script_executor = script_executor is not None
        self._stream_failures = 0
        self._streaming_disabled = False

    def request_cancel(self) -> None:
        """Request cancellation of the current operation."""
        self._cancelled = True
        logger.info("Cancel requested")

    async def connect(self) -> None:
        """Initialize and connect the Agent SDK client."""
        provider_name = normalize_provider(self.provider_config.provider)
        selected_model = resolve_model(self.provider_config)
        self._provider_name = provider_name
        self._using_claude_sdk = provider_name == "claude"
        self._system_prompt = _load_system_prompt(
            selected_model,
            compact_docs=not self._using_claude_sdk,
        )
        self._stream_failures = 0
        self._streaming_disabled = False

        logger.info("=" * 60)
        logger.info("Provider: %s", provider_name)
        if selected_model:
            logger.info("Model override: %s", selected_model)
        logger.info(f"CWD: {PROJECT_DIR}")

        if not self._using_claude_sdk:
            logger.info("Connecting to agent backend via OpenAI-compatible HTTP transport")
            self.client = None
            return

        logger.info("Connecting to agent backend via Claude Agent SDK")

        options = ClaudeAgentOptions(
            cwd=str(PROJECT_DIR),
            setting_sources=["project"],
            tools=DEFAULT_BUILTIN_TOOLS,
            disallowed_tools=["Bash"],
            permission_mode="bypassPermissions",
            env=build_provider_env(self.provider_config),
            model=selected_model,
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": self._system_prompt,
            },
            hooks={
                'PreToolUse': [
                    HookMatcher(matcher=FILE_ACCESS_TOOL_MATCHER, hooks=[_restrict_file_access])
                ]
            },
        )

        self.client = ClaudeSDKClient(options=options)
        await self.client.connect()
        logger.info("Connected successfully")

    async def disconnect(self) -> None:
        """Disconnect the Agent SDK client."""
        if self.client:
            await self.client.disconnect()
            self.client = None


    def _select_cheapest_model(self) -> str:
        """Return the cheapest known model candidate for history condensing."""
        provider_name = normalize_provider(self.provider_config.provider)
        provider_models = {
            "openai": "gpt-4.1-nano",
            "openrouter": "openai/gpt-4.1-nano",
            "gemini": "gemini-1.5-flash",
            "nim": "meta/llama-3.1-8b-instruct",
            "ollama": "llama3.1:8b",
            "claude": "claude-3-haiku-20240307",
        }
        return provider_models.get(provider_name, "gpt-4.1-nano")

    def _local_compact_summary(self, messages: list[dict[str, str]], max_lines: int = 16) -> str:
        """Generate a compact local summary fallback without extra API calls."""
        selected = messages[-max_lines:]
        lines: list[str] = []
        for item in selected:
            role = str(item.get("role", "unknown")).strip() or "unknown"
            content = str(item.get("content", "")).replace("\n", " ").strip()
            if content:
                lines.append(f"- {role}: {content[:220]}")
        return "\n".join(lines) if lines else "No earlier context to summarize."

    def _emit_context_warning_if_needed(self, messages: list[dict[str, str]]) -> None:
        """Warn once when model context is likely too small for the active prompt."""
        model_name = resolve_model(self.provider_config) or self.provider_config.model or "unknown"
        limit = get_model_context_length(model_name)
        estimated_tokens = sum(len(str(m.get("content", ""))) for m in messages) / 4
        if estimated_tokens * 2 <= limit:
            return
        if getattr(self, "_warned_context", False):
            return

        self._warned_context = True
        warning = (
            "WARNING: Model context may be inadequate. "
            f"Estimated prompt={estimated_tokens:.0f} tokens, limit={limit} tokens."
        )
        self.callback.on_error(warning)
        self.callback.on_event(
            "context_warning",
            "Context Capacity Warning",
            (
                f"Model: {model_name}\n"
                f"Estimated prompt tokens: {estimated_tokens:.0f}\n"
                f"Context limit: {limit}\n"
                "Reason: limit is less than 2x estimated prompt size."
            ),
            duration_ms=None,
        )

    def _normalize_generated_script(self, code: str) -> tuple[str, list[str]]:
        """Apply compatibility rewrites and helper shims to reduce script failures."""
        normalized = code.strip()
        fixes: list[str] = []

        rewrites = [
            (
                "db.functions.get_by_name(",
                "db.functions.get_function_by_name(",
                "Replaced db.functions.get_by_name(...) with db.functions.get_function_by_name(...)",
            ),
            (
                "db.functions.lookup_name(",
                "_compat_lookup_funcs(",
                "Replaced db.functions.lookup_name(...) with compatibility lookup helper",
            ),
            (
                "db.functions.get_function_by_addr(",
                "db.functions.get_at(",
                "Replaced db.functions.get_function_by_addr(...) with db.functions.get_at(...)",
            ),
            (
                "db.instructions.iter_range(",
                "db.instructions.get_between(",
                "Replaced db.instructions.iter_range(...) with db.instructions.get_between(...)",
            ),
            (
                "db.instructions.get_disasm(",
                "db.instructions.get_disassembly(",
                "Replaced db.instructions.get_disasm(...) with db.instructions.get_disassembly(...)",
            ),
            (
                "db.segments.get_type(",
                "_compat_segment_type(",
                "Replaced db.segments.get_type(...) with compatibility segment-type helper",
            ),
            (
                "db.entries.get_name(",
                "_compat_entry_name(",
                "Replaced db.entries.get_name(...) with compatibility entry-name helper",
            ),
        ]

        for old, new, description in rewrites:
            if old in normalized:
                normalized = normalized.replace(old, new)
                fixes.append(description)

        # Common model mistake: get_disassembly(insn.ea) instead of passing instruction object.
        normalized, disasm_subs = re.subn(
            r"db\.instructions\.get_disassembly\(\s*([A-Za-z_][A-Za-z0-9_]*)\.ea\s*\)",
            r"db.instructions.get_disassembly(\1)",
            normalized,
        )
        if disasm_subs:
            fixes.append("Adjusted get_disassembly(insn.ea) calls to get_disassembly(insn)")

        if "len(callee)" in normalized and "callees" in normalized:
            normalized = normalized.replace("len(callee)", "len(callees)")
            fixes.append("Replaced len(callee) with len(callees)")

        for module_name in ("re", "json"):
            uses_module = bool(re.search(rf"\b{module_name}\.", normalized))
            has_import = bool(
                re.search(rf"^\s*import\s+{module_name}\b", normalized, flags=re.MULTILINE)
            )
            if uses_module and not has_import:
                normalized = f"import {module_name}\n" + normalized
                fixes.append(f"Prepended 'import {module_name}' because script uses {module_name}.*")

        normalized, helper_fixes = self._inject_script_compat_helpers(normalized)
        fixes.extend(helper_fixes)

        deduped_fixes: list[str] = []
        for fix in fixes:
            if fix not in deduped_fixes:
                deduped_fixes.append(fix)

        return normalized, deduped_fixes

    def _inject_script_compat_helpers(self, code: str) -> tuple[str, list[str]]:
        """Inject helper functions when rewritten compatibility calls are present."""
        helper_blocks: list[str] = []
        fixes: list[str] = []

        if "_compat_lookup_funcs(" in code and "def _compat_lookup_funcs(" not in code:
            helper_blocks.append(
                """
def _compat_lookup_funcs(query):
    needle = str(query or '').strip().lower()
    if not needle:
        return []

    exact = []
    fuzzy = []
    for func in db.functions:
        try:
            name = str(db.functions.get_name(func) or '')
        except Exception:
            continue
        lower_name = name.lower()
        if lower_name == needle:
            exact.append(func)
        elif needle in lower_name:
            fuzzy.append(func)
        if len(exact) >= 64:
            break
        if len(fuzzy) >= 256:
            break
    return exact if exact else fuzzy
""".strip()
            )
            fixes.append("Injected compatibility helper: _compat_lookup_funcs")

        if "_compat_segment_type(" in code and "def _compat_segment_type(" not in code:
            helper_blocks.append(
                """
def _compat_segment_type(seg):
    for attr in ('type', 'seg_type', 'segment_type'):
        value = getattr(seg, attr, None)
        if value is not None:
            return str(getattr(value, 'name', value))
    try:
        return str(db.segments.get_class(seg))
    except Exception:
        return 'unknown'
""".strip()
            )
            fixes.append("Injected compatibility helper: _compat_segment_type")

        if "_compat_entry_name(" in code and "def _compat_entry_name(" not in code:
            helper_blocks.append(
                """
def _compat_entry_name(entry):
    name = getattr(entry, 'name', None)
    if name:
        return str(name)
    try:
        address = getattr(entry, 'address', None)
        if address is not None:
            resolved = db.entries.get_at(address)
            if resolved and getattr(resolved, 'name', None):
                return str(resolved.name)
    except Exception:
        pass
    return ''
""".strip()
            )
            fixes.append("Injected compatibility helper: _compat_entry_name")

        if not helper_blocks:
            return code, fixes

        helper_prefix = "\n\n".join(helper_blocks)
        return f"{helper_prefix}\n\n{code}", fixes

    def _normalize_generated_script_from_error(self, code: str, script_output: str) -> tuple[str, list[str]]:
        """Apply targeted rewrites based on a concrete script error string."""
        normalized = code
        fixes: list[str] = []
        error_text = script_output.lower()

        if "object has no attribute 'lookup_name'" in error_text and "db.functions.lookup_name(" in normalized:
            normalized = normalized.replace("db.functions.lookup_name(", "_compat_lookup_funcs(")
            fixes.append("Recovered from lookup_name error using compatibility lookup helper")

        if "object has no attribute 'get_function_by_addr'" in error_text and "db.functions.get_function_by_addr(" in normalized:
            normalized = normalized.replace("db.functions.get_function_by_addr(", "db.functions.get_at(")
            fixes.append("Recovered from get_function_by_addr error using db.functions.get_at(...)")

        if "object has no attribute 'iter_range'" in error_text and "db.instructions.iter_range(" in normalized:
            normalized = normalized.replace("db.instructions.iter_range(", "db.instructions.get_between(")
            fixes.append("Recovered from iter_range error using db.instructions.get_between(...)")

        if "object has no attribute 'get_disasm'" in error_text and "db.instructions.get_disasm(" in normalized:
            normalized = normalized.replace("db.instructions.get_disasm(", "db.instructions.get_disassembly(")
            fixes.append("Recovered from get_disasm error using get_disassembly(...)")

        if "object has no attribute 'get_type'" in error_text and "db.segments.get_type(" in normalized:
            normalized = normalized.replace("db.segments.get_type(", "_compat_segment_type(")
            fixes.append("Recovered from segments.get_type error using compatibility segment-type helper")

        if "name 're' is not defined" in error_text:
            uses_re = bool(re.search(r"\bre\.", normalized))
            has_re_import = bool(re.search(r"^\s*import\s+re\b", normalized, flags=re.MULTILINE))
            if uses_re and not has_re_import:
                normalized = "import re\n" + normalized
                fixes.append("Recovered from missing re import")

        normalized, baseline_fixes = self._normalize_generated_script(normalized)
        for fix in baseline_fixes:
            if fix not in fixes:
                fixes.append(fix)

        return normalized, fixes

    def _is_script_error_output(self, output: str) -> bool:
        """Check whether script execution output represents a runtime error."""
        return bool(output) and output.lstrip().startswith("Script error:")

    def _execute_script_batch(self, script_codes: list[str]) -> list[str]:
        """Execute multiple scripts in one executor call to reduce overhead."""
        if not script_codes:
            return []
        if len(script_codes) == 1:
            return [self._execute_script(script_codes[0])]

        encoded_scripts = json.dumps(script_codes, ensure_ascii=False)
        batch_script = f"""
import io
import json
from contextlib import redirect_stdout

scripts = json.loads({json.dumps(encoded_scripts, ensure_ascii=False)})
results = []

for code in scripts:
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            exec(code, {{"db": db, "print": print}})
        results.append(captured.getvalue())
    except Exception as exc:
        results.append(f"Script error: {{exc}}")

print(json.dumps(results, ensure_ascii=False))
"""

        raw_output = self._execute_script(batch_script).strip()
        if not raw_output:
            return ["" for _ in script_codes]

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            return [self._execute_script(code) for code in script_codes]

        if not isinstance(parsed, list) or len(parsed) != len(script_codes):
            return [self._execute_script(code) for code in script_codes]

        return [str(item) for item in parsed]

    def _execute_scripts_with_auto_fix(self, scripts_found: list[str], max_retries: int = 2) -> list[str]:
        """Execute scripts with batched passes and auto-fix retries for common failures."""
        if not scripts_found:
            return []

        states: list[dict] = []
        for index, script_code in enumerate(scripts_found, 1):
            normalized_code, fixes = self._normalize_generated_script(script_code.strip())
            if fixes:
                self.callback.on_metric(f"Auto-fixed script {index}: {', '.join(fixes)}")
                self.callback.on_event(
                    "script_fix",
                    f"Script {index} Compatibility Fixes",
                    "\n".join(fixes),
                    duration_ms=None,
                )

            self.callback.on_script_code(normalized_code)
            self.callback.on_event(
                "script_input",
                f"Script {index} Input",
                normalized_code,
                duration_ms=None,
            )

            states.append(
                {
                    "index": index,
                    "code": normalized_code,
                    "output": "",
                    "retry_count": 0,
                }
            )

        pending_indices = list(range(len(states)))
        pass_index = 0

        while pending_indices:
            pass_index += 1
            batch_codes = [states[i]["code"] for i in pending_indices]

            batch_started = time.perf_counter()
            if len(batch_codes) > 1:
                self.callback.on_metric(
                    f"Executing {len(batch_codes)} scripts in batched pass {pass_index}"
                )
                outputs = self._execute_script_batch(batch_codes)
                batch_elapsed_ms = (time.perf_counter() - batch_started) * 1000.0
                self.callback.on_event(
                    "script_batch",
                    "Script Batch Execution",
                    f"Scripts: {len(batch_codes)}\nPass: {pass_index}",
                    duration_ms=batch_elapsed_ms,
                )
            else:
                outputs = [self._execute_script(batch_codes[0])]
                batch_elapsed_ms = (time.perf_counter() - batch_started) * 1000.0

            if len(outputs) != len(batch_codes):
                outputs = [self._execute_script(code) for code in batch_codes]

            retry_indices: list[int] = []
            for local_idx, output in enumerate(outputs):
                state_idx = pending_indices[local_idx]
                state = states[state_idx]
                state["output"] = output

                if output:
                    self.callback.on_script_output(output)

                self.callback.on_event(
                    "script_output",
                    f"Script {state['index']} Output",
                    output if output else "(empty output)",
                    duration_ms=batch_elapsed_ms if len(batch_codes) == 1 else None,
                )

                if self.history:
                    self.history.append_script_execution(state["code"], output)

                if not self._is_script_error_output(output):
                    continue
                if state["retry_count"] >= max_retries:
                    continue

                fixed_code, retry_fixes = self._normalize_generated_script_from_error(
                    state["code"],
                    output,
                )
                if not retry_fixes or fixed_code == state["code"]:
                    continue

                state["retry_count"] += 1
                state["code"] = fixed_code

                self.callback.on_metric(
                    f"Retrying script {state['index']} with auto-fix ({state['retry_count']}/{max_retries})"
                )
                self.callback.on_event(
                    "script_fix",
                    f"Script {state['index']} Auto-Fix Retry {state['retry_count']}",
                    "\n".join(retry_fixes) + f"\n\nPrevious error:\n{output}",
                    duration_ms=None,
                )
                self.callback.on_script_code(fixed_code)
                self.callback.on_event(
                    "script_input",
                    f"Script {state['index']} Retry {state['retry_count']} Input",
                    fixed_code,
                    duration_ms=None,
                )
                retry_indices.append(state_idx)

            pending_indices = retry_indices

        return [str(state["output"]) for state in states]

    def _parse_tool_payload(self, payload_text: str):
        """Parse JSON payload for idatool calls, falling back to raw text."""
        text = payload_text.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _as_query_list(self, payload, key: str = "queries") -> list[str]:
        """Normalize flexible payloads into a list of query strings."""
        if isinstance(payload, str):
            return [q.strip() for q in payload.split(",") if q.strip()]
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, dict):
            value = payload.get(key) or payload.get("query") or payload.get("addr") or payload.get("name")
            return self._as_query_list(value, key)
        return []

    def _resolve_func_query(self, query: str):
        """Resolve function by address or by name."""
        query = query.strip()
        if not query:
            return None

        try:
            ea = int(query, 0)
            return self.db.functions.get_at(ea)
        except ValueError:
            return self.db.functions.get_function_by_name(query)

    def _run_idatool(self, tool_name: str, payload_text: str) -> str:
        """Execute a built-in MCP-style IDA tool and return JSON output text.

        Uses the configured script executor so all IDA operations run on the main
        thread when needed (plugin mode), avoiding thread-affinity crashes.
        """
        tool = tool_name.strip().lower()
        payload = self._parse_tool_payload(payload_text)

        script = f"""
import json

tool = {json.dumps(tool)}
payload = json.loads({json.dumps(json.dumps(payload, ensure_ascii=False))})

def as_query_list(value, key='queries'):
    if isinstance(value, str):
        return [q.strip() for q in value.split(',') if q.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        v = value.get(key) or value.get('query') or value.get('addr') or value.get('name')
        return as_query_list(v, key)
    return []

def resolve_func(query):
    query = str(query).strip()
    if not query:
        return None
    try:
        ea = int(query, 0)
        return db.functions.get_at(ea)
    except Exception:
        return db.functions.get_function_by_name(query)

result = None

if tool == 'int_convert':
    inputs = as_query_list(payload, key='inputs')
    if not inputs and isinstance(payload, dict) and payload.get('value') is not None:
        inputs = [str(payload.get('value'))]
    converted = []
    for raw in inputs:
        try:
            value = int(str(raw), 0)
            converted.append({{
                'input': raw,
                'dec': value,
                'hex': hex(value),
                'bin': bin(value),
            }})
        except Exception as exc:
            converted.append({{'input': raw, 'error': str(exc)}})
    result = {{'tool': tool, 'results': converted}}

elif tool == 'lookup_funcs':
    rows = []
    for query in as_query_list(payload):
        func = resolve_func(query)
        if not func:
            rows.append({{'query': query, 'error': 'not found'}})
            continue
        rows.append({{
            'query': query,
            'name': db.functions.get_name(func),
            'start_ea': hex(func.start_ea),
            'end_ea': hex(func.end_ea),
        }})
    result = {{'tool': tool, 'results': rows}}

elif tool == 'list_funcs':
    limit = 20
    offset = 0
    filt = ''
    if isinstance(payload, dict):
        limit = int(payload.get('limit', limit))
        offset = int(payload.get('offset', offset))
        filt = str(payload.get('filter', '')).strip().lower()
    funcs = sorted(list(db.functions), key=lambda f: f.start_ea)
    if filt:
        funcs = [f for f in funcs if filt in db.functions.get_name(f).lower()]
    selected = funcs[offset:offset + max(1, min(limit, 200))]
    rows = []
    for func in selected:
        rows.append({{
            'name': db.functions.get_name(func),
            'start_ea': hex(func.start_ea),
            'end_ea': hex(func.end_ea),
            'size': func.end_ea - func.start_ea,
        }})
    result = {{'tool': tool, 'count': len(rows), 'total_functions': len(funcs), 'results': rows}}

elif tool in ('decompile', 'disasm', 'analyze_function'):
    queries = as_query_list(payload)
    query = queries[0] if queries else ''
    func = resolve_func(query)
    if not func:
        result = {{'tool': tool, 'error': f'Function not found: {{query}}'}}
    else:
        out = {{
            'tool': tool,
            'name': db.functions.get_name(func),
            'start_ea': hex(func.start_ea),
            'end_ea': hex(func.end_ea),
            'size': func.end_ea - func.start_ea,
        }}
        if tool in ('decompile', 'analyze_function'):
            try:
                out['pseudocode'] = '\\n'.join(db.functions.get_pseudocode(func)[:120])
            except Exception as exc:
                out['pseudocode_error'] = str(exc)
        if tool in ('disasm', 'analyze_function'):
            try:
                out['disassembly'] = '\\n'.join(db.functions.get_disassembly(func)[:160])
            except Exception as exc:
                out['disasm_error'] = str(exc)
        if tool == 'analyze_function':
            try:
                callees = db.functions.get_callees(func)
                callers = db.functions.get_callers(func)
                out['callees'] = [db.functions.get_name(c) for c in callees[:40]]
                out['callers'] = [db.functions.get_name(c) for c in callers[:40]]
                out['callees_count'] = len(callees)
                out['callers_count'] = len(callers)
            except Exception as exc:
                out['xref_error'] = str(exc)
        result = out

elif tool == 'xrefs_to':
    rows = []
    for query in as_query_list(payload):
        ea = None
        try:
            ea = int(query, 0)
        except Exception:
            func = resolve_func(query)
            if func:
                ea = func.start_ea
        if ea is None:
            rows.append({{'query': query, 'error': 'invalid address or function'}})
            continue

        xrefs = []
        for xref in db.xrefs.to_ea(ea):
            xrefs.append({{
                'from_ea': hex(xref.from_ea),
                'to_ea': hex(xref.to_ea),
                'type': str(getattr(xref.type, 'name', xref.type)),
            }})
        rows.append({{'query': query, 'target': hex(ea), 'count': len(xrefs), 'xrefs': xrefs[:200]}})
    result = {{'tool': tool, 'results': rows}}

else:
    result = {{
        'tool': tool,
        'error': 'Unknown idatool',
        'available_tools': {json.dumps(AVAILABLE_IDATOOLS, ensure_ascii=False)},
    }}

print(json.dumps(result, ensure_ascii=False, indent=2))
"""
        return self._execute_script(script)

    async def _execute_idatool_calls(self, idatools_found: list[tuple[str, str]]) -> list[str]:
        """Execute MCP-style idatool calls with safe batching semantics."""
        if not idatools_found:
            return []

        call_meta: list[dict] = []
        for tool_index, (tool_name, tool_payload) in enumerate(idatools_found, 1):
            tool_display = f"idatool:{tool_name}"
            self.callback.on_tool_use(tool_display, "mcp-style tool call")

            tool_use_id = f"idatool_{int(time.time() * 1000)}_{tool_index}"
            if self.history:
                self.history.append_tool_use(
                    tool_display,
                    {"payload": tool_payload},
                    tool_use_id=tool_use_id,
                )

            self.callback.on_event(
                "idatool_request",
                f"IDATool Request ({tool_name})",
                tool_payload,
                duration_ms=None,
            )

            call_meta.append(
                {
                    "tool_name": tool_name,
                    "tool_display": tool_display,
                    "tool_payload": tool_payload,
                    "tool_use_id": tool_use_id,
                }
            )

        outputs: list[str] = []
        can_parallelize = (not self._uses_custom_script_executor) and len(call_meta) > 1

        if can_parallelize:
            self.callback.on_metric(f"Executing {len(call_meta)} idatools in parallel batch")

            async def _run_one(tool_name: str, tool_payload: str) -> tuple[str, float]:
                started = time.perf_counter()
                try:
                    output_text = await asyncio.to_thread(self._run_idatool, tool_name, tool_payload)
                except Exception as exc:
                    output_text = f"Script error: {exc}"
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                return output_text, elapsed_ms

            results = await asyncio.gather(
                *[
                    _run_one(str(meta["tool_name"]), str(meta["tool_payload"]))
                    for meta in call_meta
                ],
                return_exceptions=False,
            )

            for meta, (tool_output, tool_elapsed_ms) in zip(call_meta, results):
                outputs.append(tool_output)
                self.callback.on_script_output(tool_output)
                self.callback.on_event(
                    "idatool_response",
                    f"IDATool Response ({meta['tool_name']})",
                    tool_output,
                    duration_ms=tool_elapsed_ms,
                )
                if self.history:
                    self.history.append_tool_result(str(meta["tool_use_id"]), tool_output, is_error=False)

            return outputs

        if len(call_meta) > 1:
            self.callback.on_metric(
                f"Executing {len(call_meta)} idatools in serialized batch (main-thread safe mode)"
            )

        for meta in call_meta:
            tool_started = time.perf_counter()
            try:
                tool_output = self._run_idatool(str(meta["tool_name"]), str(meta["tool_payload"]))
            except Exception as exc:
                tool_output = f"Script error: {exc}"
            tool_elapsed_ms = (time.perf_counter() - tool_started) * 1000.0

            outputs.append(tool_output)
            self.callback.on_script_output(tool_output)
            self.callback.on_event(
                "idatool_response",
                f"IDATool Response ({meta['tool_name']})",
                tool_output,
                duration_ms=tool_elapsed_ms,
            )

            if self.history:
                self.history.append_tool_result(str(meta["tool_use_id"]), tool_output, is_error=False)

        return outputs

    def _strip_agent_tags(self, text: str) -> str:
        """Remove tool/script/delegate tags from assistant-visible plain text."""
        return DELEGATE_PATTERN.sub("", IDATOOL_PATTERN.sub("", IDASCRIPT_PATTERN.sub("", text))).strip()

    def _resolve_delegate_model(self, agent: str) -> str:
        """Resolve a stable delegate model for the current provider.

        Delegate tags may include provider-incompatible aliases (e.g. haiku on NIM),
        so default to the local cheapest known model for reliability.
        """
        provider_name = normalize_provider(self.provider_config.provider)
        agent_text = (agent or "").strip().lower()

        if provider_name == "claude":
            if "haiku" in agent_text:
                return "claude-3-haiku-20240307"
            if "sonnet" in agent_text:
                return "claude-3-7-sonnet-latest"

        return self._select_cheapest_model()

    async def _run_delegate_query(self, agent: str, task: str) -> str:
        """Execute one delegate query and return the textual delegate result."""
        provider_name = normalize_provider(self.provider_config.provider)
        delegate_model = self._resolve_delegate_model(agent)
        delegate_system = (
            "You are a delegated reverse-engineering subagent. "
            "Return concise factual results only, no chain-of-thought."
        )

        if provider_name != "claude":
            delegate_cfg = ProviderConfig(
                provider=self.provider_config.provider,
                auth_mode=self.provider_config.auth_mode,
                api_key=self.provider_config.api_key,
                model=delegate_model,
                base_url=self.provider_config.base_url,
            )
            return await _query_openai_compat(
                delegate_cfg,
                [
                    {"role": "system", "content": delegate_system},
                    {"role": "user", "content": task},
                ],
                timeout_seconds=120.0,
            )

        stderr_lines: list[str] = []

        def _collect_stderr(line: str) -> None:
            text = line.rstrip()
            if text:
                stderr_lines.append(text)

        options = ClaudeAgentOptions(
            cwd=str(PROJECT_DIR),
            permission_mode="bypassPermissions",
            allowed_tools=[],
            env=build_provider_env(self.provider_config),
            model=delegate_model,
            stderr=_collect_stderr,
        )

        client = ClaudeSDKClient(options=options)
        try:
            await client.connect()
            await client.query(task)

            response_text = ""
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text

            await client.disconnect()
            response_text = response_text.strip()
            if response_text:
                return response_text
            raise RuntimeError("Delegate returned an empty response")
        except Exception:
            try:
                await client.disconnect()
            except Exception:
                pass
            if stderr_lines:
                raise RuntimeError("\n".join(stderr_lines[-12:]))
            raise

    async def _execute_delegate_calls(self, delegations: list[tuple[str, str]]) -> list[str]:
        """Execute <delegate> calls and return delegation_result payloads."""
        if not delegations:
            return []

        outputs: list[str] = []
        for i, (agent, task) in enumerate(delegations, 1):
            agent_name = (agent or "").strip() or "delegate"
            task_text = task.strip()

            self.callback.on_tool_use("Subagent Delegation", f"{agent_name}: {task_text[:70]}")
            logger.info("Delegating task to %s", agent_name)

            delegate_id = f"delegate_{int(time.time() * 1000)}_{i}"
            if self.history:
                self.history.append_tool_use(
                    "Delegate",
                    {"agent": agent_name, "task": task_text},
                    tool_use_id=delegate_id,
                )

            self.callback.on_event(
                "delegate_request",
                f"Delegate Request ({agent_name})",
                task_text,
                duration_ms=None,
            )

            delegate_started = time.perf_counter()
            is_error = False
            try:
                delegate_text = await self._run_delegate_query(agent_name, task_text)
                if not delegate_text.strip():
                    raise RuntimeError("Delegate returned an empty response")
                delegation_output = (
                    f"<delegation_result agent='{agent_name}'>"
                    f"{delegate_text.strip()}"
                    "</delegation_result>"
                )
            except Exception as exc:
                is_error = True
                delegation_output = (
                    f"<delegation_result agent='{agent_name}'>"
                    f"Delegation failed: {str(exc).strip() or 'unknown error'}"
                    "</delegation_result>"
                )

            delegate_elapsed_ms = (time.perf_counter() - delegate_started) * 1000.0
            outputs.append(delegation_output)
            self.callback.on_script_output(delegation_output)
            self.callback.on_event(
                "delegate_response",
                f"Delegate Response ({agent_name})",
                delegation_output,
                duration_ms=delegate_elapsed_ms,
            )

            if self.history:
                self.history.append_tool_result(delegate_id, delegation_output, is_error=is_error)

        return outputs

    async def _condense_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Condense long conversations using the cheapest available summarizer model."""
        keep_recent = 10
        condense_threshold = 16
        if len(messages) <= condense_threshold:
            return messages

        system_message = messages[0]
        older_messages = messages[1:-keep_recent]
        recent_messages = messages[-keep_recent:]
        if not older_messages:
            return messages

        cheap_model = self._select_cheapest_model()
        summary_text = self._local_compact_summary(older_messages)
        request_preview = json.dumps(older_messages[-8:], ensure_ascii=False, indent=2)

        self.callback.on_metric(
            f"Condensing history via {cheap_model} ({len(older_messages)} messages -> summary)"
        )

        if normalize_provider(self.provider_config.provider) != "claude":
            started = time.perf_counter()
            try:
                summary_cfg = ProviderConfig(
                    provider=self.provider_config.provider,
                    auth_mode=self.provider_config.auth_mode,
                    api_key=self.provider_config.api_key,
                    model=cheap_model,
                    base_url=self.provider_config.base_url,
                )
                summary_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "Summarize prior assistant/user/tool context in <=160 tokens. "
                            "Keep only facts needed for reverse engineering continuity."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Conversation history JSON:\n{request_preview}",
                    },
                ]
                summary_text = await _query_openai_compat(
                    summary_cfg,
                    summary_prompt,
                    timeout_seconds=60.0,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.callback.on_event(
                    "condense",
                    "History Condensed",
                    (
                        f"Model: {cheap_model}\n"
                        f"Input messages: {len(older_messages)}\n\n"
                        f"Input preview:\n{request_preview}\n\n"
                        f"Summary:\n{summary_text}"
                    ),
                    duration_ms=elapsed_ms,
                )
            except Exception as exc:
                self.callback.on_metric(f"Condense summary fallback used: {exc}")

        condensed_notice = {
            "role": "system",
            "content": (
                "[Condensed conversation summary]\n"
                f"Source model: {cheap_model}\n"
                f"{summary_text}\n\n"
                "Prioritize the most recent messages for exact details."
            ),
        }
        return [system_message, condensed_notice, *recent_messages]

    async def _process_message_openai_compat(self, user_input: str) -> str:
        """Agentic loop for OpenAI-compatible providers without Claude SDK."""
        logger.info("-" * 60)
        logger.info("USER MESSAGE (OpenAI-compatible): %s...", user_input[:200])

        if self.history:
            self.history.append_user_message(user_input)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt or _load_system_prompt()},
        ]

        current_input = user_input
        all_script_outputs: list[str] = []
        turn = 0
        self._cancelled = False

        while turn < self.max_turns:
            if self._cancelled:
                logger.info("Operation cancelled by user")
                self.callback.on_error("Operation cancelled")
                break

            turn += 1
            logger.info("=== TURN %s/%s (OpenAI-compatible) ===", turn, self.max_turns)
            self.callback.on_turn_start(turn, self.max_turns)
            self.callback.on_thinking()

            if turn == 1 and not current_input.startswith("Script error:") and not current_input.startswith("<script_output"):
                safe_input = (
                    "[SYSTEM REMINDER: Do NOT summarize API docs. The target binary is loaded in IDA. "
                    "Use <idascript> with db.* calls only. Use db.functions.get_function_by_name(...), not get_by_name(...). "
                    "Import modules (like re) before use.]\n\n"
                    f"USER REQUEST: {current_input}"
                )
                messages.append({"role": "user", "content": safe_input})
            else:
                messages.append({"role": "user", "content": current_input})

            condensed_messages = await self._condense_history(messages)
            self._emit_context_warning_if_needed(condensed_messages)

            preview_messages: list[dict[str, str]] = []
            for msg in condensed_messages[-6:]:
                role = str(msg.get("role", ""))
                content = str(msg.get("content", ""))
                if len(content) > 1200:
                    content = content[:1200] + "\n... [truncated for log preview]"
                preview_messages.append({"role": role, "content": content})

            request_preview = json.dumps(preview_messages, ensure_ascii=False, indent=2)
            self.callback.on_metric(f"Sending prompt to model ({len(condensed_messages)} messages)...")
            self.callback.on_event(
                "model_request",
                "Model Request",
                (
                    f"Provider: {normalize_provider(self.provider_config.provider)}\n"
                    f"Model: {resolve_model(self.provider_config)}\n"
                    f"Message count: {len(condensed_messages)}\n\n"
                    f"Input preview:\n{request_preview}"
                ),
                duration_ms=None,
            )
            request_started = time.perf_counter()
            used_streaming = False
            assistant_text = ""
            if not self._streaming_disabled:
                try:
                    assistant_text = await _query_openai_compat_stream(
                        self.provider_config,
                        condensed_messages,
                        timeout_seconds=180.0,
                        on_chunk=self.callback.on_text,
                    )
                    used_streaming = True
                    self._stream_failures = 0
                    self.callback.on_metric("Streaming response enabled")
                except Exception as stream_exc:
                    self._stream_failures += 1
                    err_text = str(stream_exc).strip()
                    if err_text:
                        self.callback.on_metric(
                            f"Streaming unavailable, using non-streaming: {err_text[:180]}"
                        )
                    if self._stream_failures >= 2:
                        self._streaming_disabled = True
                        self.callback.on_metric("Streaming disabled for this chat after repeated failures")
            else:
                self.callback.on_metric("Streaming disabled for this chat; using non-streaming responses")

            if not used_streaming:
                assistant_text = await _query_openai_compat(
                    self.provider_config,
                    condensed_messages,
                    timeout_seconds=180.0,
                )
            request_elapsed_ms = (time.perf_counter() - request_started) * 1000.0

            self.callback.on_metric(f"Received response ({len(assistant_text)} chars)")
            self.callback.on_event(
                "model_response",
                "Model Response",
                assistant_text,
                duration_ms=request_elapsed_ms,
            )

            scripts_found = [script.strip() for script in IDASCRIPT_PATTERN.findall(assistant_text)]
            idatools_found = [
                (name.strip().lower(), payload.strip())
                for name, payload in IDATOOL_PATTERN.findall(assistant_text)
            ]
            delegations = [
                (agent.strip(), task.strip())
                for agent, task in DELEGATE_PATTERN.findall(assistant_text)
            ]
            cleaned = self._strip_agent_tags(assistant_text)

            # Keep a compact assistant message in rolling context to reduce token churn.
            compact_text = cleaned[:800]
            if cleaned and len(cleaned) > 800:
                compact_text += " ..."

            if scripts_found or idatools_found or delegations:
                script_bundle = "\n\n".join(
                    f"<idascript>\n{code}\n</idascript>" for code in scripts_found
                )
                tool_bundle = "\n\n".join(
                    f"<idatool name=\"{name}\">\n{payload}\n</idatool>"
                    for name, payload in idatools_found
                )
                delegate_bundle = "\n\n".join(
                    f"<delegate agent=\"{agent}\">\n{task}\n</delegate>"
                    for agent, task in delegations
                )
                combined_bundle = "\n\n".join(
                    chunk for chunk in (script_bundle, tool_bundle, delegate_bundle) if chunk.strip()
                )
                assistant_for_context = (
                    f"{compact_text}\n\n{combined_bundle}" if compact_text else combined_bundle
                )
            else:
                assistant_for_context = compact_text or assistant_text[:800]

            messages.append({"role": "assistant", "content": assistant_for_context})

            self.callback.on_thinking_done()

            if cleaned and not used_streaming:
                self.callback.on_text(cleaned)
                if self.history:
                    self.history.append_assistant_message(cleaned)
            elif cleaned and self.history:
                self.history.append_assistant_message(cleaned)

            if not scripts_found and not idatools_found and not delegations:
                logger.info("No scripts/tools in response - agent is done")
                break

            script_outputs: list[str] = []

            tool_outputs = await self._execute_idatool_calls(idatools_found)
            if tool_outputs:
                script_outputs.extend(tool_outputs)
                all_script_outputs.extend(tool_outputs)

            delegate_outputs = await self._execute_delegate_calls(delegations)
            if delegate_outputs:
                script_outputs.extend(delegate_outputs)
                all_script_outputs.extend(delegate_outputs)

            final_script_outputs = self._execute_scripts_with_auto_fix(scripts_found)
            if final_script_outputs:
                script_outputs.extend(final_script_outputs)
                all_script_outputs.extend(final_script_outputs)
                for index, output in enumerate(final_script_outputs, 1):
                    logger.debug("Script %s output (OpenAI-compatible): %s", index, output[:200])

            if script_outputs:
                formatted_outputs = []
                for i, output in enumerate(script_outputs, 1):
                    if len(script_outputs) > 1:
                        formatted_outputs.append(f"Tool output {i}:\n{output}")
                    else:
                        formatted_outputs.append(output)
                current_input = "Script output:\n\n" + "\n\n".join(formatted_outputs)
            else:
                current_input = "Script executed successfully with no output."

        if turn >= self.max_turns:
            logger.warning("Reached maximum turns (%s)", self.max_turns)
            self.callback.on_error(f"Reached maximum turns ({self.max_turns})")

        return "\n".join(all_script_outputs) if all_script_outputs else ""

    def _default_execute_script(self, code: str) -> str:
        """Default script executor - direct execution.

        Args:
            code: Python code to execute with `db` in scope.

        Returns:
            Captured stdout output or error message.
        """
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        try:
            exec(code, {"db": self.db, "print": print})
            return captured.getvalue()
        except Exception as e:
            return f"Script error: {e}"
        finally:
            sys.stdout = old_stdout

    async def _process_single_response(self) -> tuple[list[str], list[str]]:
        """Process a single agent response.

        Returns:
            Tuple of (scripts_found, script_outputs)
        """
        full_text: list[str] = []
        scripts_found: list[str] = []
        script_outputs: list[str] = []
        first_output = True

        async for message in self.client.receive_response():
            logger.debug(f"Received message type: {type(message).__name__}")

            if isinstance(message, AssistantMessage):
                logger.debug(f"AssistantMessage with {len(message.content)} blocks")
                for i, block in enumerate(message.content):
                    logger.debug(f"  Block {i}: {type(block).__name__}")

                    # Notify thinking done on first output
                    if first_output:
                        self.callback.on_thinking_done()
                        first_output = False

                    if isinstance(block, ToolUseBlock):
                        logger.info(f"TOOL USE: {block.name}")
                        logger.debug(f"  Tool input: {block.input}")

                        # Extract tool details based on tool type
                        details = ""
                        if block.name == "Read":
                            details = block.input.get("file_path", "")
                        elif block.name == "Grep":
                            details = block.input.get("pattern", "")
                        elif block.name == "Glob":
                            details = block.input.get("pattern", "")
                        elif block.name == "Task":
                            details = block.input.get("description", "")
                        else:
                            # Log unknown tools
                            logger.warning(f"  Unknown tool: {block.name}, input: {block.input}")
                            details = str(block.input)
                        self.callback.on_tool_use(block.name, details)
                        self.callback.on_event(
                            "tool_request",
                            f"Tool Call: {block.name}",
                            json.dumps(block.input, ensure_ascii=False, indent=2)
                            if isinstance(block.input, dict)
                            else str(block.input),
                            duration_ms=None,
                        )

                        # Log tool use to history
                        if self.history:
                            self.history.append_tool_use(
                                block.name,
                                block.input if isinstance(block.input, dict) else {"input": str(block.input)}
                            )

                    elif isinstance(block, TextBlock):
                        text = block.text
                        logger.debug(f"  TextBlock ({len(text)} chars): {text[:100]}...")
                        full_text.append(text)

                        # Output text excluding <idascript> blocks
                        cleaned = self._strip_agent_tags(text)
                        if cleaned:
                            self.callback.on_text(cleaned)
                            # Log assistant text to history
                            if self.history:
                                self.history.append_assistant_message(cleaned)
                    else:
                        logger.warning(f"  Unknown block type: {type(block).__name__}")

            elif isinstance(message, ResultMessage):
                logger.info(f"ResultMessage: turns={message.num_turns}, cost={message.total_cost_usd}")

                # Extract scripts from the response
                if full_text:
                    combined = "".join(full_text)
                    scripts_found = IDASCRIPT_PATTERN.findall(combined)
                    idatools_found = [
                        (name.strip().lower(), payload.strip())
                        for name, payload in IDATOOL_PATTERN.findall(combined)
                    ]
                    delegations = [
                        (agent.strip(), task.strip())
                        for agent, task in DELEGATE_PATTERN.findall(combined)
                    ]
                    logger.info(f"Found {len(scripts_found)} scripts in response")

                    # Execute idatool calls first (mcp-style high-level tools).
                    tool_outputs = await self._execute_idatool_calls(idatools_found)
                    script_outputs.extend(tool_outputs)

                    # Execute delegate calls.
                    delegate_outputs = await self._execute_delegate_calls(delegations)
                    script_outputs.extend(delegate_outputs)

                    # Execute each script
                    final_script_outputs = self._execute_scripts_with_auto_fix(scripts_found)
                    script_outputs.extend(final_script_outputs)

                    if (idatools_found or delegations) and not scripts_found:
                        # Keep outer loop alive so tool outputs are fed back to the model.
                        scripts_found = ["<tool_batch>"]

                if self.verbose:
                    self.callback.on_result(
                        message.num_turns,
                        message.total_cost_usd
                    )
            else:
                logger.warning(f"Unknown message type: {type(message).__name__}")

        return scripts_found, script_outputs

    async def process_message(self, user_input: str) -> str:
        """Agentic loop - process message and continue until agent is done.

        The agent will keep working, seeing script outputs and fixing errors,
        until either:
        - It responds without any <idascript> tags (task complete)
        - Maximum turns is reached

        Args:
            user_input: The user's message/query.

        Returns:
            Combined script outputs as a string.
        """
        if not self._using_claude_sdk:
            return await self._process_message_openai_compat(user_input)

        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        logger.info("-" * 60)
        logger.info(f"USER MESSAGE: {user_input[:200]}...")

        # Log user message to history
        if self.history:
            self.history.append_user_message(user_input)

        current_input = user_input
        all_script_outputs: list[str] = []
        turn = 0
        self._cancelled = False

        while turn < self.max_turns:
            # Check for cancellation
            if self._cancelled:
                logger.info("Operation cancelled by user")
                self.callback.on_error("Operation cancelled")
                break
            turn += 1
            logger.info(f"=== TURN {turn}/{self.max_turns} ===")
            self.callback.on_turn_start(turn, self.max_turns)
            self.callback.on_thinking()

            # Send message to agent
            logger.debug(f"Sending to agent: {current_input[:200]}...")
            await self.client.query(current_input)

            # Process response and execute any scripts
            scripts_found, script_outputs = await self._process_single_response()
            all_script_outputs.extend(script_outputs)

            if not scripts_found:
                # No scripts in response = agent is done
                logger.info("No scripts in response - agent is done")
                break

            # Feed script results back to agent for next turn
            if script_outputs:
                # Format all outputs for the agent
                formatted_outputs = []
                for i, output in enumerate(script_outputs, 1):
                    if len(scripts_found) > 1:
                        formatted_outputs.append(f"Script {i} output:\n{output}")
                    else:
                        formatted_outputs.append(output)
                current_input = "Script output:\n\n" + "\n\n".join(formatted_outputs)
                logger.debug(f"Feeding back to agent: {current_input[:200]}...")
            else:
                current_input = "Script executed successfully with no output."
                logger.debug("Script had no output, notifying agent")

        if turn >= self.max_turns:
            logger.warning(f"Reached maximum turns ({self.max_turns})")
            self.callback.on_error(f"Reached maximum turns ({self.max_turns})")

        return "\n".join(all_script_outputs) if all_script_outputs else ""
