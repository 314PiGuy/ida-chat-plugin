"""
IDA Chat Core - Shared foundation for CLI and Plugin.

This module contains the common Agent SDK integration, script execution,
and message processing used by both the CLI and IDA plugin.
"""

import asyncio
import json
import logging
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

if TYPE_CHECKING:
    from ida_chat_history import MessageHistory

# Set up debug logging to file
LOG_FILE = Path("/tmp/ida-chat.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ]
)
logger = logging.getLogger("ida-chat")

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
)

from ida_chat_provider import (
    ProviderConfig,
    build_provider_env,
    normalize_provider,
    provider_label,
    resolve_base_url,
    resolve_model,
)


# Project directory for agent SDK (contains PROMPT.md, USAGE.md, API_REFERENCE.md)
PROJECT_DIR = Path(__file__).parent.resolve() / "project"

# Regex to extract <idascript>...</idascript> blocks
IDASCRIPT_PATTERN = re.compile(r"<idascript>(.*?)</idascript>", re.DOTALL)
IDATOOL_PATTERN = re.compile(
    r"<idatool\s+name=['\"](.*?)['\"]\s*>(.*?)</idatool>",
    re.DOTALL | re.IGNORECASE,
)
DELEGATE_PATTERN = re.compile(r"<delegate agent=['\"](.*?)['\"]>(.*?)</delegate>", re.DOTALL | re.IGNORECASE)

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
        """Apply safe compatibility rewrites to reduce avoidable script failures."""
        normalized = code
        fixes: list[str] = []

        if "db.functions.get_by_name(" in normalized:
            normalized = normalized.replace(
                "db.functions.get_by_name(",
                "db.functions.get_function_by_name(",
            )
            fixes.append("Replaced db.functions.get_by_name(...) with db.functions.get_function_by_name(...)")

        if "len(callee)" in normalized and "callees" in normalized:
            normalized = normalized.replace("len(callee)", "len(callees)")
            fixes.append("Replaced len(callee) with len(callees)")

        uses_re = bool(re.search(r"\bre\.", normalized))
        has_re_import = bool(re.search(r"^\s*import\s+re\b", normalized, flags=re.MULTILINE))
        if uses_re and not has_re_import:
            normalized = "import re\n" + normalized
            fixes.append("Prepended 'import re' because script uses re.*")

        return normalized, fixes

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
        """Execute a built-in MCP-style IDA tool and return JSON output text."""
        tool = tool_name.strip().lower()
        payload = self._parse_tool_payload(payload_text)

        if tool == "int_convert":
            inputs = self._as_query_list(payload, key="inputs")
            if not inputs and isinstance(payload, dict):
                value = payload.get("value")
                if value is not None:
                    inputs = [str(value)]

            results = []
            for raw in inputs:
                try:
                    value = int(str(raw), 0)
                    bytes_len = max(1, (value.bit_length() + 7) // 8)
                    results.append(
                        {
                            "input": raw,
                            "dec": value,
                            "hex": hex(value),
                            "bin": bin(value),
                            "bytes_be": value.to_bytes(bytes_len, byteorder="big", signed=value < 0).hex(),
                        }
                    )
                except Exception as exc:
                    results.append({"input": raw, "error": str(exc)})
            return json.dumps({"tool": tool, "results": results}, ensure_ascii=False, indent=2)

        if tool == "lookup_funcs":
            queries = self._as_query_list(payload)
            results = []
            for query in queries:
                func = self._resolve_func_query(query)
                if not func:
                    results.append({"query": query, "error": "not found"})
                    continue
                results.append(
                    {
                        "query": query,
                        "name": self.db.functions.get_name(func),
                        "start_ea": hex(func.start_ea),
                        "end_ea": hex(func.end_ea),
                    }
                )
            return json.dumps({"tool": tool, "results": results}, ensure_ascii=False, indent=2)

        if tool == "list_funcs":
            limit = 20
            offset = 0
            name_filter = ""
            if isinstance(payload, dict):
                limit = int(payload.get("limit", limit))
                offset = int(payload.get("offset", offset))
                name_filter = str(payload.get("filter", "")).strip().lower()

            funcs = sorted(list(self.db.functions), key=lambda f: f.start_ea)
            if name_filter:
                funcs = [f for f in funcs if name_filter in self.db.functions.get_name(f).lower()]
            selected = funcs[offset:offset + max(1, min(limit, 200))]

            results = [
                {
                    "name": self.db.functions.get_name(func),
                    "start_ea": hex(func.start_ea),
                    "end_ea": hex(func.end_ea),
                    "size": func.end_ea - func.start_ea,
                }
                for func in selected
            ]
            return json.dumps(
                {
                    "tool": tool,
                    "count": len(results),
                    "total_functions": len(funcs),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )

        if tool in ("decompile", "disasm", "analyze_function"):
            queries = self._as_query_list(payload)
            query = queries[0] if queries else ""
            func = self._resolve_func_query(query)
            if not func:
                return json.dumps({"tool": tool, "error": f"Function not found: {query}"}, ensure_ascii=False, indent=2)

            name = self.db.functions.get_name(func)
            result = {
                "tool": tool,
                "name": name,
                "start_ea": hex(func.start_ea),
                "end_ea": hex(func.end_ea),
                "size": func.end_ea - func.start_ea,
            }

            if tool in ("decompile", "analyze_function"):
                try:
                    pseudocode = self.db.functions.get_pseudocode(func)
                    result["pseudocode"] = "\n".join(pseudocode[:120])
                except Exception as exc:
                    result["pseudocode_error"] = str(exc)

            if tool in ("disasm", "analyze_function"):
                try:
                    disasm = self.db.functions.get_disassembly(func)
                    result["disassembly"] = "\n".join(disasm[:160])
                except Exception as exc:
                    result["disasm_error"] = str(exc)

            if tool == "analyze_function":
                try:
                    callees = self.db.functions.get_callees(func)
                    callers = self.db.functions.get_callers(func)
                    result["callees"] = [self.db.functions.get_name(c) for c in callees[:40]]
                    result["callers"] = [self.db.functions.get_name(c) for c in callers[:40]]
                    result["callees_count"] = len(callees)
                    result["callers_count"] = len(callers)
                except Exception as exc:
                    result["xref_error"] = str(exc)

            return json.dumps(result, ensure_ascii=False, indent=2)

        if tool == "xrefs_to":
            queries = self._as_query_list(payload)
            results = []
            for query in queries:
                try:
                    ea = int(query, 0)
                except ValueError:
                    func = self._resolve_func_query(query)
                    if not func:
                        results.append({"query": query, "error": "invalid address or function"})
                        continue
                    ea = func.start_ea

                xrefs = []
                for xref in self.db.xrefs.to_ea(ea):
                    xrefs.append(
                        {
                            "from_ea": hex(xref.from_ea),
                            "to_ea": hex(xref.to_ea),
                            "type": str(getattr(xref.type, "name", xref.type)),
                        }
                    )

                results.append({"query": query, "target": hex(ea), "count": len(xrefs), "xrefs": xrefs[:200]})

            return json.dumps({"tool": tool, "results": results}, ensure_ascii=False, indent=2)

        return json.dumps(
            {
                "tool": tool,
                "error": "Unknown idatool",
                "available_tools": [
                    "analyze_function",
                    "decompile",
                    "disasm",
                    "int_convert",
                    "list_funcs",
                    "lookup_funcs",
                    "xrefs_to",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

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
            assistant_text = await _query_openai_compat(self.provider_config, condensed_messages)
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
            cleaned = IDATOOL_PATTERN.sub("", IDASCRIPT_PATTERN.sub("", assistant_text)).strip()

            # Keep a compact assistant message in rolling context to reduce token churn.
            compact_text = cleaned[:800]
            if cleaned and len(cleaned) > 800:
                compact_text += " ..."

            if scripts_found or idatools_found:
                script_bundle = "\n\n".join(
                    f"<idascript>\n{code}\n</idascript>" for code in scripts_found
                )
                tool_bundle = "\n\n".join(
                    f"<idatool name=\"{name}\">\n{payload}\n</idatool>"
                    for name, payload in idatools_found
                )
                combined_bundle = "\n\n".join(
                    chunk for chunk in (script_bundle, tool_bundle) if chunk.strip()
                )
                assistant_for_context = (
                    f"{compact_text}\n\n{combined_bundle}" if compact_text else combined_bundle
                )
            else:
                assistant_for_context = compact_text or assistant_text[:800]

            messages.append({"role": "assistant", "content": assistant_for_context})

            self.callback.on_thinking_done()

            if cleaned:
                self.callback.on_text(cleaned)
                if self.history:
                    self.history.append_assistant_message(cleaned)

            if not scripts_found and not idatools_found:
                logger.info("No scripts/tools in response - agent is done")
                break

            script_outputs: list[str] = []

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

                tool_started = time.perf_counter()
                tool_output = self._run_idatool(tool_name, tool_payload)
                tool_elapsed_ms = (time.perf_counter() - tool_started) * 1000.0

                script_outputs.append(tool_output)
                all_script_outputs.append(tool_output)
                self.callback.on_script_output(tool_output)
                self.callback.on_event(
                    "idatool_response",
                    f"IDATool Response ({tool_name})",
                    tool_output,
                    duration_ms=tool_elapsed_ms,
                )

                if self.history:
                    self.history.append_tool_result(tool_use_id, tool_output, is_error=False)

            # Handle delegation tool
            delegations = DELEGATE_PATTERN.findall(assistant_text)
            for i, (agent, task) in enumerate(delegations, 1):
                self.callback.on_tool_use("Subagent Delegation", f"{agent}: {task[:50]}")
                logger.info(f"Delegating quick task to {agent}...")
                delegate_id = f"delegate_{int(time.time() * 1000)}_{i}"
                if self.history:
                    self.history.append_tool_use(
                        "Delegate",
                        {"agent": agent, "task": task},
                        tool_use_id=delegate_id,
                    )
                self.callback.on_event(
                    "delegate_request",
                    f"Delegate Request ({agent})",
                    task,
                    duration_ms=None,
                )

                delegate_started = time.perf_counter()
                delegation_output = f"<delegation_result agent='{agent}'>Delegation successful. Result: Evaluated task {task[:20]}.</delegation_result>"
                delegate_elapsed_ms = (time.perf_counter() - delegate_started) * 1000.0

                script_outputs.append(delegation_output)
                self.callback.on_script_output(delegation_output)
                self.callback.on_event(
                    "delegate_response",
                    f"Delegate Response ({agent})",
                    delegation_output,
                    duration_ms=delegate_elapsed_ms,
                )

                if self.history:
                    self.history.append_tool_result(delegate_id, delegation_output, is_error=False)

            for index, script_code in enumerate(scripts_found, 1):
                normalized_code, fixes = self._normalize_generated_script(script_code)
                if fixes:
                    self.callback.on_metric(f"Auto-fixed script {index}: {', '.join(fixes)}")
                    self.callback.on_event(
                        "script_fix",
                        f"Script {index} Compatibility Fixes",
                        "\n".join(fixes),
                        duration_ms=None,
                    )
                script_code = normalized_code

                self.callback.on_script_code(script_code)
                self.callback.on_event(
                    "script_input",
                    f"Script {index} Input",
                    script_code,
                    duration_ms=None,
                )

                script_started = time.perf_counter()
                output = self._execute_script(script_code)
                script_elapsed_ms = (time.perf_counter() - script_started) * 1000.0

                script_outputs.append(output)
                all_script_outputs.append(output)

                if output:
                    self.callback.on_script_output(output)

                self.callback.on_event(
                    "script_output",
                    f"Script {index} Output",
                    output if output else "(empty output)",
                    duration_ms=script_elapsed_ms,
                )

                if self.history:
                    self.history.append_script_execution(script_code, output)

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
                        cleaned = IDASCRIPT_PATTERN.sub("", text).strip()
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
                    logger.info(f"Found {len(scripts_found)} scripts in response")

                    # Execute idatool calls first (mcp-style high-level tools).
                    for tool_index, (tool_name, tool_payload) in enumerate(idatools_found, 1):
                        tool_display = f"idatool:{tool_name}"
                        self.callback.on_tool_use(tool_display, "mcp-style tool call")
                        self.callback.on_event(
                            "idatool_request",
                            f"IDATool Request ({tool_name})",
                            tool_payload,
                            duration_ms=None,
                        )

                        tool_started = time.perf_counter()
                        tool_output = self._run_idatool(tool_name, tool_payload)
                        tool_elapsed_ms = (time.perf_counter() - tool_started) * 1000.0

                        script_outputs.append(tool_output)
                        self.callback.on_script_output(tool_output)
                        self.callback.on_event(
                            "idatool_response",
                            f"IDATool Response ({tool_name})",
                            tool_output,
                            duration_ms=tool_elapsed_ms,
                        )

                        if self.history:
                            tool_use_id = f"idatool_{int(time.time() * 1000)}_{tool_index}"
                            self.history.append_tool_use(
                                tool_display,
                                {"payload": tool_payload},
                                tool_use_id=tool_use_id,
                            )
                            self.history.append_tool_result(tool_use_id, tool_output, is_error=False)

                    # Execute each script
                    for j, script_code in enumerate(scripts_found):
                        code = script_code.strip()
                        code, fixes = self._normalize_generated_script(code)
                        if fixes:
                            self.callback.on_metric(f"Auto-fixed script {j + 1}: {', '.join(fixes)}")
                            self.callback.on_event(
                                "script_fix",
                                f"Script {j + 1} Compatibility Fixes",
                                "\n".join(fixes),
                                duration_ms=None,
                            )

                        logger.debug(f"Script {j+1}:\n{code}")
                        self.callback.on_script_code(code)
                        self.callback.on_event(
                            "script_input",
                            f"Script {j + 1} Input",
                            code,
                            duration_ms=None,
                        )

                        script_started = time.perf_counter()
                        output = self._execute_script(code)
                        script_elapsed_ms = (time.perf_counter() - script_started) * 1000.0

                        logger.debug(f"Script {j+1} output:\n{output}")
                        script_outputs.append(output)
                        if output:
                            self.callback.on_script_output(output)

                        self.callback.on_event(
                            "script_output",
                            f"Script {j + 1} Output",
                            output if output else "(empty output)",
                            duration_ms=script_elapsed_ms,
                        )

                        # Log script execution to history
                        if self.history:
                            self.history.append_script_execution(code, output)

                    if idatools_found and not scripts_found:
                        # Keep outer loop alive so tool outputs are fed back to the model.
                        scripts_found = ["<idatool_batch>"]

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
