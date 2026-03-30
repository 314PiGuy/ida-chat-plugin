"""
IDA Chat - LLM Chat Client Plugin for IDA Pro

A dockable chat interface powered by Claude Agent SDK with
multi-provider model support for AI-assisted reverse engineering within IDA Pro.
"""

import asyncio
import json
import os
import re
import sys
from io import StringIO

# Signal to core that we're running inside IDA Pro (enables UI interaction API)
os.environ["IDA_CHAT_INSIDE_IDA"] = "1"
from pathlib import Path
from typing import Callable

import ida_idaapi
import ida_kernwin
import ida_settings
from ida_domain import Database
from PySide6 import QtWidgets
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QPlainTextEdit,
    QApplication,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QComboBox,
    QTabWidget,
    QTextBrowser,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject, QTimer
from PySide6.QtGui import QKeyEvent, QPalette, QFont, QPixmap

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from ida_chat_core import IDAChatCore, ChatCallback, test_provider_connection
from ida_chat_history import MessageHistory
from ida_chat_provider import (
    ProviderConfig,
    SUPPORTED_PROVIDERS,
    apply_provider_environment,
    describe_provider,
    normalize_provider,
    provider_default_base_url,
    provider_default_model,
    provider_free_tier_note,
    provider_key_hint,
    provider_label,
    provider_recommended_models,
    requires_api_key,
    resolve_model,
    validate_provider_config,
)


# Plugin metadata
PLUGIN_NAME = "IDA Chat"
PLUGIN_COMMENT = "LLM Chat Client for IDA Pro"
PLUGIN_HELP = "A chat interface for interacting with LLMs from within IDA Pro"

# Action configuration
ACTION_ID = "ida_chat:toggle_widget"
ACTION_NAME = "Show IDA Chat"
ACTION_HOTKEY = "Ctrl+Shift+C"
ACTION_TOOLTIP = "Toggle the IDA Chat panel"

# Widget form title
WIDGET_TITLE = "IDA Chat"


def get_ida_colors():
    """Get colors from IDA's current palette."""
    app = QApplication.instance()
    palette = app.palette()

    return {
        "window": palette.color(QPalette.Window).name(),
        "window_text": palette.color(QPalette.WindowText).name(),
        "base": palette.color(QPalette.Base).name(),
        "alt_base": palette.color(QPalette.AlternateBase).name(),
        "text": palette.color(QPalette.Text).name(),
        "button": palette.color(QPalette.Button).name(),
        "button_text": palette.color(QPalette.ButtonText).name(),
        "highlight": palette.color(QPalette.Highlight).name(),
        "highlight_text": palette.color(QPalette.HighlightedText).name(),
        "mid": palette.color(QPalette.Mid).name(),
        "dark": palette.color(QPalette.Dark).name(),
        "light": palette.color(QPalette.Light).name(),
    }


# -----------------------------------------------------------------------------
# Settings Management (using ida-settings)
# -----------------------------------------------------------------------------

PROVIDER_PROFILES_KEY = "provider_profiles"


def get_show_wizard() -> bool:
    """Returns whether to show the setup wizard."""
    if ida_settings.has_current_plugin_setting("show_wizard"):
        return ida_settings.get_current_plugin_setting("show_wizard")
    return True  # Default to true


def set_show_wizard(value: bool) -> None:
    """Set whether to show the setup wizard."""
    ida_settings.set_current_plugin_setting("show_wizard", value)


def _get_setting_str(key: str) -> str | None:
    try:
        if ida_settings.has_current_plugin_setting(key):
            value = ida_settings.get_current_plugin_setting(key)
            if value is None:
                return None
            text = str(value).strip()
            return text if text else None
    except Exception:
        return None
    return None


def _set_setting_str(key: str, value: str | None) -> None:
    try:
        if value is not None and value != "":
            ida_settings.set_current_plugin_setting(key, value)
        elif ida_settings.has_current_plugin_setting(key):
            ida_settings.del_current_plugin_setting(key)
    except Exception:
        return


def _get_legacy_auth_type() -> str | None:
    value = _get_setting_str("auth_type")
    return value.lower() if value else None


def _get_legacy_api_key() -> str | None:
    return _get_setting_str("api_key")


def _load_provider_profiles() -> dict[str, dict[str, str]]:
    """Load provider profiles persisted as JSON in plugin settings."""
    raw = _get_setting_str(PROVIDER_PROFILES_KEY)
    if not raw:
        return {}

    try:
        decoded = json.loads(raw)
    except Exception:
        return {}

    if not isinstance(decoded, dict):
        return {}

    profiles: dict[str, dict[str, str]] = {}
    for provider_name, profile_data in decoded.items():
        provider = normalize_provider(str(provider_name))
        if not isinstance(profile_data, dict):
            continue

        profile: dict[str, str] = {}
        for key in ("auth_mode", "api_key", "model", "base_url"):
            value = profile_data.get(key)
            if isinstance(value, str) and value.strip():
                profile[key] = value.strip()
        if profile:
            profiles[provider] = profile

    return profiles


def _save_provider_profiles(profiles: dict[str, dict[str, str]]) -> None:
    _set_setting_str(
        PROVIDER_PROFILES_KEY,
        json.dumps(profiles, ensure_ascii=True),
    )


def get_provider_config(provider: str | None = None) -> ProviderConfig:
    """Load provider configuration from profile storage with migration support."""
    active_provider = normalize_provider(provider or _get_setting_str("provider"))
    profiles = _load_provider_profiles()

    # Backward compatibility for existing installations using flat settings.
    if provider is None and active_provider not in profiles:
        flat_auth_mode = (_get_setting_str("auth_mode") or "").lower()
        flat_api_key = _get_setting_str("api_key")
        flat_model = _get_setting_str("model")
        flat_base_url = _get_setting_str("base_url")

        legacy_auth_type = _get_legacy_auth_type()
        if not flat_auth_mode and legacy_auth_type:
            flat_auth_mode = "system" if legacy_auth_type == "system" else "api_key"
        if not flat_api_key:
            flat_api_key = _get_legacy_api_key()

        migrated_profile: dict[str, str] = {}
        if flat_auth_mode in {"system", "api_key"}:
            migrated_profile["auth_mode"] = flat_auth_mode
        if flat_api_key:
            migrated_profile["api_key"] = flat_api_key
        if flat_model:
            migrated_profile["model"] = flat_model
        if flat_base_url:
            migrated_profile["base_url"] = flat_base_url

        if migrated_profile:
            profiles[active_provider] = migrated_profile
            _save_provider_profiles(profiles)

    profile = profiles.get(active_provider, {})
    auth_mode = str(profile.get("auth_mode", "")).lower()
    if auth_mode not in {"system", "api_key"}:
        auth_mode = "api_key"
    if active_provider != "claude" and auth_mode == "system":
        auth_mode = "api_key"

    return ProviderConfig(
        provider=active_provider,
        auth_mode=auth_mode,
        api_key=profile.get("api_key"),
        model=profile.get("model"),
        base_url=profile.get("base_url"),
    )


def _is_provider_config_complete(config: ProviderConfig) -> bool:
    provider = normalize_provider(config.provider)
    api_key_present = bool((config.api_key or "").strip())

    if provider == "claude":
        return config.auth_mode == "system" or api_key_present

    if provider == "ollama":
        # Local Ollama can be keyless.
        return True

    return api_key_present


def has_configured_provider() -> bool:
    """True when at least one provider has a usable saved configuration."""
    # Trigger one-time migration from legacy flat settings when available.
    get_provider_config()
    return any(_is_provider_config_complete(get_provider_config(provider)) for provider in SUPPORTED_PROVIDERS)


def get_configured_providers() -> list[str]:
    """Return providers that have saved profiles, preserving known order."""
    configured: list[str] = []
    for provider in SUPPORTED_PROVIDERS:
        if _is_provider_config_complete(get_provider_config(provider)):
            configured.append(provider)
    return configured


def save_provider_settings(config: ProviderConfig) -> None:
    """Persist provider settings and disable onboarding wizard."""
    provider = normalize_provider(config.provider)
    profiles = _load_provider_profiles()

    profile: dict[str, str] = {
        "auth_mode": config.auth_mode,
    }
    if config.api_key:
        profile["api_key"] = config.api_key.strip()
    if config.model:
        profile["model"] = config.model.strip()
    if config.base_url:
        profile["base_url"] = config.base_url.strip()

    profiles[provider] = profile
    _save_provider_profiles(profiles)

    _set_setting_str("provider", provider)

    # Keep flat keys populated for backward compatibility with older versions.
    _set_setting_str("auth_mode", config.auth_mode)
    _set_setting_str("api_key", (config.api_key or "").strip() or None)
    _set_setting_str("model", (config.model or "").strip() or None)
    _set_setting_str("base_url", (config.base_url or "").strip() or None)

    # Keep legacy auth_type populated for backward compatibility.
    legacy_auth = "system" if provider == "claude" and config.auth_mode == "system" else "api_key"
    _set_setting_str("auth_type", legacy_auth)

    set_show_wizard(False)


def apply_auth_to_environment() -> ProviderConfig:
    """Apply the currently configured provider environment variables."""
    config = get_provider_config()
    apply_provider_environment(config)
    return config


class CollapsibleSection(QFrame):
    """Expandable/collapsible section for long content."""

    # Threshold for collapsing (lines)
    COLLAPSE_THRESHOLD = 10

    def __init__(self, title: str, content: str, collapsed: bool = True, parent=None):
        super().__init__(parent)
        self._collapsed = collapsed
        self._title = title
        self._content = content
        self._setup_ui()

    def _setup_ui(self):
        colors = get_ida_colors()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header with toggle button
        self.header = QPushButton()
        self._update_header_text()
        self.header.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {colors['mid']};
                border: none;
                text-align: left;
                padding: 2px 4px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                color: {colors['text']};
            }}
        """)
        self.header.clicked.connect(self._toggle)
        layout.addWidget(self.header)

        # Content area
        self.content_label = QLabel()
        self.content_label.setTextFormat(Qt.RichText)
        self.content_label.setWordWrap(True)
        self.content_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.content_label.setStyleSheet(f"""
            QLabel {{
                background-color: {colors['alt_base']};
                color: {colors['text']};
                padding: 8px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        self._update_content()
        layout.addWidget(self.content_label)

    def _update_header_text(self):
        arrow = "▶" if self._collapsed else "▼"
        line_count = len(self._content.strip().split('\n'))
        self.header.setText(f"{arrow} {self._title} ({line_count} lines)")

    def _update_content(self):
        if self._collapsed:
            # Show first few lines with ellipsis
            lines = self._content.strip().split('\n')
            preview = '\n'.join(lines[:3])
            if len(lines) > 3:
                preview += f"\n... ({len(lines) - 3} more lines)"
            self.content_label.setText(f"<pre>{preview}</pre>")
        else:
            self.content_label.setText(f"<pre>{self._content}</pre>")

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._update_header_text()
        self._update_content()

    @staticmethod
    def should_collapse(content: str) -> bool:
        """Check if content should be collapsed."""
        return len(content.strip().split('\n')) > CollapsibleSection.COLLAPSE_THRESHOLD


def markdown_to_html(text: str) -> str:
    """Convert markdown to HTML for display in QLabel with rich text."""
    import html

    # Get theme-aware colors
    colors = get_ida_colors()
    code_bg = colors['dark']
    code_fg = colors['text']

    # Escape HTML first
    text = html.escape(text)

    # Code blocks (``` ... ```) - must be before inline code
    def replace_code_block(match):
        code = match.group(1)
        return f'<pre style="background-color: {code_bg}; color: {code_fg}; padding: 8px; border-radius: 4px; overflow-x: auto;"><code>{code}</code></pre>'
    text = re.sub(r'```(?:\w*\n)?(.*?)```', replace_code_block, text, flags=re.DOTALL)

    # Inline code (`code`)
    text = re.sub(r'`([^`]+)`', rf'<code style="background-color: {code_bg}; color: {code_fg}; padding: 2px 4px; border-radius: 3px;">\1</code>', text)

    # Headers
    text = re.sub(r'^### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)

    # Bold (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Italic (*text* or _text_) - careful not to match inside words
    text = re.sub(r'(?<!\w)\*([^*]+)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'<i>\1</i>', text)

    # Links [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    # Bullet lists (- item or * item)
    text = re.sub(r'^[\-\*] (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    # Wrap consecutive <li> in <ul>
    text = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', text)

    # Numbered lists (1. item)
    text = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)

    # Line breaks - convert newlines to <br> (but not inside pre/code blocks)
    # Simple approach: just convert remaining newlines
    text = text.replace('\n', '<br>')

    # Clean up multiple <br> tags
    text = re.sub(r'(<br>){3,}', '<br><br>', text)

    return text


class MessageType:
    """Message type constants for visual differentiation."""
    TEXT = "text"           # Normal assistant text
    TOOL_USE = "tool_use"   # Tool invocation (muted, italic)
    SCRIPT = "script"       # Script code (monospace, dark bg)
    OUTPUT = "output"       # Script output (monospace, gray bg)
    ERROR = "error"         # Error message (red accent)
    USER = "user"           # User message


class ProgressTimeline(QFrame):
    """Compact progress timeline showing agent stages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._script_count = 0
        self._current_stage = ""
        self._is_complete = False
        self._setup_ui()

    def _setup_ui(self):
        colors = get_ida_colors()
        self.setStyleSheet(f"background-color: {colors['window']};")

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 4, 10, 4)
        self.layout.setSpacing(4)

        self.timeline_label = QLabel("")
        self.timeline_label.setStyleSheet(f"color: {colors['mid']}; font-size: 10px;")
        self.layout.addWidget(self.timeline_label)
        self.layout.addStretch()

        self.setVisible(False)

    def reset(self):
        """Reset the timeline for a new conversation."""
        self._script_count = 0
        self._current_stage = "User"
        self._is_complete = False
        self._update_display()
        self.setVisible(True)

    def add_stage(self, name: str):
        """Add a new stage to the timeline."""
        # Track scripts by parsing the number from "Script N"
        if name.startswith("Script "):
            try:
                self._script_count = int(name.split()[1])
            except (IndexError, ValueError):
                pass
        self._current_stage = name
        self._update_display()

    def complete(self):
        """Mark the timeline as complete."""
        self._is_complete = True
        self._current_stage = "Done"
        self._update_display()

    def hide_timeline(self):
        """Hide the timeline."""
        self.setVisible(False)

    def _update_display(self):
        """Update the timeline display with compact summary."""
        parts = []

        # Always show User as complete
        parts.append("<span style='color: #22c55e;'>✓ User</span>")

        # Show script count if any
        if self._script_count > 0:
            if self._is_complete:
                parts.append(f"<span style='color: #22c55e;'>✓ {self._script_count} scripts</span>")
            else:
                parts.append(f"<b style='color: #f59e0b;'>{self._script_count} scripts</b>")

        # Show current stage (Thinking, Retrying, Done)
        if self._is_complete:
            parts.append("<span style='color: #22c55e;'>✓ Done</span>")
        elif self._current_stage and self._current_stage not in ("User",) and not self._current_stage.startswith("Script"):
            parts.append(f"<b style='color: #f59e0b;'>{self._current_stage}</b>")

        self.timeline_label.setText(" → ".join(parts))


class ChatMessage(QFrame):
    """A single chat message bubble with optional status indicator."""

    def __init__(self, text: str, is_user: bool = True, is_processing: bool = False,
                 msg_type: str = MessageType.TEXT, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self._is_processing = is_processing
        self._msg_type = msg_type if not is_user else MessageType.USER
        self._blink_visible = True
        self._blink_timer = None
        self._status_indicator = None
        self._setup_ui(text)

    def _setup_ui(self, text: str):
        """Set up the message bubble UI."""
        colors = get_ida_colors()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        if self.is_user:
            # User message - right aligned, accent color background, plain QLabel
            self.message_widget = QLabel(text)
            self.message_widget.setWordWrap(True)
            self.message_widget.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            self.message_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            layout.addStretch()
            self.message_widget.setStyleSheet(f"""
                QLabel {{
                    background-color: {colors['highlight']};
                    color: {colors['highlight_text']};
                    border-radius: 10px;
                    padding: 8px 12px;
                }}
            """)
            layout.addWidget(self.message_widget)
        else:
            # Status indicator for assistant messages (small dot)
            self._status_indicator = QLabel("●")
            self._status_indicator.setFixedWidth(16)
            self._status_indicator.setAlignment(Qt.AlignCenter | Qt.AlignTop)
            self._update_indicator_style()
            layout.addWidget(self._status_indicator)

            # Assistant message - QLabel with rich text for markdown
            self.message_widget = QLabel()
            self.message_widget.setTextFormat(Qt.RichText)
            self.message_widget.setWordWrap(True)
            self.message_widget.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard | Qt.LinksAccessibleByMouse
            )
            self.message_widget.setOpenExternalLinks(True)
            self.message_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

            # Apply type-specific styling
            if self._msg_type == MessageType.TOOL_USE:
                # Tool use - muted, italic
                self.message_widget.setText(f"<i>{text}</i>")
                self.message_widget.setStyleSheet(f"""
                    QLabel {{
                        background-color: transparent;
                        color: {colors['mid']};
                        padding: 4px 8px;
                        font-size: 11px;
                    }}
                """)
            elif self._msg_type == MessageType.SCRIPT:
                # Script code - monospace, dark background
                self.message_widget.setText(f"<pre style='margin: 0; white-space: pre-wrap; word-wrap: break-word;'>{text}</pre>")
                self.message_widget.setStyleSheet(f"""
                    QLabel {{
                        background-color: #1e1e1e;
                        color: #d4d4d4;
                        border-radius: 6px;
                        padding: 8px 12px;
                        font-family: monospace;
                        font-size: 11px;
                    }}
                """)
            elif self._msg_type == MessageType.OUTPUT:
                # Script output - monospace, gray background
                self.message_widget.setText(f"<pre style='margin: 0; white-space: pre-wrap; word-wrap: break-word;'>{text}</pre>")
                self.message_widget.setStyleSheet(f"""
                    QLabel {{
                        background-color: #2d2d2d;
                        color: #a0a0a0;
                        border-radius: 6px;
                        padding: 8px 12px;
                        font-family: monospace;
                        font-size: 11px;
                    }}
                """)
            elif self._msg_type == MessageType.ERROR:
                # Error - red accent
                self.message_widget.setText(markdown_to_html(text))
                self.message_widget.setStyleSheet(f"""
                    QLabel {{
                        background-color: #2d1f1f;
                        color: #f87171;
                        border: 1px solid #dc2626;
                        border-radius: 10px;
                        padding: 8px 12px;
                    }}
                """)
            else:
                # Default text styling
                self.message_widget.setText(markdown_to_html(text))
                self.message_widget.setStyleSheet(f"""
                    QLabel {{
                        background-color: {colors['alt_base']};
                        color: {colors['text']};
                        border-radius: 10px;
                        padding: 8px 12px;
                    }}
                """)

            layout.addWidget(self.message_widget, stretch=4)
            layout.addStretch(1)  # 4:1 ratio = ~80% for message

            # Start blinking if processing
            if self._is_processing:
                self._start_blinking()

    def _update_indicator_style(self):
        """Update the status indicator color."""
        if not self._status_indicator:
            return
        if self._is_processing:
            # Yellow/orange for processing, blink visibility
            color = "#f59e0b" if self._blink_visible else "transparent"
        else:
            # Green for complete
            color = "#22c55e"
        self._status_indicator.setStyleSheet(f"QLabel {{ color: {color}; font-size: 10px; }}")

    def _start_blinking(self):
        """Start the blinking animation."""
        if self._blink_timer:
            return
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)
        self._blink_timer.start(500)  # Blink every 500ms

    def _stop_blinking(self):
        """Stop the blinking animation."""
        if self._blink_timer:
            self._blink_timer.stop()
            self._blink_timer = None
        self._blink_visible = True

    def _toggle_blink(self):
        """Toggle blink visibility."""
        self._blink_visible = not self._blink_visible
        self._update_indicator_style()

    def set_complete(self):
        """Mark this message as complete (green indicator)."""
        self._is_processing = False
        self._stop_blinking()
        self._update_indicator_style()

    def update_text(self, text: str):
        """Update the message text."""
        if self.is_user:
            self.message_widget.setText(text)
        else:
            self.message_widget.setText(markdown_to_html(text))


class ChatHistoryWidget(QScrollArea):
    """Scrollable chat history container."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_processing_message: ChatMessage | None = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the chat history UI."""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)

        # Container widget for messages
        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.addStretch(1)  # Stretch at top pushes messages to bottom

        self.setWidget(self.container)

    def add_message(self, text: str, is_user: bool = True, is_processing: bool = False,
                    msg_type: str = MessageType.TEXT) -> ChatMessage:
        """Add a message to the chat history."""
        message = ChatMessage(text, is_user, is_processing, msg_type)
        self.layout.addWidget(message)

        # Track processing message
        if is_processing:
            self._current_processing_message = message

        self.scroll_to_bottom()
        return message

    def mark_current_complete(self):
        """Mark the current processing message as complete."""
        if self._current_processing_message:
            self._current_processing_message.set_complete()
            self._current_processing_message = None

    def scroll_to_bottom(self):
        """Scroll the chat history to the bottom."""
        QTimer.singleShot(10, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        ))

    def add_collapsible(self, title: str, content: str, collapsed: bool = True) -> CollapsibleSection:
        """Add a collapsible section to the chat history."""
        section = CollapsibleSection(title, content, collapsed)
        self.layout.addWidget(section)
        self.scroll_to_bottom()
        return section

    def clear_history(self):
        """Clear all messages from the chat history."""
        self._current_processing_message = None
        # Remove all widgets except the stretch at index 0
        while self.layout.count() > 1:
            item = self.layout.takeAt(1)  # Always take from index 1, leaving stretch at 0
            if item.widget():
                item.widget().deleteLater()


class ChatInputWidget(QPlainTextEdit):
    """Multi-line text input with Enter to send and history navigation."""

    message_submitted = Signal(str)
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[str] = []
        self._history_index = -1  # -1 means not browsing history
        self._current_input = ""  # Stores current input when browsing history
        self._setup_ui()

    def set_history(self, messages: list[str]):
        """Set the message history for up/down navigation.

        Args:
            messages: List of previous user messages (oldest first).
        """
        self._history = messages
        self._history_index = -1

    def add_to_history(self, message: str):
        """Add a message to the history.

        Args:
            message: The message to add.
        """
        # Don't add duplicates of the last message
        if not self._history or self._history[-1] != message:
            self._history.append(message)
        self._history_index = -1

    def _setup_ui(self):
        """Set up the input widget UI."""
        colors = get_ida_colors()

        self.setPlaceholderText("Type a message... (↑↓ history, Enter send, Esc cancel)")
        self.setMaximumHeight(100)
        self.setMinimumHeight(40)
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {colors['base']};
                color: {colors['text']};
                border: 1px solid {colors['mid']};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QPlainTextEdit:focus {{
                border: 1px solid {colors['highlight']};
            }}
        """)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle special keys: Enter, Escape, Up/Down for history."""
        if event.key() == Qt.Key_Escape:
            # Escape: cancel current operation
            self.cancel_requested.emit()
        elif event.key() == Qt.Key_Up:
            # Up arrow: navigate to older history
            self._navigate_history(-1)
        elif event.key() == Qt.Key_Down:
            # Down arrow: navigate to newer history
            self._navigate_history(1)
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if event.modifiers() & Qt.ShiftModifier:
                # Shift+Enter: insert new line
                super().keyPressEvent(event)
            else:
                # Enter: submit message
                text = self.toPlainText().strip()
                if text:
                    self.add_to_history(text)
                    self.message_submitted.emit(text)
                    self.clear()
                    self._history_index = -1
                    self.setFocus()  # Keep focus on input
        else:
            super().keyPressEvent(event)

    def _navigate_history(self, direction: int):
        """Navigate through message history.

        Args:
            direction: -1 for older (up), +1 for newer (down)
        """
        if not self._history:
            return

        # Save current input when starting to browse
        if self._history_index == -1:
            self._current_input = self.toPlainText()

        # Calculate new index
        if direction < 0:  # Up - go to older
            if self._history_index == -1:
                # Start browsing from the end (most recent)
                new_index = len(self._history) - 1
            else:
                new_index = max(0, self._history_index - 1)
        else:  # Down - go to newer
            if self._history_index == -1:
                # Already at current input, do nothing
                return
            new_index = self._history_index + 1
            if new_index >= len(self._history):
                # Return to current input
                self._history_index = -1
                self.setPlainText(self._current_input)
                # Move cursor to end
                cursor = self.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self.setTextCursor(cursor)
                return

        # Set the history item
        self._history_index = new_index
        self.setPlainText(self._history[self._history_index])
        # Move cursor to end
        cursor = self.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.setTextCursor(cursor)


class PluginCallback(ChatCallback):
    """Qt widget output implementation of ChatCallback.

    Uses Qt signals to safely update UI from any thread.
    """

    def __init__(self, signals: "AgentSignals"):
        self.signals = signals

    def on_turn_start(self, turn: int, max_turns: int) -> None:
        self.signals.turn_start.emit(turn, max_turns)

    def on_thinking(self) -> None:
        self.signals.thinking.emit()

    def on_thinking_done(self) -> None:
        self.signals.thinking_done.emit()

    def on_tool_use(self, tool_name: str, details: str) -> None:
        self.signals.tool_use.emit(tool_name, details)

    def on_text(self, text: str) -> None:
        self.signals.text.emit(text)

    def on_script_code(self, code: str) -> None:
        self.signals.script_code.emit(code)

    def on_script_output(self, output: str) -> None:
        self.signals.script_output.emit(output)

    def on_error(self, error: str) -> None:
        self.signals.error.emit(error)

    def on_result(self, num_turns: int, cost: float | None) -> None:
        self.signals.result.emit(num_turns, cost or 0.0)


class AgentSignals(QObject):
    """Qt signals for agent callbacks."""

    turn_start = Signal(int, int)
    thinking = Signal()
    thinking_done = Signal()
    tool_use = Signal(str, str)
    text = Signal(str)
    script_code = Signal(str)
    script_output = Signal(str)
    error = Signal(str)
    result = Signal(int, float)
    finished = Signal()
    connection_ready = Signal()
    connection_error = Signal(str)


class AgentWorker(QThread):
    """Background worker for running async agent calls."""

    def __init__(self, db: Database, script_executor: Callable[[str], str],
                 history: MessageHistory, provider_config: ProviderConfig, parent=None):
        super().__init__(parent)
        self.db = db
        self.script_executor = script_executor
        self.history = history
        self.provider_config = provider_config
        self.signals = AgentSignals()
        self.callback = PluginCallback(self.signals)
        self.core: IDAChatCore | None = None
        self._pending_message: str | None = None
        self._should_connect = False
        self._should_disconnect = False
        self._should_cancel = False
        self._should_new_session = False
        self._running = True

    def request_connect(self):
        """Request connection to agent."""
        self._should_connect = True
        if not self.isRunning():
            self.start()

    def request_disconnect(self):
        """Request disconnection from agent."""
        self._should_disconnect = True
        self._running = False

    def request_cancel(self):
        """Request cancellation of current operation."""
        self._should_cancel = True
        if self.core:
            self.core.request_cancel()

    def request_new_session(self):
        """Request starting a new session for history tracking."""
        self._should_new_session = True

    def send_message(self, message: str):
        """Queue a message to be sent to the agent."""
        self._pending_message = message
        if not self.isRunning():
            self.start()

    def run(self):
        """Run the async event loop in this thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._async_run())
        finally:
            loop.close()

    async def _async_run(self):
        """Main async loop."""
        # Handle connection request
        if self._should_connect:
            self._should_connect = False
            try:
                # Start initial session for history
                self.history.start_new_session()

                self.core = IDAChatCore(
                    self.db,
                    self.callback,
                    script_executor=self.script_executor,
                    provider_config=self.provider_config,
                    history=self.history,
                )
                await self.core.connect()
                self.signals.connection_ready.emit()
            except Exception as e:
                self.signals.connection_error.emit(str(e))
                return

        # Process messages while running
        while self._running:
            # Handle new session request (e.g., after Clear)
            if self._should_new_session:
                self._should_new_session = False
                self.history.start_new_session()

            if self._pending_message:
                message = self._pending_message
                self._pending_message = None
                try:
                    await self.core.process_message(message)
                except Exception as e:
                    self.signals.error.emit(str(e))
                self.signals.finished.emit()

            # Check for disconnect request
            if self._should_disconnect:
                break

            # Small sleep to avoid busy loop
            await asyncio.sleep(0.1)

        # Handle disconnection
        if self.core:
            await self.core.disconnect()


class TestConnectionWorker(QThread):
    """Background thread for testing provider connectivity."""

    finished = Signal(bool, str)  # (success, message)

    def __init__(self, provider_config: ProviderConfig, parent=None):
        super().__init__(parent)
        self.provider_config = provider_config

    def run(self):
        """Run the connection test."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            success, message = loop.run_until_complete(test_provider_connection(self.provider_config))
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, str(e))
        finally:
            loop.close()


class OnboardingPanel(QFrame):
    """Onboarding panel for first-time setup and settings configuration."""

    onboarding_complete = Signal()  # Emitted when user clicks Save & Start

    def __init__(self, parent=None):
        super().__init__(parent)
        self._test_worker: TestConnectionWorker | None = None
        self._setup_ui()

    def _setup_ui(self):
        colors = get_ida_colors()

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {colors['base']};
                border-radius: 8px;
            }}
        """)

        # Main horizontal layout for two columns
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left column (30%) - Image
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)

        image_label = QLabel()
        splash_path = Path(__file__).parent / "splash.png"
        if splash_path.exists():
            pixmap = QPixmap(str(splash_path))
            image_label.setPixmap(pixmap.scaled(
                300, 400,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(image_label)
        image_layout.addStretch()

        main_layout.addWidget(image_container, stretch=30)

        # Right column (70%) - Settings
        settings_container = QWidget()
        layout = QVBoxLayout(settings_container)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Title
        title = QLabel("Welcome to IDA Chat")
        title.setStyleSheet(f"""
            QLabel {{
                color: {colors['text']};
                font-size: 18px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(title)

        # Instructions
        instructions = QLabel("Configure your model provider and credentials:")
        instructions.setStyleSheet(f"QLabel {{ color: {colors['mid']}; }}")
        layout.addWidget(instructions)

        # Provider selection
        provider_title_label = QLabel("Provider")
        provider_title_label.setStyleSheet(f"QLabel {{ color: {colors['text']}; font-weight: bold; }}")
        layout.addWidget(provider_title_label)

        self.provider_combo = QComboBox()
        self.provider_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {colors['alt_base']};
                color: {colors['text']};
                border: 1px solid {colors['mid']};
                border-radius: 4px;
                padding: 6px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
        """)
        for provider in SUPPORTED_PROVIDERS:
            self.provider_combo.addItem(provider_label(provider), provider)
        layout.addWidget(self.provider_combo)

        self.provider_note = QLabel("")
        self.provider_note.setWordWrap(True)
        self.provider_note.setStyleSheet(f"QLabel {{ color: {colors['mid']}; font-size: 11px; }}")
        layout.addWidget(self.provider_note)

        # Radio buttons for auth type
        self.auth_group = QButtonGroup(self)

        # Option 1: System (Claude only)
        self.radio_system = QRadioButton("Use system credentials (Claude Code)")
        self.radio_system.setStyleSheet(f"QRadioButton {{ color: {colors['text']}; }}")
        self.radio_system.setChecked(True)
        self.auth_group.addButton(self.radio_system, 0)
        layout.addWidget(self.radio_system)

        system_hint = QLabel("    Recommended for Claude when configured on this machine")
        system_hint.setStyleSheet(f"QLabel {{ color: {colors['mid']}; font-size: 11px; }}")
        layout.addWidget(system_hint)

        # Option 2: API key/token
        self.radio_api_key = QRadioButton("Use API key / token")
        self.radio_api_key.setStyleSheet(f"QRadioButton {{ color: {colors['text']}; }}")
        self.auth_group.addButton(self.radio_api_key, 1)
        layout.addWidget(self.radio_api_key)

        # Key input field (hidden for Claude+system)
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Paste your key here...")
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {colors['alt_base']};
                color: {colors['text']};
                border: 1px solid {colors['mid']};
                border-radius: 4px;
                padding: 8px;
            }}
            QLineEdit:focus {{
                border-color: {colors['highlight']};
            }}
        """)
        self.key_input.hide()
        layout.addWidget(self.key_input)

        # Model picker (editable, Copilot-style)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setStyleSheet(self.key_input.styleSheet())
        layout.addWidget(self.model_combo)

        # Optional base URL override
        self.base_url_input = QLineEdit()
        self.base_url_input.setPlaceholderText("Optional base URL override")
        self.base_url_input.setStyleSheet(self.key_input.styleSheet())
        layout.addWidget(self.base_url_input)

        # Connect controls
        self.auth_group.buttonClicked.connect(self._on_auth_type_changed)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.key_input.textChanged.connect(lambda _text: self._update_model_edit_state())

        # Buttons row
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)

        self.test_btn = QPushButton("Test Connection")
        self.test_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['button']};
                color: {colors['button_text']};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}
            QPushButton:disabled {{
                background-color: {colors['mid']};
            }}
        """)
        self.test_btn.clicked.connect(self._on_test_clicked)
        buttons_layout.addWidget(self.test_btn)

        self.save_btn = QPushButton("Save && Start")
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {colors['button']};
                color: {colors['button_text']};
            }}
            QPushButton:disabled {{
                background-color: {colors['mid']};
            }}
        """)
        self.save_btn.clicked.connect(self._on_save_clicked)
        buttons_layout.addWidget(self.save_btn)

        layout.addLayout(buttons_layout)

        # Status label
        self.status_label = QLabel("Not configured")
        self.status_label.setStyleSheet(f"QLabel {{ color: {colors['mid']}; }}")
        layout.addWidget(self.status_label)

        # Response area (for showing joke on successful test)
        self.response_label = QLabel()
        self.response_label.setWordWrap(True)
        self.response_label.setStyleSheet(f"""
            QLabel {{
                color: {colors['text']};
                background-color: {colors['alt_base']};
                border-radius: 4px;
                padding: 12px;
            }}
        """)
        self.response_label.hide()
        layout.addWidget(self.response_label)

        layout.addStretch()

        main_layout.addWidget(settings_container, stretch=70)
        self._refresh_provider_fields()

    def _get_selected_provider(self) -> str:
        value = self.provider_combo.currentData()
        if isinstance(value, str) and value:
            return normalize_provider(value)

        fallback = self.provider_combo.itemData(0)
        if isinstance(fallback, str) and fallback:
            return normalize_provider(fallback)

        return normalize_provider(SUPPORTED_PROVIDERS[0])

    def _get_auth_mode(self) -> str:
        provider = self._get_selected_provider()
        if provider == "claude" and self.radio_system.isChecked():
            return "system"
        return "api_key"

    def _refresh_provider_fields(self):
        """Update placeholders and auth controls for selected provider."""
        provider = self._get_selected_provider()

        self.provider_note.setText(provider_free_tier_note(provider))

        model_hint = provider_default_model(provider)
        current_model = self.model_combo.currentText().strip()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("")
        for model_name in provider_recommended_models(provider):
            self.model_combo.addItem(model_name)
        if model_hint and model_hint not in {self.model_combo.itemText(i) for i in range(self.model_combo.count())}:
            self.model_combo.addItem(model_hint)
        if current_model:
            self.model_combo.setCurrentText(current_model)
        elif model_hint:
            self.model_combo.setCurrentText(model_hint)
        self.model_combo.setEditable(True)
        self.model_combo.blockSignals(False)

        base_hint = provider_default_base_url(provider)
        if base_hint:
            self.base_url_input.setPlaceholderText(f"Optional base URL override (default: {base_hint})")
        else:
            self.base_url_input.setPlaceholderText("Optional base URL override")

        self.key_input.setPlaceholderText(provider_key_hint(provider))

        if provider == "claude":
            self.radio_system.setEnabled(True)
        else:
            self.radio_system.setEnabled(False)
            self.radio_api_key.setChecked(True)

        if provider == "claude" and self.radio_system.isChecked():
            self.key_input.hide()
        else:
            self.key_input.show()

        self._update_model_edit_state()

    def _on_provider_changed(self, _index: int):
        provider = self._get_selected_provider()
        cfg = get_provider_config(provider)

        if provider == "claude" and cfg.auth_mode == "system":
            self.radio_system.setChecked(True)
        else:
            self.radio_api_key.setChecked(True)

        self.key_input.setText(cfg.api_key or "")
        self.base_url_input.setText(cfg.base_url or "")
        self._refresh_provider_fields()
        self.model_combo.setCurrentText(cfg.model or resolve_model(cfg) or "")

    def _on_auth_type_changed(self, _button):
        self._refresh_provider_fields()

    def _update_model_edit_state(self):
        """Enable model picker only when selected provider auth is configured."""
        cfg = self._get_provider_config_from_ui()
        model_enabled = not requires_api_key(cfg) or bool((cfg.api_key or "").strip())
        self.model_combo.setEnabled(model_enabled)

        if model_enabled:
            self.model_combo.setToolTip("")
        else:
            self.model_combo.setToolTip("Configure provider credentials first to enable model selection")

    def _get_provider_config_from_ui(self) -> ProviderConfig:
        auth_mode = self._get_auth_mode()
        api_key = self.key_input.text().strip() or None
        if auth_mode == "system":
            api_key = None

        return ProviderConfig(
            provider=self._get_selected_provider(),
            auth_mode=auth_mode,
            api_key=api_key,
            model=self.model_combo.currentText().strip() or None,
            base_url=self.base_url_input.text().strip() or None,
        )

    def _on_test_clicked(self):
        """Run connection test."""
        config = self._get_provider_config_from_ui()
        ok, error = validate_provider_config(config)
        if not ok:
            self.status_label.setText(error)
            self.status_label.setStyleSheet("QLabel { color: #F44336; }")
            self.response_label.hide()
            return

        self.test_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("Testing connection...")
        self.response_label.hide()

        # Apply settings to environment before testing
        self._apply_current_settings(config)

        # Start test worker
        self._test_worker = TestConnectionWorker(config, self)
        self._test_worker.finished.connect(self._on_test_finished)
        self._test_worker.start()

    def _on_test_finished(self, success: bool, message: str):
        """Handle test result."""
        self.test_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        if success:
            self.status_label.setText("✓ Connected! You're all set.")
            self.status_label.setStyleSheet(f"QLabel {{ color: #4CAF50; }}")  # Green
            self.response_label.setText(message)
            self.response_label.show()
        else:
            self.status_label.setText(f"✗ Connection failed: {message}")
            self.status_label.setStyleSheet(f"QLabel {{ color: #F44336; }}")  # Red
            self.response_label.hide()

    def _apply_current_settings(self, config: ProviderConfig | None = None):
        """Apply current UI settings to environment variables."""
        apply_provider_environment(config or self._get_provider_config_from_ui())

    def _on_save_clicked(self):
        """Save settings and emit completion signal."""
        config = self._get_provider_config_from_ui()
        ok, error = validate_provider_config(config)
        if not ok:
            self.status_label.setText(error)
            self.status_label.setStyleSheet(f"QLabel {{ color: #F44336; }}")
            return

        # Save settings
        save_provider_settings(config)

        # Apply to environment
        self._apply_current_settings(config)

        # Emit completion signal
        self.onboarding_complete.emit()

    def load_current_settings(self):
        """Load current settings into the UI (for settings mode)."""
        config = get_provider_config()

        for i in range(self.provider_combo.count()):
            data = self.provider_combo.itemData(i)
            if isinstance(data, str) and normalize_provider(data) == normalize_provider(config.provider):
                self.provider_combo.setCurrentIndex(i)
                break

        if normalize_provider(config.provider) == "claude" and config.auth_mode == "system":
            self.radio_system.setChecked(True)
        else:
            self.radio_api_key.setChecked(True)

        self.key_input.setText(config.api_key or "")
        self.base_url_input.setText(config.base_url or "")

        self._refresh_provider_fields()
        self.model_combo.setCurrentText(config.model or resolve_model(config) or "")
        self._update_model_edit_state()

        # Reset status
        colors = get_ida_colors()
        self.status_label.setText("Settings loaded")
        self.status_label.setStyleSheet(f"QLabel {{ color: {colors['mid']}; }}")
        self.response_label.hide()


class IDAChatForm(ida_kernwin.PluginForm):
    """Main chat widget form."""

    def OnCreate(self, form):
        """Called when the widget is created."""
        self.parent = self.FormToPyQtWidget(form)
        self.worker: AgentWorker | None = None
        self._is_processing = False
        self._current_message = None  # Track current blinking message
        self._current_turn = 0
        self._max_turns = 20
        self._total_cost = 0.0
        self._script_count = 0
        self._last_had_error = False
        self._message_count = 0
        self._provider_config = get_provider_config()
        self._model_name = describe_provider(self._provider_config)

        # Allow horizontal resizing (IDA remembers preferred size)
        self.parent.setMinimumWidth(600)
        self.parent.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._create_ui()

        # Apply saved auth settings to environment
        self._provider_config = apply_auth_to_environment()
        self._model_name = describe_provider(self._provider_config)

        # Require onboarding until at least one provider is fully configured.
        if get_show_wizard() or not has_configured_provider():
            self._show_onboarding()
        else:
            self._init_agent()

    def _create_script_executor(self, db: Database) -> Callable[[str], str]:
        """Create a script executor that runs on the main thread.

        IDA operations must be performed on the main thread. This executor
        uses ida_kernwin.execute_sync() to ensure scripts run safely.
        """
        def execute_on_main_thread(code: str) -> str:
            result = [""]

            def run_script():
                old_stdout = sys.stdout
                sys.stdout = captured = StringIO()
                try:
                    exec(code, {"db": db, "print": print})
                    result[0] = captured.getvalue()
                except Exception as e:
                    result[0] = f"Script error: {e}"
                finally:
                    sys.stdout = old_stdout
                return 1  # Required return for execute_sync

            ida_kernwin.execute_sync(run_script, ida_kernwin.MFF_FAST)
            return result[0]

        return execute_on_main_thread

    def _init_agent(self):
        """Initialize the agent worker."""
        try:
            # Reload provider settings in case they changed in onboarding.
            self._provider_config = apply_auth_to_environment()
            self._model_name = describe_provider(self._provider_config)
            self._refresh_header_provider_controls()

            db = Database.open()
            script_executor = self._create_script_executor(db)

            # Create message history for this binary
            self.history = MessageHistory(db.path)

            if self.worker:
                self.worker.request_disconnect()
                self.worker.wait(5000)
                self.worker = None

            self.worker = AgentWorker(
                db,
                script_executor,
                self.history,
                provider_config=self._provider_config,
            )

            # Connect signals
            self.worker.signals.connection_ready.connect(self._on_connection_ready)
            self.worker.signals.connection_error.connect(self._on_connection_error)
            self.worker.signals.turn_start.connect(self._on_turn_start)
            self.worker.signals.thinking.connect(self._on_thinking)
            self.worker.signals.thinking_done.connect(self._on_thinking_done)
            self.worker.signals.tool_use.connect(self._on_tool_use)
            self.worker.signals.text.connect(self._on_text)
            self.worker.signals.script_code.connect(self._on_script_code)
            self.worker.signals.script_output.connect(self._on_script_output)
            self.worker.signals.error.connect(self._on_error)
            self.worker.signals.result.connect(self._on_result)
            self.worker.signals.finished.connect(self._on_finished)

            # Start connection
            self.worker.request_connect()
        except Exception as e:
            self.chat_history.add_message(f"Error initializing agent: {e}", is_user=False)

    def _refresh_header_model_options(self, provider: str, selected_model: str | None = None):
        """Populate the header model picker for the selected provider."""
        provider = normalize_provider(provider)
        models = provider_recommended_models(provider)
        default_model = provider_default_model(provider)

        current = selected_model or self.model_switch_combo.currentText().strip() or default_model or ""

        self.model_switch_combo.blockSignals(True)
        self.model_switch_combo.clear()
        self.model_switch_combo.addItem("")
        for model in models:
            self.model_switch_combo.addItem(model)
        if default_model and default_model not in {self.model_switch_combo.itemText(i) for i in range(self.model_switch_combo.count())}:
            self.model_switch_combo.addItem(default_model)
        if current:
            self.model_switch_combo.setCurrentText(current)
        self.model_switch_combo.setEditable(True)
        self.model_switch_combo.blockSignals(False)

    def _refresh_header_provider_controls(self):
        """Sync header provider/model controls with persisted settings."""
        active_provider = normalize_provider(self._provider_config.provider)

        self.provider_switch_combo.blockSignals(True)
        self.provider_switch_combo.clear()

        provider_order = get_configured_providers()

        if not provider_order:
            self.provider_switch_combo.addItem("Configure Provider", "")
            self.provider_switch_combo.setEnabled(False)
            self.model_switch_combo.clear()
            self.model_switch_combo.setEnabled(False)
            self.switch_model_btn.setEnabled(False)
            self.provider_switch_combo.blockSignals(False)
            return

        for provider in provider_order:
            self.provider_switch_combo.addItem(provider_label(provider), provider)

        self.provider_switch_combo.setEnabled(True)
        self.model_switch_combo.setEnabled(True)
        self.switch_model_btn.setEnabled(True)

        for i in range(self.provider_switch_combo.count()):
            data = self.provider_switch_combo.itemData(i)
            if isinstance(data, str) and normalize_provider(data) == active_provider:
                self.provider_switch_combo.setCurrentIndex(i)
                break

        selected_value = self.provider_switch_combo.currentData()
        selected_provider = normalize_provider(
            selected_value if isinstance(selected_value, str) and selected_value else provider_order[0]
        )
        selected_config = get_provider_config(selected_provider)

        self.provider_switch_combo.blockSignals(False)
        self._refresh_header_model_options(
            selected_provider,
            selected_config.model or resolve_model(selected_config),
        )

    def _on_header_provider_changed(self, _index: int):
        """Refresh model options when provider selection changes in the header."""
        value = self.provider_switch_combo.currentData()
        if not isinstance(value, str) or not value:
            self.model_switch_combo.clear()
            self.model_switch_combo.setEnabled(False)
            self.switch_model_btn.setEnabled(False)
            return

        self.model_switch_combo.setEnabled(True)
        self.switch_model_btn.setEnabled(True)
        provider = normalize_provider(value)
        cfg = get_provider_config(provider)
        self._refresh_header_model_options(provider, cfg.model or resolve_model(cfg))

    def _on_apply_provider_switch(self):
        """Apply provider/model switch from the header and reconnect the agent."""
        if self._is_processing:
            self.chat_history.add_message("Finish the current request before switching model/provider.", is_user=False)
            return

        provider_value = self.provider_switch_combo.currentData()
        if not isinstance(provider_value, str) or not provider_value:
            self.chat_history.add_message("Configure a provider in Settings first.", is_user=False)
            return

        provider = normalize_provider(provider_value)
        previous = get_provider_config(provider)

        if not _is_provider_config_complete(previous):
            self.chat_history.add_message("Selected provider is not configured. Open Settings to configure it first.", is_user=False)
            return

        switch_config = ProviderConfig(
            provider=provider,
            auth_mode=previous.auth_mode,
            api_key=previous.api_key,
            model=self.model_switch_combo.currentText().strip() or None,
            base_url=previous.base_url,
        )

        ok, error = validate_provider_config(switch_config)
        if not ok:
            self.chat_history.add_message(f"Cannot switch provider: {error}", is_user=False)
            return

        save_provider_settings(switch_config)
        self.chat_history.add_message(f"Switching to {describe_provider(switch_config)}...", is_user=False)
        self._init_agent()

    def _show_onboarding(self):
        """Show onboarding panel, hide chat UI."""
        self.onboarding_panel.show()
        self.chat_history.hide()
        self.input_container.hide()
        self.progress_timeline.hide()

    def _show_settings(self):
        """Show settings panel (re-use onboarding panel)."""
        # Load current settings into the panel
        self.onboarding_panel.load_current_settings()
        self._show_onboarding()

    def _on_onboarding_complete(self):
        """Handle successful onboarding."""
        self.onboarding_panel.hide()
        self.chat_history.show()
        self.input_container.show()
        self._init_agent()

    def _update_status_bar(self, processing_text: str | None = None):
        """Update the status bar with current stats or processing text.

        Args:
            processing_text: If provided, show this instead of idle stats.
        """
        if processing_text:
            self.status_label.setText(processing_text)
        else:
            # Idle state: show model, message count, and cost
            parts = [self._model_name]
            parts.append(f"{self._message_count} msgs")
            if self._total_cost > 0:
                parts.append(f"${self._total_cost:.4f}")
            self.status_label.setText(" · ".join(parts))

    def _on_connection_ready(self):
        """Called when agent connection is established."""
        self.chat_history.add_message(f"Agent connected and ready ({self._model_name}).", is_user=False)
        self.input_widget.setEnabled(True)
        self.input_widget.setFocus()
        self._update_status_bar()

        # Load message history for up/down arrow navigation
        if hasattr(self, 'history'):
            user_messages = self.history.get_all_user_messages()
            self.input_widget.set_history(user_messages)

    def _on_connection_error(self, error: str):
        """Called when agent connection fails."""
        self.chat_history.add_message(f"Connection error: {error}", is_user=False)

    def _log_metric(self, msg: str):
        import time
        ts = time.strftime("%H:%M:%S")
        self.metrics_browser.append(f"[{ts}] {msg}")

    def on_metric(self, text: str):
        """Route generic metric string from core."""
        self.metrics_signal.emit(text)

    def _on_turn_start(self, turn: int, max_turns: int):
        """Called at the start of each agentic turn."""
        self._current_turn = turn
        self._max_turns = max_turns
        self._log_metric(f"Turn started: {turn}/{max_turns}")

    def _on_thinking(self):
        """Called when agent starts processing."""
        self._is_processing = True
        # Mark previous message as complete before starting new turn
        if self._current_message:
            self._current_message.set_complete()
        self.input_widget.setEnabled(False)

        # Check if this is a retry after error
        if self._last_had_error:
            self._last_had_error = False
            # Update timeline
            self.progress_timeline.add_stage("Retrying")
            # Add retry message
            self._current_message = self.chat_history.add_message(
                "🔄 Retrying after error...", is_user=False, is_processing=True
            )
        else:
            # Update timeline
            self.progress_timeline.add_stage("Thinking")
            # Add thinking message with blinking indicator
            self._current_message = self.chat_history.add_message(
                "[Thinking...]", is_user=False, is_processing=True
            )

    def _on_thinking_done(self):
        """Called when agent produces first output."""
        # Remove the thinking message (last widget in layout, stretch is at index 0)
        if self.chat_history.layout.count() > 1:
            item = self.chat_history.layout.takeAt(self.chat_history.layout.count() - 1)
            if item and item.widget():
                item.widget().deleteLater()
        self._current_message = None

    def _add_processing_message(self, text: str, msg_type: str = MessageType.TEXT) -> None:
        """Add a new processing message, marking previous one as complete."""
        # Mark previous message as complete (green)
        if self._current_message:
            self._current_message.set_complete()
        # Add new blinking message
        self._current_message = self.chat_history.add_message(
            text, is_user=False, is_processing=True, msg_type=msg_type
        )

    def _on_tool_use(self, tool_name: str, details: str):
        """Called when agent uses a tool."""
        tool_msg = f"[{tool_name}]"
        if details:
            tool_msg += f" {details}"
        self._add_processing_message(tool_msg, MessageType.TOOL_USE)
        self._log_metric(f"Tool executed: {tool_name} (details: {details[:100]}...)")

    def _on_text(self, text: str):
        """Called when agent outputs text."""
        if text.strip():
            self._add_processing_message(text)

    def _on_script_code(self, code: str):
        """Called with script code before execution."""
        import html
        # Update timeline
        self._script_count += 1
        self.progress_timeline.add_stage(f"Script {self._script_count}")
        self._log_metric(f"Executing script {self._script_count} ({len(code)} bytes)")
        # Show preview of the script
        lines = code.strip().split('\n')
        preview = '\n'.join(lines[:5])
        if len(lines) > 5:
            preview += f"\n... ({len(lines) - 5} more lines)"
        self._add_processing_message(html.escape(preview), MessageType.SCRIPT)

    def _on_script_output(self, output: str):
        """Called with script output."""
        if output.strip():
            import html
            # Check if this is an error output
            is_error = output.strip().startswith("Script error:")
            if is_error:
                self._last_had_error = True
                self._add_processing_message(output, MessageType.ERROR)
            # Use collapsible section for long outputs
            elif CollapsibleSection.should_collapse(output):
                # Mark previous message as complete
                if self._current_message:
                    self._current_message.set_complete()
                self.chat_history.add_collapsible("Script Output", output, collapsed=True)
                self._current_message = None
            else:
                self._add_processing_message(html.escape(output), MessageType.OUTPUT)

    def _on_error(self, error: str):
        """Called when an error occurs."""
        self._add_processing_message(f"Error: {error}", MessageType.ERROR)
        self._log_metric(f"ERROR: {error}")

    def _on_result(self, _num_turns: int, cost: float):
        """Called when agent returns result with stats."""
        self._total_cost += cost
        self._log_metric(f"Result returned. Turn count: {_num_turns}, Request cost: ${cost:.4f}, Total cost: ${self._total_cost:.4f}")

    def _on_finished(self):
        """Called when agent finishes processing."""
        self._is_processing = False
        self._message_count += 1
        self.input_widget.setEnabled(True)
        self.input_widget.setFocus()
        self._update_status_bar()
        self.progress_timeline.complete()
        # Mark the last message as complete (green)
        if self._current_message:
            self._current_message.set_complete()
            self._current_message = None

    def _create_ui(self):
        """Create the chat interface UI."""
        colors = get_ida_colors()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 10, 6)
        header_layout.setSpacing(2)  # Tight spacing for icon buttons

        title = QLabel(PLUGIN_NAME)
        title.setStyleSheet(f"""
            QLabel {{
                color: {colors['window_text']};
                font-weight: bold;
            }}
        """)
        header_layout.addWidget(title)

        combo_style = f"""
            QComboBox {{
                background-color: {colors['alt_base']};
                color: {colors['text']};
                border: 1px solid {colors['mid']};
                border-radius: 4px;
                padding: 2px 6px;
                min-height: 22px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
        """

        self.provider_switch_combo = QComboBox()
        self.provider_switch_combo.setStyleSheet(combo_style)
        self.provider_switch_combo.setMinimumWidth(150)
        self.provider_switch_combo.currentIndexChanged.connect(self._on_header_provider_changed)
        header_layout.addWidget(self.provider_switch_combo)

        self.model_switch_combo = QComboBox()
        self.model_switch_combo.setEditable(True)
        self.model_switch_combo.setStyleSheet(combo_style)
        self.model_switch_combo.setMinimumWidth(220)
        header_layout.addWidget(self.model_switch_combo)

        self.switch_model_btn = QPushButton("Apply")
        self.switch_model_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['button']};
                color: {colors['button_text']};
                border: 1px solid {colors['mid']};
                border-radius: 4px;
                padding: 2px 8px;
                min-height: 22px;
            }}
            QPushButton:hover {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}
        """)
        self.switch_model_btn.clicked.connect(self._on_apply_provider_switch)
        header_layout.addWidget(self.switch_model_btn)

        header_layout.addStretch()

        # Icon button style (shared)
        icon_btn_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {colors['mid']};
                border: none;
                font-size: 14px;
            }}
            QPushButton:hover {{
                color: {colors['window_text']};
            }}
        """

        # Settings button (gear icon)
        settings_btn = QPushButton("⚙")
        settings_btn.setFixedSize(24, 24)
        settings_btn.setToolTip("Settings")
        settings_btn.setStyleSheet(icon_btn_style)
        settings_btn.clicked.connect(self._show_settings)
        header_layout.addWidget(settings_btn)

        # Share/export button
        share_btn = QPushButton("↗")
        share_btn.setFixedSize(24, 24)
        share_btn.setToolTip("Export chat as HTML")
        share_btn.setStyleSheet(icon_btn_style)
        share_btn.clicked.connect(self._on_share)
        header_layout.addWidget(share_btn)

        # Clear button
        clear_btn = QPushButton("✕")
        clear_btn.setFixedSize(24, 24)
        clear_btn.setToolTip("Clear chat")
        clear_btn.setStyleSheet(icon_btn_style)
        clear_btn.clicked.connect(self._on_clear)
        header_layout.addWidget(clear_btn)

        layout.addWidget(header)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"background-color: {colors['mid']};")
        separator.setFixedHeight(1)
        layout.addWidget(separator)

        # Onboarding panel (shown on first launch or when settings clicked)
        self.onboarding_panel = OnboardingPanel()
        self.onboarding_panel.onboarding_complete.connect(self._on_onboarding_complete)
        self.onboarding_panel.hide()  # Hidden by default, shown if not onboarded
        layout.addWidget(self.onboarding_panel)

        # Progress timeline (hidden by default)
        self.progress_timeline = ProgressTimeline()
        layout.addWidget(self.progress_timeline)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                border-top: 1px solid {colors['mid']};
            }}
            QTabBar::tab {{
                background: transparent;
                color: {colors['text']};
                padding: 4px 12px;
                border: none;
            }}
            QTabBar::tab:selected {{
                color: {colors['highlight']};
                border-bottom: 2px solid {colors['highlight']};
            }}
        """)

        # --- Chat Tab ---
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        # Chat history area (takes most space)
        self.chat_history = ChatHistoryWidget()
        chat_layout.addWidget(self.chat_history, stretch=1)

        # Input area at bottom
        self.input_container = QWidget()
        input_layout = QHBoxLayout(self.input_container)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(8)

        # Text input (Enter to send, Escape to cancel)
        self.input_widget = ChatInputWidget()
        self.input_widget.message_submitted.connect(self._on_message_submitted)
        self.input_widget.cancel_requested.connect(self._on_cancel)
        input_layout.addWidget(self.input_widget, stretch=1)

        chat_layout.addWidget(self.input_container)

        # Status bar at bottom
        self.status_bar = QWidget()
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(10, 4, 10, 4)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {colors['mid']}; font-size: 11px;")
        status_layout.addWidget(self.status_label)

        chat_layout.addWidget(self.status_bar)
        
        self.tabs.addTab(chat_tab, "Chat")

        # --- Metrics Tab ---
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        
        self.metrics_browser = QTextBrowser()
        self.metrics_browser.setStyleSheet(f"background-color: {colors['base']}; color: {colors['text']}; font-family: monospace; border: none; padding: 4px;")
        metrics_layout.addWidget(self.metrics_browser)
        
        self.tabs.addTab(metrics_tab, "Metrics & Events")

        layout.addWidget(self.tabs, stretch=1)

        self.parent.setLayout(layout)

        # Initialize provider/model picker controls.
        self._refresh_header_provider_controls()

        # Add welcome message
        self._add_welcome_message()

    def _add_welcome_message(self):
        """Add a welcome message to the chat."""
        welcome_text = f"Welcome to IDA Chat! Preparing {self._model_name}..."
        self.chat_history.add_message(welcome_text, is_user=False)
        # Disable input until agent is connected
        self.input_widget.setEnabled(False)

    def _on_message_submitted(self, text: str):
        """Handle message submission from input widget."""
        self._send_message(text)

    def _send_message(self, text: str):
        """Send a message to the agent."""
        if not self.worker or self._is_processing:
            return

        # Reset timeline for new conversation
        self.progress_timeline.reset()
        self._script_count = 0
        self._last_had_error = False

        # Add user message to chat
        self.chat_history.add_message(text, is_user=True)

        # Send to agent
        self.worker.send_message(text)

    def _on_cancel(self):
        """Cancel the current agent operation."""
        if self.worker and self._is_processing:
            self.worker.request_cancel()

    def _on_share(self):
        """Export the current chat session as HTML using claude-code-transcripts."""
        from pathlib import Path
        from ida_chat_core import export_transcript

        # Check if we have an active session
        if not hasattr(self, 'history') or not self.history:
            self.chat_history.add_message("No active session to export.", is_user=False)
            return

        session_file = self.history.session_file
        if not session_file or not session_file.exists():
            self.chat_history.add_message("No session file found to export.", is_user=False)
            return

        # Get the IDB path and create HTML output path
        idb_path = Path(self.history.binary_path)
        html_path = idb_path.parent / (idb_path.stem + '_chat.html')

        try:
            export_transcript(session_file, html_path)
            # Format as clickable link using file:// URL
            file_url = html_path.resolve().as_uri()
            self.chat_history.add_message(f"Chat exported to: [{html_path}]({file_url})", is_user=False)
        except Exception as e:
            self.chat_history.add_message(f"Export failed: {e}", is_user=False)

    def _on_clear(self):
        """Clear the chat history."""
        self.chat_history.clear_history()
        self._total_cost = 0.0
        self._script_count = 0
        self._message_count = 0
        self.progress_timeline.hide_timeline()

        # Start a new session for history tracking
        if self.worker:
            self.worker.request_new_session()

        # Add ready message (agent already connected)
        self.chat_history.add_message("Chat cleared. Ready for new conversation.", is_user=False)
        self.input_widget.setEnabled(True)
        self.input_widget.setFocus()
        self._update_status_bar()

    def OnClose(self, form):
        """Called when the widget is closed."""
        if self.worker:
            self.worker.request_disconnect()
            self.worker.wait(5000)  # Wait up to 5 seconds for clean shutdown
            self.worker = None


class ToggleWidgetHandler(ida_kernwin.action_handler_t):
    """Handler to toggle the dockable widget."""

    def __init__(self, plugin):
        ida_kernwin.action_handler_t.__init__(self)
        self.plugin = plugin

    def activate(self, ctx):
        """Toggle widget visibility."""
        self.plugin.toggle_widget()
        return 1

    def update(self, ctx):
        return ida_kernwin.AST_ENABLE_ALWAYS


class IDAChatPlugin(ida_idaapi.plugin_t):
    """Main plugin class."""

    flags = ida_idaapi.PLUGIN_KEEP
    comment = PLUGIN_COMMENT
    help = PLUGIN_HELP
    wanted_name = PLUGIN_NAME
    wanted_hotkey = ""

    def init(self):
        """Initialize the plugin."""
        self.form = None

        # Register toggle action
        action_desc = ida_kernwin.action_desc_t(
            ACTION_ID,
            ACTION_NAME,
            ToggleWidgetHandler(self),
            ACTION_HOTKEY,
            ACTION_TOOLTIP,
            -1
        )

        if not ida_kernwin.register_action(action_desc):
            ida_kernwin.msg(f"{PLUGIN_NAME}: Failed to register action\n")
            return ida_idaapi.PLUGIN_SKIP

        ida_kernwin.attach_action_to_menu(
            "View/",
            ACTION_ID,
            ida_kernwin.SETMENU_APP
        )

        ida_kernwin.msg(f"{PLUGIN_NAME}: Loaded (use {ACTION_HOTKEY} to toggle)\n")
        return ida_idaapi.PLUGIN_KEEP

    def toggle_widget(self):
        """Show or hide the dockable widget."""
        widget = ida_kernwin.find_widget(WIDGET_TITLE)

        if widget:
            ida_kernwin.close_widget(widget, 0)
            self.form = None
        else:
            self.form = IDAChatForm()
            self.form.Show(
                WIDGET_TITLE,
                options=(
                    ida_kernwin.PluginForm.WOPN_PERSIST |
                    ida_kernwin.PluginForm.WOPN_DP_RIGHT |
                    ida_kernwin.PluginForm.WOPN_DP_SZHINT
                )
            )
            # Dock to the right side panel
            ida_kernwin.set_dock_pos(
                WIDGET_TITLE,
                'IDATopLevelDockArea',
                ida_kernwin.DP_RIGHT | ida_kernwin.DP_SZHINT
            )

    def run(self, arg):
        """Called when plugin is invoked directly."""
        self.toggle_widget()

    def term(self):
        """Clean up when plugin is unloaded."""
        widget = ida_kernwin.find_widget(WIDGET_TITLE)
        if widget:
            ida_kernwin.close_widget(widget, 0)

        ida_kernwin.detach_action_from_menu("View/", ACTION_ID)
        ida_kernwin.unregister_action(ACTION_ID)

        ida_kernwin.msg(f"{PLUGIN_NAME}: Unloaded\n")


def PLUGIN_ENTRY():
    """Plugin entry point."""
    return IDAChatPlugin()
