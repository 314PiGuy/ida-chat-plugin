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
import time
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
    QListWidget,
    QListWidgetItem,
    QSplitter,
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
from ida_chat_ui_elements import (
    ChatHistoryWidget,
    ChatInputWidget,
    CollapsibleSection,
    EventLogWidget,
    MessageType,
    ProgressTimeline,
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


class PluginCallback(ChatCallback):
    """Qt widget output implementation of ChatCallback.

    Uses Qt signals to safely update UI from any thread.
    """

    def __init__(self, signals: "AgentSignals"):
        self.signals = signals

    def on_metric(self, text: str) -> None:
        self.signals.metric.emit(text)

    def on_event(
        self,
        kind: str,
        title: str,
        details: str,
        duration_ms: float | None = None,
    ) -> None:
        self.signals.event.emit(kind, title, details, duration_ms if duration_ms is not None else -1.0)

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

    metric = Signal(str)
    event = Signal(str, str, str, float)
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
        self._thinking_message = None
        self._current_turn = 0
        self._max_turns = 20
        self._total_cost = 0.0
        self._script_count = 0
        self._last_had_error = False
        self._message_count = 0
        self._streaming_active = False
        self._stream_buffer = ""
        self._stream_pending = ""
        self._details_expanded = False
        self._active_session_id: str | None = None
        self._updating_session_list = False
        self._provider_config = get_provider_config()
        self._model_name = describe_provider(self._provider_config)

        self._stream_flush_timer = QTimer(self.parent)
        self._stream_flush_timer.setSingleShot(True)
        self._stream_flush_timer.setInterval(80)
        self._stream_flush_timer.timeout.connect(self._flush_stream_text)

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
            self._refresh_chat_session_list()

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
            self.worker.signals.metric.connect(self._on_metric)
            self.worker.signals.event.connect(self._on_event)
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
        self.tabs.hide()
        self.progress_timeline.hide()

    def _show_settings(self):
        """Show settings panel (re-use onboarding panel)."""
        # Load current settings into the panel
        self.onboarding_panel.load_current_settings()
        self._show_onboarding()

    def _on_onboarding_complete(self):
        """Handle successful onboarding."""
        self.onboarding_panel.hide()
        self.tabs.show()
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
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(False)
        self._update_status_bar()

        # Load message history for up/down arrow navigation
        if hasattr(self, 'history'):
            user_messages = self.history.get_all_user_messages()
            self.input_widget.set_history(user_messages)
            self._active_session_id = self.history.get_current_session_id()
            self._refresh_chat_session_list(select_session_id=self._active_session_id)

    def _refresh_chat_session_list(self, select_session_id: str | None = None):
        """Refresh the chats panel list from persistent session history."""
        if not hasattr(self, "history") or not self.history:
            return

        self._updating_session_list = True
        self.chat_session_list.clear()

        sessions = self.history.list_sessions()
        for session in sessions:
            session_id = session.get("id", "")
            first_message = str(session.get("first_message", "(empty)"))
            timestamp = str(session.get("timestamp", ""))
            short_id = session_id[:8]
            label = f"{short_id}  {first_message}"
            if timestamp:
                label = f"{timestamp[:19]}  {label}"

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, session_id)
            self.chat_session_list.addItem(item)

            should_select = False
            if select_session_id and session_id == select_session_id:
                should_select = True
            elif not select_session_id and self._active_session_id and session_id == self._active_session_id:
                should_select = True

            if should_select:
                self.chat_session_list.setCurrentItem(item)

        self._updating_session_list = False

    def _extract_text_content(self, content) -> str:
        """Extract human-readable text from transcript content structures."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif item_type == "tool_result":
                result = item.get("content")
                parts.append(str(result))
        return "\n".join(part for part in parts if part)

    def _load_session_into_chat(self, session_id: str):
        """Load one persisted chat session into the chat view."""
        if not hasattr(self, "history") or not self.history:
            return

        entries = self.history.load_session(session_id)
        self.chat_history.clear_history()

        for entry in entries:
            entry_type = entry.get("type")
            message = entry.get("message", {}) if isinstance(entry, dict) else {}

            if entry_type == "user":
                text = self._extract_text_content(message.get("content", ""))
                if text:
                    self.chat_history.add_message(text, is_user=True)
                continue

            if entry_type == "assistant":
                content = message.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        item_type = item.get("type")
                        if item_type == "text":
                            text = item.get("text", "")
                            if text:
                                self.chat_history.add_message(str(text), is_user=False)
                        elif item_type == "tool_use":
                            name = str(item.get("name", "Tool"))
                            tool_input = item.get("input", {})
                            details = json.dumps(tool_input, ensure_ascii=False)
                            self.chat_history.add_message(
                                f"[{name}] {details}",
                                is_user=False,
                                msg_type=MessageType.TOOL_USE,
                            )
                continue

            if entry_type == "system":
                system_text = str(entry.get("content", "")).strip()
                if system_text:
                    self.chat_history.add_message(f"[System] {system_text}", is_user=False)

        if not entries:
            self.chat_history.add_message("Selected chat is empty.", is_user=False)

    def _start_new_chat(self):
        """Create and switch to a new chat session."""
        if self._is_processing:
            self.chat_history.add_message("Finish the current request before starting a new chat.", is_user=False)
            return
        if not hasattr(self, "history") or not self.history:
            return

        new_session_id = self.history.start_new_session()
        self._active_session_id = new_session_id
        self.chat_history.clear_history()
        self.chat_history.add_message("Started a new chat session.", is_user=False)
        self._refresh_chat_session_list(select_session_id=new_session_id)
        self.input_widget.set_history(self.history.get_all_user_messages())
        self._log_metric(f"Started new chat: {new_session_id[:8]}")

    def _delete_selected_chat(self):
        """Delete the currently selected chat session."""
        if self._is_processing:
            self.chat_history.add_message("Finish the current request before deleting chats.", is_user=False)
            return
        if not hasattr(self, "history") or not self.history:
            return

        item = self.chat_session_list.currentItem()
        if not item:
            self.chat_history.add_message("Select a chat to delete.", is_user=False)
            return

        session_id = item.data(Qt.UserRole)
        if not isinstance(session_id, str) or not session_id:
            return

        deleted = self.history.delete_session(session_id)
        if not deleted:
            self.chat_history.add_message("Unable to delete selected chat session.", is_user=False)
            return

        self._log_metric(f"Deleted chat: {session_id[:8]}")

        if self._active_session_id == session_id:
            remaining = self.history.list_sessions()
            if remaining:
                next_id = str(remaining[0].get("id", ""))
                if next_id and self.history.switch_session(next_id):
                    self._active_session_id = next_id
                    self._load_session_into_chat(next_id)
                else:
                    self._active_session_id = None
                    self.chat_history.clear_history()
            else:
                new_session_id = self.history.start_new_session()
                self._active_session_id = new_session_id
                self.chat_history.clear_history()
                self.chat_history.add_message("Started a new chat session.", is_user=False)

        self._refresh_chat_session_list(select_session_id=self._active_session_id)
        self.input_widget.set_history(self.history.get_all_user_messages())

    def _on_chat_session_selected(self):
        """Switch active chat when the user selects another session."""
        if self._updating_session_list or self._is_processing:
            return
        if not hasattr(self, "history") or not self.history:
            return

        item = self.chat_session_list.currentItem()
        if not item:
            return

        session_id = item.data(Qt.UserRole)
        if not isinstance(session_id, str) or not session_id:
            return
        if session_id == self._active_session_id:
            return

        if not self.history.switch_session(session_id):
            self.chat_history.add_message("Unable to switch to selected chat session.", is_user=False)
            return

        self._active_session_id = session_id
        self._load_session_into_chat(session_id)
        self.input_widget.set_history(self.history.get_all_user_messages())
        self._log_metric(f"Switched to chat: {session_id[:8]}")

    def _on_connection_error(self, error: str):
        """Called when agent connection fails."""
        self.chat_history.add_message(f"Connection error: {error}", is_user=False)

    def _log_metric(self, msg: str):
        self.events_log.add_event("metric", msg)

    def _on_metric(self, text: str):
        """Called for simple metric lines emitted from core."""
        self._log_metric(text)

    def _on_event(self, kind: str, title: str, details: str, duration_ms: float):
        """Called for structured, expandable core events."""
        self.events_log.add_event(kind, title, details, duration_ms if duration_ms >= 0 else None)

    def _on_turn_start(self, turn: int, max_turns: int):
        """Called at the start of each agentic turn."""
        self._current_turn = turn
        self._max_turns = max_turns
        self._log_metric(f"Turn started: {turn}/{max_turns}")

    def _on_thinking(self):
        """Called when agent starts processing."""
        self._is_processing = True
        self._streaming_active = False
        self._stream_buffer = ""
        self._stream_pending = ""
        if self._stream_flush_timer.isActive():
            self._stream_flush_timer.stop()
        # Mark previous message as complete before starting new turn
        if self._current_message:
            self._current_message.set_complete()
        self.input_widget.setEnabled(False)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(True)

        # Check if this is a retry after error
        if self._last_had_error:
            self._last_had_error = False
            # Update timeline
            self.progress_timeline.add_stage("Retrying")
            # Add retry message
            self._current_message = self.chat_history.add_message(
                "🔄 Retrying after error...", is_user=False, is_processing=True
            )
            self._thinking_message = self._current_message
        else:
            # Update timeline
            self.progress_timeline.add_stage("Thinking")
            # Add thinking message with blinking indicator
            self._current_message = self.chat_history.add_message(
                "[Thinking...]", is_user=False, is_processing=True
            )
            self._thinking_message = self._current_message

    def _on_thinking_done(self):
        """Called when agent produces first output."""
        if self._thinking_message:
            for idx in range(self.chat_history.layout.count()):
                item = self.chat_history.layout.itemAt(idx)
                if item and item.widget() is self._thinking_message:
                    taken = self.chat_history.layout.takeAt(idx)
                    if taken and taken.widget():
                        taken.widget().deleteLater()
                    break
            if self._current_message is self._thinking_message:
                self._current_message = None
            self._thinking_message = None

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
        self._flush_stream_text()
        self._streaming_active = False
        self._stream_buffer = ""
        self._stream_pending = ""
        tool_msg = f"[{tool_name}]"
        if details:
            tool_msg += " call"
        self._add_processing_message(tool_msg, MessageType.TOOL_USE)
        self._log_metric(f"Tool executed: {tool_name} (details: {details[:100]}...)")
        if details.strip():
            self.chat_history.add_collapsible(f"{tool_name} Details", details, collapsed=True)

    def _flush_stream_text(self):
        """Flush pending streamed chunks into a single UI update."""
        if not self._stream_pending:
            return

        if self._thinking_message:
            self._on_thinking_done()

        if self._streaming_active and self._current_message:
            self._stream_buffer += self._stream_pending
            self._current_message.update_text(self._stream_buffer)
        else:
            self._streaming_active = True
            self._stream_buffer = self._stream_pending
            self._add_processing_message(self._stream_buffer)

        self._stream_pending = ""

    def _on_text(self, text: str):
        """Called when agent outputs text."""
        if not text:
            return

        self._stream_pending += text
        if not self._stream_flush_timer.isActive():
            self._stream_flush_timer.start()

    def _on_script_code(self, code: str):
        """Called with script code before execution."""
        import html
        self._flush_stream_text()
        self._streaming_active = False
        self._stream_buffer = ""
        self._stream_pending = ""
        # Update timeline
        self._script_count += 1
        self.progress_timeline.add_stage(f"Script {self._script_count}")
        self._log_metric(f"Executing script {self._script_count} ({len(code)} bytes)")
        self._add_processing_message(f"Executing script {self._script_count}", MessageType.SCRIPT)
        self.chat_history.add_collapsible(f"Script {self._script_count} Code", html.escape(code), collapsed=True)

    def _on_script_output(self, output: str):
        """Called with script output."""
        if output.strip():
            import html
            # Check if this is an error output
            is_error = output.strip().startswith("Script error:")
            is_tool_json = output.lstrip().startswith("{") and '"tool"' in output[:300]
            if is_error:
                self._last_had_error = True
                self._add_processing_message(output, MessageType.ERROR)
            # Use collapsible section for long outputs
            elif is_tool_json or CollapsibleSection.should_collapse(output):
                # Mark previous message as complete
                if self._current_message:
                    self._current_message.set_complete()
                title = "Tool Output" if is_tool_json else "Script Output"
                self.chat_history.add_collapsible(title, output, collapsed=True)
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
        self._flush_stream_text()
        self._is_processing = False
        self._streaming_active = False
        self._stream_buffer = ""
        self._stream_pending = ""
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(False)
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

        # Stop button
        self.stop_btn = QPushButton("■")
        self.stop_btn.setFixedSize(24, 24)
        self.stop_btn.setToolTip("Stop current response")
        self.stop_btn.setStyleSheet(icon_btn_style)
        self.stop_btn.clicked.connect(self._on_cancel)
        self.stop_btn.setEnabled(False)
        header_layout.addWidget(self.stop_btn)

        # Clear button
        clear_btn = QPushButton("✕")
        clear_btn.setFixedSize(24, 24)
        clear_btn.setToolTip("Clear chat")
        clear_btn.setStyleSheet(icon_btn_style)
        clear_btn.clicked.connect(self._on_clear)
        header_layout.addWidget(clear_btn)

        # Collapse/expand details button
        self.details_btn = QPushButton("▤")
        self.details_btn.setFixedSize(24, 24)
        self.details_btn.setToolTip("Expand/collapse script and tool details")
        self.details_btn.setStyleSheet(icon_btn_style)
        self.details_btn.clicked.connect(self._toggle_all_details)
        header_layout.addWidget(self.details_btn)

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
                background: {colors['base']};
            }}
            QTabBar::tab {{
                background: {colors['alt_base']};
                color: {colors['text']};
                padding: 5px 12px;
                border: 1px solid {colors['mid']};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                color: {colors['highlight_text']};
                background: {colors['highlight']};
            }}
        """)

        # --- Chat Tab ---
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        chat_splitter = QSplitter(Qt.Horizontal)

        # Sessions panel (new chat + switch chats)
        sessions_panel = QWidget()
        sessions_panel.setStyleSheet(f"""
            QWidget {{
                background-color: {colors['alt_base']};
                border-right: 1px solid {colors['mid']};
            }}
        """)
        sessions_layout = QVBoxLayout(sessions_panel)
        sessions_layout.setContentsMargins(8, 8, 8, 8)
        sessions_layout.setSpacing(6)

        sessions_title = QLabel("Chats")
        sessions_title.setStyleSheet(f"color: {colors['window_text']}; font-weight: bold;")
        sessions_layout.addWidget(sessions_title)

        self.new_chat_btn = QPushButton("New Chat")
        self.new_chat_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {colors['button']};
                color: {colors['button_text']};
            }}
        """)
        self.new_chat_btn.clicked.connect(self._start_new_chat)
        sessions_layout.addWidget(self.new_chat_btn)

        self.delete_chat_btn = QPushButton("Delete Chat")
        self.delete_chat_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['button']};
                color: {colors['button_text']};
                border: 1px solid {colors['mid']};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: {colors['dark']};
                color: {colors['text']};
            }}
        """)
        self.delete_chat_btn.clicked.connect(self._delete_selected_chat)
        sessions_layout.addWidget(self.delete_chat_btn)

        self.chat_session_list = QListWidget()
        self.chat_session_list.itemSelectionChanged.connect(self._on_chat_session_selected)
        self.chat_session_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {colors['base']};
                color: {colors['text']};
                border: 1px solid {colors['mid']};
                border-radius: 6px;
                padding: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}
        """)
        sessions_layout.addWidget(self.chat_session_list, stretch=1)

        # Main chat panel
        chat_main = QWidget()
        chat_main_layout = QVBoxLayout(chat_main)
        chat_main_layout.setContentsMargins(0, 0, 0, 0)
        chat_main_layout.setSpacing(0)

        # Chat history area (takes most space)
        self.chat_history = ChatHistoryWidget()
        chat_main_layout.addWidget(self.chat_history, stretch=1)

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

        self.edit_resend_btn = QPushButton("Edit/Resend")
        self.edit_resend_btn.setToolTip("Load last message for editing and resend")
        self.edit_resend_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['button']};
                color: {colors['button_text']};
                border: 1px solid {colors['mid']};
                border-radius: 6px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}
        """)
        self.edit_resend_btn.clicked.connect(self._on_edit_resend_last)
        input_layout.addWidget(self.edit_resend_btn)

        chat_main_layout.addWidget(self.input_container)

        # Status bar at bottom
        self.status_bar = QWidget()
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(10, 4, 10, 4)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {colors['mid']}; font-size: 11px;")
        status_layout.addWidget(self.status_label)

        chat_main_layout.addWidget(self.status_bar)

        chat_splitter.addWidget(sessions_panel)
        chat_splitter.addWidget(chat_main)
        chat_splitter.setSizes([220, 900])
        chat_layout.addWidget(chat_splitter, stretch=1)
        
        self.tabs.addTab(chat_tab, "Chat")

        # --- Metrics Tab ---
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        metrics_layout.setContentsMargins(0, 0, 0, 0)

        metrics_toolbar = QWidget()
        metrics_toolbar_layout = QHBoxLayout(metrics_toolbar)
        metrics_toolbar_layout.setContentsMargins(8, 6, 8, 6)

        metrics_title = QLabel("Detailed model/tool events with expandable inputs, outputs, and timing")
        metrics_title.setStyleSheet(f"color: {colors['mid']}; font-size: 11px;")
        metrics_toolbar_layout.addWidget(metrics_title)
        metrics_toolbar_layout.addStretch()

        copy_events_btn = QPushButton("Copy Events")
        copy_events_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['button']};
                color: {colors['button_text']};
                border: 1px solid {colors['mid']};
                border-radius: 4px;
                padding: 2px 8px;
            }}
            QPushButton:hover {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}
        """)
        copy_events_btn.clicked.connect(self._on_copy_events)
        metrics_toolbar_layout.addWidget(copy_events_btn)

        clear_events_btn = QPushButton("Clear Events")
        clear_events_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['button']};
                color: {colors['button_text']};
                border: 1px solid {colors['mid']};
                border-radius: 4px;
                padding: 2px 8px;
            }}
            QPushButton:hover {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}
        """)
        clear_events_btn.clicked.connect(self._on_clear_events)
        metrics_toolbar_layout.addWidget(clear_events_btn)

        metrics_layout.addWidget(metrics_toolbar)

        self.events_log = EventLogWidget()
        metrics_layout.addWidget(self.events_log, stretch=1)
        
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
            if hasattr(self, "stop_btn"):
                self.stop_btn.setEnabled(False)
            self._log_metric("Stop requested by user")

    def _toggle_all_details(self):
        """Toggle collapsed/expanded state for all detail sections."""
        self._details_expanded = not self._details_expanded
        target_collapsed = not self._details_expanded

        for idx in range(self.chat_history.layout.count()):
            item = self.chat_history.layout.itemAt(idx)
            if not item:
                continue
            widget = item.widget()
            if isinstance(widget, CollapsibleSection):
                widget.set_collapsed(target_collapsed)

        state = "expanded" if self._details_expanded else "collapsed"
        self._log_metric(f"Detail sections {state}")

    def _on_edit_resend_last(self):
        """Load the latest user message into the input box for quick editing/resend."""
        if self._is_processing:
            self.chat_history.add_message("Finish the current request before editing/resending.", is_user=False)
            return
        if not hasattr(self, "history") or not self.history:
            self.chat_history.add_message("No message history available yet.", is_user=False)
            return

        user_messages = self.history.get_all_user_messages()
        if not user_messages:
            self.chat_history.add_message("No previous message to edit/resend.", is_user=False)
            return

        last_message = user_messages[-1]
        self.input_widget.setPlainText(last_message)
        cursor = self.input_widget.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.input_widget.setTextCursor(cursor)
        self.input_widget.setFocus()
        self._log_metric("Loaded last message for edit/resend")

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
        self._streaming_active = False
        self._stream_buffer = ""
        self._stream_pending = ""
        if self._stream_flush_timer.isActive():
            self._stream_flush_timer.stop()
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(False)
        self.chat_history.clear_history()
        self._total_cost = 0.0
        self._script_count = 0
        self._message_count = 0
        self.progress_timeline.hide_timeline()

        # Start a new session for history tracking
        if hasattr(self, "history") and self.history:
            new_session_id = self.history.start_new_session()
            self._active_session_id = new_session_id
            self._refresh_chat_session_list(select_session_id=new_session_id)
            self.input_widget.set_history(self.history.get_all_user_messages())
            self._log_metric(f"Chat cleared. New session: {new_session_id[:8]}")

        # Add ready message (agent already connected)
        self.chat_history.add_message("Chat cleared. Ready for new conversation.", is_user=False)
        self.input_widget.setEnabled(True)
        self.input_widget.setFocus()
        self._update_status_bar()

    def _on_clear_events(self):
        """Clear metrics/events panel entries."""
        self.events_log.clear_events()
        self._log_metric("Event log cleared")

    def _on_copy_events(self):
        """Copy full events log as plain text to clipboard."""
        text = self.events_log.to_plain_text()
        QApplication.clipboard().setText(text)
        self._log_metric("Copied event log to clipboard")

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
