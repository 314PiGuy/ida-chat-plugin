"""Reusable UI building blocks for IDA Chat panels."""

from __future__ import annotations

import html
import re
import time

from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QKeyEvent, QPalette


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
            Qt.TextSelectableByMouse
            | Qt.TextSelectableByKeyboard
            | Qt.LinksAccessibleByMouse
            | Qt.LinksAccessibleByKeyboard
        )
        self.content_label.setOpenExternalLinks(False)
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
        def render_pre(text: str) -> str:
            trusted_html = '<a href=' in text
            content = text if trusted_html else html.escape(text)
            return (
                "<pre style='margin: 0; white-space: pre-wrap; "
                "word-wrap: break-word; overflow-wrap: anywhere;'>"
                f"{content}</pre>"
            )

        if self._collapsed:
            # Show first few lines with ellipsis
            lines = self._content.strip().split('\n')
            preview = '\n'.join(lines[:3])
            if len(lines) > 3:
                preview += f"\n... ({len(lines) - 3} more lines)"
            self.content_label.setText(render_pre(preview))
        else:
            self.content_label.setText(render_pre(self._content))

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._update_header_text()
        self._update_content()

    def set_collapsed(self, collapsed: bool):
        """Set collapsed state without requiring user click."""
        if self._collapsed == collapsed:
            return
        self._collapsed = collapsed
        self._update_header_text()
        self._update_content()

    def set_content(self, content: str):
        """Update section content and refresh preview/header."""
        self._content = content
        self._update_header_text()
        self._update_content()

    @staticmethod
    def should_collapse(content: str) -> bool:
        """Check if content should be collapsed."""
        return len(content.strip().split('\n')) > CollapsibleSection.COLLAPSE_THRESHOLD


class ToolBatchSection(QFrame):
    """Grouped tool-call block with per-call expandable entries."""

    def __init__(self, title: str = "Tool Batch", collapsed: bool = True, parent=None):
        super().__init__(parent)
        self._title = title
        self._collapsed = collapsed
        self._calls: list[dict[str, object]] = []
        self._setup_ui()

    def _setup_ui(self):
        colors = get_ida_colors()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.header = QPushButton()
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

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 0, 0, 0)
        self.body_layout.setSpacing(4)
        self.body.setVisible(not self._collapsed)
        layout.addWidget(self.body)

        self._update_header()

    def _toggle(self):
        self._collapsed = not self._collapsed
        self.body.setVisible(not self._collapsed)
        self._update_header()

    def set_collapsed(self, collapsed: bool):
        """Set collapsed state of the batch container."""
        if self._collapsed == collapsed:
            return
        self._collapsed = collapsed
        self.body.setVisible(not self._collapsed)
        self._update_header()

    def _update_header(self):
        arrow = "▶" if self._collapsed else "▼"
        count = len(self._calls)
        self.header.setText(f"{arrow} {self._title} ({count} calls)")

    def add_call(self, tool_name: str, request_text: str) -> int:
        """Add one call entry and return its index."""
        index = len(self._calls)
        req = (request_text or "").strip() or "(empty request)"
        content = f"Request:\n{req}\n\nResponse:\n[pending]"
        section = CollapsibleSection(f"{index + 1}. {tool_name}", content, collapsed=True)
        self.body_layout.addWidget(section)
        self._calls.append(
            {
                "tool_name": tool_name,
                "request": req,
                "response": "",
                "section": section,
            }
        )
        self._update_header()
        return index

    def get_call_tool_name(self, index: int) -> str:
        """Return tool name for one indexed call."""
        if index < 0 or index >= len(self._calls):
            return ""
        return str(self._calls[index].get("tool_name", ""))

    def get_call_section(self, index: int) -> CollapsibleSection | None:
        """Return the per-call collapsible section widget."""
        if index < 0 or index >= len(self._calls):
            return None
        section = self._calls[index].get("section")
        return section if isinstance(section, CollapsibleSection) else None

    def set_call_response(self, index: int, response_text: str) -> None:
        """Attach response text to one call entry."""
        if index < 0 or index >= len(self._calls):
            return
        entry = self._calls[index]
        response = (response_text or "").strip() or "(empty response)"
        entry["response"] = response
        section = entry.get("section")
        if isinstance(section, CollapsibleSection):
            req = str(entry.get("request", ""))
            section.set_content(f"Request:\n{req}\n\nResponse:\n{response}")

    def set_all_calls_collapsed(self, collapsed: bool) -> None:
        """Set collapsed state for all nested call sections."""
        for entry in self._calls:
            section = entry.get("section")
            if isinstance(section, CollapsibleSection):
                section.set_collapsed(collapsed)


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
        return (
            f'<pre style="background-color: {code_bg}; color: {code_fg}; padding: 8px; '
            'border-radius: 4px; overflow-x: auto; white-space: pre-wrap; '
            f'word-wrap: break-word; overflow-wrap: anywhere;"><code>{code}</code></pre>'
        )
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

    return (
        "<div style='white-space: normal; word-wrap: break-word; "
        f"overflow-wrap: anywhere;'>{text}</div>"
    )


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
        self._raw_text = text
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
                if self._is_processing:
                    self.message_widget.setText(self._render_stream_text(text))
                else:
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
        if not self.is_user and self._msg_type == MessageType.TEXT:
            self.message_widget.setText(markdown_to_html(self._raw_text))

    def _render_stream_text(self, text: str) -> str:
        """Render lightweight streaming text without full markdown conversion."""
        escaped = html.escape(text).replace("\n", "<br>")
        return (
            "<div style='white-space: normal; word-wrap: break-word; "
            f"overflow-wrap: anywhere;'>{escaped}</div>"
        )

    def update_text(self, text: str):
        """Update the message text."""
        self._raw_text = text
        if self.is_user:
            self.message_widget.setText(text)
        else:
            if self._msg_type == MessageType.TEXT and self._is_processing:
                self.message_widget.setText(self._render_stream_text(text))
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

    def scroll_to_bottom_now(self):
        """Immediately scroll to the bottom without timer queuing."""
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def is_near_bottom(self, threshold: int = 60) -> bool:
        """Return whether viewport is close enough to bottom to auto-stick."""
        bar = self.verticalScrollBar()
        return (bar.maximum() - bar.value()) <= max(0, threshold)

    def add_collapsible(self, title: str, content: str, collapsed: bool = True) -> CollapsibleSection:
        """Add a collapsible section to the chat history."""
        section = CollapsibleSection(title, content, collapsed)
        self.layout.addWidget(section)
        self.scroll_to_bottom()
        return section

    def add_tool_batch(self, title: str = "Tool Batch", collapsed: bool = True) -> ToolBatchSection:
        """Add one grouped tool batch section to the chat history."""
        batch = ToolBatchSection(title=title, collapsed=collapsed)
        self.layout.addWidget(batch)
        self.scroll_to_bottom()
        return batch

    def clear_history(self):
        """Clear all messages from the chat history."""
        self._current_processing_message = None
        # Remove all widgets except the stretch at index 0
        while self.layout.count() > 1:
            item = self.layout.takeAt(1)  # Always take from index 1, leaving stretch at 0
            if item.widget():
                item.widget().deleteLater()


class EventLogWidget(QScrollArea):
    """Scrollable event log with expandable details for tool/model I/O."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[dict[str, str | float | None]] = []
        self._entry_widgets: list[tuple[dict[str, str | float | None], QFrame]] = []
        self._kind_filter = ""
        self._text_filter = ""
        self._setup_ui()

    def _setup_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setSpacing(6)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.addStretch(1)
        self.setWidget(self.container)

    def clear_events(self):
        """Clear all recorded events from the panel."""
        self._entries.clear()
        self._entry_widgets.clear()
        while self.layout.count() > 1:
            item = self.layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def add_event(
        self,
        kind: str,
        title: str,
        details: str = "",
        duration_ms: float | None = None,
    ) -> None:
        """Append one event entry with optional expandable details."""
        colors = get_ida_colors()

        frame = QFrame()
        frame.setStyleSheet(
            f"""
            QFrame {{
                border: 1px solid {colors['mid']};
                border-radius: 6px;
                background-color: {colors['window']};
            }}
            """
        )

        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 6, 8, 6)
        frame_layout.setSpacing(4)

        ts = time.strftime("%H:%M:%S")
        duration_text = f" · {duration_ms:.1f} ms" if duration_ms is not None and duration_ms >= 0 else ""
        summary = QLabel(f"<b>[{ts}] {title}</b><span style='color:{colors['mid']};'> ({kind}){duration_text}</span>")
        summary.setTextFormat(Qt.RichText)
        summary.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        summary.setStyleSheet(f"color: {colors['text']};")
        frame_layout.addWidget(summary)

        self._entries.append(
            {
                "time": ts,
                "kind": kind,
                "title": title,
                "duration_ms": duration_ms,
                "details": details,
            }
        )

        entry_ref = self._entries[-1]

        if details.strip():
            section = CollapsibleSection("Details", details, collapsed=True)
            frame_layout.addWidget(section)

        self.layout.insertWidget(self.layout.count() - 1, frame)
        self._entry_widgets.append((entry_ref, frame))
        self._apply_filters()
        QTimer.singleShot(10, lambda: self.verticalScrollBar().setValue(self.verticalScrollBar().maximum()))

    def set_kind_filter(self, kind: str) -> None:
        """Filter events by kind; empty means show all."""
        self._kind_filter = (kind or "").strip().lower()
        self._apply_filters()

    def set_text_filter(self, text: str) -> None:
        """Filter events by case-insensitive substring in title/details."""
        self._text_filter = (text or "").strip().lower()
        self._apply_filters()

    def _matches_filter(self, entry: dict[str, str | float | None]) -> bool:
        kind_filter = self._kind_filter
        if kind_filter and str(entry.get("kind", "")).strip().lower() != kind_filter:
            return False

        text_filter = self._text_filter
        if not text_filter:
            return True

        haystack = (
            f"{entry.get('title', '')}\n{entry.get('details', '')}\n{entry.get('kind', '')}"
        ).lower()
        return text_filter in haystack

    def _apply_filters(self) -> None:
        for entry, frame in self._entry_widgets:
            frame.setVisible(self._matches_filter(entry))

    def to_plain_text(self) -> str:
        """Return the complete event log as plain text for copy/paste diagnostics."""
        if not self._entries:
            return "(no events recorded)"

        chunks: list[str] = []
        for entry in self._entries:
            duration_ms = entry.get("duration_ms")
            duration_text = ""
            if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
                duration_text = f" [{duration_ms:.1f} ms]"

            header = (
                f"[{entry.get('time', '--:--:--')}] "
                f"{entry.get('title', '')} ({entry.get('kind', '')}){duration_text}"
            )
            details = str(entry.get("details", "")).strip()
            if details:
                chunks.append(f"{header}\n{details}")
            else:
                chunks.append(header)

        return "\n\n".join(chunks)


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

