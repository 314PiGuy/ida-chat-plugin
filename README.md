# IDA Chat Plugin

An AI-powered chat interface for IDA Pro using Claude Agent SDK with multi-provider support. Ask questions about your binary and get intelligent analysis powered by Claude, Gemini, OpenAI, OpenRouter, Ollama, or NVIDIA NIM.

https://github.com/user-attachments/assets/4377116d-337e-451a-8b80-85245e7501ce

## Features

- Dockable chat widget inside IDA Pro (Ctrl+Shift+C to toggle)
- AI-powered binary analysis with multiple model providers
- Automatic script generation and execution
- Markdown rendering for formatted responses
- Persistent chat history per database
- In-window provider/model switching similar to Copilot model picker

## Project Layout

The runtime is now organized as a modular package:

```text
ida_chat/
   core.py             # Agent loop, script/tool execution, provider transport
   plugin.py           # IDA plugin runtime and form lifecycle
   cli.py              # CLI runtime
   history.py          # Persistent session history
   logging_utils.py    # Shared log setup (/tmp/ida-chat.log)
   providers/config.py # Provider adaptation and environment wiring
   tools/              # Tool regex/catalog and scripting utilities
   ui/elements.py      # Reusable Qt widgets/components

ida_chat_plugin.py    # Compatibility wrapper for plugin entrypoint
ida_chat_core.py      # Compatibility wrapper for core imports
ida_chat_cli.py       # Compatibility wrapper for CLI entrypoint
ida_chat_history.py   # Compatibility wrapper for history imports
ida_chat_provider.py  # Compatibility wrapper for provider imports
```

This keeps existing entry points stable while making internals easier to extend and test.

## Requirements

- IDA Pro 9.0 or later
- `hcli` (Hex-Rays CLI tool)

## Installation

> **Note:** Make sure you have the latest version of [hcli](https://hcli.docs.hex-rays.com/) installed.

Install directly with hcli:
```bash
hcli plugin install ida-chat
```

Or install from the GitHub repository:
```bash
hcli plugin install https://github.com/hexRaysSA/ida-chat-plugin
```

Alternatively, download and install from a release:
1. Download the latest release (`ida-chat.zip`) from the [releases page](https://github.com/HexRaysSA/ida-chat-plugin/releases)
2. Install with hcli:
   ```bash
   hcli plugin install ida-chat.zip
   ```

3. On first launch, the setup wizard will guide you through provider configuration.

Free-tier recommendation order: Gemini -> OpenRouter `:free` models -> NVIDIA NIM.

After setup, switch provider/model directly from the top bar of the IDA Chat panel.

## Usage

1. Open a database in IDA Pro
2. Press **Ctrl+Shift+C** to open the chat panel (or use Edit > Plugins > IDA Chat)
3. Type your question and press Enter

![IDA Chat in action](docs/screenshot_2.png)

Example prompts:
- "List the main functions in this binary"
- "Analyze the function at the current address"
- "Find potential vulnerabilities"
- "Explain what this code does"

## Uninstalling

```bash
hcli plugin uninstall ida-chat
```

## License

MIT
