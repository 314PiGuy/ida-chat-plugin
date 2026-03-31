# IDA Chat

Chat interface for IDA Pro powered by Claude Agent SDK.

## Project Structure

```
ida_chat/             # Modular package
  core.py             # Shared runtime: agent loop, script/tool execution
  cli.py              # CLI runtime implementation
  plugin.py           # IDA plugin runtime implementation
  history.py          # Message history persistence
  logging_utils.py    # Shared logger setup
  providers/
    config.py         # Provider normalization, env wiring, defaults
  tools/
    patterns.py       # <idascript>/<idatool>/<delegate> regex extraction
    catalog.py        # Built-in idatool catalog
  ui/
    elements.py       # Reusable Qt widgets and UI primitives
ida_chat_core.py      # Compatibility wrapper -> ida_chat.core
ida_chat_cli.py       # Compatibility wrapper -> ida_chat.cli
ida_chat_plugin.py    # Compatibility wrapper -> ida_chat.plugin
ida_chat_history.py   # Compatibility wrapper -> ida_chat.history
ida_chat_provider.py  # Compatibility wrapper -> ida_chat.providers.config
project/              # Agent working directory
  PROMPT.md           # System prompt (loaded at runtime)
  USAGE.md            # API usage patterns and tips
  API_REFERENCE.md    # Complete ida-domain API reference
```

## Architecture

### Core Module (`ida_chat/core.py`)

Shared foundation for CLI and Plugin:
- `ChatCallback` protocol - abstracts output handling
- `IDAChatCore` class - Agent SDK integration, agentic loop
- Loads system prompt from `PROMPT.md`
- Extracts and executes `<idascript>` blocks
- Feeds script output back to agent until task complete

### CLI Tool (`ida_chat/cli.py`)

Standalone command-line chat for testing outside IDA:

```bash
uv run python ida_chat_cli.py <binary.i64>              # Interactive mode
uv run python ida_chat_cli.py <binary.i64> -p "prompt"  # Single prompt
```

Implements `CLICallback` for terminal output (ANSI colors, `[Thinking...]` indicator).

### IDA Plugin (`ida_chat/plugin.py`)

Dockable chat widget inside IDA Pro (Ctrl+Shift+C to toggle).

Features:
- `PluginCallback` - emits Qt signals for UI updates
- `AgentWorker(QThread)` - runs async agent in background
- Status indicators: blinking orange (processing) → green (complete)
- Markdown rendering via `QTextBrowser` (headers, code, lists, links)
- Thread-safe script execution via `ida_kernwin.execute_sync()`

## How It Works

1. Database opened with `ida_domain.Database.open()`
2. `IDAChatCore` connects to Claude via `ClaudeSDKClient`
3. Agent reads `USAGE.md` and `API_REFERENCE.md` for API knowledge
4. Agent generates analysis code in `<idascript>` XML tags
5. Core extracts scripts and runs `exec(code, {"db": db})`
6. Script output fed back to agent for next iteration
7. Loop continues until agent responds without `<idascript>` tags

## Key Pattern: Agentic Loop

The agent works autonomously until task completion:

```
User: "Analyze this binary"
  → Agent reads docs, writes script
  → Script executes, output returned to agent
  → Agent sees output, writes more scripts or concludes
  → Loop until no more <idascript> tags (max 20 turns)
```

## `<idascript>` Tags

Agent outputs analysis code in XML tags:

```xml
<idascript>
for func in db.functions:
    name = db.functions.get_name(func)
    print(f"{name}: 0x{func.start_ea:08X}")
</idascript>
```

The core module parses these tags and executes the code against the open `db` instance.

## Dependencies

- `claude-agent-sdk` - Agent SDK for Claude
- `ida-domain` - IDA Pro domain API (works standalone, spawns IDA headlessly)

## Development

```bash
# Install dependencies
uv sync

# Test CLI (outside IDA)
uv run python ida_chat_cli.py calc.exe.i64 -p "list 3 functions"
```

### Deploying the Plugin to IDA

The plugin must be packaged as a zip and installed via `hcli`. **Close IDA before redeploying.**

```bash
# Redeploy plugin (uninstall old, install new)
rm -f ida-chat.zip && \
zip -r ida-chat.zip ida-plugin.json ida_chat_plugin.py ida_chat_core.py ida_chat_history.py ida_chat_provider.py ida_chat/ splash.png project/ && \
hcli plugin uninstall ida-chat && \
hcli plugin install ida-chat.zip --config show_wizard=true
```

**Files included in the plugin zip:**
- `ida-plugin.json` - Plugin manifest
- `ida_chat_plugin.py` - IDA plugin (Qt UI)
- `ida_chat_core.py` - Compatibility wrapper for core runtime
- `ida_chat_history.py` - Compatibility wrapper for history runtime
- `ida_chat_provider.py` - Compatibility wrapper for provider runtime
- `ida_chat/` - Modular runtime package (core, ui, tools, providers)
- `splash.png` - Onboarding splash image
- `project/` - Agent prompts and documentation (PROMPT.md, USAGE.md, API_REFERENCE.md, IDA.md)

## Releasing

```bash
# Build the release zip
zip -r ida-chat.zip ida-plugin.json ida_chat_plugin.py ida_chat_core.py ida_chat_history.py ida_chat_provider.py ida_chat/ project/

# Create and push the tag
git tag -a X.Y.Z -m "Release message"
git push origin X.Y.Z

# Create GitHub release with artifact
gh release create X.Y.Z ida-chat.zip --title "vX.Y.Z - Title" --generate-notes
```

## Logging

Debug logs written to `/tmp/ida-chat.log` for troubleshooting agent behavior.
