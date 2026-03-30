"""Provider and model configuration helpers for IDA Chat."""

from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED_PROVIDERS = ["claude", "gemini", "openai", "openrouter", "ollama", "nim"]

PROVIDER_LABELS: dict[str, str] = {
    "claude": "Claude (Anthropic)",
    "gemini": "Gemini",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "ollama": "Ollama",
    "nim": "NVIDIA NIM",
}

PROVIDER_NAME_ALIASES: dict[str, str] = {
    "anthropic": "claude",
    "claude": "claude",
    "google": "gemini",
    "gemini": "gemini",
    "openai": "openai",
    "open_router": "openrouter",
    "openrouter": "openrouter",
    "ollama": "ollama",
    "nvidia": "nim",
    "nvidia_nim": "nim",
    "nim": "nim",
}

PROVIDER_DEFAULT_MODELS: dict[str, str | None] = {
    # Keep Claude default unset so Claude Code can pick the best local default.
    "claude": None,
    # Free-tier friendly defaults where possible.
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "openrouter": "google/gemini-2.5-flash-lite-preview:free",
    "ollama": "qwen2.5-coder:7b",
    "nim": "meta/llama-3.1-70b-instruct",
}

PROVIDER_RECOMMENDED_MODELS: dict[str, list[str]] = {
    "claude": [
        "claude-sonnet-4",
        "claude-opus-4",
        "claude-3-7-sonnet-latest",
    ],
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ],
    "openai": [
        "gpt-4.1-mini",
        "gpt-4.1",
        "o4-mini",
    ],
    "openrouter": [
        "google/gemini-2.5-flash-lite-preview:free",
        "deepseek/deepseek-r1:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ],
    "ollama": [
        "qwen2.5-coder:7b",
        "deepseek-r1:8b",
        "llama3.1:8b",
    ],
    "nim": [
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "mistralai/mixtral-8x7b-instruct-v0.1",
    ],
}

PROVIDER_DEFAULT_BASE_URLS: dict[str, str | None] = {
    "claude": None,
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://127.0.0.1:11434/v1",
    "nim": "https://integrate.api.nvidia.com/v1",
}

PROVIDER_KEY_HINTS: dict[str, str] = {
    "claude": "Anthropic API key",
    "gemini": "Gemini API key",
    "openai": "OpenAI API key",
    "openrouter": "OpenRouter API key",
    "ollama": "Optional Ollama token (leave blank for local default)",
    "nim": "NVIDIA NIM API key",
}

PROVIDER_FREE_TIER_NOTES: dict[str, str] = {
    "claude": "System auth is supported when Claude Code is installed.",
    "gemini": "Strong free-tier compatibility: use gemini-2.5-flash when available.",
    "openai": "Use your OpenAI project key and optional base URL.",
    "openrouter": "Free-tier friendly via :free models, e.g. Gemini Flash Lite variants.",
    "ollama": "Runs local models and does not require a remote API key.",
    "nim": "NVIDIA NIM offers free evaluation quotas in many accounts.",
}

# Keys we manage so provider switches do not leak stale credentials between sessions.
MANAGED_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OLLAMA_API_KEY",
    "NVIDIA_API_KEY",
    "NIM_API_KEY",
}


@dataclass
class ProviderConfig:
    provider: str = "claude"
    auth_mode: str = "system"  # system | api_key
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None


def normalize_provider(provider: str | None) -> str:
    if not provider:
        return "claude"

    normalized = provider.strip().lower().replace("-", "_").replace(" ", "_")
    return PROVIDER_NAME_ALIASES.get(normalized, "claude")


def provider_label(provider: str | None) -> str:
    return PROVIDER_LABELS.get(normalize_provider(provider), "Claude (Anthropic)")


def provider_key_hint(provider: str | None) -> str:
    return PROVIDER_KEY_HINTS.get(normalize_provider(provider), "API key")


def provider_free_tier_note(provider: str | None) -> str:
    return PROVIDER_FREE_TIER_NOTES.get(normalize_provider(provider), "")


def provider_default_model(provider: str | None) -> str | None:
    return PROVIDER_DEFAULT_MODELS.get(normalize_provider(provider))


def provider_default_base_url(provider: str | None) -> str | None:
    return PROVIDER_DEFAULT_BASE_URLS.get(normalize_provider(provider))


def provider_recommended_models(provider: str | None) -> list[str]:
    return list(PROVIDER_RECOMMENDED_MODELS.get(normalize_provider(provider), []))


def resolve_model(config: ProviderConfig) -> str | None:
    explicit = (config.model or "").strip()
    if explicit:
        return explicit
    return provider_default_model(config.provider)


def resolve_base_url(config: ProviderConfig) -> str | None:
    explicit = (config.base_url or "").strip()
    if explicit:
        return explicit
    return provider_default_base_url(config.provider)


def requires_api_key(config: ProviderConfig) -> bool:
    provider = normalize_provider(config.provider)
    if provider == "claude" and config.auth_mode == "system":
        return False
    if provider == "ollama" and not (config.api_key or "").strip():
        return False
    return True


def validate_provider_config(config: ProviderConfig) -> tuple[bool, str]:
    provider = normalize_provider(config.provider)
    auth_mode = (config.auth_mode or "api_key").strip().lower()

    if provider not in SUPPORTED_PROVIDERS:
        return False, f"Unsupported provider: {provider}"

    if auth_mode not in {"system", "api_key"}:
        return False, "Authentication mode must be 'system' or 'api_key'"

    if auth_mode == "system" and provider != "claude":
        return False, "System authentication is only supported for Claude"

    if requires_api_key(config) and not (config.api_key or "").strip():
        return False, "This provider requires an API key/token"

    return True, ""


def build_provider_env(config: ProviderConfig) -> dict[str, str]:
    provider = normalize_provider(config.provider)
    auth_mode = (config.auth_mode or "api_key").strip().lower()
    api_key = (config.api_key or "").strip()
    base_url = resolve_base_url(config)

    env: dict[str, str] = {}

    if provider == "claude":
        if auth_mode == "api_key" and api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url

    elif provider == "gemini":
        if api_key:
            env["GOOGLE_API_KEY"] = api_key
            env["GEMINI_API_KEY"] = api_key
            env["OPENAI_API_KEY"] = api_key
        if base_url:
            env["OPENAI_BASE_URL"] = base_url

    elif provider == "openai":
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        if base_url:
            env["OPENAI_BASE_URL"] = base_url

    elif provider == "openrouter":
        if api_key:
            env["OPENROUTER_API_KEY"] = api_key
            env["OPENAI_API_KEY"] = api_key
        if base_url:
            env["OPENAI_BASE_URL"] = base_url

    elif provider == "ollama":
        env["OPENAI_API_KEY"] = api_key or "ollama"
        if api_key:
            env["OLLAMA_API_KEY"] = api_key
        if base_url:
            env["OPENAI_BASE_URL"] = base_url

    elif provider == "nim":
        if api_key:
            env["NVIDIA_API_KEY"] = api_key
            env["NIM_API_KEY"] = api_key
            env["OPENAI_API_KEY"] = api_key
        if base_url:
            env["OPENAI_BASE_URL"] = base_url

    return env


def clear_managed_provider_env() -> None:
    for key in MANAGED_ENV_KEYS:
        os.environ.pop(key, None)


def apply_provider_environment(config: ProviderConfig) -> dict[str, str]:
    clear_managed_provider_env()
    env_updates = build_provider_env(config)
    os.environ.update(env_updates)
    return env_updates


def describe_provider(config: ProviderConfig) -> str:
    label = provider_label(config.provider)
    model = resolve_model(config)
    if model:
        return f"{label} · {model}"
    return label
