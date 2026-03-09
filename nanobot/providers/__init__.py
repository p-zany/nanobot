"""LLM provider abstraction module."""

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import OpenAICodexProvider
from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

# ClaudeCodeProvider is imported lazily to avoid loading claude-agent-sdk when unused.
# Import directly from nanobot.providers.claude_code_provider if needed.

__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider", "OpenAICodexProvider", "AzureOpenAIProvider"]
