"""Tests for ClaudeCodeProvider (Phase 5 validation)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.providers.claude_code_provider import ClaudeCodeProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**kwargs) -> ClaudeCodeProvider:
    defaults = dict(
        bridge_message_tool=False,
        bridge_cron_tool=False,
        resume_sessions=True,
    )
    defaults.update(kwargs)
    return ClaudeCodeProvider(**defaults)


def _user_msg(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def _sys_msg(content: str) -> dict[str, Any]:
    return {"role": "system", "content": content}


async def _fake_query(texts: list[str], session_id: str = "sess-abc", is_error: bool = False):
    """Yield fake SDK messages matching a successful (or error) run."""
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    for t in texts:
        msg = AssistantMessage.__new__(AssistantMessage)
        block = TextBlock.__new__(TextBlock)
        object.__setattr__(block, "text", t)
        object.__setattr__(msg, "content", [block])
        yield msg

    result = ResultMessage.__new__(ResultMessage)
    object.__setattr__(result, "is_error", is_error)
    object.__setattr__(result, "session_id", session_id)
    yield result


# ---------------------------------------------------------------------------
# 5.1 Basic chat — no resume, no MCP
# ---------------------------------------------------------------------------


class TestBasicChat:
    @pytest.mark.asyncio
    async def test_returns_assistant_text(self):
        provider = _make_provider()
        provider.set_turn_context("cli", "user1")

        async def fake_gen(**_):
            return _fake_query(["Hello there!"])

        with patch(
            "claude_agent_sdk.query", side_effect=lambda **kw: _fake_query(["Hello there!"])
        ):
            response = await provider.chat([_user_msg("Hi")])

        assert response.content == "Hello there!"
        assert response.tool_calls == []
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_joined(self):
        provider = _make_provider()
        provider.set_turn_context("cli", "u1")

        with patch(
            "claude_agent_sdk.query", side_effect=lambda **kw: _fake_query(["Part A", "Part B"])
        ):
            response = await provider.chat([_user_msg("Go")])

        assert response.content == "Part A\nPart B"

    @pytest.mark.asyncio
    async def test_no_user_message_returns_error(self):
        provider = _make_provider()
        response = await provider.chat([_sys_msg("system only")])
        assert "no user message" in response.content.lower()
        assert response.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_import_error_returns_friendly_message(self):
        provider = _make_provider()
        provider.set_turn_context("cli", "u1")

        import builtins

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "claude_agent_sdk":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocking_import):
            response = await provider.chat([_user_msg("Hi")])

        assert "claude-agent-sdk" in response.content
        assert response.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_system_prompt_from_messages(self):
        provider = _make_provider()
        provider.set_turn_context("cli", "u1")

        captured_options = []

        def capture_query(*, prompt, options, **_):
            captured_options.append(options)
            return _fake_query(["ok"])

        with patch("claude_agent_sdk.query", side_effect=capture_query):
            await provider.chat([_sys_msg("Be helpful"), _user_msg("Hello")])

        assert captured_options
        assert captured_options[0].system_prompt == "Be helpful"

    @pytest.mark.asyncio
    async def test_config_system_prompt_used_when_no_message_system(self):
        """system_prompt from config is passed even without a system role message."""
        provider = _make_provider(system_prompt="You are an expert assistant")
        provider.set_turn_context("cli", "u1")

        captured_options = []

        def capture_query(*, prompt, options, **_):
            captured_options.append(options)
            return _fake_query(["ok"])

        with patch("claude_agent_sdk.query", side_effect=capture_query):
            await provider.chat([_user_msg("Hello")])

        assert captured_options[0].system_prompt == "You are an expert assistant"

    @pytest.mark.asyncio
    async def test_config_and_message_system_prompts_merged(self):
        """Config-level and message-level system prompts are combined."""
        provider = _make_provider(system_prompt="Config-level instruction")
        provider.set_turn_context("cli", "u1")

        captured_options = []

        def capture_query(*, prompt, options, **_):
            captured_options.append(options)
            return _fake_query(["ok"])

        with patch("claude_agent_sdk.query", side_effect=capture_query):
            await provider.chat([_sys_msg("Message-level instruction"), _user_msg("Hello")])

        sp = captured_options[0].system_prompt
        assert "Config-level instruction" in sp
        assert "Message-level instruction" in sp


# ---------------------------------------------------------------------------
# 5.2 Session resume — multi-turn context persistence
# ---------------------------------------------------------------------------


class TestSessionResume:
    @pytest.mark.asyncio
    async def test_session_id_persisted_after_first_turn(self):
        provider = _make_provider()
        provider.set_turn_context("tg", "chat42")

        with patch(
            "claude_agent_sdk.query",
            side_effect=lambda **kw: _fake_query(["Hi"], session_id="sid-1"),
        ):
            await provider.chat([_user_msg("Hello")])

        assert provider._session_map.get("tg:chat42") == "sid-1"

    @pytest.mark.asyncio
    async def test_resume_id_sent_on_second_turn(self):
        provider = _make_provider()
        provider.set_turn_context("tg", "chat42")
        provider._session_map["tg:chat42"] = "sid-prev"

        captured_options = []

        def capture_query(*, prompt, options, **_):
            captured_options.append(options)
            return _fake_query(["ok"], session_id="sid-prev")

        with patch("claude_agent_sdk.query", side_effect=capture_query):
            await provider.chat([_user_msg("Second message")])

        assert captured_options[0].resume == "sid-prev"

    @pytest.mark.asyncio
    async def test_stale_resume_cleared_and_retried(self):
        provider = _make_provider()
        provider.set_turn_context("tg", "chat42")
        provider._session_map["tg:chat42"] = "stale-sid"

        call_count = 0

        def query_with_error(*, prompt, options, **_):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (with stale resume) → is_error
                return _fake_query([], session_id="stale-sid", is_error=True)
            else:
                # Second call (fresh) → success
                return _fake_query(["Fresh response"], session_id="sid-new")

        with patch("claude_agent_sdk.query", side_effect=query_with_error):
            response = await provider.chat([_user_msg("Hello")])

        assert call_count == 2
        assert response.content == "Fresh response"
        assert response.finish_reason == "stop"
        # stale session cleared, new one stored
        assert provider._session_map.get("tg:chat42") == "sid-new"

    @pytest.mark.asyncio
    async def test_resume_disabled_never_stores_session(self):
        provider = _make_provider(resume_sessions=False)
        provider.set_turn_context("tg", "chat42")

        with patch(
            "claude_agent_sdk.query",
            side_effect=lambda **kw: _fake_query(["ok"], session_id="sid-x"),
        ):
            await provider.chat([_user_msg("Hi")])

        assert "tg:chat42" not in provider._session_map

    @pytest.mark.asyncio
    async def test_different_sessions_use_different_resume_ids(self):
        provider = _make_provider()
        provider._session_map["ch1:u1"] = "sid-A"
        provider._session_map["ch2:u2"] = "sid-B"

        captured = []

        def cap(*, prompt, options, **_):
            captured.append(options.resume)
            return _fake_query(["ok"], session_id=options.resume or "new")

        provider.set_turn_context("ch1", "u1")
        with patch("claude_agent_sdk.query", side_effect=cap):
            await provider.chat([_user_msg("msg")])

        provider.set_turn_context("ch2", "u2")
        with patch("claude_agent_sdk.query", side_effect=cap):
            await provider.chat([_user_msg("msg")])

        assert captured[0] == "sid-A"
        assert captured[1] == "sid-B"


# ---------------------------------------------------------------------------
# 5.3 nanobot_message tool bridge
# ---------------------------------------------------------------------------


class TestMessageToolBridge:
    def _provider_with_bus(self):
        bus = MagicMock()
        bus.publish_outbound = AsyncMock()
        provider = ClaudeCodeProvider(
            bridge_message_tool=True,
            bridge_cron_tool=False,
        )
        provider._bus = bus
        provider._mcp_server = provider._build_mcp_server()
        return provider, bus

    def test_mcp_server_built_when_bridge_enabled(self):
        provider, _ = self._provider_with_bus()
        assert provider._mcp_server is not None

    def test_mcp_server_none_when_no_bus(self):
        provider = ClaudeCodeProvider(bridge_message_tool=True, bridge_cron_tool=False)
        # No bus assigned → no tools → None
        assert provider._mcp_server is None

    @pytest.mark.asyncio
    async def test_message_tool_publishes_outbound(self):
        from nanobot.bus.events import OutboundMessage

        provider, bus = self._provider_with_bus()
        provider.set_turn_context("telegram", "chat99")

        # Extract the nanobot_message tool handler from the MCP server tools
        mcp_tools = (
            provider._mcp_server.server._tools if hasattr(provider._mcp_server, "server") else None
        )

        # Instead, call the inner async function directly by finding it in the tool list
        # We'll test via the registered handler stored in the server's tool_map
        # Simpler: rebuild and grab the handler
        from claude_agent_sdk import tool as sdk_tool

        handler_ref = []

        original_create = __import__("claude_agent_sdk").create_sdk_mcp_server

        def capturing_create(name, version="1.0.0", tools=None):
            if tools:
                handler_ref.extend(tools)
            return original_create(name, version=version, tools=tools)

        provider2, bus2 = self._provider_with_bus()
        bus2.publish_outbound = AsyncMock()

        with patch("claude_agent_sdk.create_sdk_mcp_server", side_effect=capturing_create):
            provider2._mcp_server = provider2._build_mcp_server()

        provider2.set_turn_context("telegram", "chat99")

        if handler_ref:
            msg_tool = next((t for t in handler_ref if t.name == "nanobot_message"), None)
            if msg_tool:
                result = await msg_tool.handler(
                    {
                        "content": "Hello from Claude Code!",
                        "channel": "",
                        "chat_id": "",
                        "media": [],
                    }
                )
                bus2.publish_outbound.assert_called_once()
                call_args = bus2.publish_outbound.call_args[0][0]
                assert isinstance(call_args, OutboundMessage)
                assert call_args.content == "Hello from Claude Code!"
                assert call_args.channel == "telegram"
                assert call_args.chat_id == "chat99"
                assert "is_error" not in result

    @pytest.mark.asyncio
    async def test_message_tool_uses_explicit_channel_if_provided(self):
        provider, bus = self._provider_with_bus()
        provider.set_turn_context("telegram", "chat99")

        handler_ref = []
        original_create = __import__("claude_agent_sdk").create_sdk_mcp_server

        def capturing_create(name, version="1.0.0", tools=None):
            if tools:
                handler_ref.extend(tools)
            return original_create(name, version=version, tools=tools)

        with patch("claude_agent_sdk.create_sdk_mcp_server", side_effect=capturing_create):
            provider._mcp_server = provider._build_mcp_server()

        from nanobot.bus.events import OutboundMessage

        msg_tool = next((t for t in handler_ref if t.name == "nanobot_message"), None)
        if msg_tool:
            await msg_tool.handler(
                {
                    "content": "Hi override",
                    "channel": "discord",
                    "chat_id": "ch123",
                    "media": [],
                }
            )
            call_args = bus.publish_outbound.call_args[0][0]
            assert call_args.channel == "discord"
            assert call_args.chat_id == "ch123"

    @pytest.mark.asyncio
    async def test_message_tool_error_when_no_context(self):
        provider, _ = self._provider_with_bus()
        # No set_turn_context called → channel/chat_id empty

        handler_ref = []
        original_create = __import__("claude_agent_sdk").create_sdk_mcp_server

        def capturing_create(name, version="1.0.0", tools=None):
            if tools:
                handler_ref.extend(tools)
            return original_create(name, version=version, tools=tools)

        with patch("claude_agent_sdk.create_sdk_mcp_server", side_effect=capturing_create):
            provider._mcp_server = provider._build_mcp_server()

        msg_tool = next((t for t in handler_ref if t.name == "nanobot_message"), None)
        if msg_tool:
            result = await msg_tool.handler(
                {"content": "test", "channel": "", "chat_id": "", "media": []}
            )
            assert result.get("is_error") is True


# ---------------------------------------------------------------------------
# 5.4 nanobot_cron tool bridge
# ---------------------------------------------------------------------------


class TestCronToolBridge:
    def _make_cron_provider(self):
        from nanobot.cron.service import CronService

        cron = MagicMock(spec=CronService)
        cron.list_jobs.return_value = []
        cron.remove_job.return_value = True
        job = MagicMock()
        job.name = "Reminder"
        job.id = "job-1"
        cron.add_job.return_value = job

        provider = ClaudeCodeProvider(
            bridge_message_tool=False,
            bridge_cron_tool=True,
        )
        provider._cron_service = cron
        provider._mcp_server = provider._build_mcp_server()
        return provider, cron

    def _get_cron_tool(self, provider):
        handler_ref = []
        original_create = __import__("claude_agent_sdk").create_sdk_mcp_server

        def capturing_create(name, version="1.0.0", tools=None):
            if tools:
                handler_ref.extend(tools)
            return original_create(name, version=version, tools=tools)

        with patch("claude_agent_sdk.create_sdk_mcp_server", side_effect=capturing_create):
            provider._mcp_server = provider._build_mcp_server()

        return next((t for t in handler_ref if t.name == "nanobot_cron"), None)

    @pytest.mark.asyncio
    async def test_cron_list_empty(self):
        provider, cron = self._make_cron_provider()
        cron_tool = self._get_cron_tool(provider)
        if cron_tool is None:
            pytest.skip("cron tool not captured")
        result = await cron_tool.handler({"action": "list"})
        text = result["content"][0]["text"]
        assert "No scheduled" in text

    @pytest.mark.asyncio
    async def test_cron_add_every_seconds(self):
        provider, cron = self._make_cron_provider()
        provider.set_turn_context("telegram", "chat1")
        cron_tool = self._get_cron_tool(provider)
        if cron_tool is None:
            pytest.skip("cron tool not captured")

        result = await cron_tool.handler(
            {
                "action": "add",
                "message": "Take a break",
                "every_seconds": 3600,
            }
        )
        cron.add_job.assert_called_once()
        text = result["content"][0]["text"]
        assert "job-1" in text

    @pytest.mark.asyncio
    async def test_cron_add_at_datetime(self):
        provider, cron = self._make_cron_provider()
        provider.set_turn_context("telegram", "chat1")
        cron_tool = self._get_cron_tool(provider)
        if cron_tool is None:
            pytest.skip("cron tool not captured")

        result = await cron_tool.handler(
            {
                "action": "add",
                "message": "Meeting reminder",
                "at": "2026-03-10T09:00:00",
            }
        )
        cron.add_job.assert_called_once()
        assert "job-1" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_cron_add_invalid_datetime(self):
        provider, cron = self._make_cron_provider()
        provider.set_turn_context("telegram", "chat1")
        cron_tool = self._get_cron_tool(provider)
        if cron_tool is None:
            pytest.skip("cron tool not captured")

        result = await cron_tool.handler(
            {
                "action": "add",
                "message": "Bad time",
                "at": "not-a-date",
            }
        )
        assert result.get("is_error") is True

    @pytest.mark.asyncio
    async def test_cron_remove(self):
        provider, cron = self._make_cron_provider()
        cron_tool = self._get_cron_tool(provider)
        if cron_tool is None:
            pytest.skip("cron tool not captured")

        result = await cron_tool.handler({"action": "remove", "job_id": "job-1"})
        cron.remove_job.assert_called_with("job-1")
        assert "Removed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_cron_unknown_action(self):
        provider, _ = self._make_cron_provider()
        cron_tool = self._get_cron_tool(provider)
        if cron_tool is None:
            pytest.skip("cron tool not captured")

        result = await cron_tool.handler({"action": "unknown"})
        assert "Unknown action" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# 5.5 Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_cli_exception_returns_error_response(self):
        from claude_agent_sdk import CLINotFoundError

        provider = _make_provider()
        provider.set_turn_context("cli", "u1")

        async def raising_query(**_):
            raise CLINotFoundError("claude not found")
            return  # make it a generator
            yield  # noqa: unreachable

        # Patch query to raise exception
        with patch("claude_agent_sdk.query", side_effect=CLINotFoundError("not found")):
            response = await provider.chat([_user_msg("Hello")])

        assert response.finish_reason == "error"
        assert "not found" in response.content.lower() or "Error" in response.content

    @pytest.mark.asyncio
    async def test_is_error_result_without_resume_returns_error(self):
        """Non-resume error in ResultMessage → error response (no retry)."""
        provider = _make_provider()
        provider.set_turn_context("cli", "u1")

        with patch(
            "claude_agent_sdk.query",
            side_effect=lambda **kw: _fake_query(
                ["Something went wrong"], session_id="sid-x", is_error=True
            ),
        ):
            response = await provider.chat([_user_msg("Hi")])

        assert response.finish_reason == "error"

    @pytest.mark.asyncio
    async def test_no_session_key_when_channel_empty(self):
        """Without set_turn_context, session_key is empty → no resume stored."""
        provider = _make_provider()

        with patch(
            "claude_agent_sdk.query",
            side_effect=lambda **kw: _fake_query(["ok"], session_id="sid-y"),
        ):
            await provider.chat([_user_msg("Hello")])

        # Empty session key should not be stored
        assert "" not in provider._session_map


# ---------------------------------------------------------------------------
# Message extraction unit tests
# ---------------------------------------------------------------------------


class TestExtractors:
    def test_extract_plain_user_message(self):
        msgs = [_user_msg("Hello world")]
        assert ClaudeCodeProvider._extract_prompt(msgs) == "Hello world"

    def test_extract_last_user_message(self):
        msgs = [_user_msg("First"), {"role": "assistant", "content": "reply"}, _user_msg("Second")]
        assert ClaudeCodeProvider._extract_prompt(msgs) == "Second"

    def test_strip_runtime_context_prefix(self):
        tag = ClaudeCodeProvider._RUNTIME_CONTEXT_TAG
        msg = f"{tag}\nchannel: cli\n\nActual user text here"
        result = ClaudeCodeProvider._strip_runtime_context(msg)
        assert result == "Actual user text here"

    def test_no_prefix_unchanged(self):
        msg = "plain message"
        assert ClaudeCodeProvider._strip_runtime_context(msg) == "plain message"

    def test_extract_system_message_string(self):
        msgs = [_sys_msg("You are helpful"), _user_msg("Hi")]
        assert ClaudeCodeProvider._extract_system(msgs) == "You are helpful"

    def test_extract_system_message_list_content(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "Be concise"}]},
            _user_msg("Hi"),
        ]
        assert ClaudeCodeProvider._extract_system(msgs) == "Be concise"

    def test_no_system_returns_none(self):
        msgs = [_user_msg("Hi")]
        assert ClaudeCodeProvider._extract_system(msgs) is None


# ---------------------------------------------------------------------------
# Config / property tests
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_session_key_property(self):
        p = _make_provider()
        p.set_turn_context("telegram", "99")
        assert p._current_session_key == "telegram:99"

    def test_cli_path_empty_string_becomes_none(self):
        p = _make_provider(cli_path="")
        assert p.cli_path is None

    def test_cli_path_set_when_provided(self):
        p = _make_provider(cli_path="/usr/local/bin/claude")
        assert p.cli_path == "/usr/local/bin/claude"

    def test_get_default_model(self):
        p = _make_provider(default_model="claude-opus-4-5")
        assert p.get_default_model() == "claude-opus-4-5"

    def test_mcp_server_none_when_bridge_disabled(self):
        p = _make_provider(bridge_message_tool=False, bridge_cron_tool=False)
        assert p._mcp_server is None
