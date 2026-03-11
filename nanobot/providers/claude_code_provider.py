"""Claude Code CLI provider — delegates the full agent loop to Claude Code via claude-agent-sdk."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, Literal

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.providers.base import LLMProvider, LLMResponse

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus
    from nanobot.cron.service import CronService


class ClaudeCodeProvider(LLMProvider):
    """LLM provider that runs Claude Code CLI as the agentic backend.

    Unlike LiteLLM/CustomProvider, this provider does NOT participate in
    nanobot's tool-call loop.  Instead it delegates the entire agent loop
    to Claude Code CLI via claude-agent-sdk.  The response always has
    tool_calls=[] so AgentLoop exits after a single iteration.

    nanobot-specific tools (message, cron) are exposed to Claude Code
    via an in-process SDK MCP server when bridge_* options are enabled.
    """

    def __init__(
        self,
        cli_path: str = "",
        default_model: str = "claude-opus-4-5",
        permission_mode: Literal[
            "default", "acceptEdits", "bypassPermissions"
        ] = "bypassPermissions",
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
        cwd: str | None = None,
        max_turns: int | None = None,
        env: dict[str, str] | None = None,
        bridge_message_tool: bool = True,
        bridge_cron_tool: bool = True,
        resume_sessions: bool = True,
        bus: "MessageBus | None" = None,
        cron_service: "CronService | None" = None,
    ) -> None:
        super().__init__(api_key=None, api_base=None)
        self.cli_path = cli_path or None  # empty string → None → SDK auto-discovery
        self.default_model = default_model
        self.permission_mode = permission_mode
        self.allowed_tools = allowed_tools or []
        self.disallowed_tools = disallowed_tools or []
        self.system_prompt = system_prompt or None
        self.cwd = cwd or None
        self.max_turns = max_turns
        self.env = env or {}
        self.bridge_message_tool = bridge_message_tool
        self.bridge_cron_tool = bridge_cron_tool
        self.resume_sessions = resume_sessions
        self._bus = bus
        self._cron_service = cron_service

        # Per-turn runtime context (updated by set_turn_context before each chat())
        self._current_channel: str = ""
        self._current_chat_id: str = ""

        # Session store callbacks injected by AgentLoop; backed by nanobot Session.metadata.
        # reader(session_key) -> cc_session_id | None
        # writer(session_key, cc_session_id) -> None
        self._session_reader: Callable[[str], str | None] | None = None
        self._session_writer: Callable[[str, str], None] | None = None

        # Built once; re-built if bridge settings change (they don't at runtime)
        self._mcp_server = (
            self._build_mcp_server() if (bridge_message_tool or bridge_cron_tool) else None
        )

        # Per-session worker tasks.  Each worker task owns one ClaudeSDKClient
        # (connect + disconnect happen inside the same task to satisfy anyio's
        # cancel-scope rule).  chat() communicates with the worker via a Queue.
        self._session_workers: dict[str, asyncio.Task] = {}
        self._prompt_queues: dict[str, asyncio.Queue] = {}
        # Tracks which session_keys have already had their CC session_id saved to
        # metadata (written once, on the first successful turn).
        self._session_id_saved: set[str] = set()

    # ------------------------------------------------------------------
    # Context injection (called by AgentLoop._set_tool_context)
    # ------------------------------------------------------------------

    @property
    def _current_session_key(self) -> str:
        return f"{self._current_channel}:{self._current_chat_id}"

    def set_turn_context(self, channel: str, chat_id: str) -> None:
        """Update per-turn delivery context for bridged MCP tools."""
        self._current_channel = channel
        self._current_chat_id = chat_id

    def set_session_store(
        self,
        reader: Callable[[str], str | None],
        writer: Callable[[str, str], None],
    ) -> None:
        """Inject persistent session store callbacks.

        reader(session_key) -> session_id | None
        writer(session_key, session_id) -> None
        """
        self._session_reader = reader
        self._session_writer = writer

    # ------------------------------------------------------------------
    # Session worker — one long-lived asyncio Task per session_key
    # ------------------------------------------------------------------

    async def _session_worker(
        self,
        session_key: str,
        options: Any,
        connect_future: "asyncio.Future[None]",
        prompt_queue: "asyncio.Queue[tuple[str, asyncio.Future[LLMResponse]] | None]",
    ) -> None:
        """Worker task that owns a ClaudeSDKClient for the lifetime of one nanobot session.

        connect() and disconnect() both run inside this task, satisfying anyio's
        requirement that cancel scopes are entered and exited in the same task.
        chat() communicates via prompt_queue; each item is (prompt_str, response_future).
        A None item signals shutdown (/new or process exit).
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
        )

        client = ClaudeSDKClient(options)
        try:
            await client.connect()
            if not connect_future.done():
                connect_future.set_result(None)
            logger.info("cc worker: connected (session_key={})", session_key)
        except Exception as exc:
            if not connect_future.done():
                connect_future.set_exception(exc)
            logger.error("cc worker: connect failed (session_key={}): {}", session_key, exc)
            self._session_workers.pop(session_key, None)
            self._prompt_queues.pop(session_key, None)
            return  # worker exits; no disconnect needed (connect never succeeded)

        try:
            while True:
                item = await prompt_queue.get()
                if item is None:  # shutdown signal from close_session()
                    logger.info("cc worker: shutdown signal (session_key={})", session_key)
                    break

                prompt_str, response_future = item
                logger.debug("cc worker: processing prompt ({} chars)", len(prompt_str))

                text_parts: list[str] = []
                thinking_parts: list[str] = []
                new_session_id: str | None = None
                is_error = False

                try:
                    await client.query(prompt_str)
                    msg_count = 0
                    async for msg in client.receive_response():
                        msg_count += 1
                        logger.debug(
                            "cc worker: msg #{} type={}", msg_count, type(msg).__name__
                        )
                        if isinstance(msg, AssistantMessage):
                            for block in msg.content:
                                if isinstance(block, TextBlock):
                                    text_parts.append(block.text)
                                elif isinstance(block, ThinkingBlock):
                                    thinking_parts.append(block.thinking)
                        elif isinstance(msg, ResultMessage):
                            new_session_id = msg.session_id
                            is_error = msg.is_error
                            logger.debug(
                                "cc worker: ResultMessage session_id={} is_error={}",
                                new_session_id,
                                is_error,
                            )

                    logger.debug(
                        "cc worker: done — {} msgs, {} texts, {} thoughts",
                        msg_count,
                        len(text_parts),
                        len(thinking_parts),
                    )

                    # Persist CC session_id once so nanobot can resume after restart.
                    if new_session_id and self.resume_sessions and self._session_writer:
                        if session_key not in self._session_id_saved:
                            self._session_writer(session_key, new_session_id)
                            self._session_id_saved.add(session_key)
                            logger.debug(
                                "cc worker: saved session_id={} (first turn)", new_session_id
                            )
                    logger.info("cc session id: {}", new_session_id)

                    if is_error:
                        logger.error(
                            "cc worker: is_error=True, text={!r}",
                            "\n".join(text_parts)[:500],
                        )
                        result = LLMResponse(
                            content="\n".join(text_parts)
                            or f"Claude Code error (session={new_session_id})",
                            finish_reason="error",
                        )
                    else:
                        result = LLMResponse(
                            content="\n".join(text_parts) or None,
                            tool_calls=[],
                            finish_reason="stop",
                            reasoning_content="\n".join(thinking_parts) or None,
                        )

                    if not response_future.done():
                        response_future.set_result(result)

                except Exception as exc:
                    logger.exception("cc worker: error processing prompt: {}", exc)
                    if not response_future.done():
                        response_future.set_result(
                            LLMResponse(content=f"Error: {exc}", finish_reason="error")
                        )
                    raise  # unrecoverable; exit worker loop

        except Exception as exc:
            logger.exception(
                "cc worker: fatal error, exiting (session_key={}): {}", session_key, exc
            )
            # Drain any items queued after the fatal error and resolve their futures.
            while True:
                try:
                    item = prompt_queue.get_nowait()
                    if item is not None:
                        _, future = item
                        if not future.done():
                            future.set_result(
                                LLMResponse(
                                    content=f"cc worker died: {exc}", finish_reason="error"
                                )
                            )
                except asyncio.QueueEmpty:
                    break

        finally:
            self._session_workers.pop(session_key, None)
            self._prompt_queues.pop(session_key, None)
            self._session_id_saved.discard(session_key)
            try:
                await client.disconnect()
                logger.info("cc worker: disconnected (session_key={})", session_key)
            except Exception as exc:
                logger.warning(
                    "cc worker: disconnect error (session_key={}): {}", session_key, exc
                )

    async def _ensure_worker(
        self, session_key: str, make_options: Callable[[], Any]
    ) -> None:
        """Start a session worker for session_key if one is not already running.

        Raises if the worker fails to connect (e.g. stale resume ID).
        make_options() is called only when a new worker is needed.
        """
        existing = self._session_workers.get(session_key)
        if existing is not None and not existing.done():
            return  # healthy worker already running

        # Build options in the current task context (captures resume_id closure).
        options = make_options()
        connect_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        prompt_queue: asyncio.Queue = asyncio.Queue()

        self._prompt_queues[session_key] = prompt_queue

        task = asyncio.create_task(
            self._session_worker(session_key, options, connect_future, prompt_queue)
        )
        self._session_workers[session_key] = task
        logger.info("cc worker: started (session_key={})", session_key)

        # Wait for connect() to succeed or fail.
        await connect_future  # raises on connect failure

    async def close_session(self, session_key: str) -> None:
        """Shut down the session worker for session_key.

        Sends a shutdown signal via the prompt queue so the worker's
        disconnect() runs inside the worker task itself (avoiding the anyio
        cross-task cancel-scope error).
        """
        prompt_queue = self._prompt_queues.get(session_key)
        worker_task = self._session_workers.get(session_key)

        if prompt_queue is not None:
            await prompt_queue.put(None)  # poison pill

        if worker_task is not None and not worker_task.done():
            try:
                await worker_task
            except Exception:
                pass  # worker may have exited with an error

        # Belt-and-suspenders cleanup (worker's finally also removes these).
        self._session_workers.pop(session_key, None)
        self._prompt_queues.pop(session_key, None)
        self._session_id_saved.discard(session_key)
        logger.info("cc session: closed (session_key={})", session_key)

    # ------------------------------------------------------------------
    # In-process MCP server for nanobot-unique tools
    # ------------------------------------------------------------------

    def _build_mcp_server(self):
        """Build an in-process SDK MCP server exposing nanobot tools to Claude Code."""
        try:
            from claude_agent_sdk import create_sdk_mcp_server, tool
        except ImportError:
            return None

        sdk_tools = []

        if self.bridge_message_tool and self._bus is not None:

            @tool(
                "nanobot_message",
                "Send a message to the user via the current nanobot channel (Telegram, Discord, etc.).",
                {
                    "content": str,
                    "channel": str,
                    "chat_id": str,
                    "media": list,
                },
            )
            async def _message_tool(args: dict[str, Any]) -> dict[str, Any]:
                from nanobot.bus.events import OutboundMessage

                channel = args.get("channel") or self._current_channel
                chat_id = args.get("chat_id") or self._current_chat_id
                media = args.get("media") or []
                if not channel or not chat_id:
                    return {
                        "content": [{"type": "text", "text": "Error: no channel/chat_id context"}],
                        "is_error": True,
                    }
                try:
                    await self._bus.publish_outbound(
                        OutboundMessage(
                            channel=channel,
                            chat_id=chat_id,
                            content=args.get("content", ""),
                            media=media,
                        )
                    )
                    return {
                        "content": [
                            {"type": "text", "text": f"Message sent to {channel}:{chat_id}"}
                        ]
                    }
                except Exception as e:
                    return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}

            sdk_tools.append(_message_tool)

        if self.bridge_cron_tool and self._cron_service is not None:

            @tool(
                "nanobot_cron",
                (
                    "Schedule reminders and recurring tasks via nanobot. "
                    "Actions: add, list, remove. "
                    "For add: provide message and one of every_seconds / cron_expr / at (ISO datetime)."
                ),
                {
                    "action": str,
                    "message": str,
                    "every_seconds": int,
                    "cron_expr": str,
                    "tz": str,
                    "at": str,
                    "job_id": str,
                },
            )
            async def _cron_tool(args: dict[str, Any]) -> dict[str, Any]:
                from nanobot.cron.types import CronSchedule

                action = args.get("action", "")
                channel = self._current_channel
                chat_id = self._current_chat_id

                if action == "list":
                    jobs = self._cron_service.list_jobs()
                    if not jobs:
                        text = "No scheduled jobs."
                    else:
                        text = "Scheduled jobs:\n" + "\n".join(
                            f"- {j.name} (id: {j.id}, {j.schedule.kind})" for j in jobs
                        )
                    return {"content": [{"type": "text", "text": text}]}

                if action == "remove":
                    job_id = args.get("job_id")
                    if not job_id:
                        return {
                            "content": [{"type": "text", "text": "Error: job_id required"}],
                            "is_error": True,
                        }
                    ok = self._cron_service.remove_job(job_id)
                    text = f"Removed job {job_id}" if ok else f"Job {job_id} not found"
                    return {"content": [{"type": "text", "text": text}]}

                if action == "add":
                    message = args.get("message", "")
                    if not message:
                        return {
                            "content": [{"type": "text", "text": "Error: message required"}],
                            "is_error": True,
                        }
                    if not channel or not chat_id:
                        return {
                            "content": [{"type": "text", "text": "Error: no session context"}],
                            "is_error": True,
                        }
                    every_seconds = args.get("every_seconds")
                    cron_expr = args.get("cron_expr")
                    tz = args.get("tz")
                    at = args.get("at")
                    delete_after = False
                    if every_seconds:
                        schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
                    elif cron_expr:
                        schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
                    elif at:
                        from datetime import datetime

                        try:
                            dt = datetime.fromisoformat(at)
                        except ValueError:
                            return {
                                "content": [
                                    {"type": "text", "text": f"Error: invalid datetime '{at}'"}
                                ],
                                "is_error": True,
                            }
                        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
                        delete_after = True
                    else:
                        return {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Error: every_seconds, cron_expr, or at required",
                                }
                            ],
                            "is_error": True,
                        }
                    job = self._cron_service.add_job(
                        name=message[:30],
                        schedule=schedule,
                        message=message,
                        deliver=True,
                        channel=channel,
                        to=chat_id,
                        delete_after_run=delete_after,
                    )
                    return {
                        "content": [
                            {"type": "text", "text": f"Created job '{job.name}' (id: {job.id})"}
                        ]
                    }

                return {
                    "content": [{"type": "text", "text": f"Unknown action: {action}"}],
                    "is_error": True,
                }

            sdk_tools.append(_cron_tool)

        if not sdk_tools:
            return None
        return create_sdk_mcp_server("nanobot", version="1.0.0", tools=sdk_tools)

    # ------------------------------------------------------------------
    # Message extraction helpers
    # ------------------------------------------------------------------

    _RUNTIME_CONTEXT_TAG = ContextBuilder._RUNTIME_CONTEXT_TAG

    @classmethod
    def _strip_runtime_context(cls, text: str) -> str:
        """Strip nanobot's runtime context prefix from a user message string."""
        if text.startswith(cls._RUNTIME_CONTEXT_TAG):
            parts = text.split("\n\n", 1)
            return parts[1] if len(parts) > 1 else ""
        return text

    _IMAGE_SIZE_LIMIT = 1024 * 1024  # 1 MB raw bytes; compress if larger

    @staticmethod
    def _compress_image(raw: bytes, media_type: str) -> tuple[bytes, str]:
        """Return (possibly compressed) bytes and the resulting media_type.

        If raw size <= _IMAGE_SIZE_LIMIT the original bytes are returned unchanged.
        Otherwise the image is re-encoded as JPEG with progressively lower quality
        until it fits under the limit (minimum quality 30).
        """
        if len(raw) <= ClaudeCodeProvider._IMAGE_SIZE_LIMIT:
            return raw, media_type

        try:
            from io import BytesIO

            from PIL import Image

            img = Image.open(BytesIO(raw))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            for quality in (85, 70, 55, 40, 30):
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                compressed = buf.getvalue()
                logger.debug(
                    "Compressed image: {} KB → {} KB (quality={})",
                    len(raw) // 1024,
                    len(compressed) // 1024,
                    quality,
                )
                if len(compressed) <= ClaudeCodeProvider._IMAGE_SIZE_LIMIT:
                    return compressed, "image/jpeg"

            # Even at quality=30 still too large — return smallest result
            return compressed, "image/jpeg"
        except Exception as exc:
            logger.warning("Image compression failed ({}); using original", exc)
            return raw, media_type

    @staticmethod
    def _image_url_to_anthropic(block: dict[str, Any]) -> dict[str, Any] | None:
        """Convert an OpenAI-style image_url block to Anthropic's native image format.

        Compresses the image if it exceeds _IMAGE_SIZE_LIMIT before encoding.

        nanobot's ContextBuilder encodes images as:
          {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}

        Claude Code CLI (Anthropic API) expects:
          {"type": "image", "source": {"type": "base64", "media_type": "<mime>", "data": "<data>"}}
        """
        import base64

        url = block.get("image_url", {}).get("url", "")
        if not url.startswith("data:"):
            return None
        try:
            meta, b64data = url.split(",", 1)
            media_type = meta.split(":")[1].split(";")[0]
        except (ValueError, IndexError):
            return None

        raw = base64.b64decode(b64data)
        raw, media_type = ClaudeCodeProvider._compress_image(raw, media_type)
        data = base64.b64encode(raw).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }

    @classmethod
    def _extract_content(cls, messages: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        """Return the last user-role message content, preserving image blocks.

        Returns a plain string if there are no images, or a list of content blocks
        in Anthropic native format (image + text) when images are present.
        """
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return cls._strip_runtime_context(content)
            if isinstance(content, list):
                result: list[dict[str, Any]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "image_url":
                        converted = cls._image_url_to_anthropic(block)
                        if converted:
                            result.append(converted)
                    elif btype in ("text", "input_text"):
                        text = block.get("text", "")
                        stripped = cls._strip_runtime_context(text)
                        if stripped:
                            result.append({"type": "text", "text": stripped})
                if not result:
                    return ""
                if all(b.get("type") == "text" for b in result):
                    return "\n".join(b.get("text", "") for b in result)
                return result
        return ""

    @classmethod
    def _extract_prompt(cls, messages: list[dict[str, Any]]) -> str:
        """Return the last user-role message content as a plain string, without runtime context prefix."""
        content = cls._extract_content(messages)
        if isinstance(content, list):
            return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
        return content

    @staticmethod
    def _extract_system(messages: list[dict[str, Any]]) -> str | None:
        """Return the first system-role message content, or None."""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content or None
                if isinstance(content, list):
                    parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") in ("text", "input_text")
                    ]
                    text = "\n".join(p for p in parts if p)
                    return text or None
        return None

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        try:
            from claude_agent_sdk import ClaudeAgentOptions
        except ImportError:
            return LLMResponse(
                content="Error: claude-agent-sdk is not installed. Run: pip install claude-agent-sdk",
                finish_reason="error",
            )

        logger.debug(
            "cc chat(): {} messages, roles={}",
            len(messages),
            [m.get("role") for m in messages],
        )

        content = self._extract_content(messages)
        if not content:
            logger.warning("cc chat(): no user message found in {} messages", len(messages))
            return LLMResponse(
                content="Error: no user message found in context", finish_reason="error"
            )

        if isinstance(content, list):
            logger.debug("cc chat(): multimodal content, {} blocks", len(content))
        else:
            logger.debug("cc chat(): prompt ({} chars): {!r}", len(content), content[:200])

        # Use the system message from the conversation as the system prompt for Claude Code SDK.
        # Config-level system_prompt is a fallback when no system message is present.
        system = self._extract_system(messages) or self.system_prompt or None
        logger.debug(
            "cc chat(): system prompt ({} chars): {!r}",
            len(system or ""),
            (system or "")[:200],
        )

        session_key = self._current_session_key
        is_fresh = (
            session_key not in self._session_workers
            or self._session_workers[session_key].done()
        )
        logger.debug(
            "cc chat(): session_key={!r}, is_fresh={}, resume_sessions={}, reader={}",
            session_key,
            is_fresh,
            self.resume_sessions,
            self._session_reader is not None,
        )

        # resume_id is only needed when spinning up a fresh worker (e.g. after restart).
        # Once a worker is live, the subprocess holds the full conversation context.
        resume_id: str | None = None
        if is_fresh and self.resume_sessions and session_key and self._session_reader:
            resume_id = self._session_reader(session_key) or None

        if not is_fresh:
            logger.info("cc session: reusing worker (session_key={})", session_key)
        elif resume_id:
            logger.info(
                "cc session: new worker with resume={} (session_key={})",
                resume_id,
                session_key,
            )
        else:
            logger.info("cc session: new worker (session_key={})", session_key)

        def _make_options() -> ClaudeAgentOptions:
            """Build ClaudeAgentOptions for the worker's initial connect()."""
            kwargs: dict[str, Any] = dict(
                model=model or self.default_model,
                permission_mode=self.permission_mode,
                env=self.env,
                stderr=lambda line: logger.warning("claude-cli stderr: {}", line),
            )
            if self.cli_path:
                kwargs["cli_path"] = self.cli_path
            if system:
                kwargs["system_prompt"] = system
            if self.cwd:
                kwargs["cwd"] = self.cwd
            if self.max_turns is not None:
                kwargs["max_turns"] = self.max_turns
            if self.allowed_tools:
                kwargs["allowed_tools"] = self.allowed_tools
            if self.disallowed_tools:
                kwargs["disallowed_tools"] = self.disallowed_tools
            if resume_id:
                kwargs["resume"] = resume_id
            if self._mcp_server is not None:
                kwargs["mcp_servers"] = {"nanobot": self._mcp_server}
            logger.debug(
                "cc options: model={}, permission_mode={}, cwd={}, max_turns={}, "
                "resume={}, mcp={}, allowed_tools={}, disallowed_tools={}",
                kwargs.get("model"),
                kwargs.get("permission_mode"),
                kwargs.get("cwd"),
                kwargs.get("max_turns"),
                bool(resume_id),
                self._mcp_server is not None,
                kwargs.get("allowed_tools"),
                kwargs.get("disallowed_tools"),
            )
            return ClaudeAgentOptions(**kwargs)

        # Build the prompt string. If content contains image blocks, write each image
        # to a temporary file and reference it via @filename (the approach recommended
        # in https://github.com/anthropics/claude-agent-sdk-python/issues/44).
        # Claude Code CLI does not support multimodal content blocks in the
        # --input-format stream-json protocol, but @filename injects image content
        # directly into the prompt without requiring the Read tool.
        tmp_paths: list[str] = []
        if isinstance(content, list):
            import base64
            import os
            import tempfile
            from pathlib import Path

            at_refs: list[str] = []
            text_parts_inline: list[str] = []
            for block in content:
                if block.get("type") == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        ext = (
                            src.get("media_type", "image/jpeg")
                            .split("/")[-1]
                            .replace("jpeg", "jpg")
                        )
                        raw = base64.b64decode(src["data"])
                        fd, path = tempfile.mkstemp(suffix=f".{ext}")
                        os.close(fd)
                        Path(path).write_bytes(raw)
                        tmp_paths.append(path)
                        at_refs.append(f"@{path}")
                elif block.get("type") == "text" and block.get("text"):
                    text_parts_inline.append(block["text"])
            prompt_str: str = " ".join(at_refs + text_parts_inline)
        else:
            prompt_str = content

        # Ensure the session worker is running (starts it if needed).
        try:
            await self._ensure_worker(session_key, _make_options)
        except Exception as e:
            logger.error("cc chat(): worker connect failed: {}", e)
            # Clear stale session ID so the next turn doesn't retry the same bad resume.
            if self._session_writer and session_key:
                self._session_writer(session_key, "")
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass
            return LLMResponse(
                content=f"Error connecting to Claude Code: {e}", finish_reason="error"
            )

        # Deliver the prompt to the worker and await its response via a Future.
        response_future: asyncio.Future[LLMResponse] = (
            asyncio.get_running_loop().create_future()
        )
        try:
            await self._prompt_queues[session_key].put((prompt_str, response_future))
            return await response_future
        except Exception as e:
            logger.exception("cc chat(): error awaiting response: {}", e)
            return LLMResponse(content=f"Error: {e}", finish_reason="error")
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass

    async def consolidate_memory(
        self,
        session: Any,
        memory_store: Any,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> "bool | None":
        """Run memory consolidation via a fresh CC session with a one-off MCP server."""
        try:
            from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query, tool
        except ImportError:
            return None  # fall back to standard path

        # Replicate early-exit logic from MemoryStore.consolidate()
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(
                "ClaudeCodeProvider consolidate_memory (archive_all): {} messages",
                len(session.messages),
            )
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated : -keep_count]
            if not old_messages:
                return True
            logger.info(
                "ClaudeCodeProvider consolidate_memory: {} to consolidate, {} keep",
                len(old_messages),
                keep_count,
            )

        # Build conversation text (same format as memory.py lines 97-103)
        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(
                f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}"
            )

        current_memory = memory_store.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        saved = False

        # Build a one-off MCP server with a single save_memory tool
        @tool(
            "save_memory",
            "Save the memory consolidation result to persistent storage.",
            {
                "history_entry": str,
                "memory_update": str,
            },
        )
        async def _save_memory_tool(args: dict[str, Any]) -> dict[str, Any]:
            nonlocal saved
            try:
                entry = args.get("history_entry", "")
                update = args.get("memory_update", "")
                if entry:
                    memory_store.append_history(entry)
                if update and update != current_memory:
                    memory_store.write_long_term(update)
                saved = True
                return {"content": [{"type": "text", "text": "Memory saved successfully."}]}
            except Exception as e:
                return {
                    "content": [{"type": "text", "text": f"Error saving memory: {e}"}],
                    "is_error": True,
                }

        mcp_server = create_sdk_mcp_server(
            "nanobot_memory", version="1.0.0", tools=[_save_memory_tool]
        )

        kwargs: dict[str, Any] = dict(
            model=self.default_model,
            permission_mode=self.permission_mode,
            system_prompt="You are a memory consolidation agent. Call the save_memory tool with your consolidation of the conversation.",
            mcp_servers={"nanobot_memory": mcp_server},
            allowed_tools=["mcp__nanobot_memory__save_memory"],
            max_turns=5,
            env=self.env,
            stderr=lambda line: logger.debug("claude-cli consolidate stderr: {}", line),
        )
        if self.cli_path:
            kwargs["cli_path"] = self.cli_path
        if self.cwd:
            kwargs["cwd"] = self.cwd

        options = ClaudeAgentOptions(**kwargs)

        try:
            async for _ in query(prompt=prompt, options=options):
                pass
        except Exception:
            logger.exception("ClaudeCodeProvider consolidate_memory: CC session failed")
            return False

        if saved:
            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info(
                "ClaudeCodeProvider consolidate_memory done: last_consolidated={}",
                session.last_consolidated,
            )
        else:
            logger.warning("ClaudeCodeProvider consolidate_memory: CC did not call save_memory")

        return saved

    def get_default_model(self) -> str:
        return self.default_model
