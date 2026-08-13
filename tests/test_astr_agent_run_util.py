import asyncio
from types import SimpleNamespace

import pytest

from astrbot.core.agent.response import AgentResponse
from astrbot.core.astr_agent_run_util import (
    _build_tool_call_status_message,
    _build_tool_result_status_message,
    _record_tool_call_name,
    _simulated_stream_tts,
    run_agent,
)
from astrbot.core.message.components import Json
from astrbot.core.message.message_event_result import MessageChain


class _FakeEvent:
    """Minimal event surface used by the agent stream bridge."""

    def is_stopped(self) -> bool:
        return False

    def get_extra(self, key: str):
        del key
        return None

    def get_platform_name(self) -> str:
        return "test"


class _StreamingErrorRunner:
    """Agent runner that finishes with one provider error response."""

    streaming = True
    req = None

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text
        self.finished = False
        self.run_context = SimpleNamespace(context=SimpleNamespace(event=_FakeEvent()))

    async def step(self):
        self.finished = True
        yield AgentResponse(
            type="err",
            data={"chain": MessageChain().message(self.error_text)},
        )

    def done(self) -> bool:
        return self.finished


class _MalformedStreamingErrorRunner(_StreamingErrorRunner):
    """Agent runner that returns an invalid provider error payload."""

    async def step(self):
        self.finished = True
        yield AgentResponse(type="err", data={})


@pytest.mark.asyncio
async def test_run_agent_forwards_streaming_provider_error():
    error_text = (
        "LLM 响应错误: Not found the model k2.7-code-highspeed or Permission denied"
    )
    runner = _StreamingErrorRunner(error_text)

    chains = [chain async for chain in run_agent(runner)]

    assert len(chains) == 1
    assert chains[0].get_plain_text() == error_text


@pytest.mark.asyncio
async def test_run_agent_replaces_malformed_streaming_provider_error():
    runner = _MalformedStreamingErrorRunner("unused")

    chains = [chain async for chain in run_agent(runner)]

    assert len(chains) == 1
    assert chains[0].get_plain_text() == "Error occurred during AI execution."


def test_tool_call_status_includes_unique_nested_tool_slugs():
    tool_info = {
        "id": "call-1",
        "name": "multi_execute",
        "args": {
            "tools": [
                {"tool_slug": "GMAIL_SEND_EMAIL"},
                {"nested": {"tool_slug": "SLACK_SEND_MESSAGE"}},
                {"tool_slug": "GMAIL_SEND_EMAIL"},
            ]
        },
    }
    tool_names: dict[str, str] = {}

    _record_tool_call_name(tool_info, tool_names)

    expected_name = "multi_execute(GMAIL_SEND_EMAIL, SLACK_SEND_MESSAGE)"
    assert tool_names == {"call-1": expected_name}
    assert (
        _build_tool_call_status_message(tool_info)
        == f"🔨 Calling tool: {expected_name}"
    )


def test_tool_result_status_uses_recorded_display_name_and_english_labels():
    tool_names = {"call-1": "multi_execute(GMAIL_SEND_EMAIL)"}
    chain = MessageChain(chain=[Json(data={"id": "call-1", "result": "Email sent"})])

    status = _build_tool_result_status_message(chain, tool_names)

    assert status == (
        "🔨 Calling tool: multi_execute(GMAIL_SEND_EMAIL)\n📎 Result: Email sent"
    )
    assert tool_names == {}


def test_tool_call_status_handles_missing_tool_info():
    assert _build_tool_call_status_message(None) == "🔨 Calling tool..."


@pytest.mark.asyncio
async def test_simulated_stream_tts_leaves_audio_for_deferred_cleanup(tmp_path):
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"audio")

    class _TTSProvider:
        async def get_audio(self, text: str) -> str:
            assert text == "hello"
            return str(audio_path)

    text_queue: asyncio.Queue[str | None] = asyncio.Queue()
    audio_queue: asyncio.Queue[bytes | tuple[str, bytes] | None] = asyncio.Queue()
    await text_queue.put("hello")
    await text_queue.put(None)

    await _simulated_stream_tts(_TTSProvider(), text_queue, audio_queue)

    assert await audio_queue.get() == ("hello", b"audio")
    assert await audio_queue.get() is None
    assert audio_path.exists()
