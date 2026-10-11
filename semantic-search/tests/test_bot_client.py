import asyncio
import json

import httpx
import pytest

from app.bot.api_client import ApiRejected, ApiUnavailable, QAClient


def run(coro):
    return asyncio.run(coro)


def client_with(handler) -> QAClient:
    return QAClient("http://api", transport=httpx.MockTransport(handler))


def test_ask_sends_bot_source():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        return httpx.Response(200, json={"status": "answer"})

    assert run(client_with(handler).ask("soliq", top_k=3)) == {"status": "answer"}
    assert seen == {"path": "/ask", "body": {"question": "soliq", "top_k": 3, "source": "bot"}}


def test_answer_feedback_stats_paths():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(201 if request.url.path == "/feedback" else 200, json={"ok": 1})

    qa = client_with(handler)
    run(qa.answer(5))
    run(qa.feedback("q", 5, True))
    run(qa.stats())
    assert calls == [("GET", "/answers/5"), ("POST", "/feedback"), ("GET", "/stats")]


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")], ids=["connect", "timeout"]
)
def test_network_errors_are_unavailable(error):
    def handler(request):
        raise error

    with pytest.raises(ApiUnavailable):
        run(client_with(handler).ask("soliq"))


def test_server_error_is_unavailable_and_422_is_rejected():
    with pytest.raises(ApiUnavailable):
        run(client_with(lambda r: httpx.Response(503, text="down")).stats())
    rejected = client_with(lambda r: httpx.Response(422, json={"detail": [{"msg": "too long"}]}))
    with pytest.raises(ApiRejected):
        run(rejected.ask("x" * 600))
