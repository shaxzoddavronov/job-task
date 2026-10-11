import asyncio
import re
import string
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot import handlers, texts
from app.bot.api_client import ApiUnavailable
from app.bot.formatting import about_text, button_label, split_message, unknown_laws
from app.bot.handlers import QueryStore
from app.bot.keyboards import ExampleCb, FeedbackCb, SuggestCb, feedback_kb, suggestions_kb


def run(coro):
    return asyncio.run(coro)


# --- Formatting ------------------------------------------------------------------------------


def test_split_short_text_unchanged():
    assert split_message("salom") == ["salom"]


def test_split_long_text_respects_limit_and_keeps_words():
    text = "\n\n".join(" ".join(f"so'z{i}{j}." for j in range(80)) for i in range(20))
    parts = split_message(text, limit=500)
    assert len(parts) > 1 and all(0 < len(p) <= 500 for p in parts)
    assert " ".join(" ".join(parts).split()) == " ".join(text.split())


def test_split_handles_one_huge_word():
    parts = split_message("a" * 1200, limit=500)
    assert [len(p) for p in parts] == [500, 500, 200]


def test_button_label_trims():
    assert button_label("qisqa savol") == "qisqa savol"
    assert len(button_label("x" * 200)) == 60


def test_unknown_laws_lists_surfaces():
    result = {
        "entities": [{"type": "LAW", "surface": "PQ-9999"}, {"type": "ORG", "surface": "DXM"}]
    }
    assert unknown_laws(result) == "PQ-9999"


def test_about_text_with_and_without_metrics():
    stats = {
        "eval": {
            "realistic": {"top1": 0.673, "top3": 0.822},
            "ood": {"not_found_rate": 0.913},
            "test": {"latency_p50_ms": 5.2},
        }
    }
    text = about_text(stats)
    assert "67%" in text and "82%" in text and "91%" in text and "5 ms" in text
    assert texts.ABOUT_NO_METRICS in about_text({"eval": None})


def test_texts_have_no_stray_placeholders():
    fields = {
        name: {f for _, f, _, _ in string.Formatter().parse(value) if f}
        for name, value in vars(texts).items()
        if isinstance(value, str)
    }
    assert fields["UNKNOWN_LAW"] == {"laws"}
    assert fields["ABOUT"] == {"metrics"}
    assert all(
        not f for name, f in fields.items() if name not in {"UNKNOWN_LAW", "ABOUT", "ABOUT_METRICS"}
    )


# --- Keyboards / store -----------------------------------------------------------------------


def test_callback_data_fits_telegram_limit():
    for cb in (
        ExampleCb(idx=99),
        SuggestCb(answer_id=99999, qid=10**9),
        FeedbackCb(helpful=False, answer_id=99999, qid=10**9),
    ):
        assert len(cb.pack().encode()) <= 64


def test_keyboards():
    cands = [{"answer_id": i, "question": f"savol {i}", "preview": "p"} for i in range(5)]
    kb = suggestions_kb(cands, qid=7)
    assert [row[0].text for row in kb.inline_keyboard] == ["savol 0", "savol 1", "savol 2"]
    assert SuggestCb.unpack(kb.inline_keyboard[1][0].callback_data) == SuggestCb(answer_id=1, qid=7)
    fb = feedback_kb(answer_id=3, qid=7).inline_keyboard[0]
    assert FeedbackCb.unpack(fb[0].callback_data).helpful is True
    assert FeedbackCb.unpack(fb[1].callback_data).helpful is False


def test_query_store_is_bounded():
    store = QueryStore(max_size=2)
    first = store.put("a")
    store.put("b")
    third = store.put("c")
    assert store.get(first) is None and store.get(third) == "c"


# --- Handlers --------------------------------------------------------------------------------


def fake_message(text="savol"):
    message = MagicMock()
    message.text = text
    message.bot = None
    message.answer = AsyncMock()
    message.edit_reply_markup = AsyncMock()
    return message


def fake_qa(**responses):
    qa = MagicMock()
    for name in ("ask", "answer", "feedback", "stats"):
        value = responses.get(name, {})
        setattr(
            qa,
            name,
            AsyncMock(side_effect=value)
            if isinstance(value, Exception)
            else AsyncMock(return_value=value),
        )
    return qa


def sent(message):
    return [(c.args[0], c.kwargs.get("reply_markup")) for c in message.answer.call_args_list]


def test_answer_status_sends_answer_with_feedback_buttons():
    qa = fake_qa(ask={"status": "answer", "answer": "Javob matni", "answer_id": 5})
    message = fake_message("ID-karta muddati")
    run(handlers.on_text(message, qa=qa, queries=QueryStore()))
    ((text, markup),) = sent(message)
    assert text == "Javob matni"
    assert FeedbackCb.unpack(markup.inline_keyboard[0][0].callback_data).answer_id == 5


def test_long_answer_is_split_and_buttons_go_last():
    qa = fake_qa(ask={"status": "answer", "answer": "So'z. " * 900, "answer_id": 5})
    message = fake_message()
    run(handlers.on_text(message, qa=qa, queries=QueryStore()))
    messages = sent(message)
    assert len(messages) == 2 and messages[0][1] is None and messages[1][1] is not None


def test_suggest_status_sends_buttons():
    cands = [
        {"answer_id": i, "question": f"savol {i}", "preview": "p", "score": 0.6} for i in range(3)
    ]
    message = fake_message()
    run(
        handlers.on_text(
            message,
            qa=fake_qa(ask={"status": "suggest", "candidates": cands}),
            queries=QueryStore(),
        )
    )
    ((text, markup),) = sent(message)
    assert text == texts.SUGGEST and len(markup.inline_keyboard) == 3


def test_not_found_and_unknown_law():
    message = fake_message()
    run(
        handlers.on_text(
            message, qa=fake_qa(ask={"status": "not_found", "reason": None}), queries=QueryStore()
        )
    )
    assert sent(message)[0][0] == texts.NOT_FOUND
    law = {
        "status": "not_found",
        "reason": "unknown_law",
        "entities": [{"type": "LAW", "surface": "PQ-9999"}],
    }
    message = fake_message()
    run(handlers.on_text(message, qa=fake_qa(ask=law), queries=QueryStore()))
    assert sent(message)[0][0] == texts.UNKNOWN_LAW.format(laws="PQ-9999")


def test_api_down_message():
    message = fake_message()
    run(handlers.on_text(message, qa=fake_qa(ask=ApiUnavailable("down")), queries=QueryStore()))
    assert sent(message)[0][0] == texts.API_DOWN


def test_suggestion_tap_sends_answer_and_removes_buttons():
    callback = MagicMock(answer=AsyncMock(), message=fake_message())
    qa = fake_qa(answer={"answer": "To'liq javob"})
    run(handlers.on_suggestion(callback, SuggestCb(answer_id=9, qid=1), qa=qa))
    qa.answer.assert_awaited_once_with(9)
    callback.message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
    assert sent(callback.message)[0][0] == "To'liq javob"


def test_feedback_uses_stored_query():
    store = QueryStore()
    qid = store.put("soliq qarzi")
    callback = MagicMock(answer=AsyncMock(), message=fake_message())
    qa = fake_qa(feedback={"id": 1})
    run(
        handlers.on_feedback(
            callback, FeedbackCb(helpful=True, answer_id=3, qid=qid), qa=qa, queries=store
        )
    )
    qa.feedback.assert_awaited_once_with("soliq qarzi", 3, True)
    callback.answer.assert_awaited_once_with(texts.FEEDBACK_THANKS)
    callback.message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)


def test_example_button_asks_example_question():
    callback = MagicMock(answer=AsyncMock(), message=fake_message())
    qa = fake_qa(ask={"status": "not_found"})
    run(handlers.on_example(callback, ExampleCb(idx=0), qa=qa, queries=QueryStore()))
    qa.ask.assert_awaited_once_with(texts.EXAMPLE_QUESTIONS[0])


def test_about_and_start():
    message = fake_message()
    run(handlers.cmd_about(message, qa=fake_qa(stats=ApiUnavailable("down"))))
    assert texts.ABOUT_NO_METRICS in sent(message)[0][0]
    message = fake_message()
    run(handlers.cmd_start(message))
    text, markup = sent(message)[0]
    assert text == texts.START and len(markup.inline_keyboard) == len(texts.EXAMPLE_QUESTIONS)


@pytest.mark.parametrize("module", ["handlers", "keyboards", "formatting", "main"])
def test_user_facing_uzbek_only_in_texts(module):
    source = Path(f"app/bot/{module}.py").read_text(encoding="utf-8")
    strings = re.findall(r'"([^"\n]{12,})"', source)
    uzbek = [
        s for s in strings if re.search(r"\b(qanday|iltimos|kechirasiz|savol|javob)\b", s, re.I)
    ]
    assert not uzbek, uzbek
