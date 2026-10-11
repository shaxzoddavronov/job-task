"""Bot handlers. Dependencies (qa, queries) come from the dispatcher's workflow data, so the
handlers are plain async functions that tests can call with fakes."""

from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from itertools import count

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message
from aiogram.utils.chat_action import ChatActionSender

from app.bot import texts
from app.bot.api_client import ApiRejected, ApiUnavailable, QAClient
from app.bot.formatting import about_text, split_message, unknown_laws
from app.bot.keyboards import (
    ExampleCb,
    FeedbackCb,
    SuggestCb,
    examples_kb,
    feedback_kb,
    suggestions_kb,
)

router = Router()


class QueryStore:
    """Bounded map qid → original question; callback data is too small to carry the text."""

    def __init__(self, max_size: int = 2000) -> None:
        self.max_size = max_size
        self._items: OrderedDict[int, str] = OrderedDict()
        self._ids = count(1)

    def put(self, question: str) -> int:
        qid = next(self._ids)
        self._items[qid] = question
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)
        return qid

    def get(self, qid: int) -> str | None:
        return self._items.get(qid)


@asynccontextmanager
async def typing(message: Message) -> AsyncIterator[None]:
    """Show "typing…" while the API works (skipped when there is no bot, e.g. in tests)."""
    if message.bot is None:
        yield
        return
    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        yield


async def send_answer(message: Message, text: str, answer_id: int, qid: int) -> None:
    parts = split_message(text)
    for part in parts[:-1]:
        await message.answer(part)
    await message.answer(parts[-1], reply_markup=feedback_kb(answer_id, qid))


async def answer_question(
    message: Message, question: str, qa: QAClient, queries: QueryStore
) -> None:
    async with typing(message):
        try:
            result = await qa.ask(question)
        except ApiUnavailable:
            await message.answer(texts.API_DOWN)
            return
        except ApiRejected:
            await message.answer(texts.TOO_LONG)
            return
    qid = queries.put(question)
    if result["status"] == "answer":
        await send_answer(message, result["answer"], result["answer_id"], qid)
    elif result["status"] == "suggest":
        await message.answer(texts.SUGGEST, reply_markup=suggestions_kb(result["candidates"], qid))
    elif result.get("reason") == "unknown_law":
        await message.answer(texts.UNKNOWN_LAW.format(laws=unknown_laws(result)))
    else:
        await message.answer(texts.NOT_FOUND)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(texts.START, reply_markup=examples_kb())


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(texts.HELP)


@router.message(Command("about"))
async def cmd_about(message: Message, qa: QAClient) -> None:
    try:
        stats = await qa.stats()
    except (ApiUnavailable, ApiRejected):
        await message.answer(texts.ABOUT.format(metrics=texts.ABOUT_NO_METRICS))
        return
    await message.answer(about_text(stats))


@router.message(F.text)
async def on_text(message: Message, qa: QAClient, queries: QueryStore) -> None:
    await answer_question(message, message.text, qa, queries)


@router.message()
async def on_other(message: Message) -> None:
    await message.answer(texts.NOT_TEXT)


@router.callback_query(ExampleCb.filter())
async def on_example(
    callback: CallbackQuery, callback_data: ExampleCb, qa: QAClient, queries: QueryStore
) -> None:
    await callback.answer()
    question = texts.EXAMPLE_QUESTIONS[callback_data.idx]
    await callback.message.answer(f"❓ {question}")
    await answer_question(callback.message, question, qa, queries)


@router.callback_query(SuggestCb.filter())
async def on_suggestion(callback: CallbackQuery, callback_data: SuggestCb, qa: QAClient) -> None:
    await callback.answer()
    try:
        data = await qa.answer(callback_data.answer_id)
    except ApiUnavailable:
        await callback.message.answer(texts.API_DOWN)
        return
    except ApiRejected:
        await callback.message.answer(texts.ANSWER_MISSING)
        return
    await callback.message.edit_reply_markup(reply_markup=None)
    await send_answer(callback.message, data["answer"], callback_data.answer_id, callback_data.qid)


@router.callback_query(FeedbackCb.filter())
async def on_feedback(
    callback: CallbackQuery, callback_data: FeedbackCb, qa: QAClient, queries: QueryStore
) -> None:
    query = queries.get(callback_data.qid) or "?"
    try:
        await qa.feedback(query, callback_data.answer_id, callback_data.helpful)
    except (ApiUnavailable, ApiRejected):
        await callback.answer(texts.API_DOWN)
        return
    await callback.answer(texts.FEEDBACK_THANKS)
    await callback.message.edit_reply_markup(reply_markup=None)
