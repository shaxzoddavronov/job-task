"""Inline keyboards and their callback data (each packs to ≤ 64 bytes, Telegram's limit)."""

from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from app.bot import texts
from app.bot.formatting import button_label


class ExampleCb(CallbackData, prefix="ex"):
    idx: int


class SuggestCb(CallbackData, prefix="sg"):
    answer_id: int
    qid: int


class FeedbackCb(CallbackData, prefix="fb"):
    helpful: bool
    answer_id: int
    qid: int


def examples_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=q, callback_data=ExampleCb(idx=i).pack())]
            for i, q in enumerate(texts.EXAMPLE_QUESTIONS)
        ]
    )


def suggestions_kb(candidates: list[dict], qid: int, limit: int = 3) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=button_label(c["question"] or c["preview"]),
                    callback_data=SuggestCb(answer_id=c["answer_id"], qid=qid).pack(),
                )
            ]
            for c in candidates[:limit]
        ]
    )


def feedback_kb(answer_id: int, qid: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=texts.HELPFUL,
                    callback_data=FeedbackCb(helpful=True, answer_id=answer_id, qid=qid).pack(),
                ),
                InlineKeyboardButton(
                    text=texts.NOT_HELPFUL,
                    callback_data=FeedbackCb(helpful=False, answer_id=answer_id, qid=qid).pack(),
                ),
            ]
        ]
    )
