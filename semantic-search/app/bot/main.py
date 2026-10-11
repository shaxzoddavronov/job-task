"""Telegram bot entry point (polling). Talks to the QA API over HTTP only.

Run: `make bot` (API_URL defaults to http://localhost:8000; Docker sets http://api:8000).
The token comes from BOT_TOKEN or bot_token.txt (Settings.resolve_bot_token) and is never logged.
"""

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand

from app.bot import texts
from app.bot.api_client import QAClient
from app.bot.handlers import QueryStore, router
from app.core.config import get_settings

log = logging.getLogger("app.bot")


async def run() -> None:
    settings = get_settings()
    token = settings.resolve_bot_token()
    if not token:
        sys.exit("No bot token: set BOT_TOKEN or put the token in BOT_TOKEN_FILE (bot_token.txt).")

    qa = QAClient(settings.api_url, timeout=settings.bot_api_timeout)
    bot = Bot(token, default=DefaultBotProperties(parse_mode=None))
    dp = Dispatcher(qa=qa, queries=QueryStore())
    dp.include_router(router)
    try:
        me = await bot.get_me()
        log.info("Bot @%s started (polling), API at %s", me.username, settings.api_url)
        await bot.set_my_commands(
            [BotCommand(command=c, description=d) for c, d in texts.COMMANDS.items()]
        )
        await dp.start_polling(bot)
    finally:
        await qa.aclose()
        await bot.session.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run())


if __name__ == "__main__":
    main()
