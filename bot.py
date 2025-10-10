import asyncio
import logging
import sys
from os import getenv

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from main import run_ollama, load_knowledge

# Bot token can be obtained via https://t.me/BotFather
TOKEN = getenv("BOT_TOKEN")
print(TOKEN)

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer("Привіт!")
    response = run_ollama("Розкажи про себе. Відповідай тільки українською мовою!")
    await message.answer(response)

@dp.message()
async def echo_handler(message: Message) -> None:
    try:
        await message.answer("Думаю...")
        response = run_ollama(f"\n\nUser: {message.text}\n\n відповідай українською мовою.")
        await message.answer(response)
    except TypeError:
        await message.answer("Nice try!")


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())