import asyncio
import logging
import os
import time
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from database.models import Base, User, Conversation
from services.search_service import search, generate_answer
from dotenv import load_dotenv

load_dotenv()

# Настройки
API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")

# Логирование
logging.basicConfig(level=logging.INFO)

# Инициализация БД
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Временное хранилище для Rate Limit {user_id: [timestamps]}
user_requests = {}

def check_rate_limit(user_id: int) -> bool:
    """Проверка: не более 3 запросов в минуту."""
    now = time.time()
    if user_id not in user_requests:
        user_requests[user_id] = []
    
    # Очищаем старые таймстампы (старше 60 сек)
    user_requests[user_id] = [t for t in user_requests[user_id] if now - t < 60]
    
    if len(user_requests[user_id]) >= 3:
        return False
    
    user_requests[user_id].append(now)
    return True

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Хендлер /start: регистрация и приветствие [cite: 17-20]."""
    async with async_session() as session:
        # Проверяем, есть ли пользователь
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()

        if not user:
            user = User(
                telegram_id=message.from_user.id,
                username=message.from_user.username,
                full_name=message.from_user.full_name
            )
            session.add(user)
            await session.commit()

    welcome_text = (
        f"Здравствуйте, {message.from_user.first_name}! 👋\n\n"
        "Я — ваш ИИ Бухгалтерский ассистент. Я могу отвечать на вопросы по законодательству, "
        "опираясь на загруженную базу документов.\n\n"
        "⚠️ *Внимание: мои ответы носят информационный характер и не являются юридической консультацией.*"
    )
    await message.answer(welcome_text, parse_mode="Markdown")

@dp.message(F.text)
async def handle_question(message: types.Message):
    """Обработка текстовых вопросов с RAG-логикой [cite: 21-22]."""
    user_id = message.from_user.id

    # 1. Проверка лимитов
    if not check_rate_limit(user_id):
        await message.answer("⚠️ Вы исчерпали лимит запросов (3 в минуту). Пожалуйста, подождите.")
        return

    # Отправляем статус "печатает", так как ИИ может думать долго
    await bot.send_chat_action(message.chat.id, "typing")
    
    # 2. Поиск и генерация
    query = message.text
    try:
        search_results = await search(query)
        if not search_results:
            answer = "К сожалению, в моей базе знаний нет информации по вашему вопросу."
        else:
            answer = await generate_answer(query, search_results)

        # 3. Сохранение в БД [cite: 81-82]
        async with async_session() as session:
            # Находим ID пользователя в нашей БД
            res = await session.execute(select(User.id).where(User.telegram_id == user_id))
            db_user_id = res.scalar()
            
            new_conv = Conversation(
                user_id=db_user_id,
                message_text=query,
                bot_response=answer,
                message_date=datetime.utcnow()
            )
            session.add(new_conv)
            await session.commit()

        # 4. Отправка ответа
        await message.answer(answer, parse_mode="HTML")

    except Exception as e:
        logging.error(f"Error handling message: {e}")
        await message.answer("Произошла ошибка при обработке запроса. Попробуйте позже.")

async def main():
    logging.info("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())