import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from database.models import Base
import os
from dotenv import load_dotenv

load_dotenv()

# Получаем URL из .env
DATABASE_URL = os.getenv("DATABASE_URL")

async def init_models():
    if not DATABASE_URL:
        print("Ошибка: Не задан DATABASE_URL в .env")
        return

    engine = create_async_engine(DATABASE_URL, echo=True)
    
    async with engine.begin() as conn:
        # Удаляем старые таблицы (если нужно для тестов) и создаем новые
        # await conn.run_sync(Base.metadata.drop_all) 
        print("Создание таблиц...")
        await conn.run_sync(Base.metadata.create_all)
        print("Таблицы users и conversations созданы успешно.")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_models())