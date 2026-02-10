import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import select
from database.models import DocumentChunk, AdminDocument
import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL)

async def view_chunks():
    async with AsyncSession(engine) as session:
        result = await session.execute(select(DocumentChunk, AdminDocument).join(AdminDocument))
        chunks = result.all()
        with open("all_chunks_debug.txt", "w", encoding="utf-8") as f:
            for chunk, doc in chunks:
                f.write(f"ФАЙЛ: {doc.document_name} | ИНДЕКС: {chunk.chunk_index}\n")
                f.write(f"{chunk.chunk_text}\n")
                f.write("-" * 50 + "\n")
    print("Готово. Все чанки сохранены в файл all_chunks_debug.txt внутри контейнера.")

if __name__ == "__main__":
    asyncio.run(view_chunks())