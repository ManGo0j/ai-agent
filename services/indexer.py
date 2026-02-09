import os
import glob
import uuid
import hashlib
import pdfplumber
import asyncio
from docx import Document
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Исправленные импорты для SQLAlchemy 2.0
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Импорт моделей БД
from database.models import AdminDocument, DocumentChunk

# Импорт инструментов для обработки текста
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- КОНФИГУРАЦИЯ ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Настройки Qdrant и БД
QDRANT_URL = os.getenv("QDRANT_URL", "http://accountant_qdrant:6333")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = "knowledge_base"
MODEL_NAME = "all-MiniLM-L6-v2" 

# Настройки разбиения текста
CHUNK_SIZE = 2000 
CHUNK_OVERLAP = 300 

# --- ИНИЦИАЛИЗАЦИЯ ---
qdrant_client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(MODEL_NAME)
engine = create_async_engine(DATABASE_URL)

# Правильное создание фабрики асинхронных сессий
AsyncSessionLocal = async_sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Инициализируем умный сплиттер
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_file_hash(file_path: str) -> str:
    """Вычисляет хеш файла для проверки на дубликаты."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_text_from_docx(file_path: str) -> str:
    """Извлекает текст из Word."""
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Ошибка чтения DOCX {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    """Извлекает текст из PDF."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Ошибка чтения PDF {file_path}: {e}")
        return ""

async def process_files():
    """Основная функция обработки документов."""
    
    # 1. Проверка/Создание коллекции в Qdrant
    try:
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )
            print(f"✅ Создана коллекция Qdrant: {COLLECTION_NAME}")
    except Exception as e:
        print(f"❌ Ошибка подключения к Qdrant: {e}")
        return

    # 2. Поиск всех файлов
    files = []
    extensions = ['*.pdf', '*.docx', '*.doc', '*.txt']
    for ext in extensions:
        files.extend(glob.glob(os.path.join(DOCS_DIR, ext)))

    if not files:
        print(f"⚠️ Файлы в директории {DOCS_DIR} не найдены.")
        return

    print(f"Найдено файлов для обработки: {len(files)}")

    # 3. Обработка файлов
    async with AsyncSessionLocal() as session:
        for file_path in files:
            file_name = os.path.basename(file_path)
            
            try:
                # Проверка хеша
                file_hash = get_file_hash(file_path)
                result = await session.execute(
                    select(AdminDocument).where(AdminDocument.file_hash == file_hash)
                )
                if result.scalar_one_or_none():
                    print(f"⏩ Пропуск: {file_name} уже в базе.")
                    continue

                print(f"📂 Обработка: {file_name}...")

                # Чтение текста
                text_content = ""
                doc_type = "документ"
                
                if file_path.lower().endswith('.pdf'):
                    text_content = extract_text_from_pdf(file_path)
                    doc_type = "scan/pdf"
                elif file_path.lower().endswith(('.docx', '.doc')):
                    text_content = extract_text_from_docx(file_path)
                    doc_type = "word"
                elif file_path.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()

                if not text_content.strip():
                    print(f"⚠️ Файл {file_name} пуст или не прочитан.")
                    continue

                # Создание записи о документе
                db_doc = AdminDocument(
                    document_name=file_name,
                    file_path=file_path,
                    file_hash=file_hash,
                    document_type=doc_type,
                    upload_date=datetime.utcnow()
                )
                session.add(db_doc)
                await session.flush() 

                # 4. Чанкинг и Эмбеддинги
                chunks = text_splitter.split_text(text_content)
                embeddings = model.encode(chunks)

                points = []
                for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
                    point_id = str(uuid.uuid4())
                    
                    db_chunk = DocumentChunk(
                        document_id=db_doc.id,
                        chunk_index=i,
                        chunk_text=chunk_text,
                        embedding_id=point_id
                    )
                    session.add(db_chunk)

                    points.append(models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "document_id": db_doc.id,
                            "document_name": file_name,
                            "text": chunk_text
                        }
                    ))

                # 5. Загрузка в Qdrant
                if points:
                    for k in range(0, len(points), 100):
                        qdrant_client.upsert(
                            collection_name=COLLECTION_NAME, 
                            points=points[k : k + 100]
                        )
                
                await session.commit()
                print(f"✅ Успешно: {file_name} (создано {len(chunks)} чанков)")

            except Exception as e:
                await session.rollback()
                print(f"❌ Ошибка с файлом {file_name}: {str(e)}")
                continue

if __name__ == "__main__":
    asyncio.run(process_files())