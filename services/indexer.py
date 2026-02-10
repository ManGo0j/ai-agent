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
from fastembed import SparseTextEmbedding

# Исправленные импорты для SQLAlchemy 2.0
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Импорт моделей БД
from database.models import Base, AdminDocument, DocumentChunk

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

# Модель для плотных векторов (смысл)
model = SentenceTransformer(MODEL_NAME)

# Модель для разреженных векторов (ключевые слова)
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

engine = create_async_engine(DATABASE_URL)

# Создание фабрики асинхронных сессий
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
    
    # 1. Проверка или создание коллекции в Qdrant с поддержкой гибридного поиска
    try:
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                # Плотные векторы
                vectors_config={
                    "default": models.VectorParams(
                        size=384, 
                        distance=models.Distance.COSINE
                    )
                },
                # Разреженные векторы
                sparse_vectors_config={
                    "sparse-text": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )
            print(f"Успешно создана гибридная коллекция Qdrant: {COLLECTION_NAME}")
        else:
            print(f"Коллекция {COLLECTION_NAME} уже существует.")
    except Exception as e:
        print(f"Ошибка при настройке Qdrant: {e}")
        return

    # 2. Поиск всех файлов
    files = []
    extensions = ['*.pdf', '*.docx', '*.doc', '*.txt']
    for ext in extensions:
        files.extend(glob.glob(os.path.join(DOCS_DIR, ext)))

    if not files:
        print(f"Файлы в директории {DOCS_DIR} не найдены.")
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
                    print(f"Пропуск: {file_name} уже проиндексирован.")
                    continue

                print(f"Обработка документа: {file_name}")

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
                    print(f"Файл {file_name} пуст.")
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

                # 4. Создание контекстных чанков
                raw_chunks = text_splitter.split_text(text_content)
                contextualized_chunks = []
                for chunk in raw_chunks:
                    context_header = (
                        f"ИСТОЧНИК: {file_name}\n"
                        f"ТИП ДОКУМЕНТА: {doc_type}\n"
                        f"ГРАНИЦА ФРАГМЕНТА\n"
                    )
                    full_text = context_header + chunk
                    contextualized_chunks.append(full_text)

                # 5. Генерация двух типов эмбеддингов
                dense_embeddings = model.encode(contextualized_chunks)
                sparse_embeddings = list(sparse_model.embed(contextualized_chunks))

                points = []
                for i, (chunk_text, dense_vec, sparse_vec) in enumerate(zip(
                    contextualized_chunks, 
                    dense_embeddings, 
                    sparse_embeddings
                )):
                    point_id = str(uuid.uuid4())
                    
                    db_chunk = DocumentChunk(
                        document_id=db_doc.id,
                        chunk_index=i,
                        chunk_text=chunk_text,
                        embedding_id=point_id
                    )
                    session.add(db_chunk)

                    # Формируем структуру точки для Qdrant
                    points.append(models.PointStruct(
                        id=point_id,
                        vector={
                            "default": dense_vec.tolist(),
                            "sparse-text": models.SparseVector(
                                indices=sparse_vec.indices.tolist(),
                                values=sparse_vec.values.tolist()
                            )
                        },
                        payload={
                            "document_id": db_doc.id,
                            "document_name": file_name,
                            "text": chunk_text
                        }
                    ))

                # 6. Загрузка в Qdrant порциями
                if points:
                    for k in range(0, len(points), 100):
                        qdrant_client.upsert(
                            collection_name=COLLECTION_NAME, 
                            points=points[k : k + 100]
                        )
                
                await session.commit()
                print(f"Завершена обработка {file_name}. Создано чанков: {len(contextualized_chunks)}")

            except Exception as e:
                await session.rollback()
                print(f"Ошибка при обработке файла {file_name}: {str(e)}")
                continue

if __name__ == "__main__":
    asyncio.run(process_files())