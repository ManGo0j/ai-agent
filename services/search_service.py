import os
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
import openai

load_dotenv()

# Настройки конфигурации
QDRANT_URL = os.getenv("QDRANT_URL", "http://accountant_qdrant:6333")
COLLECTION_NAME = "knowledge_base"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Модель для плотных векторов
MODEL_NAME = "all-MiniLM-L6-v2"

# Модель для разреженных векторов
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"

# Модель для Переранжирования
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Инициализация клиентов
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)
encoder = SentenceTransformer(MODEL_NAME)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

ai_client = openai.AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

async def rewrite_query(original_query: str) -> str:
    """
    Функция пересборки вопроса для улучшения поиска в юридической базе
    """
    try:
        response = await ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Ты эксперт по юридическому поиску. Твоя задача превратить вопрос пользователя "
                        "в оптимальный поисковый запрос для базы данных нормативных актов РФ. "
                        "Используй официальные термины и ключевые слова. Не отвечай на вопрос, "
                        "просто верни строку с поисковыми ключами."
                    )
                },
                {"role": "user", "content": original_query}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception:
        return original_query

async def search(query: str) -> List[Dict]:
    """
    Улучшенный поиск с пересборкой вопроса и увеличенными лимитами
    """
    # 1. ПЕРЕСБОРКА ВОПРОСА
    # Это позволяет найти профессиональные статьи по обывательским вопросам
    search_query = await rewrite_query(query)
    
    # 2. КОДИРОВАНИЕ
    query_dense_vector = encoder.encode(search_query).tolist()
    query_sparse_embeddings = list(sparse_model.embed([search_query]))
    sparse_emb = query_sparse_embeddings[0]

    try:
        # Увеличиваем лимит prefetch до 40 для большего охвата кандидатов
        search_result = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=query_dense_vector,
                    using="default",
                    limit=40,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist()
                    ),
                    using="sparse-text",
                    limit=40,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=40 
        )

        if not search_result.points:
            return []

        # 3. ПОДГОТОВКА К ПЕРЕРАНЖИРОВАНИЮ
        points = search_result.points
        # Используем исходный запрос пользователя для финальной сверки релевантности
        cross_inp = [[query, hit.payload.get("text")] for hit in points]

        # 4. РЕРАНКИНГ
        scores = reranker.predict(cross_inp)
        scored_hits = []
        for hit, score in zip(points, scores):
            scored_hits.append({
                "hit": hit,
                "score": score
            })

        scored_hits.sort(key=lambda x: x["score"], reverse=True)

        # Увеличиваем количество итоговых чанков до 8 для полноты контекста
        top_hits = scored_hits[:8]

        results = []
        for item in top_hits:
            hit = item["hit"]
            results.append({
                "text": hit.payload.get("text"),
                "source": hit.payload.get("document_name"),
                "score": float(item["score"])
            })
        
        return results
        
    except Exception:
        return []

async def generate_answer(query: str, search_results: List[Dict]) -> str:
    """Генерация ответа с расширенным контекстом"""
    if not search_results:
        return "В базе знаний нет информации по вашему вопросу"

    context_parts = []
    sources = set()
    
    for i, res in enumerate(search_results, 1):
        if res['text']:
            context_parts.append(f"=== ФРАГМЕНТ №{i} ===\n{res['text']}")
        if res['source']:
            sources.add(res['source'])

    context_text = "\n\n".join(context_parts)
    
    system_prompt = (
        "Ты профессиональный бухгалтерский ассистент. Отвечай "
        "строго на основе предоставленного контекста. "
        "Учитывай названия источников. Приоритет имеют Кодексы РФ. "
        "Если ответа в тексте нет, сообщи о нехватке данных. "
        "Пиши структурировано."
    )
    
    user_prompt = f"КОНТЕКСТ:\n{context_text}\n\nВОПРОС: {query}\n\nОТВЕТ:"

    try:
        response = await ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2
        )
        
        answer = response.choices[0].message.content
        sources_list = "\n\n<b>Источники:</b>\n" + "\n".join([f"• {s}" for s in sources])
        
        return f"{answer}{sources_list}"

    except Exception as e:
        return f"Ошибка при генерации ответа {str(e)}"