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

# Модель для плотных векторов (смысл)
MODEL_NAME = "all-MiniLM-L6-v2"

# Модель для разреженных векторов (ключевые слова)
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"

# Модель для Переранжирования (Re-ranking)
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Инициализация клиентов
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)
encoder = SentenceTransformer(MODEL_NAME)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# Инициализация ИИ клиента DeepSeek
ai_client = openai.AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

async def search(query: str) -> List[Dict]:
    """
    Гибридный поиск (Dense + Sparse) с RRF и последующим реранкингом.
    """
    # 1. Кодируем вопрос двумя способами
    query_dense_vector = encoder.encode(query).tolist()
    
    # Получаем разреженный вектор (берем первый элемент из итератора)
    query_sparse_embeddings = list(sparse_model.embed([query]))
    sparse_emb = query_sparse_embeddings[0]

    try:
        # ГИБРИДНЫЙ ЗАПРОС К QDRANT С ИСПОЛЬЗОВАНИЕМ RRF
        search_result = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                # Ветка 1: Векторный поиск по смыслу
                models.Prefetch(
                    query=query_dense_vector,
                    using="default",
                    limit=20,
                ),
                # Ветка 2: Поиск по ключевым словам
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist()
                    ),
                    using="sparse-text",
                    limit=20,
                ),
            ],
            # Объединение результатов через Reciprocal Rank Fusion
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20 
        )

        if not search_result.points:
            return []

        # 2. ПОДГОТОВКА К ПЕРЕРАНЖИРОВАНИЮ
        points = search_result.points
        cross_inp = [[query, hit.payload.get("text")] for hit in points]

        # 3. ОЦЕНКА РЕЛЕВАНТНОСТИ (Re-ranking)
        scores = reranker.predict(cross_inp)

        scored_hits = []
        for hit, score in zip(points, scores):
            scored_hits.append({
                "hit": hit,
                "score": score
            })

        # Сортировка по весу реранкера
        scored_hits.sort(key=lambda x: x["score"], reverse=True)

        # Выбираем лучшие 5 фрагментов
        top_hits = scored_hits[:5]

        results = []
        for item in top_hits:
            hit = item["hit"]
            results.append({
                "text": hit.payload.get("text"),
                "source": hit.payload.get("document_name"),
                "score": float(item["score"])
            })
        
        print(f"DEBUG: Гибридный поиск нашел {len(points)} кандидатов. Реранкер отобрал {len(results)} лучших.")
        return results
        
    except Exception as e:
        print(f"Ошибка при гибридном поиске: {e}")
        return []

async def generate_answer(query: str, search_results: List[Dict]) -> str:
    """Генерация ответа через DeepSeek с использованием найденного контекста."""
    if not search_results:
        return "К сожалению, в базе знаний не найдено информации по вашему вопросу."

    context_parts = []
    sources = set()
    
    for i, res in enumerate(search_results, 1):
        if res['text']:
            context_parts.append(f"=== ФРАГМЕНТ №{i} (Релевантность: {res.get('score', 0):.2f}) ===\n{res['text']}")
        if res['source']:
            sources.add(res['source'])

    context_text = "\n\n".join(context_parts)
    
    system_prompt = (
        "Ты профессиональный бухгалтерский ассистент. Твоя задача отвечать на вопросы, "
        "строго опираясь на предоставленный контекст. "
        "ВАЖНО: "
        "1. Каждый фрагмент контекста начинается с указания 'ИСТОЧНИК' и 'ТИП ДОКУМЕНТА'. "
        "Обязательно учитывай эту информацию. Приоритет отдавай Кодексам РФ и Федеральным законам. "
        "2. Если в контексте нет прямого ответа на вопрос, честно сообщи, что данных недостаточно. "
        "Не выдумывай информацию. "
        "3. Ответ должен быть структурированным, с четкими формулировками."
    )
    
    user_prompt = f"НАЙДЕННЫЕ ДОКУМЕНТЫ (отсортированы по важности):\n{context_text}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n\nОТВЕТ:"

    try:
        response = await ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        sources_list = "\n\n<b>Библиотека источников:</b>\n" + "\n".join([f"• {s}" for s in sources])
        disclaimer = "\n\n<i>Примечание: Ответ носит справочный характер.</i>"

        return f"{answer}{sources_list}{disclaimer}"

    except Exception as e:
        return f"Ошибка при обращении к ИИ сервису: {str(e)}"