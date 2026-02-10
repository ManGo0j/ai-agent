import os
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import openai

load_dotenv()

# Настройки конфигурации
QDRANT_URL = os.getenv("QDRANT_URL", "http://accountant_qdrant:6333")
COLLECTION_NAME = "knowledge_base"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Модель для поиска (Embeddings)
MODEL_NAME = "all-MiniLM-L6-v2"

# Модель для Переранжирования (Re-ranking)
# Используем mMARCO - легкая многоязычная модель, отличная для RAG
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Инициализация клиентов
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)
encoder = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# Инициализация ИИ-клиента DeepSeek
ai_client = openai.AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

async def search(query: str) -> List[Dict]:
    """
    Поиск с двухступенчатой системой:
    1. Retrieval (Qdrant): достаем топ-20 кандидатов.
    2. Re-ranking (CrossEncoder): выбираем топ-5 лучших по смыслу.
    """
    # 1. Кодируем вопрос пользователя в вектор
    query_vector = encoder.encode(query).tolist()

    try:
        # ЗАПРОС К QDRANT
        # Увеличиваем лимит до 20, чтобы захватить больше кандидатов
        search_result = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=20 
        )

        if not search_result.points:
            return []

        # 2. ПОДГОТОВКА К ПЕРЕРАНЖИРОВАНИЮ
        # Собираем пары [Запрос, Текст документа] для оценки нейросетью
        points = search_result.points
        cross_inp = [[query, hit.payload.get("text")] for hit in points]

        # 3. ОЦЕНКА РЕЛЕВАНТНОСТИ (Re-ranking)
        # Модель выдает число (score) для каждой пары. Чем больше, тем лучше.
        scores = reranker.predict(cross_inp)

        # Объединяем результаты с их оценками
        scored_hits = []
        for hit, score in zip(points, scores):
            scored_hits.append({
                "hit": hit,
                "score": score
            })

        # Сортируем по убыванию оценки (от лучшего к худшему)
        scored_hits.sort(key=lambda x: x["score"], reverse=True)

        # Берем ТОП-5 победителей
        top_hits = scored_hits[:5]

        # Формируем итоговый ответ
        results = []
        for item in top_hits:
            hit = item["hit"]
            results.append({
                "text": hit.payload.get("text"),
                "source": hit.payload.get("document_name"),
                "score": float(item["score"]) # Можно использовать для отладки
            })
        
        print(f"DEBUG: Найдено {len(search_result.points)} кандидатов -> Оставлено {len(results)} лучших.")
        return results
        
    except Exception as e:
        print(f"❌ Ошибка при поиске/реранкинге: {e}")
        return []

async def generate_answer(query: str, search_results: List[Dict]) -> str:
    """Генерация ответа через DeepSeek с использованием найденного контекста."""
    if not search_results:
        return "К сожалению, в базе знаний не найдено информации по вашему вопросу."

    # Сборка контекста
    context_parts = []
    sources = set()
    
    for i, res in enumerate(search_results, 1):
        if res['text']:
            # Текст уже содержит метаданные благодаря indexer.py
            context_parts.append(f"=== ФРАГМЕНТ №{i} (Релевантность: {res.get('score', 0):.2f}) ===\n{res['text']}")
        if res['source']:
            sources.add(res['source'])

    context_text = "\n\n".join(context_parts)
    
    # Системный промпт
    system_prompt = (
        "Ты — профессиональный бухгалтерский ассистент. Твоя задача — отвечать на вопросы, "
        "строго опираясь на предоставленный контекст.\n"
        "ВАЖНО:\n"
        "1. Каждый фрагмент контекста начинается с указания 'ИСТОЧНИК' и 'ТИП ДОКУМЕНТА'. "
        "Обязательно учитывай эту информацию. Приоритет отдавай Кодексам РФ и Федеральным законам.\n"
        "2. Если в контексте нет прямого ответа на вопрос, честно сообщи, что данных недостаточно. "
        "Не выдумывай информацию.\n"
        "3. Ответ должен быть структурированным, с четкими формулировками."
    )
    
    user_prompt = f"НАЙДЕННЫЕ ДОКУМЕНТЫ (отсортированы по важности):\n{context_text}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n\nОТВЕТ:"

    try:
        # Запрос к нейросети DeepSeek
        response = await ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        sources_list = "\n\n<b>📚 Источники:</b>\n" + "\n".join([f"• {s}" for s in sources])
        disclaimer = "\n\n<i>⚠️ Внимание: Ответ носит справочный характер.</i>"

        return f"{answer}{sources_list}{disclaimer}"

    except Exception as e:
        return f"Ошибка при обращении к ИИ-сервису: {str(e)}"