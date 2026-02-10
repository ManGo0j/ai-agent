import os
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
import openai

load_dotenv()

# Настройки конфигурации
QDRANT_URL = os.getenv("QDRANT_URL", "http://accountant_qdrant:6333")
COLLECTION_NAME = "knowledge_base"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "all-MiniLM-L6-v2"

# Инициализация клиентов
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)
encoder = SentenceTransformer(MODEL_NAME)

# Инициализация ИИ-клиента DeepSeek
ai_client = openai.AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

async def search(query: str) -> List[Dict]:
    """Поиск релевантных чанков в векторной базе Qdrant."""
    # Кодируем вопрос пользователя в вектор
    query_vector = encoder.encode(query).tolist()

    try:
        search_result = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5
        )

        results = []
        for hit in search_result.points:
            results.append({
                "text": hit.payload.get("text"),
                "source": hit.payload.get("document_name")
            })
        
        print(f"DEBUG: Найдено чанков: {len(results)}")
        return results
        
    except Exception as e:
        print(f"❌ Ошибка при поиске в Qdrant: {e}")
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
            # Текст уже содержит "ИСТОЧНИК: ..." благодаря Contextual Retrieval в indexer.py
            context_parts.append(f"=== ФРАГМЕНТ №{i} ===\n{res['text']}")
        if res['source']:
            sources.add(res['source'])

    context_text = "\n\n".join(context_parts)
    
    # Обновленный промпт для работы с контекстными чанками
    system_prompt = (
        "Ты — профессиональный бухгалтерский ассистент. Твоя задача — отвечать на вопросы, "
        "строго опираясь на предоставленный контекст.\n"
        "ВАЖНО:\n"
        "1. Каждый фрагмент контекста начинается с указания 'ИСТОЧНИК' и 'ТИП ДОКУМЕНТА'. "
        "Обязательно учитывай эту информацию. Если информация из Налогового кодекса противоречит "
        "старому письму Минфина, отдавай приоритет Кодексу.\n"
        "2. Если в контексте нет прямого ответа на вопрос, честно сообщи, что данных недостаточно. "
        "Не выдумывай законы.\n"
        "3. Ответ должен быть структурированным, с четкими формулировками."
    )
    
    user_prompt = f"НАЙДЕННЫЕ ДОКУМЕНТЫ:\n{context_text}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}\n\nОТВЕТ:"

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