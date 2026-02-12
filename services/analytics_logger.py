import aiofiles
import os

async def log_request_details(
    original_query,
    rewritten_query,
    chunks,
    final_answer,
    embedding_model,
    chunk_size,
    num_fragments
):
    """
    Записывает подробную информацию о процессе RAG в файл аналитики
    """
    # Файл будет создан в корневой папке проекта
    log_file_path = "analitycs.txt"
    
    # Формируем блоки данных
    log_entry = [
        "ОТЧЕТ ПО ЗАПРОСУ",
        f"Модель эмбеддинга: {embedding_model}",
        f"Размер чанка: {chunk_size}",
        f"Количество фрагментов в контексте: {num_fragments}",
        "",
        "ЭТАПЫ ОБРАБОТКИ ЗАПРОСА",
        f"Исходный вопрос пользователя: {original_query}",
        f"Пересобранный вопрос (DeepSeek): {rewritten_query}",
        "",
        "ВЫБРАННЫЕ ФРАГМЕНТЫ ИЗ БАЗЫ"
    ]
    
    # Добавляем информацию о каждом чанке
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Источник не определен")
        text = chunk.get("text", "Контент отсутствует")
        log_entry.append(f"Чанк №{i}")
        log_entry.append(f"Источник: {source}")
        log_entry.append(f"Текст чанка: {text}")
        log_entry.append("")
        
    log_entry.append("ИТОГОВЫЙ ОТВЕТ МОДЕЛИ")
    log_entry.append(final_answer)
    log_entry.append("_" * 60)
    log_entry.append("\n")
    
    # Собираем всё в одну строку
    final_log_text = "\n".join(log_entry)
    
    # Асинхронная запись в режиме добавления
    async with aiofiles.open(log_file_path, mode="a", encoding="utf-8") as f:
        await f.write(final_log_text)