from app.rag import RAGService
from app.schemas import CallRecord, TranscriptFragment


def follow_up(answer: str) -> str:
    if "блок" in answer.casefold() or "event loop" in answer.casefold():
        return "Как ограничить конкурентность и отменить задачу по таймауту?"
    return "Расскажите о конкретном случае и о том, как проверили результат."


def run_demo(call: CallRecord, rag: RAGService) -> None:
    """Synthetic text interview. No phone, audio or model requests."""
    call.consent = "granted"
    dialogue = [
        (
            "agent",
            "Я AI-рекрутер. Это тестовое интервью с ИИ. Удобно говорить? "
            "Разрешаете сохранить текст и проанализировать ответы?",
        ),
        ("candidate", "Да, удобно, согласен на сохранение текста и анализ ответов."),
        ("agent", "Расскажите, как вы строили API на FastAPI."),
        (
            "candidate",
            "Я использую Pydantic для валидации и dependency injection для сессии БД; "
            "пишу тесты REST API и возвращаю корректные HTTP статусы.",
        ),
        ("agent", "Как исследовать медленный SQL запрос?"),
        (
            "candidate",
            "Смотрю EXPLAIN ANALYZE, сравниваю оценки и строки, выбираю индекс "
            "по фильтрам и проверяю стоимость записи.",
        ),
        ("agent", "Как собирать и запускать контейнер?"),
        (
            "candidate",
            "Использую multi-stage build, фиксирую зависимости, запускаю non-root "
            "и передаю секреты при запуске, а не в образе.",
        ),
        ("agent", "Что происходит при блокирующем I/O внутри async функции?"),
        ("candidate", "Блокируется event loop. Нужен асинхронный клиент или to_thread."),
    ]
    dialogue.append(("agent", follow_up(dialogue[-1][1])))
    dialogue.extend(
        [
            (
                "candidate",
                "Использую Semaphore и таймаут asyncio.timeout; event loop не должен "
                "блокироваться. При отмене освобождаю ресурсы в finally.",
            ),
            ("candidate", "Какой формат работы и зарплата?"),
        ]
    )
    chunks = rag.search_vacancy("формат работы зарплата удаленная")
    call.retrievals.extend(chunks)
    dialogue.append(("agent", "По тестовой базе: " + " ".join(c.text for c in chunks)))
    dialogue.append(("agent", "Спасибо за беседу. Результаты рассмотрит человек HR. До свидания."))
    for index, (role, text) in enumerate(dialogue):
        call.transcript.append(
            TranscriptFragment(
                event_id=f"demo_{index}",
                role="agent" if role == "agent" else "candidate",
                text=text + "\n",
                start_ms=index * 3000,
                end_ms=(index + 1) * 3000,
            )
        )
    call.close_reason = "demo_finished"
    call.finalization_confirmed = True
