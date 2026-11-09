# elecroHR — локальное голосовое HR-интервью (Docker, RU/EN)

Независимый от GPT-Live/SIP браузерный режим: микрофон → Silero VAD →
faster-whisper → Ollama/Qwen3 → Chatterbox Multilingual V3 → динамики.

Это браузерное голосовое интервью, не звонок на мобильный телефон.
В старом backend облачный SIP/GPT-Live остаётся без изменений.

## Что изменилось

- Мультиязычный TTS: Chatterbox Multilingual V3, русский и английский
  в одной модели, отдельный контейнер `tts`. Определение языка
  выполняется по каждому фрагменту текста. Русские фразы с латинскими
  названиями технологий обычно синтезируются как русский текст.
- 119 вопросов из файла: `knowledge/faq.md` содержит вопросы
  `RAG-Q001`–`RAG-Q119`, с исходными разделами и тегами.
  В файле нет эталонных ответов, каждый вопрос индексируется отдельно.
- ReAct выбирает новый вопрос из банка, уточнение предыдущего ответа,
  поиск фактов о вакансии или завершение. ID вопросов сохраняются.
- Вступление: приветствие → удобство разговора → имя →
  обращение «вы»/«ты» → подтверждение → технический вопрос из банка.
- Отдельный RAG по вакансии: `knowledge/vacancy.md` используется
  для ответов об условиях работы. `faq.md` для таких ответов исключён.
- Скоринг по теме LLM/RAG: основы LLM (25), RAG/retrieval (35),
  оценка качества (20), развёртывание/агенты (20). Баллы обосновываются
  дословными ответами кандидата. Финальное решение принимает HR.

## Запуск: RTX 4070 Ti, Windows / Docker Desktop WSL2

Docker Desktop должен иметь доступ к видеокарте:

    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

В папке проекта:

    git switch feat/local-voice-docker-mvp
    git pull origin feat/local-voice-docker-mvp
    docker-compose -f compose.yaml -f compose.gpu.yaml up -d --build
    docker-compose -f compose.yaml -f compose.gpu.yaml logs -f tts app

Открыть http://localhost:8000. Рекомендуются наушники.

Первый запуск требует интернета для Docker-образов, исходников Chatterbox
на зафиксированной ревизии GitHub и весов моделей. CUDA PyTorch-образ
большой; проверьте свободное место Docker/WSL. Веса сохраняются в
persistent volumes, при последующих запусках не скачиваются заново.

Сервисы:
- ollama и ollama-init: Qwen3 4B, RTX 4070 Ti.
- model-init: загрузка Whisper Small (CPU).
- tts-model-init: загрузка Chatterbox Multilingual V3.
- tts: отдельный CUDA TTS, внутренний HTTP API (без публичного порта).
- app: FastAPI + WebSocket + UI + STT/VAD/RAG/scoring.

Ollama и Chatterbox используют общую RTX 4070 Ti 12 ГБ. При OOM
понадобится уменьшать контекст и/или переносить TTS на CPU (медленно).

Полезная диагностика:

    docker-compose -f compose.yaml -f compose.gpu.yaml ps
    docker-compose -f compose.yaml -f compose.gpu.yaml logs -f tts
    docker-compose -f compose.yaml -f compose.gpu.yaml exec ollama ollama ps

GET http://localhost:8000/health показывает размер question bank.
Интервью и отчёты хранятся в Docker volume `interview_data`.

## Диалоговая система

Стадии: availability → name → address → confirmation → screening → finished.

На этапе screening ReAct выдаёт запрос, затем backend выбирает
подходящий вопрос из `knowledge/faq.md` и исключает уже заданные ID.
В UI появляется ID вопроса. При необходимости агент задаёт один
уточняющий вопрос по предыдущему ответу. Вопросы основного скрининга
сохраняют формулировки из исходного файла.

После завершения скоринг получает техническую часть диалога.
Файл вопросов не содержит ответов, поэтому оценка показывает
наблюдаемую глубину объяснений, а не подтверждение эталонной правильности.

## Ограничения

- Голосовой end-to-end, GPU-сборка Chatterbox и задержка не проверены
  автоматически на вашем компьютере. CI проверяет код и Compose.
- Обе модели на общей видеокарте могут расходовать VRAM одновременно.
- Локальные транскрипты сохраняются в нешифрованном JSON. Для реальных
  кандидатов потребуются отдельные меры безопасности и хранения.
- Chatterbox — архивированный исследовательский проект под лицензией MIT.
  Используется зафиксированная ревизия для воспроизводимости.
- Для устойчивого barge-in используйте наушники.

Остановить без удаления моделей:

    docker-compose -f compose.yaml -f compose.gpu.yaml down

Не запускайте down -v, если нужны данные из volumes.
