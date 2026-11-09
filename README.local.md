# elecroHR: локальное голосовое интервью в Docker

Этот браузерный режим **не меняет** прежний GPT-Live/SIP backend.
Сценарий: один тестовый голосовой разговор, RAG по существующей вакансии,
отдельный локальный скоринг и JSON-отчёт. Реального звонка на мобильный номер
пока нет — браузер использует микрофон и динамики.

## NVIDIA RTX 4070 Ti (12 ГБ)

На Windows требуются Docker Desktop с WSL2 и NVIDIA-драйвер для WSL2.
На Linux потребуется NVIDIA Container Toolkit. Проверка GPU:

    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

Запуск с видеокартой:

    docker compose -f compose.yaml -f compose.gpu.yaml up --build

После запуска открой http://localhost:8000. Используй наушники,
разреши браузеру доступ к микрофону, отметь согласие на локальное
сохранение текста и нажми «Начать интервью».

Ollama работает на GPU, Whisper Small и Silero TTS на CPU. При первом запуске
model-init скачает веса STT/TTS, а ollama-init — Qwen3:4b. Веса хранятся
в постоянных Docker volumes; при повторном запуске не скачиваются заново.
После первого скачивания инференс не требует внешних AI API.

Проверка фактического использования GPU:

    docker compose -f compose.yaml -f compose.gpu.yaml exec ollama ollama ps

CPU-запуск (заметно медленнее):

    docker compose up --build

Остановить и сохранить модели:

    docker compose -f compose.yaml -f compose.gpu.yaml down

Не используйте down -v, если нужно сохранить модели и результаты.

## API локального режима

- GET / — Web UI.
- GET /health — health check локального процесса.
- POST /sessions с телом {"consent": true} — создать одну сессию.
- WS /ws/{session_id} — 16 kHz PCM16 (512 samples/frame), JSON-события.
- POST /sessions/{session_id}/finish — завершение и скоринг.
- GET /sessions/{session_id}/report — результат с цитатами.
- GET /sessions/{session_id}/transcript — расшифровка.

Данные лежат в Docker volume interview_data в файлах JSON. Порт
опубликован только на 127.0.0.1:8000; для публичной сети потребуются
аутентификация, HTTPS и настройки хранения персональных данных.

Модели и голос можно сменить через локальную конфигурацию Compose;
образец переменных — .env.local.example. Исходная SIP-реализация запускается
отдельно по инструкциям основного README.md.

## Известные ограничения

- Это браузерный тестовый звонок, а не реальный PSTN/SIP.
- STT начинает распознавание после паузы, определённой Silero VAD.
- LLM отдаёт текст потоково; TTS начинает воспроизводить законченные
  предложения. Это компромисс по задержке.
- Воспроизведение прерывается по barge-in, но уже начатое вычисление
  TTS в отдельном потоке отменить мгновенно нельзя.
- Используйте наушники: браузерная echo cancellation не всегда удаляет
  обратную связь от колонок.
- Скоринг валидирует веса и цитаты, но результаты LLM надо проверять HR.
- CI проверяет Python-контракты и бизнес-логику; запуск Docker и
  полноценный голосовой сценарий необходимо проверить на локальном GPU.
- Для демонстрации рекомендуется тестовый профиль вместо реальных
  персональных данных кандидата.

Команды диагностики:

    docker compose -f compose.yaml -f compose.gpu.yaml ps
    docker compose -f compose.yaml -f compose.gpu.yaml logs -f app
    docker compose -f compose.yaml -f compose.gpu.yaml logs -f ollama
