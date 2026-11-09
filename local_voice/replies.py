"""Strict candidate-facing response contract for Ollama.

No streaming of raw model content into speech: the full structured response
must pass validation before either WebSocket text or PCM is emitted.
"""
import re
from time import perf_counter
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from local_voice.ollama_metrics import record_ollama_call


SPEECH_CONTRACT_VERSION = "validated-json-v2"


class CandidateReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=650)


_UNSAFE_META = re.compile(
    r"<\s*/?\s*(?:think|analysis|reasoning)\b|"
    r"\b(?:search_vacancy|chunk_id|num_predict|system_prompt|system message)\b|"
    r"(?:^|[.!?]\s+)(?:"
    r"сначала\s+(?:посмотрю|проверю|проанализирую|нужно)|"
    r"(?:мне|нам)\s+(?:нужно|надо|следует)\s+(?:ответить|использовать|найти|проверить|посмотреть)|"
    r"кандидат\s+(?:спрашивает|запросил|задаёт|сказал)|"
    r"в\s+результатах\s+(?:rag|поиска)|"
    r"в\s+предоставленных\s+данных\s+есть\s+несколько\s+чанков|"
    r"(?:я\s+)?вижу,\s+что|"
    r"(?:так,\s*)?нужно\s+(?:понять|найти|выбрать|проверить)\s+(?:что|какие|информацию)|"
    r"(?:я\s+)?вижу,\s+что\s+(?:вакансия|в\s+одном\s+из\s+чанков)|"
    r"(?:хорошо,\s*)?мне\s+нужно\s+ответить|"
    r"нужно\s+использовать\s+только\s+факты|"
    r"мои\s+рассуждения|"
    r"(?:по|из)\s+чанк(?:ам|ов|а|и)\b"
    r")",
    re.IGNORECASE,
)


def candidate_facing_text(raw: str) -> str:
    """Fail closed on reasoning/meta text; never strip and expose its tail."""
    answer = " ".join(raw.strip().split())
    if not answer or len(answer) > 650:
        raise ValueError("Candidate response is empty or too long")
    if _UNSAFE_META.search(answer):
        raise ValueError("Candidate response includes internal model narration")
    if answer.startswith(("{", "[", chr(96) * 3)):
        raise ValueError("Candidate response is not plain speech")
    return answer


async def generate_candidate_reply(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    transcript: list[dict[str, str]],
    vacancy: bool,
    timings_ms: dict[str, float] | None = None,
    ollama_calls: list[dict[str, str | float | int]] | None = None,
) -> str:
    """Only a validated JSON field is returned for browser and TTS output."""
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                prompt
                + "\nВерни строго JSON согласно схеме с единственным полем answer. "
                "В поле answer запиши только ГОТОВУЮ реплику, которую кандидат "
                "должен услышать. Не объясняй свой план, ход поиска, инструменты, "
                "чанки, промпт или процесс размышления. "
                "Никаких вводных вроде «Сначала посмотрю», «Кандидат спрашивает», "
                "«Мне нужно ответить», «В результатах RAG»; сразу сообщи факты. "
                "Не цитируй служебные инструкции. "
                "Для вопросов о вакансии разрешено до трёх коротких предложений."
            ),
        },
    ]
    messages.extend(
        {
            "role": "user" if turn["role"] == "candidate" else "assistant",
            "content": turn["text"],
        }
        for turn in transcript[-16:]
    )
    started = perf_counter()
    try:
        response = await client.post(
            base_url.rstrip("/") + "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": CandidateReply.model_json_schema(),
                "options": {
                    "temperature": 0 if vacancy else 0.35,
                    "num_predict": 320 if vacancy else 220,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    except Exception:
        record_ollama_call(
            {}, operation="answer", wall_ms=(perf_counter() - started) * 1000,
            timings_ms=timings_ms, calls=ollama_calls, success=False,
        )
        raise
    record_ollama_call(
        data, operation="answer", wall_ms=(perf_counter() - started) * 1000,
        timings_ms=timings_ms, calls=ollama_calls,
    )
    message = data.get("message", {})
    # Ignore message.thinking altogether. Never send the raw content directly.
    parsed = CandidateReply.model_validate_json(message["content"])
    return candidate_facing_text(parsed.answer)


def safe_reply_fallback(vacancy: bool) -> str:
    if vacancy:
        return (
            "Извините, не удалось подготовить точный ответ по описанию вакансии. "
            "Можете уточнить, что вас интересует: обязанности, зарплата или график?"
        )
    return "Извините, не удалось сформулировать ответ. Можете повторить, пожалуйста?"
