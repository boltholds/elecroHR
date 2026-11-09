"""Ollama output must be validated before it can reach TTS or transcript."""
import json

import httpx
import pytest

from local_voice.replies import (
    CandidateReply, candidate_facing_text, generate_candidate_reply,
    safe_reply_fallback,
)


async def test_uses_answer_only_and_never_exposes_thinking():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["format"] == CandidateReply.model_json_schema()
        assert payload["messages"][0]["role"] == "system"
        assert payload["options"]["temperature"] == 0
        return httpx.Response(200, json={
            "message": {
                "thinking": "Сначала посмотрю на результаты RAG. Кандидат спрашивает.",
                "content": json.dumps({
                    "answer": "Это тестовая вакансия LLM-разработчика с удалённой работой."
                }, ensure_ascii=False),
            }
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        answer = await generate_candidate_reply(
            client, "http://ollama:11434", "qwen3:4b",
            "Ответь на вопрос по вакансии.", [{"role": "candidate",
            "text": "Расскажи подробнее о вакансии."}], vacancy=True,
        )
    assert answer == "Это тестовая вакансия LLM-разработчика с удалённой работой."
    assert "Сначала" not in answer
    assert "Кандидат" not in answer


@pytest.mark.parametrize("bad", [
    "Сначала посмотрю на результаты RAG.",
    "Мне нужно ответить на запрос кандидата.",
    "Кандидат спрашивает: расскажите больше о вакансии.",
    "В результатах RAG указаны условия работы.",
    "<think>Сначала рассмотрим чанки</think> Зарплата составляет 200 тысяч.",
    "Ответ: search_vacancy вернул 3 чанка.",
    "Вижу, что вакансия демонстрационная, потому что так указано в чанке.",
    "По чанкам нужно выбрать информацию.",
])
def test_rejects_meta_narration(bad):
    with pytest.raises(ValueError):
        candidate_facing_text(bad)


def test_valid_candidate_reply_is_preserved():
    assert candidate_facing_text(
        "Работа удалённая, график — пять дней в неделю."
    ) == "Работа удалённая, график — пять дней в неделю."


async def test_json_field_cannot_be_used_for_internal_analysis():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {
                "content": json.dumps({
                    "answer": "Сначала посмотрю на результаты RAG. "
                              "В вакансии указан удалённый формат."
                }, ensure_ascii=False)
            }
        })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="internal"):
            await generate_candidate_reply(
                client, "http://ollama:11434", "qwen3:4b",
                "Отвечай по вакансии.",
                [{"role": "candidate", "text": "Какие условия?"}], vacancy=True,
            )


async def test_plain_text_from_llm_is_not_accepted_as_validated_reply():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"content": "Сначала посмотрю на результаты RAG."}
        })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            await generate_candidate_reply(
                client, "http://ollama:11434", "qwen3:4b",
                "Отвечай по вакансии.",
                [{"role": "candidate", "text": "Расскажи о вакансии"}], vacancy=True,
            )


def test_fallback_contains_no_internal_details():
    assert "RAG" not in safe_reply_fallback(True)
    assert "think" not in safe_reply_fallback(False)



async def test_interview_never_sends_model_reasoning_to_websocket_or_tts():
    from app.schemas import RetrievedChunk
    from local_voice.dialogue import HRDialogue, Phase
    from local_voice.server import Interview
    from local_voice.settings import LocalSettings

    class FakeAudio:
        spoken = []

        def transcribe(self, pcm):
            return "Расскажите больше о вакансии."

        def synthesize(self, text):
            self.spoken.append(text)
            return b"\x00\x00"

    class FakeSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)

        async def send_bytes(self, payload):
            self.sent.append(payload)

    class FakeRAG:
        def search_vacancy(self, query, top_k=5):
            return [RetrievedChunk(
                chunk_id="overview", source="vacancy.md",
                text="Должность: LLM-разработчик. Формат: удалённо.",
                score=3.0,
            )]

    def handler(request: httpx.Request):
        return httpx.Response(200, json={
            "message": {
                "thinking": "Хорошо, мне нужно ответить на запрос кандидата.",
                "content": json.dumps({
                    "answer": "Сначала посмотрю на результаты RAG. Вижу данные."
                }, ensure_ascii=False),
            },
        })

    audio = FakeAudio()
    sock = FakeSocket()
    dialogue = HRDialogue(
        phase=Phase.screening, preferred_name="Антон", candidate_name="Антон",
        confirmed=True, form="ты",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        interview = Interview(
            session_id="test", settings=LocalSettings(_env_file=None),
            audio=audio, rag=FakeRAG(), http=client,
            dialogue=dialogue, state="thinking",
        )
        interview.ws = sock
        await interview.handle_turn(b"fake pcm")
    text_messages = [x["text"] for x in sock.sent
                     if isinstance(x, dict) and x.get("type") == "agent_text"]
    assert text_messages == [safe_reply_fallback(True)]
    assert audio.spoken == text_messages
    assert all("Сначала посмотрю" not in t for t in text_messages)
    assert all("кандидата" not in t for t in text_messages)
