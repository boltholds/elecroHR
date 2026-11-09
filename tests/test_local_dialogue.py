"""State and ReAct tests use a fake Ollama HTTP response; no GPU is necessary."""
import json

import httpx
import pytest

from app.schemas import RetrievedChunk
from local_voice.dialogue import HRDialogue, Phase, verified_name
from local_voice.settings import LocalSettings


class FakeRAG:
    def __init__(self):
        self.calls: list[str] = []

    def search_vacancy(self, query: str, top_k: int = 3):
        self.calls.append(query)
        if "удал" in query.casefold():
            return [
                RetrievedChunk(chunk_id="chunk-001", source="vacancy.md",
                               text="Формат работы: удалённый.", score=4.0)
            ]
        return []


def ollama_handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    assert payload["stream"] is False
    assert payload["think"] is False
    assert isinstance(payload["format"], dict)
    prompt = payload["messages"][0]["content"]
    utterance = payload["messages"][1]["content"]
    if "СЕЙЧАС" in prompt:
        answer = {"answer": "yes" if "да" in utterance.casefold() else "no"}
    elif "Извлеки только" in prompt:
        if "Александр" in utterance:
            answer = {"name": "Александр", "evidence": "Меня зовут Александр"}
        else:
            answer = {"name": "Никита", "evidence": "Меня зовут Никита"}
    elif "Найди предпочтительное" in prompt:
        answer = {
            "preferred_name": "Саша" if "Саша" in utterance else "",
            "form": "ты" if "ты" in utterance else "вы",
            "evidence": utterance,
        }
    elif "Кандидат подтверждает" in prompt:
        answer = {"answer": "no" if "нет" in utterance.casefold() else "yes"}
    elif "Ты управляешь" in prompt:
        latest = json.loads(utterance)["latest_candidate_utterance"].lower()
        if "удал" in latest:
            answer = {
                "action": "search_vacancy", "focus": "", "query": "удалённая работа",
                "instruction": "",
            }
        elif "законч" in latest:
            answer = {
                "action": "end_interview", "focus": "", "query": "", "instruction": "",
            }
        else:
            answer = {
                "action": "ask_question", "focus": "rag_retrieval", "query": "",
                "instruction": "Попроси пример SQL индекса.",
            }
    else:
        raise AssertionError(f"Unexpected structured prompt: {prompt[:150]}")
    return httpx.Response(200, json={"message": {"content": json.dumps(answer, ensure_ascii=False)}})


async def test_complete_introduction_then_react_and_rag():
    dialogue = HRDialogue()
    rag = FakeRAG()
    settings = LocalSettings(_env_file=None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(ollama_handler)) as client:
        transcript = []
        answer = await dialogue.respond("Да, удобно.", transcript, client, settings, rag)
        assert "имя" in answer.text.lower()
        assert dialogue.phase == Phase.name
        assert not rag.calls

        answer = await dialogue.respond("Меня зовут Александр", transcript, client, settings, rag)
        assert "Александр" in answer.text
        assert dialogue.phase == Phase.address
        assert not dialogue.public_view()["confirmed"]

        answer = await dialogue.respond("Саша, на ты", transcript, client, settings, rag)
        assert dialogue.phase == Phase.confirmation
        assert "Саша" in answer.text
        assert dialogue.public_view()["name"] == ""  # still unconfirmed

        answer = await dialogue.respond("Да, всё верно.", transcript, client, settings, rag)
        assert dialogue.phase == Phase.screening
        assert dialogue.public_view()["name"] == "Саша"
        assert dialogue.public_view()["form"] == "ты"
        assert answer.instruction or answer.text
        assert not rag.calls

        dialogue.playback_confirmed()
        assert dialogue.asked  # first technical question was heard
        plan = await dialogue.respond(
            "А работа удалённая?", transcript, client, settings, rag
        )
        assert plan.sources[0]["chunk_id"] == "chunk-001"
        assert rag.calls == ["удалённая работа"]
        assert "search_vacancy" in plan.knowledge
        assert "Саша" in dialogue.reply_prompt(plan)
        assert "«ты»" in dialogue.reply_prompt(plan)

        question = await dialogue.respond(
            "Я работал с FastAPI", transcript, client, settings, rag
        )
        assert question.instruction
        assert dialogue.pending_focus == "rag_retrieval"
        dialogue.playback_interrupted()
        assert dialogue.pending_focus == ""

        ending = await dialogue.respond(
            "Давайте закончим.", transcript, client, settings, rag
        )
        assert dialogue.phase == Phase.finished
        assert ending.text
        assert "Саша" in ending.text


async def test_missing_name_does_not_advance():
    dialogue = HRDialogue(phase=Phase.name)
    rag = FakeRAG()
    settings = LocalSettings(_env_file=None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(ollama_handler)) as client:
        answer = await dialogue.respond(
            "Я хочу работать с Python.", [], client, settings, rag
        )
    assert dialogue.phase == Phase.name
    assert dialogue.candidate_name == ""
    assert "имя" in answer.text.lower()


async def test_correction_resets_preference():
    dialogue = HRDialogue(
        phase=Phase.confirmation,
        candidate_name="Александр",
        preferred_name="Саша",
        form="ты",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(ollama_handler)) as client:
        reply = await dialogue.respond(
            "Нет, исправьте.", [], client, LocalSettings(_env_file=None), FakeRAG()
        )
    assert dialogue.phase == Phase.name
    assert not dialogue.confirmed
    assert dialogue.preferred_name == ""
    assert "правильно" in reply.text


def test_cannot_infer_name_from_unrelated_sentence():
    assert verified_name("Никита", "Меня зовут Никита", "Я хочу работать с Python") == ""
    assert verified_name("Александр", "Меня зовут Александр", "Меня зовут Александр") == "Александр"
    assert verified_name("123", "123", "123") == ""


async def test_first_and_next_questions_are_from_source_bank():
    from local_voice.question_bank import QuestionBank, parse_questions

    markdown = (
        "## 01. Основы LLM\n"
        "### RAG-Q001 — Что такое большая языковая модель?\n\n"
        "### RAG-Q002 — В чём отличие RAG от fine-tuning?\n\n"
        "### RAG-Q003 — Как оценить качество retrieval?\n"
    )
    bank = QuestionBank(parse_questions(markdown))
    dialogue = HRDialogue(
        phase=Phase.confirmation,
        candidate_name="Антон", preferred_name="Антон", form="ты",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(ollama_handler)) as client:
        first = await dialogue.respond(
            "Да", [], client, LocalSettings(_env_file=None), FakeRAG(), bank
        )
        assert first.question_id in bank.by_id
        assert bank.by_id[first.question_id].text in first.text
        assert dialogue.asked_question_ids == []
        dialogue.playback_confirmed()
        assert dialogue.asked_question_ids == [first.question_id]
        another = await dialogue.respond(
            "Я проектировал RAG retrieval", [], client,
            LocalSettings(_env_file=None), FakeRAG(), bank
        )
        assert another.question_id in bank.by_id
        assert another.question_id != first.question_id
        assert another.text == bank.by_id[another.question_id].text
        dialogue.playback_confirmed()
        assert len(dialogue.asked_question_ids) == 2
