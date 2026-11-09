"""Vacancy Q&A must be grounded in public RAG and preserve dialogue state."""
from pathlib import Path

import httpx
import pytest

from local_voice.dialogue import HRDialogue, Phase
from local_voice.settings import LocalSettings
from local_voice.vacancy import (
    VacancyRAG, is_vacancy_question, public_vacancy_chunks,
)


@pytest.fixture
def rag() -> VacancyRAG:
    markdown = Path("knowledge/vacancy.md").read_text(encoding="utf-8")
    return VacancyRAG(public_vacancy_chunks(markdown))


@pytest.mark.parametrize("utterance", [
    "Какая зарплата?",
    "Сколько платят?",
    "А работа удалённая?",
    "Какой график?",
    "Какой у вас стек?",
    "Нужно ли высшее образование?",
    "Есть ли ДМС?",
    "Что за компания?",
    "Сколько этапов собеседования?",
    "Можно ли работать из Сербии?",
])
def test_detects_candidate_vacancy_questions(utterance):
    assert is_vacancy_question(utterance)


@pytest.mark.parametrize("utterance", [
    "Я работал удалённо с FastAPI.",
    "У меня опыт работы в команде.",
    "Что такое RAG?",
    "Как устроен BM25?",
    "Я писал технологические сервисы с Docker.",
])
def test_does_not_confuse_technical_answers_and_vacancy_questions(utterance):
    assert not is_vacancy_question(utterance)


def test_only_public_sections_are_indexed(rag):
    assert len(rag.chunks) >= 20
    assert {chunk.source for chunk in rag.chunks} == {"vacancy.md"}
    corpus = "\n".join(chunk.text for chunk in rag.chunks)
    assert "200 000–280 000" in corpus
    assert "HR-бот обязан" not in corpus
    assert "Шесть основных вопросов" not in corpus
    assert "согласие на запись" not in corpus
    assert "Не выдавайте" not in corpus
    assert "Как тестировать звонок" not in corpus
    assert all("internal_policy" not in chunk.text for chunk in rag.chunks)


def test_salary_and_dms_are_grounded_in_public_facts(rag):
    salary = rag.search_vacancy("А какая зарплата?", top_k=5)
    assert salary and any("200 000" in hit.text for hit in salary)
    dms = rag.search_vacancy("Есть ли ДМС?", top_k=5)
    assert dms and any("не определены" in hit.text for hit in dms)


def test_bm25_returns_nothing_for_absent_benefit(rag):
    assert rag.search_vacancy("Есть ли опционы?") == []


def test_normal_markdown_without_internal_metadata_is_supported():
    text = "## Условия работы\n\nОплата 150 000 ₽.\n\n## Служебные правила\n<!-- record_type: internal_policy -->\nЭто внутренний секрет."
    chunks = public_vacancy_chunks(text)
    assert any("150 000" in c.text for c in chunks)
    assert not any("внутренний секрет" in c.text for c in chunks)


async def test_vacancy_question_before_confirmation_preserves_intro_phase(rag):
    dialogue = HRDialogue(phase=Phase.name)
    settings = LocalSettings(_env_file=None)
    async with httpx.AsyncClient() as client:
        plan = await dialogue.respond("Какая зарплата?", [], client, settings, rag)
    assert plan.response_type == "vacancy"
    assert plan.instruction and not plan.text
    assert plan.sources
    assert "search_vacancy" in plan.knowledge
    assert all("vacancy.md" == item["source"] for item in plan.sources)
    assert dialogue.phase == Phase.name
    assert not dialogue.confirmed
    assert "ещё не подтверждены" in dialogue.reply_prompt(plan)


async def test_unknown_vacancy_condition_has_fixed_safe_answer(rag):
    dialogue = HRDialogue(phase=Phase.screening, confirmed=True, form="вы",
                          preferred_name="Тест", asked_question_ids=["RAG-Q001"])
    settings = LocalSettings(_env_file=None)
    async with httpx.AsyncClient() as client:
        plan = await dialogue.respond(
            "Есть ли опционы?", [], client, settings, rag
        )
    assert plan.response_type == "vacancy"
    assert "нет в описании вакансии" in plan.text
    assert plan.knowledge == ""
    assert plan.sources == []
    assert dialogue.phase == Phase.screening
    assert dialogue.asked_question_ids == ["RAG-Q001"]


async def test_followup_to_vacancy_does_not_consume_question(rag):
    dialogue = HRDialogue(phase=Phase.screening, confirmed=True, form="ты",
                          preferred_name="Тест", asked_question_ids=["RAG-Q007"])
    async with httpx.AsyncClient() as client:
        plan = await dialogue.respond(
            "Нужно ли высшее образование?", [], client, LocalSettings(_env_file=None),
            rag
        )
    assert plan.response_type == "vacancy"
    assert plan.sources
    assert dialogue.asked_question_ids == ["RAG-Q007"]
    assert dialogue.last_question_id == ""
    assert dialogue.phase == Phase.screening
