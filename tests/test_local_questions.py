"""Questions must originate from the source file and retain stable IDs."""
from pathlib import Path

import pytest

from local_voice.question_bank import QuestionBank, parse_questions


def test_real_bank_loads_all_119_questions():
    markdown = Path("knowledge/faq.md").read_text(encoding="utf-8")
    bank = QuestionBank(parse_questions(markdown))
    assert len(bank.questions) == 119
    assert len(bank.by_id) == 119
    assert bank.by_id["RAG-Q001"].text.startswith("В чём разница")
    assert bank.by_id["RAG-Q119"].text.startswith("Кейс 2.")
    assert all(q.source == "faq.md" for q in bank.questions)
    assert all(q.section for q in bank.questions)


def test_question_retrieval_and_no_repeats():
    bank = QuestionBank(parse_questions(
        "## 01. Основы\n"
        "### RAG-Q001 — Что такое LLM?\n"
        "**Теги:** " + chr(96) + "llm" + chr(96) + "\n\n"
        "### RAG-Q002 — Что такое эмбеддинги?\n"
        "**Теги:** " + chr(96) + "embeddings" + chr(96) + "\n"
    ))
    q1 = bank.next_question("эмбеддинги embeddings", set())
    assert q1.question_id == "RAG-Q002"
    q2 = bank.next_question("эмбеддинги embeddings", {"RAG-Q002"})
    assert q2.question_id == "RAG-Q001"
    assert bank.next_question("llm", {"RAG-Q001", "RAG-Q002"}) is None


def test_duplicate_id_is_rejected():
    with pytest.raises(ValueError, match="Duplicate question"):
        parse_questions(
            "### RAG-Q001 — First?\n\n"
            "### RAG-Q001 — Second?\n"
        )
