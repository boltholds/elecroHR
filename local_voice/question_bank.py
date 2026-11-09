"""Structured question retrieval from knowledge/faq.md.

Source contains 119 RAG-Q IDs, section names and tags, but no verified answers.
Each question is indexed independently with its original source identifier.
"""
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from app.rag import RAGService
from app.schemas import RetrievedChunk

_HEADER = re.compile(r"^###\s+(RAG-Q\d{3})\s*[—–-]\s*(.+?)\s*$", re.M)
_SECTION = re.compile(r"^##\s+(\d{2})\.\s+(.+?)\s*$", re.M)
_TAGS = re.compile(r"\*\*Теги:\*\*\s*(.*)")


@dataclass(frozen=True, slots=True)
class InterviewQuestion:
    question_id: str
    text: str
    section: str
    tags: tuple[str, ...]
    source: str = "faq.md"

    @property
    def search_text(self) -> str:
        return " ".join([self.text, self.section, " ".join(self.tags)])


def parse_questions(markdown: str) -> list[InterviewQuestion]:
    sections = list(_SECTION.finditer(markdown))
    heads = list(_HEADER.finditer(markdown))
    questions: list[InterviewQuestion] = []
    seen: set[str] = set()
    for index, header in enumerate(heads):
        question_id, title = header.group(1), header.group(2).strip()
        if question_id in seen:
            raise ValueError(f"Duplicate question ID: {question_id}")
        seen.add(question_id)
        body_end = heads[index + 1].start() if index + 1 < len(heads) else len(markdown)
        body = markdown[header.end():body_end]
        tags_match = _TAGS.search(body)
        tags = tuple(re.findall(chr(96) + "([^" + chr(96) + "]+)" + chr(96), tags_match.group(1))) if tags_match else ()
        section = next(
            (s.group(2).strip() for s in reversed(sections) if s.start() < header.start()),
            "Общие вопросы",
        )
        if title:
            questions.append(InterviewQuestion(question_id, title, section, tags))
    return questions


class QuestionBank:
    def __init__(self, questions: list[InterviewQuestion]):
        if not questions:
            raise ValueError("Question bank contains no questions")
        self.questions = questions
        self.by_id = {q.question_id: q for q in questions}
        self.index = RAGService([
            RetrievedChunk(
                chunk_id=q.question_id,
                source=q.source,
                text=q.search_text,
                score=0,
            )
            for q in questions
        ])

    @classmethod
    async def load(cls, path: Path) -> "QuestionBank":
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return cls(parse_questions(content))

    def retrieve(
        self, query: str, excluded: set[str] | None = None, top_k: int = 5
    ) -> list[InterviewQuestion]:
        excluded_ids = excluded or set()
        candidates = self.index.search_vacancy(query, top_k=len(self.questions))
        ranked = [
            self.by_id[hit.chunk_id]
            for hit in candidates
            if hit.chunk_id not in excluded_ids
        ]
        if ranked:
            return ranked[:top_k]
        return [q for q in self.questions if q.question_id not in excluded_ids][:top_k]

    def next_question(self, query: str, excluded: set[str]) -> InterviewQuestion | None:
        matches = self.retrieve(query, excluded, top_k=1)
        return matches[0] if matches else None
