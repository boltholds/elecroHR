"""Public vacancy RAG for the local HR interview.

The question bank (faq.md) is a separate index. Candidate answers may
only use public vacancy/FAQ sections, not embedded HR internal_policy.
"""
import asyncio
import hashlib
import re
from pathlib import Path

from app.rag import RAGService, tokens
from app.schemas import RetrievedChunk

_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_METADATA = re.compile(r"<!--.*?-->", re.DOTALL)
_KIND = re.compile(r"\brecord_type\s*:\s*([a-z_]+)")
_SOURCE_ID = re.compile(r"\bchunk_id\s*:\s*([\w-]+)")
_LINE_ITEM = re.compile(r"^(?:[-*]\s+|\d+[.)]\s+|\*\*[^*]+\*\*)")
_INTERNAL_HEADING = re.compile(
    r"^(?:рекомендац.*rag|правила\s+hr|сценарий\s+первичного|"
    r"практический\s+кейс\s+для\s+разговора)",
    re.IGNORECASE,
)
_STOPWORDS = frozenset({
    "а", "и", "или", "ли", "вы", "вас", "мне", "меня", "мой", "мы",
    "какая", "какой", "какие", "какое", "как", "что", "чем", "это",
    "в", "во", "на", "о", "об", "у", "по", "для", "из", "от", "с", "со",
    "не", "но", "же", "бы", "то", "есть", "будет", "могу", "можно",
    "расскажите", "скажите", "пожалуйста", "хотел", "хочу",
    "узнать", "лишь", "тут", "там", "вообще", "сейчас", "ещё",
})
_VACANCY_TOPIC = re.compile(
    r"ваканс|должност|зарплат|оклад|оплат|доход|ставк|вилк|"
    r"сколько\s+плат|удал[её]н|дистанцион|офис|гибрид|релокац|"
    r"переезд|работать\s+из|график|смен|выходн|занятост|"
    r"час(?:ов|а)?\s+в\s+день|оформлен|договор|трудоустройств|"
    r"дмс|больничн|отпуск|соцпакет|преми|бонус|льгот|опцион|"
    r"оборудован|ноутбук|компенсац|компан|работодател|"
    r"команд|сотрудник|обязанност|задач[аи]|"
    r"требован|стек|технологи[ия]|фреймворк|"
    r"этап(?:ы|ов)?\s+отбор|собеседован|интервью|тестов|"
    r"фидбек|обратн(?:ую|ой)\s+связ|образован|диплом|"
    r"стаж|опыт\s+нуж|английск",
    re.IGNORECASE,
)
_OVERVIEW_REQUEST = re.compile(
    r"(?:ваканс|об\s+этой\s+работе|о\s+должности)",
    re.IGNORECASE,
)
_OVERVIEW_DETAIL = re.compile(
    r"(?:подроб|побольш|больше|расскаж|описан|что\s+за|в\s+ч[её]м\s+суть)",
    re.IGNORECASE,
)


def is_vacancy_overview_request(text: str) -> bool:
    """General request for a vacancy description rather than one specific condition."""
    return bool(_OVERVIEW_REQUEST.search(text) and _OVERVIEW_DETAIL.search(text))


_QUESTION_FORM = re.compile(
    r"\?|(?:^|\s)(?:подробн(?:ее|о)|побольш(?:е)?|больше|сколько|какой|какая|какие|какое|где|когда|"
    r"можно|подскажите|расскажи(?:те)?|нужен|нужно|нужны|"
    r"требуется|есть|будет|дают|предусмотрен|платят|"
    r"интересует|полагается|проводится|можете|"
    r"обязательно|достаточно|возможно)(?:\s|$)|"
    r"^\s*(?:а\s+)?(?:зарплата|оклад|удалёнка|удаленка|"
    r"оформление|дмс|тестовое|соцпакет|график|вакансия|условия)\b",
    re.IGNORECASE,
)
_ALIASES: tuple[tuple[str, str], ...] = (
    (r"зарплат|оклад|оплат|доход|ставк|вилк|сколько\s+плат", "зарплата платят вилка"),
    (r"удал|дистанцион|из\s+дома|работать\s+из|офис|гибрид", "удалённо формат офис"),
    (r"график|час|смен|занятост|выходн", "график пересечение часов"),
    (r"оформлен|договор|трудоустройств|самозанят", "оформление трудовой договор"),
    (r"дмс|преми|бонус|соцпакет|оборудован|ноутбук|льгот", "премии ДМС оборудование"),
    (r"отпуск|больнич", "отпуск больничные"),
    (r"компан|работодател|продукт|команд", "компания продукт команда"),
    (r"обязанност|задач", "обязанности задачи"),
    (r"технолог|стек|фреймворк", "технологический стек Python"),
    (r"тестов|отбор|собеседован|интервью|фидбек|обратн", "этапы интервью тестовое"),
    (r"диплом|образован|вышк", "образование диплом"),
    (r"стаж|опыт", "опыт разработка"),
    (r"английск", "английский документации"),
)


def is_vacancy_question(utterance: str) -> bool:
    """Recognize clear questions about employment without hijacking tech answers."""
    value = utterance.strip()
    if not value or not _VACANCY_TOPIC.search(value):
        return False
    return bool(_QUESTION_FORM.search(value))


def _split_items(body: str) -> list[str]:
    """Keep each FAQ question or field with its answer for precise retrieval."""
    plain = _METADATA.sub("", body)
    chunks: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            chunks.append(" ".join(paragraph).strip())
            paragraph.clear()

    for raw in plain.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif _LINE_ITEM.match(line):
            flush()
            chunks.append(line)
        else:
            paragraph.append(line)
    flush()
    return [part for part in chunks if len(part) >= 12]


def public_vacancy_chunks(markdown: str) -> list[RetrievedChunk]:
    """Parse only candidate-visible sections and drop all internal-policy records."""
    headers = list(_SECTION.finditer(markdown))
    result: list[RetrievedChunk] = []
    for number, header in enumerate(headers):
        section = re.sub(r"^\d+[.)]\s*", "", header.group(1)).strip()
        if _INTERNAL_HEADING.search(section):
            continue
        end = headers[number + 1].start() if number + 1 < len(headers) else len(markdown)
        body = markdown[header.end():end]
        # Discard trailing internal QA instructions outside the public FAQ.
        # The demo vacancy ends section 16 with an operational test checklist.
        body = re.split(r"(?m)^---\s*$|^###\s+Как тестировать", body, maxsplit=1)[0]
        meta = _METADATA.search(body)
        metadata = meta.group(0) if meta else ""
        kind = _KIND.search(metadata)
        if kind and kind.group(1) not in {"vacancy", "faq"}:
            continue
        source_id = _SOURCE_ID.search(metadata)
        prefix = source_id.group(1) if source_id else hashlib.sha256(
            section.encode("utf-8")
        ).hexdigest()[:12]
        pieces = _split_items(body)
        if section.casefold().startswith("как отвечать, если информации"):
            # This section is tagged faq but also contains HR instructions.
            # Only explicit candidate-facing Q&A entries are source facts.
            pieces = [part for part in pieces if part.startswith("**")]
        for part_number, piece in enumerate(pieces, start=1):
            result.append(RetrievedChunk(
                chunk_id=f"{prefix}-{part_number:02d}",
                source="vacancy.md",
                text=f"{section}: {piece}",
                score=0,
            ))
    if not result:
        raise ValueError("Vacancy contains no candidate-visible information")
    return result


class VacancyRAG(RAGService):
    """BM25 over public vacancy facts with limited Russian term expansion."""

    @classmethod
    async def load(cls, path: Path) -> "VacancyRAG":
        markdown = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return cls(public_vacancy_chunks(markdown))

    def search_vacancy(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if is_vacancy_overview_request(query):
            # A broad question has little lexical specificity. Provide a compact
            # factual overview, not arbitrary BM25 matches on "вакансия".
            priorities = (
                "**Компания:**", "**Должность:**", "**Уровень:**",
                "**Главная задача:**", "**Формат:**", "**Зарплата:**",
                "Продукт и команда:",
            )
            overview: list[RetrievedChunk] = []
            chosen: set[str] = set()
            for marker in priorities:
                chunk = next(
                    (item for item in self.chunks
                     if marker.casefold() in item.text.casefold()
                     and item.chunk_id not in chosen),
                    None,
                )
                if chunk is not None:
                    chosen.add(chunk.chunk_id)
                    overview.append(chunk)
            if overview:
                return overview[:max(0, top_k)]
        meaningful = [term for term in tokens(query) if term not in _STOPWORDS]
        extra = [
            alias for pattern, alias in _ALIASES if re.search(pattern, query, re.IGNORECASE)
        ]
        if not meaningful and not extra:
            return []
        return super().search_vacancy(" ".join(meaningful + extra), top_k=top_k)
