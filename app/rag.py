import asyncio
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from app.schemas import RetrievedChunk


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w+#]+", text.casefold())


class RAGService:
    def __init__(self, chunks: list[RetrievedChunk]):
        if not chunks:
            raise ValueError("knowledge directory must contain nonempty Markdown documents")
        self.chunks = chunks
        self.terms = [Counter(tokens(c.text)) for c in chunks]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.average = sum(self.lengths) / len(chunks) or 1
        self.df: Counter[str] = Counter()
        for terms in self.terms:
            self.df.update(terms.keys())

    @classmethod
    async def load(cls, directory: Path) -> "RAGService":
        def read() -> list[RetrievedChunk]:
            chunks = []
            for path in sorted(directory.rglob("*.md")):
                source = path.relative_to(directory).as_posix()
                for paragraph in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
                    words = paragraph.split()
                    for offset in range(0, len(words), 160):
                        text = " ".join(words[offset : offset + 200])
                        digest = hashlib.sha256(f"{source}:{offset}:{text}".encode()).hexdigest()[
                            :16
                        ]
                        chunks.append(
                            RetrievedChunk(chunk_id=digest, source=source, text=text, score=0)
                        )
            return chunks

        return cls(await asyncio.to_thread(read))

    def search_vacancy(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        results = []
        for chunk, terms, length in zip(self.chunks, self.terms, self.lengths, strict=True):
            score = 0.0
            for token in set(tokens(query)):
                frequency = terms[token]
                if frequency:
                    idf = math.log(
                        1 + (len(self.chunks) - self.df[token] + 0.5) / (self.df[token] + 0.5)
                    )
                    score += (
                        idf
                        * frequency
                        * 2.5
                        / (frequency + 1.5 * (0.25 + 0.75 * length / self.average))
                    )
            if score > 0:
                results.append(chunk.model_copy(update={"score": score}))
        return sorted(results, key=lambda c: (-c.score, c.chunk_id))[: max(0, min(top_k, 10))]
