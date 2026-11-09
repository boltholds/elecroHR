"""Pure audio turn detection and report contracts; safe to test without ML weights."""
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LOCAL_WEIGHTS = {
    "llm_fundamentals": 25,
    "rag_retrieval": 35,
    "evaluation_quality": 20,
    "deployment_agents": 20,
}
LocalCriterion = Literal[
    "llm_fundamentals", "rag_retrieval", "evaluation_quality", "deployment_agents",
]


class LocalCriterionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion: LocalCriterion
    score: int = Field(ge=0, strict=True)
    max_score: int = Field(gt=0, strict=True)
    evidence: list[str]
    assessed: bool
    explanation: str

    @model_validator(mode="after")
    def validate_score(self) -> "LocalCriterionScore":
        if self.max_score != LOCAL_WEIGHTS[self.criterion] or self.score > self.max_score:
            raise ValueError("Invalid local criterion weight or score")
        if not self.assessed and (self.score != 0 or self.evidence):
            raise ValueError("Unassessed criterion must have zero score and no evidence")
        if self.assessed and not any(text.strip() for text in self.evidence):
            raise ValueError("Assessed criterion must have candidate evidence")
        return self


class LocalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criteria: list[LocalCriterionScore]
    strengths: list[str]
    weaknesses: list[str]
    summary: str

    @model_validator(mode="after")
    def validate_criteria(self) -> "LocalAssessment":
        keys = [item.criterion for item in self.criteria]
        if len(keys) != len(LOCAL_WEIGHTS) or set(keys) != set(LOCAL_WEIGHTS):
            raise ValueError("Exactly the four local evaluation criteria are required")
        return self



class LocalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    status: Literal["completed"] = "completed"
    score: int = Field(ge=0, le=100)
    max_score: Literal[100] = 100
    assessment: LocalAssessment
    unassessed_criteria: list[str]
    assessment_complete: bool
    recommendation: Literal["manual_review"] = "manual_review"
    scoring_method: Literal["ollama", "insufficient_data"]

    @model_validator(mode="after")
    def validate_totals(self) -> "LocalReport":
        assert sum(LOCAL_WEIGHTS.values()) == 100
        if self.score != sum(c.score for c in self.assessment.criteria):
            raise ValueError("Score must match criterion total")
        absent = [c.criterion for c in self.assessment.criteria if not c.assessed]
        if absent != self.unassessed_criteria or self.assessment_complete != (not absent):
            raise ValueError("Unassessed criteria mismatch")
        return self


@dataclass
class EndpointDetector:
    """Accept exactly 32 ms 16kHz mono PCM16 frames and VAD probabilities."""

    threshold: float = 0.5
    min_speech_frames: int = 3
    silence_frames: int = 22
    max_frames: int = 625
    preroll: deque[bytes] = field(default_factory=lambda: deque(maxlen=10))
    frames: list[bytes] = field(default_factory=list)
    voiced: int = 0
    silent: int = 0
    active: bool = False

    def reset(self) -> None:
        self.preroll.clear()
        self.frames.clear()
        self.voiced = 0
        self.silent = 0
        self.active = False

    def feed(self, frame: bytes, probability: float) -> tuple[bool, bytes]:
        """Return (speech_started, completed_pcm); empty bytes means no endpoint."""
        if len(frame) != 1024:
            raise ValueError("PCM frame must contain 512 samples")
        if not self.active:
            self.preroll.append(frame)
            self.voiced = self.voiced + 1 if probability >= self.threshold else 0
            if self.voiced < self.min_speech_frames:
                return False, b""
            self.active = True
            self.frames = list(self.preroll)
            self.preroll.clear()
            return True, b""
        self.frames.append(frame)
        self.silent = self.silent + 1 if probability < self.threshold else 0
        if self.silent >= self.silence_frames or len(self.frames) >= self.max_frames:
            completed = b"".join(self.frames)
            self.reset()
            return False, completed
        return False, b""


def empty_assessment(reason: str) -> LocalAssessment:
    return LocalAssessment(
        criteria=[
            LocalCriterionScore(
                criterion=name, max_score=weight, score=0,
                assessed=False, evidence=[], explanation=reason,
            )
            for name, weight in LOCAL_WEIGHTS.items()
        ],
        strengths=[], weaknesses=[], summary=reason,
    )


def make_report(session_id: str, assessment: LocalAssessment, method: Literal["ollama", "insufficient_data"]) -> LocalReport:
    missing = [item.criterion for item in assessment.criteria if not item.assessed]
    return LocalReport(
        session_id=session_id, score=sum(item.score for item in assessment.criteria),
        assessment=assessment, unassessed_criteria=missing,
        assessment_complete=not missing, scoring_method=method,
    )
