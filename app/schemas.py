from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class DTO(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CallStatus(StrEnum):
    created = "created"
    dialing = "dialing"
    ringing = "ringing"
    active = "active"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"


TERMINAL = {CallStatus.completed, CallStatus.failed, CallStatus.canceled}
TRANSITIONS = {
    CallStatus.created: {CallStatus.dialing, CallStatus.failed, CallStatus.canceled},
    CallStatus.dialing: {
        CallStatus.ringing,
        CallStatus.active,
        CallStatus.failed,
        CallStatus.canceled,
    },
    CallStatus.ringing: {CallStatus.active, CallStatus.failed, CallStatus.canceled},
    CallStatus.active: TERMINAL,
}
WEIGHTS = {"python_fastapi": 35, "postgresql_sql": 25, "docker": 20, "async_python": 20}
CriterionName = Literal["python_fastapi", "postgresql_sql", "docker", "async_python"]
Mode = Literal["demo", "live"]


class DemoRequest(DTO):
    mode: Literal["demo"] = "demo"


class LiveRequest(DTO):
    mode: Literal["live"]
    destination: str = Field(pattern=r"^\+[1-9]\d{7,14}$")
    confirm_test_call: Literal[True]


CallRequest = Annotated[DemoRequest | LiveRequest, Field(discriminator="mode")]


class TranscriptFragment(DTO):
    event_id: str
    role: Literal["candidate", "agent"]
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def interval(self) -> Self:
        if self.end_ms < self.start_ms:
            raise ValueError("Invalid transcript interval")
        return self


class RetrievedChunk(DTO):
    chunk_id: str
    source: str
    text: str
    score: float


class CriterionScore(DTO):
    criterion: CriterionName
    score: int = Field(ge=0, strict=True)
    max_score: int = Field(gt=0, strict=True)
    evidence: list[str]
    assessed: bool
    explanation: str

    @model_validator(mode="after")
    def validate_score(self) -> Self:
        if self.max_score != WEIGHTS[self.criterion] or self.score > self.max_score:
            raise ValueError("Invalid criterion weight or score")
        if not self.assessed and (self.score != 0 or self.evidence):
            raise ValueError("Unassessed criteria must have zero score and no evidence")
        if self.assessed and not any(e.strip() for e in self.evidence):
            raise ValueError("Assessment requires evidence")
        return self


class Assessment(DTO):
    criteria: list[CriterionScore]
    strengths: list[str]
    weaknesses: list[str]
    summary: str

    @model_validator(mode="after")
    def all_criteria(self) -> Self:
        names = [c.criterion for c in self.criteria]
        if len(names) != len(WEIGHTS) or set(names) != set(WEIGHTS):
            raise ValueError("Exactly one of each criterion is required")
        return self


class Report(Assessment):
    call_id: str
    status: CallStatus
    mode: Mode
    score: int = Field(ge=0, le=100, strict=True)
    max_score: Literal[100] = 100
    unassessed_criteria: list[CriterionName]
    assessment_complete: bool
    recommendation: Literal["manual_review"] = "manual_review"
    scoring_method: Literal["llm", "demo_rules", "insufficient_data"]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.status not in TERMINAL:
            raise ValueError("Report requires terminated call")
        if self.score != sum(c.score for c in self.criteria):
            raise ValueError("Score must equal criterion total")
        missing = [c.criterion for c in self.criteria if not c.assessed]
        if missing != self.unassessed_criteria or self.assessment_complete != (not missing):
            raise ValueError("Incomplete assessment must be explicit")
        return self


class CallRecord(DTO):
    call_id: str
    mode: Mode
    status: CallStatus = CallStatus.created
    history: list[CallStatus] = Field(default_factory=lambda: [CallStatus.created])
    provider_session_id: str = ""
    consent: Literal["pending", "granted", "declined", "not_required"] = "pending"
    transcript: list[TranscriptFragment] = Field(default_factory=list)
    retrievals: list[RetrievedChunk] = Field(default_factory=list)
    finalization_confirmed: bool = False
    hangup_confirmed: bool = False
    close_reason: str = ""
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    error_code: str = ""
    report_state: Literal["pending", "ready", "failed"] = "pending"
    report_error: str = ""

    def transition(self, target: CallStatus) -> None:
        if target not in TRANSITIONS.get(self.status, set()):
            raise ValueError(f"Invalid transition: {self.status} -> {target}")
        self.status = target
        self.history.append(target)

    def candidate_text(self) -> str:
        return "".join(
            f.text
            for f in sorted(self.transcript, key=lambda x: x.start_ms)
            if f.role == "candidate"
        )


class CallView(DTO):
    call_id: str
    mode: Mode
    status: CallStatus
    history: list[CallStatus]
    consent: str
    finalization_confirmed: bool
    close_reason: str
    error_code: str
    report_state: str
    report_error: str

    @classmethod
    def from_record(cls, call: CallRecord) -> Self:
        return cls.model_validate({name: getattr(call, name) for name in cls.model_fields})
