import pytest

from local_voice.core import (
    LOCAL_WEIGHTS, EndpointDetector, LocalAssessment, LocalCriterionScore,
    LocalReport, empty_assessment, make_report,
)
from local_voice.server import sentences


def test_endpointing_and_preroll():
    detector = EndpointDetector(threshold=0.5, min_speech_frames=3, silence_frames=5)
    frame = bytes(1024)
    for _ in range(2):
        started, completed = detector.feed(frame, 0.8)
        assert not started and not completed
    started, completed = detector.feed(frame, 0.8)
    assert started and not completed
    for _ in range(4):
        started, completed = detector.feed(frame, 0.1)
        assert not started and not completed
    started, completed = detector.feed(frame, 0.1)
    assert not started and completed == frame * 8
    assert not detector.active


def test_bad_frame_is_rejected():
    with pytest.raises(ValueError, match="512 samples"):
        EndpointDetector().feed(b"wrong", 0.9)


def test_sentence_streaming():
    ready, tail = sentences("Привет! Как дела? Что с")
    assert ready == ["Привет!", "Как дела?"]
    assert tail == "Что с"


def test_incomplete_score_is_explicit():
    assessment = empty_assessment("нет данных")
    report = make_report("test", assessment, "insufficient_data")
    assert report.score == 0
    assert len(report.unassessed_criteria) == 4
    assert not report.assessment_complete


def test_score_sum_and_weight_validation():
    criteria = [
        LocalCriterionScore(
            criterion=key, score=weight, max_score=weight,
            evidence=["Показал рабочий пример."], assessed=True, explanation="Пример",
        )
        for key, weight in LOCAL_WEIGHTS.items()
    ]
    assessment = LocalAssessment(
        criteria=criteria, strengths=[], weaknesses=[], summary="Хороший пример",
    )
    report = make_report("score-test", assessment, "ollama")
    assert report.score == 100
    with pytest.raises(ValueError, match="Score must match"):
        LocalReport.model_validate({**report.model_dump(), "score": 95})
