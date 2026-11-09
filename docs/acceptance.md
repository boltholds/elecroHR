# Acceptance record

## Local checks (2026-11-09)

- Python environment: Python 3.12.13; dependency management migrated to Poetry 2.2.1
  with locked dependency resolution in poetry.lock.
- Offline unit/API/provider-fixture tests: **25 passed** (1.11 s); one upstream
  Starlette deprecation warning about its httpx TestClient bridge.
- Ruff lint and formatting, strict mypy (without disk cache): passed.
- Separate uvicorn process at 127.0.0.1:18739: health, create demo, state and report
  endpoints succeeded. Call `96ae1fbf-bd03-40ee-b0c6-67346c49797c` completed, report
  persisted in `data/`, total 50, method demo_rules, recommendation manual_review.
- No real phone call or OpenAI model request was made.

## Real test checklist — not yet verified

- [ ] Free disk space / writable DATA_DIR and one worker only.
- [ ] OpenAI project/model access; successful create request against real API.
- [ ] Provider TLS, digest authentication, authorized caller ID, SDES-SRTP, PSTN routing.
- [ ] Correct ringing/answered events and immediate sideband attachment.
- [ ] AI disclosure and consent/refusal interpretation heard by the participant.
- [ ] Bidirectional Russian audio, latency, and barge-in without clipped candidate speech.
- [ ] Context-aware technical follow-up without repeated questions.
- [ ] Vacancy questions use actual RAG tool results; unknown salary stays unknown.
- [ ] Natural farewell and hangup timing; provider confirms the call has ended.
- [ ] session.closed received, final usage and reason persisted.
- [ ] Actual Responses scoring output validates, quotations match candidate speech,
  missing competencies are explicit and no non-professional traits are scored.
- [ ] HTTP report agrees with durable JSON, reviewed by HR.

Mocked transports verify application behavior and documented shapes only; they are
not proof of provider interoperability, model behavior or live call success.
