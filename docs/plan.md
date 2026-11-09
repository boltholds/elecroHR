# AI HR Voice Recruiter — implementation plan

Goal: one explicit outbound test call, a native GPT-Live interview, vacancy RAG,
and a validated report for human review. User specification is authoritative;
implementation proceeds autonomously as requested.

Architecture: one FastAPI process, direct Live SIP creation, Responses delegation
for custom tools, sideband transcripts, BM25 in memory, atomic JSON persistence.
No audio relay or separate STT/TTS. Offline demo uses an explicitly labeled text
fixture and deterministic rubric probes; it is not a live voice acceptance test.

## Tasks
- [x] DTOs, checked transitions, configuration, persistence; tests for invalid transitions,
  restart interruption, report totals, missing credentials, destination restrictions.
- [x] RAG and rubric; tests for ranking, empty corpus, unknown queries, missing evidence,
  invented evidence, invalid JSON and partial assessment.
- [x] HTTP SIP adapter and sideband agent; tests for actual documented request shapes,
  duplicate events/tools, transcript ordering, consent, disconnect and finalization.
- [x] Controller/API lifecycle; tests for demo report, concurrency, cancellation,
  network errors, safe error responses, shutdown and report persistence.
- [x] README/env, lint/type checking, full tests and real local HTTP demo smoke.

## Review focus
Never retry an ambiguous SIP creation; avoid an accidental second call. A successful
create response is not answered. Sideband disconnect is not successful completion.
Unassessed criteria are not proof of low ability. Model/tool output is untrusted.
No transcript persistence/scoring before configured consent; no provider audio recording.
One worker only; no resume of an interrupted interview after process restart.

## Design decisions
Direct Live API is documented; Realtime call creation is WebRTC only. SIP provider
must supply TLS + SDES-SRTP and digest credentials. Responses delegation reduces
custom orchestration. Tool execution stays local and allowlisted. A real test needs
operator credentials and a permitted test number; none is inferred or dialed here.

## Verification and review ledger
- Core tests were run before their modules existed (expected missing-module failures).
- Incremental suites: 9, 13, 17, 21, then 23 passing tests.
- Regression: crash after ambiguous SIP creation must preserve the no-redial marker;
  reproduced failure, fixed restart preservation, suite green.
- Independent read-only reviewer found two issues: disk failure leaving report pending,
  and successful recovery retaining cleanup block. Both reproduced in failing tests,
  fixed with supervisor failure state and explicit hangup confirmation; suite green.
- Added checks for refusal clearing transcripts and safe RAG errors.
- Strict mypy succeeded with cache disabled. Its initial SQLite cache failure was
  caused by nearly full C: drive. Automatic permission review denied cache deletion;
  no cleanup was performed.
- Live localhost smoke on port 18739: health=ok, demo call completed, persisted report
  score=50, scoring_method=demo_rules, assessment_complete=true, manual_review.
- No actual OpenAI/SIP requests: live disabled, key/SIP credentials/test number absent.
- Real acceptance remains open: provider TLS/SRTP, account/model access, native audio,
  consent interpretation, barge-in, tool latency, farewell timing, transcript fidelity,
  actual structured scoring and terminal session event.
