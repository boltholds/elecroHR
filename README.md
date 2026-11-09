# AI HR Voice Recruiter MVP

Python 3.12+ / FastAPI backend for **one explicitly initiated test call at a time**.
Live uses native GPT-Live and an outbound SIP trunk. Demo is fully offline and
uses a synthetic text interview; it does not prove voice quality or real scoring.

**Real phone acceptance has not been performed.** API contracts were checked
against official documentation on 2026-11-09; account access and provider
compatibility still need a real test. No credentials or phone number are bundled.

## Run locally

Requires Python 3.12+ and Poetry 2.2+. From this repository directory (PowerShell):

```powershell
poetry install
Copy-Item .env.example .env
poetry run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Use exactly **one worker and one application instance per DATA_DIR**. Do not use
`--reload` during a live call. Open <http://127.0.0.1:8000/docs> for API documentation.
The default data directory is `data/`, relative to the source root; it is gitignored.
On a full disk, set `DATA_DIR` to a writable volume with free space before calling.

## Offline demo

In another terminal:

```powershell
$call = Invoke-RestMethod http://127.0.0.1:8000/calls -Method Post `
  -ContentType 'application/json' -Body '{"mode":"demo"}'
Invoke-RestMethod "http://127.0.0.1:8000/calls/$($call.call_id)"
Invoke-RestMethod "http://127.0.0.1:8000/calls/$($call.call_id)/report"
```

If the report is still processing, the report endpoint returns HTTP 409; poll again.
When `API_TOKEN` is set, include `-Headers @{Authorization="Bearer $env:API_TOKEN"}`
on all `/calls` requests. A `.env` file is read by the app, not by PowerShell;
set the corresponding `$env:API_TOKEN` in the terminal separately.

The demo traverses created → dialing → ringing → active → completed, performs
BM25 vacancy lookup, branches one follow-up based on the fixture answer, and
produces a persisted JSON report. `scoring_method=demo_rules` clearly identifies
simple marker-based fixture probes. This score must not be used to assess a real
candidate. It checks plumbing without making any OpenAI or SIP request.

## First real phone call

1. Obtain an OpenAI project API key with access to `gpt-live-1` and the configured
   Responses models. Set billing/usage limits in that project as appropriate.
2. Configure an outbound SIP trunk with your provider: **TLS signaling and
   SDES-SRTP media**, digest username/password, outbound PSTN routing and an
   authorized E.164 caller ID. Check provider network requirements using the
   official SIP page below. An ordinary SIP account without SRTP is insufficient.
3. Choose a single test participant and a number you are permitted to call.
   Set `ALLOWED_TEST_NUMBER` to that exact E.164 number. No bulk list is accepted.
4. Fill `.env`: `OPENAI_API_KEY`, `API_TOKEN`, `SIP_PROVIDER_URL`, `SIP_USERNAME`,
   `SIP_PASSWORD`, `SIP_CALLER_NUMBER`, `ALLOWED_TEST_NUMBER`; set `LIVE_ENABLED=true`.
   `SIP_PROVIDER_URL` looks like `sips:provider.example:5061`, without credentials.
   `LIVE_MODEL`, `BACKEND_MODEL`, `SCORING_MODEL`, and `LIVE_VOICE` have defaults
   in `.env.example` and can be changed for models enabled on your project.
5. Keep `REQUIRE_CONSENT=true` for the supplied scenario. The agent discloses AI
   use and asks permission to retain the transcript and analyze answers. Audio
   recording is disabled (`store=false`); audio/text still reach OpenAI to operate
   the conversation. Local persistence/scoring waits for the consent tool.
6. Start the server using the command above. Do not use shared/public deployment
   for this local MVP. Health is public; all call/report routes use `API_TOKEN`.
7. Set the same API token and permitted number in the terminal, then explicitly dial:

```powershell
$env:API_TOKEN = 'your-local-api-token'
$testNumber = '+YOUR_PERMITTED_NUMBER'
$headers = @{ Authorization = "Bearer $env:API_TOKEN" }
$body = @{ mode='live'; destination=$testNumber; confirm_test_call=$true } | ConvertTo-Json
$call = Invoke-RestMethod http://127.0.0.1:8000/calls -Method Post `
  -Headers $headers -ContentType 'application/json' -Body $body
Invoke-RestMethod "http://127.0.0.1:8000/calls/$($call.call_id)" -Headers $headers
```

8. Answer the phone. Check the AI disclosure, consent, two-way Russian speech,
   interruption during agent speech, a concrete technical follow-up, and a vacancy
   question (e.g. remote work or salary). Verify that unsupported facts are not invented.
9. End the conversation normally, or cancel explicitly:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/calls/$($call.call_id)/cancel" `
  -Method Post -Headers $headers
```

10. Poll state, then fetch the report:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/calls/$($call.call_id)/report" -Headers $headers |
  ConvertTo-Json -Depth 12
```

Verify `finalization_confirmed=true`, the close reason, transcript accuracy,
RAG sources in `data/<call_id>.call.json`, evidence quotations and criterion scores
in `data/<call_id>.report.json`. Partial assessment is explicit. Only HR makes a decision.
Check that the provider has ended the call, especially after any local/network failure.

## Architecture and files

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI, authentication, lifespan and safe HTTP errors |
| `app/controller.py` | One-call reservation, lifecycle, cancellation, shutdown, post-call scoring |
| `app/telephony.py` | Actual Live HTTP SIP create/hangup adapter, never simulates Live |
| `app/voice_agent.py` | Sideband, transcript deltas, deduplication, delegated tools, finalization |
| `app/prompts.py` | Interview and backend instructions, strict tool argument schemas |
| `app/rag.py` | Markdown chunking and in-memory BM25 with source IDs |
| `app/scoring.py` | Responses structured output, rubric checks and exact evidence validation |
| `app/schemas.py` | Pydantic DTOs, discriminated demo/live request, transitions, report invariants |
| `app/call_store.py` | In-memory state plus atomic disk snapshots and restart recovery |
| `app/demo.py` | Synthetic text fixture with a context-dependent follow-up |
| `app/config.py` | Environment settings and live preflight |
| `knowledge/*.md` | Clearly marked test vacancy and FAQ |
| `scoring_rubric.json` | Four weighted criteria, evaluation anchors |
| `tests/` | Unit, HTTP flow and mocked provider failure tests |
| `docs/api-contracts.md` | Verified official endpoints and event contract notes |

Media path: participant ↔ SIP provider ↔ GPT-Live. Native duplex audio and
interruptions belong to GPT-Live/SIP. The backend does not transcode audio, construct
STT → LLM → TTS, or send microphone audio over the sideband.

Control path: API → CallController → Live create → immediate sideband attach →
transcripts and Responses delegation → local tools → session.closed → separate scoring.
File I/O is moved off the event loop with `asyncio.to_thread`; HTTP/WebSocket I/O is async.

## API and failures

| Route | Behavior |
|---|---|
| `GET /health` | Process health and live-enabled flag, not external provider readiness |
| `POST /calls` | 202 accepted, 409 busy/ambiguous prior call, 503 configuration problem |
| `GET /calls/{id}` | State, transition history, consent, report status; no phone or transcript |
| `GET /calls/{id}/report` | 200 report; 409 pending; 503 scoring failed; 404 unknown |
| `POST /calls/{id}/cancel` | 202 cancellation requested; state changes asynchronously |

HTTP creation success does **not** imply the phone was answered. Only
`transport.answered` transitions to active. Replayed events/tool calls are deduplicated.
Socket disconnection alone never counts as successful finalization.
Scoring errors are recorded separately from call outcome; invalid JSON or unsupported
evidence is rejected instead of replaced with an invented assessment.

An ambiguous creation timeout sets `dialing_outcome_unknown`; new live calls are
blocked even after restart. No request retry is performed. Reconcile with the provider
and end any surviving call first. With the app stopped, archive that call's JSON records
outside DATA_DIR after verification, then restart. Do not clear this protection blindly.
`cleanup_unconfirmed` similarly requires provider-side reconciliation.
On restart, known unfinalized session IDs receive a best-effort hangup; interviews
are never resumed. A process killed before it learns the provider session ID cannot
hang up that unknown session automatically.

## Data and evaluation boundaries

- Only exact candidate quotations can substantiate criterion scores; totals and all
  four weights (35/25/20/20) are validated. No-evidence criteria have `assessed=false`
  and zero as an unassessed placeholder. `assessment_complete=false` makes this explicit.
- Recommendation is always `manual_review`. No automatic rejection or job offer.
- Pending-consent transcripts exist transiently in memory to interpret the answer;
  they are excluded from disk snapshots and scoring. Refusal clears them.
  Consent interpretation is model-driven and must be checked during real acceptance.
- Documents are returned as tool data, never system instructions. Source text cannot
  authorize operations. Tools are allowlisted and arguments are validated.
- Personal transcript/report files are local plaintext, not encrypted storage. Restrict
  OS access and manage retention/deletion according to your deployment scenario.
- The tool requests a brief farewell and uses a bounded grace period before hangup.
  Sideband transcript events do not prove that the participant heard all audio.
  Verify farewell timing and interruption behavior on the real provider.
- No distributed queue, database, cross-process lock, reconnect/resume, or guaranteed
  crash recovery. The service is deliberately a single-process MVP.
- BM25 is lexical; vocabulary mismatch can return no results. The agent must admit
  missing information. Replace demo content before any non-demo recruitment usage.

## Checks

```powershell
poetry run pytest -q
poetry run ruff check app tests
poetry run ruff format --check app tests
poetry run mypy app --cache-dir=nul --no-incremental
```

The final command disables mypy's disk cache on Windows; on Linux use
`--cache-dir=/dev/null`. Tests use mocked HTTP/WebSocket providers and do not place
real calls. A Starlette warning about its httpx TestClient bridge is a dependency
deprecation, not a failing test.

## Verified official API references

- [SIP telephony and outbound calls](https://developers.openai.com/api/docs/guides/voice-sip)
- [Live Create session](https://developers.openai.com/api/reference/resources/live/methods/create)
- [Sideband controls](https://developers.openai.com/api/docs/guides/voice-server-controls)
- [Delegation and tools](https://developers.openai.com/api/docs/guides/live-delegation)
- [Sessions, transcripts and graceful close](https://developers.openai.com/api/docs/guides/live-conversations)
- [Structured outputs for scoring](https://developers.openai.com/api/docs/guides/structured-outputs)
