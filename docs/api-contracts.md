# Verified API contract — 2026-11-09

Read official pages and their Markdown versions before implementing the adapter.
No live credentials were used to validate these shapes against the service.

| Action | Contract | Source |
|---|---|---|
| Outbound call | `POST https://api.openai.com/v1/live/sessions`; `session.model=gpt-live-1`, `transport.type=sip`, E.164 destination, trunk with `provider_url`, digest `auth`, `caller_number` | [Create](https://developers.openai.com/api/reference/resources/live/methods/create), [SIP](https://developers.openai.com/api/docs/guides/voice-sip) |
| Audio | Omit audio.format; SIP negotiates TLS + SDES-SRTP. `audio.output.voice` set at startup | [SIP](https://developers.openai.com/api/docs/guides/voice-sip) |
| Attach | `wss://api.openai.com/v1/live/sessions/{session_id}/attach`, bearer API key; no `session.start` | [Sideband](https://developers.openai.com/api/docs/guides/voice-server-controls) |
| Progress | `transport.ringing`, `transport.answered`, `transport.failed`; deduplicate event_id; attach immediately (short replay window) | [SIP](https://developers.openai.com/api/docs/guides/voice-sip) |
| Tools | Configure `delegation.responses.tools`; nested `response.event` / `response.output_item.done` contains completed function_call | [Delegation](https://developers.openai.com/api/docs/guides/live-delegation) |
| Tool result | `response.item.create` with function_call_output and call_id; then `response.create` with no standalone Responses request body | [Sideband schema](https://developers.openai.com/api/reference/resources/live/sideband-websocket) |
| Transcript | `session.input_transcript.delta` and `session.output_transcript.delta`: delta, start_ms, end_ms, event_id; preserve fragments, no assumed turn-completed event | [Sessions](https://developers.openai.com/api/docs/guides/live-conversations) |
| Hangup | `POST /v1/live/sessions/{session_id}/hangup`, no body; keep receiving until session.closed with reason and usage | [SIP](https://developers.openai.com/api/docs/guides/voice-sip), [Sessions](https://developers.openai.com/api/docs/guides/live-conversations) |
| Scoring | `POST /v1/responses`, `text.format.type=json_schema`, strict schema, inspect completed output message text | [Structured output](https://developers.openai.com/api/docs/guides/structured-outputs) |

Documentation inconsistency: the SIP guide describes HTTP 200 at creation, while
the API reference describes 201. The implementation accepts either successful status
and validates `session.id`; neither status is interpreted as an answered call.

Realtime `/v1/realtime/calls` creation is not used for outbound SIP. There is no
invented provider status polling endpoint or assumed transcript-final event.
No inbound webhook is needed for this outbound-only design.
