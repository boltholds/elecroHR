import asyncio
import json
import os
from pathlib import Path

from app.schemas import TERMINAL, CallRecord, CallStatus, Report


class CallStore:
    """Single-process state; atomic disk snapshots, no cross-process locking."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.calls: dict[str, CallRecord] = {}
        self.reports: dict[str, Report] = {}
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        def read() -> tuple[list[CallRecord], list[Report]]:
            self.directory.mkdir(parents=True, exist_ok=True)
            calls = [
                CallRecord.model_validate_json(p.read_text(encoding="utf-8"))
                for p in self.directory.glob("*.call.json")
            ]
            reports = [
                Report.model_validate_json(p.read_text(encoding="utf-8"))
                for p in self.directory.glob("*.report.json")
            ]
            return calls, reports

        calls, reports = await asyncio.to_thread(read)
        self.reports = {r.call_id: r for r in reports}
        for call in calls:
            if call.status not in TERMINAL:
                call.transition(CallStatus.failed)
                call.error_code = call.error_code or "process_interrupted"
            if call.call_id in self.reports:
                call.report_state = "ready"
            elif call.report_state == "pending":
                call.report_state = "failed"
                call.report_error = "process_interrupted"
            await self.save(call)

    def get(self, call_id: str) -> CallRecord:
        return self.calls[call_id]

    async def _write(self, filename: str, content: str) -> None:
        def write() -> None:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / filename
            temporary = target.with_suffix(".tmp")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, target)

        async with self._lock:
            await asyncio.to_thread(write)

    async def save(self, call: CallRecord) -> None:
        self.calls[call.call_id] = call
        data = call.model_dump(mode="json")
        if call.consent not in {"granted", "not_required"}:
            data["transcript"] = []
        await self._write(f"{call.call_id}.call.json", json.dumps(data, ensure_ascii=False))

    async def save_report(self, report: Report) -> None:
        await self._write(f"{report.call_id}.report.json", report.model_dump_json(indent=2))
        self.reports[report.call_id] = report
