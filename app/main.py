import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.call_store import CallStore
from app.config import Settings
from app.controller import BusyError, CallController
from app.rag import RAGService
from app.schemas import CallRequest, CallView, Report
from app.scoring import ScoringService


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()
    store = CallStore(config.data_dir)
    controllers: list[CallController] = []

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        await store.load()
        rag = await RAGService.load(config.knowledge_dir)
        async with httpx.AsyncClient(
            timeout=config.http_timeout_seconds, follow_redirects=False
        ) as client:
            scoring = await ScoringService.create(config, client)
            controller = CallController(config, store, rag, scoring, client)
            await controller.recover()
            controllers.append(controller)
            try:
                yield
            finally:
                await controller.shutdown()
                controllers.clear()

    application = FastAPI(title="AI HR Voice Recruiter", version="0.1.0", lifespan=lifespan)

    async def authorized(authorization: Annotated[str, Header()] = "") -> None:
        token = config.api_token.get_secret_value()
        if token and not hmac.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(401, "Invalid API token")

    auth = [Depends(authorized)]

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]}
                    for e in error.errors()
                ]
            },
        )

    @application.get("/health")
    async def health() -> dict[str, str | bool]:
        return {"status": "ok", "live_enabled": config.live_enabled}

    @application.post("/calls", response_model=CallView, status_code=202, dependencies=auth)
    async def create_call(body: CallRequest) -> CallView:
        try:
            call = await controllers[0].start(body)
        except ValueError as error:
            raise HTTPException(503, str(error)) from None
        except BusyError as error:
            raise HTTPException(409, str(error)) from None
        except OSError:
            raise HTTPException(503, "storage_unavailable") from None
        return CallView.from_record(call)

    @application.get("/calls/{call_id}", response_model=CallView, dependencies=auth)
    async def get_call(call_id: str) -> CallView:
        try:
            return CallView.from_record(store.get(call_id))
        except KeyError:
            raise HTTPException(404, "Call not found") from None

    @application.get("/calls/{call_id}/report", response_model=Report, dependencies=auth)
    async def get_report(call_id: str) -> Report:
        try:
            call = store.get(call_id)
        except KeyError:
            raise HTTPException(404, "Call not found") from None
        if call.report_state == "failed":
            raise HTTPException(503, call.report_error)
        if call.report_state != "ready":
            raise HTTPException(409, "Report is not ready")
        return store.reports[call_id]

    @application.post(
        "/calls/{call_id}/cancel", response_model=CallView, status_code=202, dependencies=auth
    )
    async def cancel_call(call_id: str) -> CallView:
        try:
            controllers[0].cancel(call_id)
            return CallView.from_record(store.get(call_id))
        except KeyError:
            raise HTTPException(404, "Call not found") from None
        except BusyError as error:
            raise HTTPException(409, str(error)) from None

    return application


app = create_app()
