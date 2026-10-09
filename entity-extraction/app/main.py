"""FastAPI app: GET / (HTML), POST /analyze, GET /health.

Plan section 11, Phase 7, and the lifecycle of section 8.3.

Deliberately small: no auth, no database, no job queue. One upload is analysed
synchronously and the JSON of section 12 comes back.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.pipeline import analyze
from app.schemas import Analysis

logger = logging.getLogger("uzdoc")

STATIC = Path(__file__).parent / "static"

#: Plan section 11: uploads are capped at 20 MB.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK = 1024 * 1024

PDF_MAGIC = b"%PDF-"

#: The PDF header need not sit at byte 0 — the spec lets a reader scan for it,
#: and real files do have bytes in front (one sample carried its own filename
#: as leading text, putting %PDF- at offset 26). Checking only byte 0 rejects
#: perfectly readable documents.
PDF_MAGIC_SEARCH_BYTES = 1024


class State:
    """What the app knows about its model (plan section 8.3).

    Starts "starting"; the lifespan task fills this in. ``/analyze`` refuses
    with 503 until it is ready, so a request never waits on a model pull.
    """

    def __init__(self) -> None:
        self.status = "starting"
        self.model: str | None = None
        self.tier: str | None = None
        self.reason = "model selection has not run yet"


state = State()


async def _prepare_model() -> None:
    """Pick, pull, warm up and verify the model (section 8.3)."""
    from app.model_select import prepare

    try:
        await prepare(state)
    except Exception as exc:  # the app still serves /health and says why
        state.status = "error"
        state.reason = f"{type(exc).__name__}: {exc}"
        logger.exception("model preparation failed")


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_prepare_model())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Uzbek document analysis", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health() -> dict:
    return {
        "status": state.status,
        "model": state.model,
        "tier": state.tier,
        "reason": state.reason,
    }


async def _save_upload(upload: UploadFile) -> Path:
    """Stream the upload to a temp file, enforcing the size cap as we go."""
    handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    total = 0
    first = b""
    try:
        while chunk := await upload.read(UPLOAD_CHUNK):
            if len(first) < PDF_MAGIC_SEARCH_BYTES:
                first += chunk[: PDF_MAGIC_SEARCH_BYTES - len(first)]
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            handle.write(chunk)
    finally:
        handle.close()

    path = Path(handle.name)
    if total == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The file is empty.")
    if PDF_MAGIC not in first:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="That file is not a PDF.")
    return path


@app.post("/analyze", response_model=Analysis)
async def analyze_upload(file: UploadFile = File(...)) -> Analysis:
    if state.status != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"The model is not ready yet ({state.status}): {state.reason}. "
            "The first start downloads it; try again shortly.",
        )

    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    path = await _save_upload(file)
    try:
        return await analyze(str(path), state.model)
    except Exception as exc:
        logger.exception("analysis failed for %s", file.filename)
        raise HTTPException(
            status_code=500, detail=f"Could not analyse the document: {exc}"
        ) from exc
    finally:
        path.unlink(missing_ok=True)


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
