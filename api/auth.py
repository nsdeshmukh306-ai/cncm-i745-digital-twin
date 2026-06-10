"""
Lightweight API-key authentication for the CNCM I-745 Digital Twin API (v4.0.0).

A single shared key is read from the ``DT_API_KEY`` environment variable and
checked via the ``X-API-Key`` request header. The dependency is intended to be
applied to **POST** (state-mutating / compute-heavy) endpoints only; GET
endpoints remain public.

Fail-open design: if ``DT_API_KEY`` is unset or empty, authentication is
*disabled* so that local development and the bundled dashboard keep working out
of the box. When the variable is set (as in the systemd unit), the key is
enforced and a missing/incorrect key yields HTTP 401 ``{"error": "Unauthorized"}``.
"""

from __future__ import annotations

import os

from fastapi import Header
from fastapi.responses import JSONResponse
from starlette.requests import Request

API_KEY_ENV = "DT_API_KEY"


class UnauthorizedError(Exception):
    """Raised when a required API key is missing or incorrect."""


def configured_key() -> str | None:
    """Return the configured API key, or ``None`` when auth is disabled."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def auth_enabled() -> bool:
    return configured_key() is not None


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing the API key on protected routes."""
    expected = configured_key()
    if expected is None:
        return  # auth disabled — accept all
    if not x_api_key or x_api_key != expected:
        raise UnauthorizedError()


async def unauthorized_handler(_: Request, __: UnauthorizedError) -> JSONResponse:
    """Return the exact ``{"error": "Unauthorized"}`` body with HTTP 401."""
    return JSONResponse(status_code=401, content={"error": "Unauthorized"})
