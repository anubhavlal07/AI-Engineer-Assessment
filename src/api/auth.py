"""Simple shared-secret auth via the X-API-Key header.

For a production system this would be replaced with OAuth2/JWT and per-user
identity, but a shared key is sufficient to demonstrate the auth boundary.
"""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from src.config import get_settings


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = get_settings().api_key
    if not expected:
        # No key configured -> auth disabled (dev convenience).
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
