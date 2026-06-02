import os
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client

_bearer = HTTPBearer(auto_error=False)

_supabase: Client | None = None


async def init_supabase() -> None:
    global _supabase
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("WARNING: Supabase credentials not configured. DB operations will fail.")
        return
    _supabase = create_client(url, key)


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        _supabase = create_client(url, key)
    return _supabase


def get_admin_client() -> Client:
    """
    Returns a fresh service-role client for admin operations (user deletion, etc.).
    Always fresh — never shares state with the singleton that verify_token touches.
    Calling auth.get_user(jwt) on the singleton mutates its auth header, which breaks
    subsequent auth.admin calls on that same instance.
    """
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    return create_client(url, key)


def verify_token(token: str) -> Optional[dict]:
    """
    Verify a Supabase JWT and return the user payload.
    Uses the singleton client — note: get_user() mutates auth state on the client,
    so admin operations must use get_admin_client() instead.
    """
    try:
        supabase = get_supabase()
        res = supabase.auth.get_user(token)
        if res and res.user:
            return {"id": res.user.id, "email": res.user.email}
        return None
    except Exception:
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """
    FastAPI dependency — extracts and verifies the Bearer JWT from every request.
    Use with: user = Depends(get_current_user)
    Works with Swagger's global Authorize button automatically.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    user = verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token — please log in again")
    return user
