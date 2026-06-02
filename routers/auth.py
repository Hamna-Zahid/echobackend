from fastapi import APIRouter, Depends, HTTPException
from typing import Literal, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator

from db.supabase import get_supabase, get_admin_client, get_current_user

router = APIRouter()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=80)
    target_exam: Literal["IELTS", "TOEFL", "INTERVIEW", "Job Interview", "General English"] = "IELTS"
    preferred_accent: Literal["american", "british"] = "american"

    @field_validator("full_name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name is required")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class ProfileUpdateRequest(BaseModel):
    cefr_level: Optional[Literal["A1", "A2", "B1", "B2", "C1", "C2"]] = None
    target_exam: Optional[Literal["IELTS", "TOEFL", "INTERVIEW", "General English"]] = None
    preferred_accent: Optional[Literal["american", "british"]] = None
    goal: Optional[str] = None


@router.post("/register")
async def register(req: RegisterRequest):
    supabase = get_supabase()
    try:
        # admin.create_user with email_confirm=True guarantees the auth.users row
        # is fully committed before we touch public.users — sign_up() via service
        # role key leaves the row in a pending state that breaks the FK insert.
        res = supabase.auth.admin.create_user({
            "email": req.email,
            "password": req.password,
            "email_confirm": True,
        })
        if res.user is None:
            raise HTTPException(status_code=400, detail="Registration failed")

        user_id = res.user.id

        # Normalize target_exam to the DB-allowed set (the check constraint predates
        # the "Job Interview" label the app sends).
        exam_map = {
            "IELTS": "IELTS", "TOEFL": "TOEFL",
            "Job Interview": "INTERVIEW", "INTERVIEW": "INTERVIEW",
            "General English": "General English",
        }
        db_exam = exam_map.get(req.target_exam, "IELTS")

        # Upsert profile fields onto the row the trigger already created.
        upsert_data = {
            "id": user_id,
            "full_name": req.full_name,
            "target_exam": db_exam,
            "exam_target": req.target_exam,   # original label (for accent-mode routing)
            "preferred_accent": req.preferred_accent,
        }
        supabase.table("users").upsert(upsert_data, on_conflict="id").execute()

        # Sign the user in to return a usable session token immediately.
        sign_in = supabase.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password,
        })

        return {
            "user_id": user_id,
            "access_token": sign_in.session.access_token if sign_in.session else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def login(req: LoginRequest):
    supabase = get_supabase()
    try:
        res = supabase.auth.sign_in_with_password({"email": req.email, "password": req.password})
        if res.session is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {
            "access_token": res.session.access_token,
            "user_id": res.user.id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/logout")
async def logout():
    supabase = get_supabase()
    supabase.auth.sign_out()
    return {"status": "logged out"}


@router.patch("/profile")
async def update_profile(
    req: ProfileUpdateRequest,
    user: dict = Depends(get_current_user),
):
    supabase = get_supabase()
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        return {"status": "no changes"}
    supabase.table("users").update(updates).eq("id", user["id"]).execute()
    return {"status": "updated"}


@router.get("/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    supabase = get_supabase()
    res = supabase.table("users").select("*").eq("id", user["id"]).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Profile not found")
    return res.data


@router.delete("/account")
async def delete_account(user: dict = Depends(get_current_user)):
    """
    Permanently delete the authenticated user's account and all data.
    FK cascades handle sessions, errors, and learning_plans automatically.
    """
    # Fresh admin client — the singleton was mutated by verify_token's get_user() call.
    admin = get_admin_client()
    user_id = user["id"]

    try:
        admin.table("users").delete().eq("id", user_id).execute()
    except Exception:
        pass  # Row may not exist for brand-new users

    try:
        admin.auth.admin.delete_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auth deletion failed: {str(e)}")

    return {"status": "deleted"}
