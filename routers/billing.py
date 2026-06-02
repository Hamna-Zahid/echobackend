"""
Billing & entitlements API.

Pre-launch: the purchase is mocked (POST /billing/subscribe flips the user to
premium). Real Apple/Google IAP via RevenueCat gets wired here at store-submission
time — the rest of the app already gates on the entitlement state this returns.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from db.supabase import get_current_user
from services.user_state import get_memory
from services.entitlements import entitlements, get_subscription, set_premium, cancel_premium
from services.gamification import get_progress

router = APIRouter()


class SubscribeRequest(BaseModel):
    plan: str = "monthly"   # "monthly" | "yearly" (mock)


# Display catalog the paywall renders (prices are illustrative for the mock).
PLANS = [
    {"id": "monthly", "label": "Monthly", "price": "$9.99", "period": "/month", "months": 1, "badge": None},
    {"id": "yearly",  "label": "Yearly",  "price": "$59.99", "period": "/year", "months": 12, "badge": "Save 50%"},
]


@router.get("/status")
async def billing_status(user=Depends(get_current_user)):
    mem = get_memory(user["id"])
    return {
        "subscription": get_subscription(mem),
        "entitlements": entitlements(mem),
        "gamification": get_progress(user["id"], memory=mem),
        "plans": PLANS,
    }


@router.post("/subscribe")
async def subscribe(req: SubscribeRequest, user=Depends(get_current_user)):
    """Mock purchase — flips the user to premium. Replace with RevenueCat verify later."""
    months = 12 if req.plan == "yearly" else 1
    sub = set_premium(user["id"], months=months)
    mem = get_memory(user["id"])
    return {"status": "subscribed", "subscription": sub, "entitlements": entitlements(mem)}


@router.post("/cancel")
async def cancel(user=Depends(get_current_user)):
    sub = cancel_premium(user["id"])
    mem = get_memory(user["id"])
    return {"status": "cancelled", "subscription": sub, "entitlements": entitlements(mem)}
