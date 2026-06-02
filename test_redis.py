"""
Step 8 verification — Redis session state.
Run: python test_redis.py
Requires REDIS_URL (and REDIS_TOKEN for Upstash) in .env
"""
import asyncio
import time
from dotenv import load_dotenv
load_dotenv()


async def main():
    from services.redis_state import get_session_state, set_session_state, delete_session_state

    sid = f"test-{int(time.time())}"
    print(f"Testing Redis with session key: session:{sid}")

    # Write
    state = {
        "user_id": "test-user",
        "mode": "training",
        "tts_is_speaking": False,
        "diagnostic_turn_count": 0,
        "top_recurring_errors": [],
        "cefr_level": "B2",
        "session_db_id": "test-db-id",
    }
    await set_session_state(sid, state)
    print("[OK] State written")

    # Read
    retrieved = await get_session_state(sid)
    assert retrieved is not None, "State not found after write"
    assert retrieved["user_id"] == "test-user"
    assert retrieved["cefr_level"] == "B2"
    print("[OK] State read correctly")

    # Update
    retrieved["tts_is_speaking"] = True
    await set_session_state(sid, retrieved)
    updated = await get_session_state(sid)
    assert updated["tts_is_speaking"] is True
    print("[OK] State update works")

    # Delete
    await delete_session_state(sid)
    gone = await get_session_state(sid)
    assert gone is None, "State should be gone after delete"
    print("[OK] State deleted")

    print("\nStep 8 PASS — Redis session state working.")


if __name__ == "__main__":
    asyncio.run(main())
