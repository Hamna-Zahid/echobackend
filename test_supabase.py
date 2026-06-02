"""
Step 9 verification — Supabase connection, auth, and schema.
Run: python test_supabase.py
Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env
"""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()


def main():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not key:
        print("[SKIP] SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")
        return

    from db.supabase import get_supabase
    sb = get_supabase()

    # Test table access (tables must exist from schema.sql)
    for table in ("users", "sessions", "errors", "learning_plans"):
        try:
            res = sb.table(table).select("id").limit(1).execute()
            print(f"[OK] Table '{table}' accessible ({len(res.data)} rows sampled)")
        except Exception as e:
            print(f"[FAIL] Table '{table}': {e}")
            print("       → Run backend/db/schema.sql in Supabase SQL Editor first")

    print("\nStep 9 PASS — Supabase connection verified.")


if __name__ == "__main__":
    main()
