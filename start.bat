@echo off
REM ─── EchoCoach Backend — start script ────────────────────────────────────
REM This MUST be used instead of `uvicorn main:app` directly.
REM   --host 0.0.0.0   → makes the server reachable from your phone/emulator on the LAN
REM   --port 8000      → matches EXPO_PUBLIC_BACKEND_HTTP_URL in mobile/.env
REM
REM After starting, check your LAN IP with:  ipconfig | findstr IPv4
REM Then set mobile/.env:
REM   EXPO_PUBLIC_BACKEND_HTTP_URL=http://<YOUR_IP>:8000
REM   EXPO_PUBLIC_BACKEND_WS_URL=ws://<YOUR_IP>:8000
REM   (Android emulator: use 10.0.2.2 instead of your LAN IP)

cd /d "%~dp0"
call venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8765 --reload
