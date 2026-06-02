"""
Step 2 verification — WebSocket echo test.
Run with: python test_ws_echo.py
Server must be running: uvicorn main:app --port 8000
"""
import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)


async def test_echo():
    uri = "ws://localhost:8000/ws/echo"
    print(f"Connecting to {uri} ...")

    # Starlette 1.0 checks Origin on WS upgrade — send localhost to pass CORS
    async with websockets.connect(uri, additional_headers={"Origin": "http://localhost"}) as ws:
        # Should receive connected message
        raw = await ws.recv()
        msg = json.loads(raw)
        assert msg["type"] == "connected", f"Expected connected, got {msg}"
        print(f"[OK] Connected: {msg['message']}")

        # Send a text message, expect echo
        payload = "Hello EchoCoach — Step 2 test"
        await ws.send(payload)
        raw = await ws.recv()
        msg = json.loads(raw)
        assert msg["type"] == "echo", f"Expected echo, got {msg}"
        assert msg["data"] == payload, f"Echo mismatch: {msg['data']}"
        print(f"[OK] Echo received: {msg['data']}")

        # Send JSON payload
        await ws.send(json.dumps({"type": "ping", "ts": 12345}))
        raw = await ws.recv()
        msg = json.loads(raw)
        assert msg["type"] == "echo"
        print(f"[OK] JSON echo: {msg['data']}")

    print("\nStep 2 PASS — WebSocket echo working correctly.")


if __name__ == "__main__":
    asyncio.run(test_echo())
