"""
Step 2 — WebSocket echo test.
No auth required. Use to verify WS infrastructure before wiring real services.

Test with wscat:
  npx wscat -c ws://localhost:8000/ws/echo

Or with Python:
  python -c "
  import asyncio, websockets
  async def t():
      async with websockets.connect('ws://localhost:8000/ws/echo') as ws:
          await ws.send('Hello EchoCoach')
          print(await ws.recv())
  asyncio.run(t())
  "
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

router = APIRouter()

# Starlette 1.0 enforces CORS on WebSocket upgrades. The echo endpoint
# is test-only and explicitly accepts any origin.



@router.websocket("/ws/echo")
async def echo(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "connected", "message": "EchoCoach echo ready"}))
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"type": "echo", "data": data}))
    except WebSocketDisconnect:
        pass
