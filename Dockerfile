# EchoCoach backend — production image
# Portable across Railway / Render / Fly.io / Cloud Run.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEV_TTS=edge

WORKDIR /app

# Install deps first (layer cached unless requirements change).
COPY requirements-deploy.txt .
RUN pip install --upgrade pip && pip install -r requirements-deploy.txt

# App source.
COPY . .

# The platform injects $PORT (Railway/Render/Fly). Default 8000 for `docker run` locally.
ENV PORT=8000
EXPOSE 8000

# Start via Python so PORT is read from the environment in-process. This is exec-form
# safe — it works even if a platform overrides the start command without a shell.
CMD ["python", "main.py"]
