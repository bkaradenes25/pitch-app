# Stage 1: build the React app (relative API URLs, so one server serves everything)
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
ENV VITE_API_URL=""
RUN npm run build

# Stage 2: FastAPI serving the API and the built UI
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 DB_PATH=/tmp/pitch.db
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY backend/ .
COPY --from=web /web/dist ./static
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
