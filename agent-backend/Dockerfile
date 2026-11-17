# Builds backend + serves frontend/datasets. Build from repo root:
#   docker build -f agent-backend/Dockerfile -t football-agents .
FROM python:3.11-slim
WORKDIR /srv
COPY datasets ./datasets
COPY frontend ./frontend
COPY agent-backend ./agent-backend
WORKDIR /srv/agent-backend
RUN pip install --no-cache-dir ".[api,agents,observability]"
ENV USE_MOCK_LLM=true
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
