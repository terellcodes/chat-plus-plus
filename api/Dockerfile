FROM python:3.9 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN python -m venv .venv
COPY pyproject.toml ./
RUN .venv/bin/pip install .
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .
CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
