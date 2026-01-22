FROM python:3.11
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/

ENV UV_NO_DEV=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

COPY models ./models
COPY data ./data
COPY src ./src

RUN uv sync --frozen

CMD [ "uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]