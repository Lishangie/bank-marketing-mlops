FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MPLBACKEND=Agg

WORKDIR /app

# Сначала только зависимости — этот слой кешируется, пока не меняется uv.lock.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Затем код проекта.
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

# Данные не входят в образ: data/ монтируется при запуске (после dvc pull).
CMD ["uv", "run", "--no-sync", "python", "scripts/eda.py"]
