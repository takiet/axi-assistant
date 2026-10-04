FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

RUN useradd --create-home --uid 1000 axis \
 && mkdir -p /data/crewai \
 && chown axis:axis /data/crewai
USER axis

EXPOSE 8080
