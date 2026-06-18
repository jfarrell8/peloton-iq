FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache \
    fastapi uvicorn[standard] pydantic-settings \
    anthropic \
    qdrant-client \
    sentence-transformers \
    rank-bm25 rapidfuzz \
    langgraph langchain langchain-anthropic \
    xgboost scikit-learn \
    pandas numpy pyarrow \
    boto3 gpxpy youtube-transcript-api

# Install the package itself
RUN uv pip install --system -e "." --no-deps

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "peloton_iq.api.app:app", "--host", "0.0.0.0", "--port", "8000"]