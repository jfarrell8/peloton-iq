FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install uv --no-cache-dir

# Copy dependency files first (layer caching)
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system --no-cache \
    fastapi \
    uvicorn[standard] \
    pydantic-settings \
    anthropic \
    qdrant-client \
    sentence-transformers \
    rank-bm25 \
    rapidfuzz \
    langgraph \
    langchain \
    langchain-anthropic \
    lightgbm \
    xgboost \
    optuna \
    scikit-learn \
    pandas \
    numpy \
    pyarrow \
    google-api-python-client \
    youtube-transcript-api \
    prefect \
    python-dotenv

# Copy source
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY models/ ./models/

# Environment
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Start API
CMD ["uvicorn", "peloton_iq.api.app:app", "--host", "0.0.0.0", "--port", "8000"]