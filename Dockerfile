FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (git for repo cloning, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Semgrep & Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt semgrep

# Copy application code, models, and assets
COPY . .

# Expose port 8000 (serves both API and Web UI)
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Launch FastAPI backend with static frontend mounted
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
