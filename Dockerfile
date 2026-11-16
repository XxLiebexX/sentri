FROM python:3.11-slim

# Install system dependencies (git for repo cloning, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

# Install CPU PyTorch first (much smaller & faster build), then remaining dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt semgrep

# Copy application code and ensure ownership
COPY --chown=user:user . $HOME/app

USER user

# Ensure cache directory exists and is writable
RUN mkdir -p $HOME/app/.stage0_cache

# Expose Space default port 7860
EXPOSE 7860

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/api/health || exit 1

# Launch FastAPI backend with static frontend mounted
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
