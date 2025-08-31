# # Use Python 3.13 slim image
# FROM python:3.13-slim

# # Set working directory
# WORKDIR /app

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PYTHONPATH=/app

# # Install system dependencies
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#         build-essential \
#         curl \
#         git \
#     && rm -rf /var/lib/apt/lists/*

# # Install uv
# RUN pip install uv

# # Copy dependency files
# COPY pyproject.toml uv.lock ./

# # Install Python dependencies
# RUN uv sync --frozen --no-dev

# # Copy application code
# COPY . .

# # Create necessary directories
# RUN mkdir -p logs models

# # Expose port
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# # Run the application
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]



# Use Python 3.13 slim image
FROM python:3.13-slim

# Workdir
WORKDIR /app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# System deps (curl kept for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy lock/metadata early (+ README so hatchling is happy)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies only (skip installing local project now for better caching)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY . .

# Install local project (now README & sources are present)
# Choose one of the following; keeping uv sync for consistency with lockfile:
RUN uv sync --frozen --no-dev
# Alternative: RUN uv pip install -e .

# Ensure runtime dirs exist
RUN mkdir -p logs models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
