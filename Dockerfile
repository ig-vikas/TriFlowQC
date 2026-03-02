FROM python:3.11-slim

LABEL maintainer="TriFlowQC"
LABEL description="TriFlowQC - Qutrit Quantum Neural Network Framework"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# Copy the package
COPY . .

# Install TriFlowQC in editable mode
RUN pip install --no-cache-dir -e ".[dev]"

# Run tests by default
CMD ["pytest", "tests/unit/", "tests/math_validation/", "-v", "--tb=short"]
