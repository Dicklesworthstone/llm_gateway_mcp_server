FROM python:3.13-slim as builder

# Set working directory
WORKDIR /app

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python build tools and core dependencies (including torch for runtime)
# Assuming torch is needed at runtime by dependencies like sentence-transformers
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    # Install CPU-specific torch first if needed by dependencies
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    # Install the project and its dependencies defined in pyproject.toml
    pip install .

# Create a lightweight runtime image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables for runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime system dependencies
# Add libgomp1 commonly needed by numpy/torch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl libgomp1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create directories needed by the application before changing user
RUN mkdir -p logs .cache .embeddings

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
# Copy the hatchling/pip generated entrypoint scripts
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy necessary configuration files
COPY marqo_index_config.json .

# Create non-root user and group
RUN groupadd -r llmgateway && \
    useradd --no-log-init -r -g llmgateway llmgateway && \
    # Change ownership of app directories
    chown -R llmgateway:llmgateway /app

# Switch to non-root user
USER llmgateway

# Expose application port
EXPOSE 8013

# Use the installed script from pyproject.toml as entrypoint
ENTRYPOINT ["llm-gateway"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8013"]

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8013/healthz || exit 1