# Use a more specific version tag for reproducibility
# FROM python:3.9-slim@sha256:<specific-sha256-hash>
FROM python:3.11-slim


WORKDIR /app

# Combine apt commands and clean up in single RUN layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies in one RUN
COPY epde/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/ITMO-NSS-team/EPDE.git

# Copy only what's needed (consider .dockerignore)
COPY epde/run.py .
COPY epde/config.py .
COPY epde/dataloader.py .
COPY data /app/data

# Create results directory at runtime rather than build time
VOLUME /app/results

# Specify a non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Set default command if appropriate

CMD ["python", "run.py"]