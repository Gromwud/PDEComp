FROM python:3.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libnuma1 ocl-icd-opencl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY discover/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install tensorboard==1.15 protobuf==3.20

RUN git clone https://github.com/Gromwud/DISCOVER.git /tmp/discover && \
    cd /tmp/discover/dso && \
    pip install -e . && \
    echo "/tmp/discover" > /usr/local/lib/python3.7/site-packages/dso.pth && \
    cd .. 

# Copy application files
COPY discover/run.py .
COPY data /app/data

# Setup non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

VOLUME /app/results

CMD ["python", "run.py"]