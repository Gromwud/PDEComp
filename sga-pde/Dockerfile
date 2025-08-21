FROM python:3.7-slim

WORKDIR /app
ENV PYTHONPATH="/app:/tmp/sga-pde/codes:$PYTHONPATH"
# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libnuma1 ocl-icd-opencl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY sga-pde/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --default-timeout=100 --no-cache-dir -r requirements.txt && \
    pip install tensorboard==1.15 protobuf==3.20

RUN git clone https://github.com/Gromwud/SGA-PDE.git /tmp/sga-pde && \
    cd /tmp/sga-pde && \
    pip install -e . && \
    echo "/tmp/sga-pde" > /usr/local/lib/python3.7/site-packages/dso.pth 

# Copy application files
COPY sga-pde/run.py .
COPY data /app/data

# Setup non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

VOLUME /app/results

CMD ["python", "run.py"]