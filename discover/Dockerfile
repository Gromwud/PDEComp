FROM python:3.7-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/discover:/app/discover/discover/dso
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY discover/requirements.txt /tmp/discover-requirements.txt
RUN pip install --no-cache-dir --upgrade "pip<24" \
    && pip install --no-cache-dir -r /tmp/discover-requirements.txt

COPY . .

RUN mkdir -p /app/results \
    && useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "clean_run_metrics.py", "discover"]
