FROM python:3.7-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/discover
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/discover-requirements.txt
RUN pip install --no-cache-dir --upgrade "pip<24" \
    && pip install --no-cache-dir -r /tmp/discover-requirements.txt

CMD ["python", "clean_run_metrics.py", "discover"]
