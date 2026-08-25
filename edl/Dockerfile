FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/edl/EDL/evaluation

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY edl/requirements.txt /tmp/edl-requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/edl-requirements.txt

COPY . .

RUN mkdir -p /app/results \
    && useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "clean_run_metrics.py", "edl"]
