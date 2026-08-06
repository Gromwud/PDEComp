FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/epde/EPDE

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        auto_mix_prep==0.2.0 \
        matplotlib==3.10.1 \
        numpy==1.26.4 \
        pandas==1.5.3 \
        psutil==7.0.0 \
        "PyYAML>=6.0.1" \
        scikit-learn==1.6.1 \
        scipy==1.15.2 \
        seaborn==0.13.2 \
        torch==2.4.0

COPY . .

RUN mkdir -p /app/results \
    && useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "clean_run_metrics.py", "epde"]
