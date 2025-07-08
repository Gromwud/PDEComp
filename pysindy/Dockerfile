FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

# Copy requirements first for better caching
COPY pysindy/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/dynamicslab/pysindy.git
# Copy ALL necessary files
COPY pysindy/run.py .
COPY data /app/data

# Create results directory
RUN mkdir -p /app/results

CMD ["python", "run.py"]