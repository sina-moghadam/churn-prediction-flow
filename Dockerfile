FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MLFLOW_ALLOW_FILE_STORE=true
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
CMD ["python", "run_pipeline.py"]