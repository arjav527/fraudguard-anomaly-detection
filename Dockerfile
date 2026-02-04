# Phase 1 – Stable x86 Dockerfile (NO Lambda, NO ARM)
FROM python:3.10-slim

WORKDIR /usr/src/app

# Minimal system deps
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Download embeddings (same as repo)
RUN curl -L -o embeddings_anomalies.csv https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/8fc42c16-64b9-4b11-ae2b-20fe38ea021c/embeddings_anomalies.csv \
 && curl -L -o embeddings_transactions_01.csv https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/8fc42c16-64b9-4b11-ae2b-20fe38ea021c/embeddings_transactions_01.csv \
 && curl -L -o embeddings_transactions_02.csv https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/8fc42c16-64b9-4b11-ae2b-20fe38ea021c/embeddings_transactions_02.csv \
 && curl -L -o embeddings_transactions_03.csv https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/8fc42c16-64b9-4b11-ae2b-20fe38ea021c/embeddings_transactions_03.csv

# Copy model code
COPY app/anomaly-detector/anomaly_detector.py .
COPY app/anomaly-detector/requirements.txt .

# Install Python deps
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

CMD ["python", "anomaly_detector.py"]
