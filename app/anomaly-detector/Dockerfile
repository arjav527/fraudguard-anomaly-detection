# Simple, stable base image (NO ARM, NO Lambda)
FROM python:3.10-slim

# Set working directory
WORKDIR /usr/src/app

# System deps (only required ones)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download embeddings (same as original, untouched)
RUN curl -L -o embeddings_anomalies.csv https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/8fc42c16-64b9-4b11-ae2b-20fe38ea021c/embeddings_anomalies.csv \
 && curl -L -o embeddings_transactions_01.csv https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/8fc42c16-64b9-4b11-ae2b-20fe38ea021c/embeddings_transactions_01.csv \
 && curl -L -o embeddings_transactions_02.csv https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/8fc42c16-64b9-4b11-ae2b-20fe38ea021c/embeddings_transactions_02.csv \
 && curl -L -o embeddings_transactions_03.csv https://ws-assets-prod-iad-r-iad-ed304a55c2ca1aee.s3.us-east-1.amazonaws.com/8fc42c16-64b9-4b11-ae2b-20fe38ea021c/embeddings_transactions_03.csv

# Copy only required files
COPY anomaly_detector.py .
COPY requirements.txt .

# Install Python dependencies
# IMPORTANT: psycopg2 removed requirement of pg_config by using binary
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Run model
CMD ["python", "anomaly_detector.py"]
