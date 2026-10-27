FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common common
COPY simulator simulator
COPY ingestion ingestion
COPY features features
COPY models models
ENV PYTHONUNBUFFERED=1
