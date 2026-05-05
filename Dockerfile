FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY predict_credit_risk ./
COPY README.md ./

RUN mkdir -p /app/models

ENV CREDIT_RISK_MODEL_PATH=/app/models/credit_risk_pipeline.joblib

EXPOSE 8000

CMD ["uvicorn", "predict_credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
