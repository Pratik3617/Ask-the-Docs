#!/bin/bash
set -e

echo "Starting FastAPI backend..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit frontend..."
streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0

wait
