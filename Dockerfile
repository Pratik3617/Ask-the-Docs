# Base Image
FROM python:3.12.3

# Env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache

# system dependencies
RUN apt-get update && apt-get install -y\
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Hugging Face model
# (prevents runtime timeout)
RUN python - <<EOF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = "google/flan-t5-large"
AutoTokenizer.from_pretrained(model)
AutoModelForSeq2SeqLM.from_pretrained(model)
EOF

# COPY application code
COPY . .

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Start Application
# Copy startup script
COPY start.sh /app/start.sh

# Run both FastAPI + Streamlit
CMD ["bash", "/app/start.sh"]