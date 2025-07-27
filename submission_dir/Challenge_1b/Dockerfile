# ================================
# Stage 1: Build Environment  
# ================================
FROM --platform=linux/amd64 python:3.11-slim AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies for Challenge 1B
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu \
    transformers==4.35.2 \
    PyMuPDF==1.26.3 \
    numpy==1.26.0

# Pre-download models for offline execution
RUN python -c "import os; from transformers import AutoTokenizer, AutoModel, pipeline; print('Downloading models for Challenge 1B...'); tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny'); model = AutoModel.from_pretrained('prajjwal1/bert-tiny'); summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6'); print('All models cached successfully')"

# ================================
# Stage 2: Production Runtime
# ================================
FROM --platform=linux/amd64 python:3.11-slim AS production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the cached Hugging Face models for offline access
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

# Copy application code
COPY process_pdfs.py .

# Create input and output directories with proper permissions (same as 1A)
RUN mkdir -p /app/input /app/output && \
    chown -R appuser:appuser /app /home/appuser/.cache

# Switch to non-root user
USER appuser

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# Add health check to verify dependencies
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=2 \
    CMD python -c "import fitz, torch, transformers, numpy; print('✅ All dependencies OK')" || exit 1

# Set the command to run Challenge 1B application
CMD ["python", "process_pdfs.py"]
