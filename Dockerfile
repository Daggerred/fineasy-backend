# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including build tools and libraries
RUN apt-get update && apt-get install -y \
    adduser \
    curl \
    g++ \
    gcc \
    git \
    libffi-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libjpeg-dev \
    liblcms2-dev \
    libmagic-dev \
    libmagic1 \
    libopenjp2-7-dev \
    libpng-dev \
    libpq-dev \
    libssl-dev \
    libtiff5-dev \
    libwebp-dev \
    libxcb1-dev \
    make \
    passwd \
    pkg-config \
    redis-tools \
    supervisor \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with increased timeout and retries
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs /app/cache /app/temp && \
    chmod 755 /app/uploads /app/logs /app/cache /app/temp

# Create non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check with proper curl installation check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import requests; import os; port=os.environ.get('PORT', '8000'); requests.get(f'http://localhost:{port}/health', timeout=5)" || exit 1

# Run the application with proper error handling
CMD ["python", "start_server.py"]