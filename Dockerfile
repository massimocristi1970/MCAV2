FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV APP_ENTRYPOINT=app/main.py
ENV PORT=8501
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f "http://localhost:${PORT}/_stcore/health" || exit 1

# Expose port
EXPOSE 8501

# Run the selected Streamlit application. Render provides PORT at runtime.
CMD ["sh", "-c", "streamlit run \"$APP_ENTRYPOINT\" --server.port=\"${PORT:-8501}\" --server.address=0.0.0.0 --server.headless=true"]
