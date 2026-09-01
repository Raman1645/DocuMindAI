# Use slim Python 3.11 image for Hugging Face Spaces & Docker compatibility
FROM python:3.11-slim

WORKDIR /app

# Install system utilities & build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Pre-generate sample data so container starts with demo fixtures ready
RUN python generate_sample_data.py

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Set Streamlit server settings for containerized deployment
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py"]
