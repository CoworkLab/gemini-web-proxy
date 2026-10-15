# Dockerfile for Hugging Face Spaces deployment
# Creates an OpenAI-compatible proxy for Gemini Web API

FROM python:3.11-slim

# Install system deps for curl_cffi
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY gemini_openai_server.py .

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Run the server
CMD ["python", "gemini_openai_server.py"]
