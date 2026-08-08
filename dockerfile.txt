FROM python:3.10-slim

# Prevent interactive prompts during apt compilation loops
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system native binary dependencies for OpenCV and Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install requirements in a cached layout layer
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application script assets into the active working tree
COPY . .

# Expose port and configure the launch parameters
EXPOSE 10000
CMD ["uvicorn", "premium_main:app", "--host", "0.0.0.0", "--port", "10000"]
