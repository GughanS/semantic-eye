# Use a lightweight Python base
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies for OpenCV (essential for video processing)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port
EXPOSE 8002

# Run the app
CMD ["python", "main.py"]