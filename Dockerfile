# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Default to mock mode (can be overridden by docker-compose)
ENV MOCK_MODE=true
# Default port
ENV PORT=5000

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for markitdown, torch, or other building steps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# Large files and envs are excluded via .dockerignore
COPY . .

# Expose the Flask port
EXPOSE 5000

# Start Flask application
CMD ["python", "app.py"]
