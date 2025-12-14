FROM python:3.11-slim

WORKDIR /app

# Install system deps for some packages (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command prints help; run tests or jupyter externally
CMD ["python", "-m", "pytest", "-q"]
