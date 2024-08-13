# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system packages and dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    libx11-dev \
    gcc \
    python3-tk \
    tk-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install backtrader and other dependencies
RUN pip install --no-cache-dir \
    backtrader \
    confluent-kafka[avro] \
    prefect\
    psycopg2-binary 

# Copy the rest of the code into the container
COPY . .

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
