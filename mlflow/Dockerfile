# Use an official Python runtime as a parent image
FROM python:3.9-slim

RUN apt-get update && apt-get upgrade -y && apt-get install -y git


# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container's workspace
COPY . /app

RUN pip install psycopg2-binary

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080


# Run app.py when the container launches
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]
