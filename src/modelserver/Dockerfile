# Use the latest Python 3.10 image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install core dependencies first
COPY requirements.txt .

RUN pip install --no-cache-dir fastapi uvicorn numpy torch torchvision

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application files
COPY . .

# Expose the port used by the service
EXPOSE 8002

# Run the FastAPI server
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8002"]
