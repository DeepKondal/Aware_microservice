# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /broken_microservices

# Copy the current directory contents into the container at /app
COPY user_service_v1/ /broken_microservices/user_service_v1
COPY shared/config/rabbitmq_config.py /broken_microservices/shared/config/
COPY shared/config/__init__.py /broken_microservices/shared/config/
COPY shared/__init__.py /broken_microservices/shared/

# Add a dummy __init__.py file to ensure the directory is treated as a package
# RUN touch /broken_microservices/__init__.py

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r user_service_v1/requirements.txt

# Define environment variables
ENV FLASK_APP=user_service_v1.wsgi
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV RABBITMQ_HOST=${RABBITMQ_HOST}
ENV RABBITMQ_PORT=${RABBITMQ_PORT}
ENV RABBITMQ_USER=${RABBITMQ_ORDER_USER}
ENV RABBITMQ_PASSWORD=${RABBITMQ_ORDER_PASSWORD}
ENV RABBITMQ_QUEUE_NAME=${RABBITMQ_QUEUE_NAME}
ENV PYTHONUBNBUFFERED=1

# Make port running flask available to the world outside this container
EXPOSE 5000

# Run the application
# CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "user_service_v1.wsgi:app"]