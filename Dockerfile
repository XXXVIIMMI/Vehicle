# Use an official Python 3.10 image from Docker Hub
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 5000

# Command to run the FastAPI app
CMD ["python3", "app.py"]
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]