# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy the dependency list
COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port your Flask app will run on
EXPOSE 9696

# Start the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]

