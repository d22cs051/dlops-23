# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container and install the dependencies
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Gradio app code into the container
COPY * /app

# Expose port 80 for the Gradio app to listen on
EXPOSE 8001

# Set the entry point to run the Gradio app
CMD ["python", "app.py"]
