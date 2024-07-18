# Use an official Python runtime as a parent image
FROM python:3.10.6-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy api and ml contents into the container at /app
COPY ./api /app/api
COPY ./ml_logic /app/ml_logic

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run app.py when the container launches
CMD ["uvicorn", "api.gold_main:app", "--host", "0.0.0.0", "--port", "80"]
