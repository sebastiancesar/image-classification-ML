# Use an official Python runtime as a parent image
FROM python:2.7-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y libglib2.0 libsm6 libxrender-dev libxext-dev

# Make port 80 available to the world outside this container
EXPOSE 5000

# Define environment varable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "src/web-server.py"]
