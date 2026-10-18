FROM python:3.14.8-slim

WORKDIR /usr/src/myapp

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the local directory into the Docker container
COPY . .

EXPOSE 8501
ENV PREDICTOR_API=http://localhost:5000/predict

# Using Streamlit as it was already used in the base project :)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]