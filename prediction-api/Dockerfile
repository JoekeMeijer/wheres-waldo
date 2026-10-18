FROM python:3.14.8-slim
LABEL author="group 10"

# Install system libraries req'd for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/myapp
COPY prediction-api/requirements.txt .
COPY prediction-api/app.py .
COPY src/ ./src/
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python3", "app.py"]