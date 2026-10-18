FROM python:3.14.8-slim
LABEL author="group 10"

WORKDIR /usr/src/myapp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the local directory into the Docker container
COPY . .

EXPOSE 5000
CMD ["python3", "app.py"]