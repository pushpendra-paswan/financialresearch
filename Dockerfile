FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached until the requirements files change.
# Dev-only project for now, so one image includes the test and lint tools
# (requirements-dev.txt starts with "-r requirements.txt")
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

# The command is set per service in docker-compose.yml (api and worker share this image)
