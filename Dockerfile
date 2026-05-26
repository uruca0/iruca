FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    cmake \
    libsndfile1 \
    fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "openjtalk_bot.py"]