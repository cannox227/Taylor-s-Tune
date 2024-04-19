FROM python:3.11-slim

WORKDIR /

COPY . .

RUN if [ ! -f /secrets.toml ]; then echo "File not found!" && exit 1; fi
COPY secrets.toml /.streamlit/secrets.toml
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "💬_Chat.py", "--browser.gatherUsageStats", "false", "--server.headless", "true", "--server.fileWatcherType", "none", "--server.port=8501", "--server.address=0.0.0.0"]