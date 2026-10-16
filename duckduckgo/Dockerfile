FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
UN pip install --no-cache-dir -r requirements.txt duckduckgo-search langchain-community
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
