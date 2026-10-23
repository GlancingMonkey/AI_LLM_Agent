FROM python:3.10-slim

WORKDIR /app

# 시스템 필수 패키지 및 빌드 도구 설치
RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run 기본 포트인 8080 개방
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
