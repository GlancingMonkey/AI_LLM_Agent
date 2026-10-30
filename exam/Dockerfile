# 1. 파이썬 3.10 슬림 버전 사용 (가볍고 배포가 빠름)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 파이썬 로그가 버퍼링 없이 즉시 출력되도록 설정 (Cloud Run 로그 확인용)
ENV PYTHONUNBUFFERED=1

# 4. Chroma DB 내부 SQLite 등 C++ 기반 라이브러리 빌드에 필요한 최소 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. 의존성 파일 복사 및 패키지 설치 (캐시를 남기지 않아 이미지 용량 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 소스 코드(main.py 등) 복사
COPY . .

# 7. Cloud Run 기본 포트(8080) 명시
ENV PORT=8080
EXPOSE $PORT

# 8. Uvicorn을 사용해 FastAPI 서버 실행
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
