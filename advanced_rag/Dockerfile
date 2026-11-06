# 1. Base Image: 경량화된 Python 3.10-slim 이미지 사용
FROM python:3.10-slim

# 2. 시스템 환경변수 설정 (버퍼링 방지 및 pyc 생성 방지)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. FAISS 및 수치 연산 라이브러리에 필요한 최소 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. 의존성 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 애플리케이션 소스 코드 복사
COPY . .

# 7. Google Cloud Run 기본 포트(8080) 노출 및 Uvicorn 서버 실행
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

