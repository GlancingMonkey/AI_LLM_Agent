# 1. 공식 Python 3.11 슬림 베이스 이미지 사용
FROM python:3.11-slim

# 2. 필수 시스템 환경 변수 설정 (로그 실시간 출력 및 Cloud Run 기본 포트 8080 지정)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    CHROMA_PERSIST_DIR=/tmp/chroma_db

# 3. 컨테이너 내부 작업 디렉토리 설정
WORKDIR /app

# 4. ChromaDB 구동에 필요한 최소 C++ 런타임 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. 의존성 파일 복사 및 패키지 설치 (도커 레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 애플리케이션 소스 코드 복사
COPY main.py .

# 7. Cloud Run 인바운드 포트 노출
EXPOSE 8080

# 8. Uvicorn 프로덕션 서버 실행 (모듈명 main:app 지정 필수)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

