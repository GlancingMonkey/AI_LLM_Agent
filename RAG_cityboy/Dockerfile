# 1. 공식 Python 3.11 슬림 베이스 이미지
FROM python:3.11-slim

# 2. 필수 시스템 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 소스 코드 복사
COPY main.py .

# 6. 포트 노출
EXPOSE 8080

# 7. Uvicorn 프로덕션 서버 실행 (파일명 main.py가 아닌 모듈명 main:app 필수)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
