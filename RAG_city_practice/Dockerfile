# 파이썬 3.10 슬림 버전 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 패키지 설치 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 복사
COPY . .

# 구글 클라우드 런 기본 포트 8080 노출
EXPOSE 8080

# FastAPI 앱 실행 (Uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
