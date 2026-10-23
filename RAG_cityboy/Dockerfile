# 1. 공식 가벼운 Python 3.11 슬림 베이스 이미지 사용
FROM python:3.11-slim

# 2. 필수 시스템 환경 변수 설정 (버퍼링 방지 및 Cloud Run 포트 지정)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# 3. 컨테이너 내부 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치 (도커 빌드 캐시 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 소스 코드 복사
COPY main.py .

# 6. 구글 클라우드 런 포트 노출
EXPOSE 8080

# 7. Uvicorn 프로덕션 서버 비동기 실행 (Cloud Run 표준 진입 명령)
CMD uvicorn main.py:app --host 0.0.0.0 --port ${PORT}
