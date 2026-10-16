# 가벼운 파이썬 3.9 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 노출 (문서화 용도)
EXPOSE 8080

# 앱 실행 명령어
CMD ["python", "main.py"]
