from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "안녕하세요! GitHub Actions를 통해 Cloud Run에 배포된 앱입니다."}

if __name__ == "__main__":
    import uvicorn
    # Cloud Run은 환경 변수 PORT를 통해 포트 번호를 전달합니다. (기본값 8080)
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
