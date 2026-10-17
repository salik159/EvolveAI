from fastapi import FastAPI

app = FastAPI(title="EvolveAI API")


@app.get("/")
def home():
    return {"message": "EvolveAI backend is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}