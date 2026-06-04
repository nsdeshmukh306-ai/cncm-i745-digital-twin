from fastapi import FastAPI
app = FastAPI(title="CNCM I-745 Digital Twin API", version="0.1.0")

@app.get("/")
def root():
    return {"status": "online", "project": "CNCM I-745 Digital Twin"}

@app.get("/health")
def health():
    return {"status": "healthy"}
