import os
from fastapi import FastAPI

app = FastAPI(title="Classifier Metadata Dependency")
INSTANCE = os.getenv("INSTANCE_NAME", "primary")

@app.get("/health")
def health():
    return {"status": "ok", "instance": INSTANCE}

@app.get("/metadata")
def metadata():
    return {
        "status": "ok",
        "instance": INSTANCE,
        "model_family": "AlexNet",
        "task": "ImageNet-1K image classification",
    }
