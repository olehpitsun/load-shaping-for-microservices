import os
import random
from pathlib import Path
from locust import HttpUser, task, between

IMAGE_DIR = Path("/mnt/locust/images")
IMAGES = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

class ImageClassifierUser(HttpUser):
    wait_time = between(0.1, 0.8)

    @task(10)
    def classify(self):
        path = random.choice(IMAGES)
        with path.open("rb") as f:
            self.client.post(
                "/predict",
                files={"file": (path.name, f, "image/jpeg")},
                name="/predict",
            )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
