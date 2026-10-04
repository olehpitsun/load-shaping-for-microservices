import asyncio
import io
import json
import math
import os
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal

import httpx
import torch
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel
from torchvision.models import AlexNet_Weights, alexnet


APP_DIR = Path(__file__).resolve().parent
LAB_KEY = os.getenv("LAB_KEY", "local-recovery-lab-key")
DEVICE_NAME = os.getenv("ALEXNET_DEVICE", "cpu")
PRIMARY_URL = os.getenv("DEPENDENCY_PRIMARY", "http://metadata-primary:8100")
STANDBY_URL = os.getenv("DEPENDENCY_STANDBY", "http://metadata-standby:8100")
DEPLOYMENT_STATE_FILE = Path(
    os.getenv("DEPLOYMENT_STATE_FILE", "/lab-state/deployment.json")
)

app = FastAPI(title="AlexNet Image Classification Recovery Lab")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.mount("/metrics", make_asgi_app())
templates = Environment(loader=FileSystemLoader(APP_DIR / "templates"))

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests processed by the classifier",
    ["method", "endpoint", "status"],
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 5, 10),
)
INFERENCE_DURATION = Histogram(
    "inference_duration_seconds",
    "AlexNet inference duration",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 5),
)
CLASSIFICATION_ERRORS = Counter(
    "classification_errors_total",
    "Image-classification failures",
)
QUEUE_LENGTH = Gauge(
    "queue_length",
    "Number of classifier requests currently in the inference section",
)
MODEL_READY = Gauge("model_ready", "Whether the AlexNet model is loaded")
FAULT_ACTIVE = Gauge(
    "fault_active",
    "Whether a controlled fault is active",
    ["kind"],
)

for label in ("cpu", "memory", "errors", "latency", "deployment"):
    FAULT_ACTIVE.labels(kind=label).set(0)

_model = None
_weights = None
_preprocess = None
_categories = None
_device = torch.device("cpu")

fault_lock = threading.Lock()
fault_state = {
    "latency_ms": 0,
    "error_probability": 0.0,
    "memory_mb": 0,
    "cpu_threads": 0,
}
memory_chunks = []
cpu_processes = []
dependency_target = "primary"


class FaultRequest(BaseModel):
    kind: Literal["cpu", "memory", "errors", "latency", "deployment"]
    severity: Literal["Low", "Medium", "High"]


def ensure_lab_key(x_lab_key: str | None):
    if x_lab_key != LAB_KEY:
        raise HTTPException(status_code=403, detail="Invalid X-Lab-Key")


def endpoint_label(path: str) -> str:
    if path == "/predict":
        return "/predict"
    if path == "/health":
        return "/health"
    if path.startswith("/admin/"):
        return "/admin"
    if path == "/":
        return "/"
    return "other"


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    if request.url.path.startswith("/metrics"):
        return await call_next(request)
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        label = endpoint_label(request.url.path)
        HTTP_REQUESTS.labels(request.method, label, status).inc()
        HTTP_DURATION.labels(label).observe(time.perf_counter() - start)


def load_deployment_state() -> dict:
    try:
        return json.loads(DEPLOYMENT_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"faulty": False}


def write_deployment_state(faulty: bool):
    DEPLOYMENT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_STATE_FILE.write_text(
        json.dumps({"faulty": bool(faulty)}),
        encoding="utf-8",
    )
    FAULT_ACTIVE.labels(kind="deployment").set(1 if faulty else 0)


def stop_cpu_fault():
    global cpu_processes
    for p in cpu_processes:
        try:
            p.terminate()
            p.wait(timeout=1)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    cpu_processes = []
    fault_state["cpu_threads"] = 0
    FAULT_ACTIVE.labels(kind="cpu").set(0)


def set_cpu_fault(count: int):
    global cpu_processes
    stop_cpu_fault()
    count = max(0, count)
    burner = (
        "import math\n"
        "x=0.1234567\n"
        "while True:\n"
        "    x=math.sin(x)*math.cos(x)+1.000001\n"
    )
    for _ in range(count):
        p = subprocess.Popen(
            ["python", "-c", burner],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cpu_processes.append(p)
    fault_state["cpu_threads"] = count
    FAULT_ACTIVE.labels(kind="cpu").set(1 if count else 0)


def set_memory_fault(mb: int):
    global memory_chunks
    memory_chunks = []
    if mb > 0:
        # Separate chunks reduce the chance of a single huge allocation failure.
        remaining = mb
        while remaining > 0:
            chunk = min(32, remaining)
            memory_chunks.append(bytearray(chunk * 1024 * 1024))
            remaining -= chunk
    fault_state["memory_mb"] = mb
    FAULT_ACTIVE.labels(kind="memory").set(1 if mb else 0)


def reset_transient_faults():
    with fault_lock:
        stop_cpu_fault()
        set_memory_fault(0)
        fault_state["latency_ms"] = 0
        fault_state["error_probability"] = 0.0
        FAULT_ACTIVE.labels(kind="errors").set(0)
        FAULT_ACTIVE.labels(kind="latency").set(0)


def apply_fault(kind: str, severity: str):
    level = {"Low": 1, "Medium": 2, "High": 3}[severity]
    with fault_lock:
        if kind == "cpu":
            set_cpu_fault(level)
        elif kind == "memory":
            set_memory_fault({1: 128, 2: 256, 3: 384}[level])
        elif kind == "errors":
            fault_state["error_probability"] = {1: 0.05, 2: 0.15, 3: 0.30}[level]
            FAULT_ACTIVE.labels(kind="errors").set(1)
        elif kind == "latency":
            fault_state["latency_ms"] = {1: 300, 2: 1000, 3: 2500}[level]
            FAULT_ACTIVE.labels(kind="latency").set(1)
        elif kind == "deployment":
            write_deployment_state(True)


async def dependency_check():
    target = PRIMARY_URL if dependency_target == "primary" else STANDBY_URL
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(target + "/metadata")
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Active dependency '{dependency_target}' unavailable",
        ) from exc


def current_fault_effects():
    with fault_lock:
        latency_ms = int(fault_state["latency_ms"])
        error_probability = float(fault_state["error_probability"])

    deployment = load_deployment_state()
    if deployment.get("faulty"):
        latency_ms += 700
        error_probability = max(error_probability, 0.22)

    return latency_ms, min(error_probability, 0.95)


@app.on_event("startup")
def startup():
    global _model, _weights, _preprocess, _categories, _device

    _weights = AlexNet_Weights.DEFAULT
    _preprocess = _weights.transforms()
    _categories = _weights.meta["categories"]
    _model = alexnet(weights=_weights)
    _model.eval()

    if DEVICE_NAME.startswith("cuda") and torch.cuda.is_available():
        _device = torch.device(DEVICE_NAME)
    else:
        _device = torch.device("cpu")

    _model.to(_device)

    # Ensure the persistent deployment-state file exists.
    if not DEPLOYMENT_STATE_FILE.exists():
        write_deployment_state(False)
    else:
        FAULT_ACTIVE.labels(kind="deployment").set(
            1 if load_deployment_state().get("faulty") else 0
        )

    MODEL_READY.set(1)


@app.on_event("shutdown")
def shutdown():
    stop_cpu_fault()


@app.get("/", response_class=HTMLResponse)
def home():
    template = templates.get_template("index.html")
    return template.render()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "alexnet",
        "weights": "AlexNet_Weights.DEFAULT",
        "device": str(_device),
        "dependency_target": dependency_target,
        "deployment_faulty": bool(load_deployment_state().get("faulty")),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    QUEUE_LENGTH.inc()
    try:
        latency_ms, error_probability = current_fault_effects()

        if latency_ms:
            await asyncio.sleep(latency_ms / 1000.0)

        if random.random() < error_probability:
            CLASSIFICATION_ERRORS.inc()
            raise HTTPException(status_code=500, detail="Injected application failure")

        dependency = await dependency_check()

        raw = await file.read()
        if not raw:
            CLASSIFICATION_ERRORS.inc()
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            CLASSIFICATION_ERRORS.inc()
            raise HTTPException(status_code=400, detail="Invalid image") from exc

        batch = _preprocess(image).unsqueeze(0).to(_device)

        started = time.perf_counter()
        with torch.inference_mode():
            logits = _model(batch)[0]
            probabilities = torch.softmax(logits, dim=0)
        INFERENCE_DURATION.observe(time.perf_counter() - started)

        values, indices = torch.topk(probabilities, 5)
        predictions = [
            {
                "class_id": int(idx),
                "label": _categories[int(idx)],
                "probability": round(float(prob), 6),
            }
            for prob, idx in zip(values.cpu(), indices.cpu())
        ]

        return {
            "filename": file.filename,
            "model": "AlexNet",
            "dependency": dependency.get("instance"),
            "predictions": predictions,
        }
    finally:
        QUEUE_LENGTH.dec()


@app.post("/admin/fault")
def admin_fault(payload: FaultRequest, x_lab_key: str | None = Header(default=None)):
    ensure_lab_key(x_lab_key)
    apply_fault(payload.kind, payload.severity)
    return {"ok": True, "kind": payload.kind, "severity": payload.severity}


@app.post("/admin/reset-transient")
def admin_reset_transient(x_lab_key: str | None = Header(default=None)):
    ensure_lab_key(x_lab_key)
    reset_transient_faults()
    return {"ok": True}


@app.post("/admin/deployment/rollback")
def admin_rollback(x_lab_key: str | None = Header(default=None)):
    ensure_lab_key(x_lab_key)
    write_deployment_state(False)
    return {"ok": True, "deployment_faulty": False}


@app.post("/admin/dependency/failover")
async def admin_failover(x_lab_key: str | None = Header(default=None)):
    global dependency_target
    ensure_lab_key(x_lab_key)

    # Do not report a successful failover unless the standby dependency is
    # actually reachable. This makes the recovery action observable and
    # prevents silent false-positive failover commands.
    previous_target = dependency_target
    dependency_target = "standby"
    try:
        await dependency_check()
    except Exception:
        dependency_target = previous_target
        raise

    return {"ok": True, "dependency_target": dependency_target}


@app.post("/admin/dependency/primary")
def admin_primary(x_lab_key: str | None = Header(default=None)):
    global dependency_target
    ensure_lab_key(x_lab_key)
    dependency_target = "primary"
    return {"ok": True, "dependency_target": dependency_target}


@app.get("/admin/state")
def admin_state(x_lab_key: str | None = Header(default=None)):
    ensure_lab_key(x_lab_key)
    return {
        "fault_state": fault_state,
        "deployment": load_deployment_state(),
        "dependency_target": dependency_target,
        "memory_chunks": len(memory_chunks),
    }
