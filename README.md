# AlexNet Recovery Lab

A small, reproducible image-classification web application integrated with a
fault-injection, load-testing, observability, and recovery-experiment bench.

Target research use:
**An Explainable and Uncertainty-Aware AI Framework for Autonomous Recovery
Decision-Making in Microservice Systems**

## What is included

- FastAPI image-classification website
- pretrained TorchVision AlexNet (ImageNet-1K)
- HTML upload form and JSON `/predict` API
- Prometheus application metrics
- cAdvisor container metrics
- Nginx gateway
- primary + standby metadata dependencies
- Locust workload generation
- controlled test-only fault injection:
  - CPU Resource Exhaustion
  - Memory Pressure
  - Application Error Injection
  - Application Latency
  - Service Dependency Failure
  - Failed Software Deployment
  - Temporary Workload Spike
- recovery actions:
  - Restart
  - ScaleOut
  - Rollback
  - Failover
  - NoAction
- experiment plan generator
- automatic experiment runner
- raw empirical CSV output

## Important safety note

The `/admin/*` endpoints intentionally inject failures. They are for a local or
isolated research environment only. Do not expose this stack directly to the
public Internet. The lab uses `X-Lab-Key` for accidental-access protection, but
it is not intended as production security.

## Architecture

```text
                     ┌───────────────┐
                     │    Locust     │
                     └───────┬───────┘
                             │
                             ▼
                     ┌───────────────┐
                     │ Nginx Gateway │ :8000
                     └───────┬───────┘
                             │
              Docker DNS / scaled classifier replicas
                             │
                             ▼
                 ┌──────────────────────┐
                 │ FastAPI + AlexNet    │
                 │ /predict /metrics    │
                 └──────┬────────┬──────┘
                        │        │
             primary ───┘        └── standby
              metadata              metadata

Prometheus <── classifier metrics + cAdvisor
Experiment runner ──> faults / recovery / CSV
```

## Quick start

### 1. Requirements

- Docker + Docker Compose v2
- Python 3.10+ on the host for the experiment runner

The first classifier start needs Internet access so TorchVision can download
AlexNet ImageNet weights. They are cached in the Docker `model-cache` volume.

### 2. Configure

Copy:

```bash
cp .env.example .env
```

The default lab key is suitable only for local testing.

### 3. Build and start

```bash
docker compose build
docker compose up -d
```

On first start, wait for the AlexNet weights to download.

Open:

- Website: http://localhost:8000
- Prometheus: http://localhost:9090
- cAdvisor: http://localhost:8081

Health check:

```bash
curl http://localhost:8000/health
```

### 4. Test image classification

Open the website and upload a photo.

Or:

```bash
curl -F "file=@photo.jpg" http://localhost:8000/predict
```

The result contains the top-5 ImageNet predictions.

TorchVision's `AlexNet_Weights.DEFAULT` is used together with the transforms and
class labels supplied by those weights.

### 5. Install experiment-runner dependencies

Recommended:

```bash
python -m venv .venv
```

Activate it and run:

```bash
pip install -r experiment/requirements.txt
```

### 6. Generate an experiment plan

A small pilot campaign:

```bash
python experiment/make_plan.py --replicates 2
```

This produces:

7 fault types × 3 severities × 2 repetitions × 5 actions = 210 runs.

A larger campaign:

```bash
python experiment/make_plan.py --replicates 10
```

produces 1050 runs.

### 7. Dry run

```bash
python experiment/run_experiments.py --dry-run --max-runs 5
```

### 8. Real experiment

```bash
python experiment/run_experiments.py
```

Output:

```text
experiment/data/recovery_experiments_raw.csv
```

## Experimental logic

One `incident_group_id` describes a reproducible configuration:

```text
fault type + severity + workload profile + seed
```

Five independent clean runs are performed for that group:

```text
same configured incident + Restart
same configured incident + ScaleOut
same configured incident + Rollback
same configured incident + Failover
same configured incident + NoAction
```

This is scientifically preferable to pretending that five different actions
can be applied to exactly the same running incident.

Each run follows:

```text
reset
  ↓
start workload
  ↓
healthy baseline
  ↓
inject fault
  ↓
degraded telemetry
  ↓
execute one recovery action
  ↓
post-action telemetry
  ↓
recovery-success decision
  ↓
CSV
  ↓
clean reset
```

## Recovery success

Recovery is assessed from a stable sequence of post-action samples. The default
condition requires:

- classifier availability >= 0.99
- p95 latency <= max(1000 ms, baseline p95 × 1.5)
- error rate <= max(3%, baseline error rate + 2 percentage points)

Change these values only using pilot/validation experiments. Freeze them before
the final test campaign.

## Fault semantics

### CPU Resource Exhaustion
A background CPU burner runs inside the original classifier replica.
`Restart` removes the transient burner; `ScaleOut` introduces healthy replicas.

### Memory Pressure
The classifier allocates controlled blocks of memory.
Restart is a meaningful candidate because process memory is released.

### Application Error Injection
A configured percentage of `/predict` calls return HTTP 500.
Restart clears the transient injected state.

### Application Latency
Artificial latency is added to inference requests.

### Service Dependency Failure
The experiment stops `metadata-primary`.
Classifier requests fail until the dependency is restored or `Failover`
switches the classifier to `metadata-standby`.

### Failed Software Deployment
A persistent shared deployment-state file introduces latency and errors.
Restart does **not** remove this fault. `Rollback` clears the persistent faulty
deployment state.

### Temporary Workload Spike
An additional short-lived Locust process produces a burst of requests.
`NoAction` can become successful after the burst naturally ends.

## Raw empirical vs derived ML data

The experiment runner stores measured quantities such as:

- baseline CPU/memory/latency/error/request rate
- degraded CPU/memory/latency/error/request rate
- post-action metrics
- recovery success
- recovery time
- SLA violation

Do not write SHAP values, utility, decision confidence, or learned success
probabilities into the raw measurement file. Derive them later in the ML
pipeline and keep the distinction explicit in the paper.

## Scale-out note

The gateway is restarted after scaling so it resolves all current classifier
replicas. Prometheus uses Docker DNS service discovery to scrape classifier
replicas.

## Load images

Synthetic images are included under `locust/images/` only to generate requests.
Their semantic labels are irrelevant to load testing. For model-quality
experiments, replace them with a properly licensed labeled dataset.

## Suggested next research step

After collecting real runs:

1. split by `incident_group_id`;
2. train `P(recovery_success | telemetry, candidate_action)`;
3. calibrate probabilities;
4. calculate action utilities;
5. add confidence margin + entropy;
6. tune the abstention threshold on validation data;
7. evaluate on a held-out test set;
8. use SHAP only after selecting the final model.
