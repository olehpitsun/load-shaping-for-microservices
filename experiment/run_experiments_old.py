import argparse
import csv
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    "cpu_pct",
    "memory_mb",
    "latency_ms",
    "error_rate_pct",
    "request_rate",
    "availability",
    "inference_latency_ms",
    "queue_length",
]

FAULT_TO_KIND = {
    "CPU Resource Exhaustion": "cpu",
    "Memory Pressure": "memory",
    "Application Error Injection": "errors",
    "Application Latency": "latency",
    "Failed Software Deployment": "deployment",
}


def docker_compose(*args, capture=False, check=True):
    cmd = ["docker", "compose", *args]
    print("$", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def admin_post(cfg, path, payload=None, tolerate=False):
    headers = {"X-Lab-Key": cfg["project"]["lab_key"]}
    url = cfg["project"]["gateway_url"].rstrip("/") + path
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json() if r.content else {}
    except Exception:
        if tolerate:
            return {}
        raise


def wait_for_health(cfg, timeout=120):
    url = cfg["project"]["gateway_url"].rstrip("/") + "/health"
    end = time.time() + timeout
    while time.time() < end:
        try:
            r = requests.get(url, timeout=4)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Classifier did not become healthy in time")


def prom_query(cfg, query):
    url = cfg["project"]["prometheus_url"].rstrip("/") + "/api/v1/query"
    r = requests.get(url, params={"query": query}, timeout=10)
    r.raise_for_status()
    results = r.json()["data"]["result"]
    if not results:
        return float("nan")
    vals = []
    for item in results:
        try:
            vals.append(float(item["value"][1]))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else float("nan")


def sample(cfg):
    out = {}
    for name, query in cfg["prometheus_queries"].items():
        try:
            out[name] = prom_query(cfg, query)
        except Exception as exc:
            print("metric unavailable:", name, exc)
            out[name] = float("nan")
    return out


def collect(cfg, seconds, dry_run=False):
    if dry_run:
        return [{
            "cpu_pct": 42.0,
            "memory_mb": 410.0,
            "latency_ms": 340.0,
            "error_rate_pct": 0.4,
            "request_rate": 3.0,
            "availability": 1.0,
            "inference_latency_ms": 95.0,
            "queue_length": 0.0,
        }]

    interval = int(cfg["timing"]["sample_interval_s"])
    end = time.time() + seconds
    samples = []
    while time.time() < end:
        samples.append(sample(cfg))
        time.sleep(interval)
    return samples


def mean(values):
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def summarize(samples, prefix):
    return {
        f"{prefix}_{metric}": mean([s.get(metric, float("nan")) for s in samples])
        for metric in METRICS
    }


def start_load(cfg, profile, duration="10m", users=None, spawn_rate=None):
    p = cfg["workloads"][profile]
    users = users if users is not None else p["users"]
    spawn_rate = spawn_rate if spawn_rate is not None else p["spawn_rate"]

    result = docker_compose(
        "run", "--rm", "-d", "locust",
        "-f", "/mnt/locust/locustfile.py",
        "--headless",
        "-u", str(users),
        "-r", str(spawn_rate),
        "--host", "http://gateway:8000",
        "--run-time", str(duration),
        capture=True,
    )
    cid = result.stdout.strip()
    print("load container:", cid)
    return cid


def stop_container(cid):
    if cid:
        subprocess.run(["docker", "rm", "-f", cid], check=False, capture_output=True)


def reset_environment(cfg, dry_run=False):
    if dry_run:
        return

    # Return the classifier to exactly one replica and restore dependencies.
    docker_compose("up", "-d", "--scale", "classifier=1",
                   "classifier", "metadata-primary", "metadata-standby",
                   "gateway", "prometheus", "cadvisor")
    docker_compose("restart", "classifier", "gateway")
    time.sleep(cfg["timing"]["reset_wait_s"])
    wait_for_health(cfg)

    admin_post(cfg, "/admin/reset-transient", tolerate=True)
    admin_post(cfg, "/admin/deployment/rollback", tolerate=True)
    admin_post(cfg, "/admin/dependency/primary", tolerate=True)


def inject_fault(cfg, fault, severity, dry_run=False):
    if dry_run:
        print(f"[dry-run] inject {fault} / {severity}")
        return None

    if fault in FAULT_TO_KIND:
        return admin_post(
            cfg,
            "/admin/fault",
            {"kind": FAULT_TO_KIND[fault], "severity": severity},
        )

    if fault == "Service Dependency Failure":
        docker_compose("stop", "metadata-primary")
        return None

    if fault == "Temporary Workload Spike":
        spike = cfg["temporary_spike"][severity]
        return start_load(
            cfg,
            "light",
            duration=f"{spike['duration_s']}s",
            users=spike["users"],
            spawn_rate=spike["spawn_rate"],
        )

    raise ValueError(f"Unknown fault: {fault}")


def perform_action(cfg, action, dry_run=False):
    if dry_run:
        print(f"[dry-run] action {action}")
        return

    if action == "Restart":
        docker_compose("restart", "classifier")
        time.sleep(5)
        wait_for_health(cfg)

    elif action == "ScaleOut":
        docker_compose("up", "-d", "--scale", "classifier=3", "classifier")
        time.sleep(4)
        docker_compose("restart", "gateway")
        time.sleep(4)
        wait_for_health(cfg)

    elif action == "Rollback":
        admin_post(cfg, "/admin/deployment/rollback")

    elif action == "Failover":
        admin_post(cfg, "/admin/dependency/failover")

    elif action == "NoAction":
        pass

    else:
        raise ValueError(f"Unknown action: {action}")


def thresholds(cfg, baseline):
    rules = cfg["recovery_rules"]
    b_lat = baseline["baseline_latency_ms"]
    b_err = baseline["baseline_error_rate_pct"]

    lat_limit = float(rules["latency_floor_ms"])
    if math.isfinite(b_lat):
        lat_limit = max(
            lat_limit,
            b_lat * float(rules["latency_factor_over_baseline"]),
        )

    err_limit = float(rules["error_ceiling_pct"])
    if math.isfinite(b_err):
        err_limit = max(
            err_limit,
            b_err + float(rules["error_delta_pct_points"]),
        )

    return {
        "availability_min": float(rules["availability_min"]),
        "latency_limit_ms": lat_limit,
        "error_limit_pct": err_limit,
    }


def healthy(s, t):
    required = ("availability", "latency_ms", "error_rate_pct")
    if not all(math.isfinite(s.get(k, float("nan"))) for k in required):
        return False
    return (
        s["availability"] >= t["availability_min"]
        and s["latency_ms"] <= t["latency_limit_ms"]
        and s["error_rate_pct"] <= t["error_limit_pct"]
    )


def recovery_time_ms(samples, t, interval_s, stable_n):
    for end in range(stable_n, len(samples) + 1):
        window = samples[end-stable_n:end]
        if all(healthy(s, t) for s in window):
            return int(end * interval_s * 1000)
    return -1


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiment/config.yaml")
    ap.add_argument("--plan", default="experiment/data/experiment_plan.csv")
    ap.add_argument("--out", default="experiment/data/recovery_experiments_raw.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    with (ROOT / args.plan).open(encoding="utf-8") as f:
        plan = list(csv.DictReader(f))

    if args.max_runs:
        plan = plan[:args.max_runs]

    out = ROOT / args.out

    for idx, item in enumerate(plan, start=1):
        print("\n" + "=" * 80)
        print(
            f"{idx}/{len(plan)} {item['run_id']} | "
            f"{item['fault_type']} | {item['severity']} | "
            f"{item['candidate_action']}"
        )

        load_id = None
        spike_id = None

        try:
            reset_environment(cfg, args.dry_run)

            if not args.dry_run:
                load_id = start_load(cfg, item["workload_profile"])
                time.sleep(cfg["timing"]["workload_warmup_s"])

            baseline_samples = collect(
                cfg, cfg["timing"]["baseline_duration_s"], args.dry_run
            )
            baseline = summarize(baseline_samples, "baseline")

            spike_id = inject_fault(
                cfg, item["fault_type"], item["severity"], args.dry_run
            )

            degraded_samples = collect(
                cfg, cfg["timing"]["degraded_duration_s"], args.dry_run
            )
            pre = summarize(degraded_samples, "pre")

            t = thresholds(cfg, baseline)

            perform_action(cfg, item["candidate_action"], args.dry_run)
            post_samples = collect(
                cfg, cfg["timing"]["post_action_duration_s"], args.dry_run
            )
            post = summarize(post_samples, "post")

            rt = recovery_time_ms(
                post_samples,
                t,
                int(cfg["timing"]["sample_interval_s"]),
                int(cfg["timing"]["stable_samples_required"]),
            )
            success = int(rt >= 0)
            sla_violation = int(success == 0)

            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **item,
                **baseline,
                **pre,
                **post,
                "latency_limit_ms": t["latency_limit_ms"],
                "error_limit_pct": t["error_limit_pct"],
                "availability_min": t["availability_min"],
                "recovery_success": success,
                "recovery_time_ms": rt,
                "sla_violation": sla_violation,
            }

            append_row(out, row)
            print("saved:", out)
            print("recovery_success:", success, "recovery_time_ms:", rt)

        finally:
            stop_container(spike_id)
            stop_container(load_id)

            # Best-effort cleanup between independent runs.
            if not args.dry_run:
                try:
                    docker_compose("start", "metadata-primary", check=False)
                    reset_environment(cfg, False)
                except Exception as exc:
                    print("cleanup warning:", exc)


if __name__ == "__main__":
    main()
