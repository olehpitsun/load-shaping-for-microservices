import argparse
import csv
from pathlib import Path

FAULTS = [
    "CPU Resource Exhaustion",
    "Memory Pressure",
    "Application Error Injection",
    "Application Latency",
    "Service Dependency Failure",
    "Failed Software Deployment",
    "Temporary Workload Spike",
]
SEVERITIES = ["Low", "Medium", "High"]
ACTIONS = ["Restart", "ScaleOut", "Rollback", "Failover", "NoAction"]
WORKLOADS = ["light", "medium", "heavy"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--out", default="experiment/data/experiment_plan.csv")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    group_no = 0
    run_no = 0

    for fidx, fault in enumerate(FAULTS):
        for sidx, severity in enumerate(SEVERITIES):
            for rep in range(1, args.replicates + 1):
                group_no += 1
                group = f"IG_{group_no:05d}"
                workload = WORKLOADS[(fidx + sidx + rep) % len(WORKLOADS)]
                seed = 900000 + group_no

                for action in ACTIONS:
                    run_no += 1
                    rows.append({
                        "run_id": f"RUN_{run_no:06d}",
                        "incident_group_id": group,
                        "fault_type": fault,
                        "severity": severity,
                        "workload_profile": workload,
                        "replicate": rep,
                        "seed": seed,
                        "candidate_action": action,
                    })

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} runs / {group_no} incident groups to {out}")

if __name__ == "__main__":
    main()
