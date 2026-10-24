"""Offline integrity and numerical-summary checks for the frozen snapshot."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "benchmarks/2026-10-23"

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def main():
    manifest = json.loads((FOLDER / "provenance.json").read_text(encoding="utf-8"))
    for name, identity in manifest["files_sha256"].items():
        require(hashlib.sha256((FOLDER / name).read_bytes()).hexdigest() == identity, "snapshot hash: " + name)
    for name, identity in manifest["source_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == identity,
                "C++ source differs from measured snapshot: " + name)
    total = 0
    for kind, expected in [("serial",225),("parallel",200)]:
        with (FOLDER / (kind + ".csv")).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == expected, "wrong row count")
        require(all(r["consistent"] == "true" for r in rows), "reference mismatch")
        summary = json.loads((FOLDER / (kind + "_summary.json")).read_text(encoding="utf-8"))
        for group in summary["rows"]:
            keys = ["n","dim","queries","k","leaf","engine"] + (["threads"] if kind == "parallel" else [])
            subset = [r for r in rows if all(str(group[k]) == r[k] for k in keys)]
            require(len(subset) == 5, "repeat coverage")
            metric = "p50_us" if kind == "serial" else "batch_ms"
            require(median(float(r[metric]) for r in subset) == group[metric + "_median"], "summary median mismatch")
        total += len(rows)
    print(f"Snapshot: {total} measured rows, hashes and summary medians verified")

if __name__ == "__main__":
    main()
