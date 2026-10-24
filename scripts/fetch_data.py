"""Fetch the 1797-row UCI digit subset distributed by scikit-learn (data only)."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "optdigits"
URL = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/sklearn/datasets/data/digits.csv.gz"
EXPECTED_SOURCE_SHA256 = "09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "digits.csv.gz"
    if not archive.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "ExactSearch/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response, archive.with_suffix(".part").open("wb") as f:
            while block := response.read(1 << 20):
                f.write(block)
        archive.with_suffix(".part").replace(archive)
    if sha256(archive) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("data identity differs from the frozen experiment source")
    with gzip.open(archive, "rt", encoding="ascii") as f:
        rows = [[float(x) for x in row] for row in csv.reader(f)]
    if len(rows) != 1797 or any(len(row) != 65 for row in rows):
        raise RuntimeError("unexpected subset shape")
    if any(not all(0 <= x <= 16 and x.is_integer() for x in row[:64]) or
           not (0 <= row[-1] <= 9 and row[-1].is_integer()) for row in rows):
        raise RuntimeError("invalid feature or label range")
    ids = list(range(len(rows)))
    random.Random(42).shuffle(ids)
    files = []
    for name, subset in [("base", ids[:1400]), ("queries", ids[1400:])]:
        destination = OUT / (name + ".fvecs")
        with destination.open("wb") as f:
            for i in subset:
                f.write(struct.pack("<i64f", 64, *rows[i][:64]))
        (OUT / (name + "_labels.json")).write_text(json.dumps([int(rows[i][-1]) for i in subset]), encoding="utf-8")
        (OUT / (name + "_source_ids.json")).write_text(json.dumps(subset), encoding="utf-8")
        files.append({"name": destination.name, "rows": len(subset), "dim": 64,
                      "bytes": destination.stat().st_size, "sha256": sha256(destination)})
    manifest = {
        "source_url": URL, "source_sha256": sha256(archive), "files": files,
        "hash_note": "Locally computed identity hashes; not a publisher signature.",
        "citation": "Alpaydin, E. & Kaynak, C. (1998), Optical Recognition of Handwritten Digits, UCI, DOI 10.24432/C50P49",
        "license": "UCI dataset CC BY 4.0; source subset distributed in scikit-learn 1.5.2 (BSD-3-Clause project).",
        "split": "1797-row scikit-learn subset only; shuffled source IDs with Python random seed 42; first 1400 corpus, remaining 397 queries. Labels excluded from distance. Not original UCI train/test split.",
        "scope": "Exact nearest-vector retrieval; no model training, semantic-quality or classification-accuracy claim.",
    }
    target = ROOT / "results" / "dataset_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
