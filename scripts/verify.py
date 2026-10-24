"""Independent numerical oracle on the frozen, public integer-feature dataset."""
import json
import re
import subprocess
import sys
import numpy as np
from common import ROOT, executable


def read_fvecs(path, dim):
    raw = np.fromfile(path, dtype="<u4").reshape(-1, dim + 1)
    if not np.all(raw[:, 0] == dim):
        raise RuntimeError("invalid reference file headers")
    return raw[:, 1:].view("<f4").astype(np.float64)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    subprocess.run([str(executable("exactsearch_tests"))], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/check_cli.py"], cwd=ROOT, check=True)
    from run_experiments import regression_result_isolation
    regression_result_isolation()
    folder = ROOT / "data/raw/optdigits"
    if not (folder / "base.fvecs").exists():
        raise RuntimeError("Run python scripts/fetch_data.py before the public-data oracle")
    base = read_fvecs(folder / "base.fvecs", 64)
    queries = read_fvecs(folder / "queries.fvecs", 64)
    checks = []
    for index in list(range(19)) + [len(queries) - 1]:
        distances = np.sum((base - queries[index]) ** 2, axis=1, dtype=np.float64)
        ids = np.lexsort((np.arange(len(base)), distances))[:10]
        expected = [(int(i), float(distances[i])) for i in ids]
        for engine in ("flat", "kdtree"):
            result = subprocess.run([str(executable("exactsearch")), "query", "--base",
                str(folder / "base.fvecs"), "--queries", str(folder / "queries.fvecs"),
                "--query-index", str(index), "--k", "10", "--engine", engine],
                cwd=ROOT, encoding="utf-8", capture_output=True, check=True, timeout=30)
            actual = []
            for line in result.stdout.splitlines():
                match = re.fullmatch(r"(\d+),([0-9.eE+-]+)", line)
                if match:
                    actual.append((int(match[1]), float(match[2])))
            if actual != expected:
                raise RuntimeError(f"Oracle mismatch: query {index}, engine {engine}")
            checks.append({"query_index":index, "engine":engine, "passed":True})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/numeric_verification.json").write_text(
        json.dumps({"passed":True,"comparisons":checks}, indent=2), encoding="utf-8")
    print(f"Independent NumPy oracle: {len(checks)} comparisons passed")


if __name__ == "__main__":
    main()
