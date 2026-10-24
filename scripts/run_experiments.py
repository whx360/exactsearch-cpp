"""Run the serial exact-search protocol; stdlib and NumPy are the only dependencies.

Default: 15 configurations, five repeats, 128 synthetic queries, all 397 real
queries. --quick is a smoke protocol (fewer configurations, one repeat and 16
synthetic queries); its manifest explicitly marks it as unsuitable for final
performance claims. Each CLI invocation runs synchronously, never concurrently.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np


from common import ROOT, executable as find_executable
ENGINES = ("sort", "flat", "kdtree")
CSV_FIELDS = (
    "dataset,n,dim,queries,k,leaf,engine,threads,repeat,p50_us,p95_us,batch_ms,qps,"
    "mean_distance_evals,mean_nodes,mean_box_evals,mean_heap_updates,index_build_ms,"
    "index_bytes,data_bytes,consistent,checksum"
).split(",")
METRICS = (
    "p50_us", "p95_us", "batch_ms", "qps", "mean_distance_evals", "mean_nodes",
    "mean_box_evals", "mean_heap_updates", "index_build_ms", "index_bytes", "data_bytes",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_fvecs(path, matrix):
    """Write independent little-endian row records, each int32 dim + dim float32."""
    values = np.ascontiguousarray(matrix, dtype="<f4")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("generated matrix must be nonempty and two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("generated matrix contains a non-finite value")
    records = np.empty((values.shape[0], values.shape[1] + 1), dtype="<u4")
    records[:, 0] = values.shape[1]
    records[:, 1:] = values.view("<u4")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    records.tofile(temporary)
    os.replace(temporary, path)


def inspect_fvecs(path, expected_rows, expected_dim, kind):
    """Validate every row header and value before accepting an input identity."""
    path = Path(path)
    if path.stat().st_size != expected_rows * (expected_dim + 1) * 4:
        raise ValueError(f"unexpected fvecs byte count: {path}")
    records = np.fromfile(path, dtype="<u4").reshape(expected_rows, expected_dim + 1)
    if not np.all(records[:, 0] == expected_dim):
        raise ValueError(f"inconsistent fvecs dimensions: {path}")
    if not np.isfinite(records[:, 1:].view("<f4")).all():
        raise ValueError(f"non-finite fvecs coordinates: {path}")
    return {"path": display_path(path), "rows": expected_rows, "dim": expected_dim,
            "bytes": path.stat().st_size, "sha256": sha256_file(path), "kind": kind}


def prepare_data(quick, manifest, save_manifest):
    generated = ROOT / "data" / "generated" / "serial_seed42"
    generator = np.random.default_rng(42)
    # Generate in precisely this order, independently of the chosen dimensions.
    base = generator.uniform(-1.0, 1.0, size=(10000, 128)).astype("<f4")
    queries = generator.uniform(-1.0, 1.0, size=(128, 128)).astype("<f4")
    identity = {
        "seed": 42, "bit_generator": generator.bit_generator.__class__.__name__,
        "numpy_version": np.__version__, "distribution": "uniform(-1,1), float64 draw then float32 cast",
        "draw_order": ["base[10000,128]", "queries[128,128]"], "dtype": "little-endian float32",
        "base_raw_sha256": array_hash(base), "queries_raw_sha256": array_hash(queries),
        "dimension_control": "each d uses the first d columns of these same parent arrays",
        "scale_control": "N=1000 is a prefix; N=50000 retains the first 10000 rows and appends seed43 rows",
    }
    identity["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest["synthetic_generation"] = identity
    files = manifest["input_files"]
    cache = {}

    def add_generated(n, dim, values):
        base_path = generated / f"base_n{n}_d{dim}.fvecs"
        query_path = generated / f"queries_q128_d{dim}.fvecs"
        write_fvecs(base_path, values)
        base_info = inspect_fvecs(base_path, n, dim, "synthetic_uniform")
        files[display_path(base_path)] = base_info
        if display_path(query_path) not in files:
            write_fvecs(query_path, queries[:, :dim])
            files[display_path(query_path)] = inspect_fvecs(query_path, 128, dim, "synthetic_uniform")
        cache[(n, dim)] = (base_info, files[display_path(query_path)])
        save_manifest()

    dimension_sweep = [2, 16, 128] if quick else [2, 4, 8, 16, 32, 64, 128]
    # d=8 is also needed by the leaf-size intervention, including quick mode.
    for dim in sorted(set(dimension_sweep + [8])):
        add_generated(10000, dim, base[:, :dim])
    add_generated(1000, 16, base[:1000, :16])
    if not quick:
        extension_rng = np.random.default_rng(43)
        extension = extension_rng.uniform(-1.0, 1.0, size=(40000, 16)).astype("<f4")
        expanded = np.concatenate((base[:, :16], extension), axis=0)
        manifest["scale_extension_generation"] = {
            "seed": 43, "bit_generator": extension_rng.bit_generator.__class__.__name__,
            "shape": [40000, 16], "raw_sha256": array_hash(extension),
            "expanded_base_raw_sha256": array_hash(expanded),
        }
        add_generated(50000, 16, expanded)

    real_base = ROOT / "data" / "raw" / "optdigits" / "base.fvecs"
    real_queries = ROOT / "data" / "raw" / "optdigits" / "queries.fvecs"
    real_pair = (inspect_fvecs(real_base, 1400, 64, "real_optdigits_subset"),
                 inspect_fvecs(real_queries, 397, 64, "real_optdigits_subset"))
    for information in real_pair:
        files[information["path"]] = information
    source_manifest = ROOT / "results" / "dataset_manifest.json"
    if source_manifest.exists():
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        expected = {item["name"]: item["sha256"] for item in source["files"]}
        for information in real_pair:
            if expected[Path(information["path"]).name] != information["sha256"]:
                raise ValueError("real dataset SHA256 differs from dataset_manifest.json")
        manifest["real_dataset_source"] = {
            "manifest_path": display_path(source_manifest), "manifest_sha256": sha256_file(source_manifest),
            "source_url": source.get("source_url"), "split": source.get("split"),
            "scope": source.get("scope"),
        }

    repeats, query_limit = (1, 16) if quick else (5, 128)
    configurations = []
    seen = set()

    def add(n, dim, k, leaf, intervention, real=False):
        query_count = 397 if real else query_limit
        key = ("real" if real else "synthetic", n, dim, query_count, k, leaf)
        if key in seen:
            return
        seen.add(key)
        base_info, query_info = real_pair if real else cache[(n, dim)]
        identifier = ("optdigits" if real else "uniform") + f"_n{n}_d{dim}_q{query_count}_k{k}_leaf{leaf}"
        configurations.append({
            "id": identifier, "dataset": "optdigits_real_subset" if real else "synthetic_uniform_nested_seed42",
            "data_kind": "real_optdigits_subset" if real else "synthetic_uniform",
            "intervention": intervention, "n": n, "dim": dim, "queries": query_count,
            "k": k, "leaf": leaf, "repeats": repeats, "engines": list(ENGINES),
            "base": base_info, "query_input": query_info,
        })

    for dim in dimension_sweep:
        add(10000, dim, 10, 16, "dimension")
    for k in ([1, 1000] if quick else [1, 100, 1000]):
        add(10000, 16, k, 16, "k")
    for leaf in ([64] if quick else [4, 64]):
        add(10000, 8, 10, leaf, "leaf_size")
    for n in ([1000] if quick else [1000, 50000]):
        add(n, 16, 10, 16, "corpus_size")
    add(1400, 64, 10, 16, "real_descriptor_validation", real=True)
    manifest["configurations"] = configurations
    save_manifest()
    return configurations


def validate_csv(path, config):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"unexpected CSV schema in {path}: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != len(ENGINES) * config["repeats"]:
        raise ValueError(f"missing or additional engine/repeat rows: {path}")
    observed = set()
    checksums = set()
    for row in rows:
        for field in ("n", "dim", "queries", "k", "leaf"):
            if int(row[field]) != config[field]:
                raise ValueError(f"CSV {field} differs from the requested configuration")
        if row["dataset"] != config["dataset"] or int(row["threads"]) != 1:
            raise ValueError("CSV dataset/threads differs from serial protocol")
        engine, repeat = row["engine"], int(row["repeat"])
        if engine not in ENGINES or not 1 <= repeat <= config["repeats"] or (engine, repeat) in observed:
            raise ValueError("CSV contains an invalid/duplicate engine or repeat")
        observed.add((engine, repeat))
        for field in METRICS:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"CSV contains invalid numerical metric: {field}")
        if float(row["p50_us"]) <= 0 or float(row["qps"]) <= 0:
            raise ValueError("zero-duration measurement cannot support a speedup claim")
        if float(row["p95_us"]) < float(row["p50_us"]):
            raise ValueError("p95 is below p50")
        distances = float(row["mean_distance_evals"])
        if distances > config["n"] + 1e-9:
            raise ValueError("distance count exceeds corpus size")
        if engine in ("sort", "flat") and distances != config["n"]:
            raise ValueError("exhaustive baseline did not scan every point")
        if row["consistent"] != "true":
            raise ValueError("an engine failed exact reference verification")
        checksums.add(int(row["checksum"]))
        row["experiment_id"] = config["id"]
        row["data_kind"] = config["data_kind"]
        row["base_sha256"] = config["base"]["sha256"]
        row["queries_sha256"] = config["query_input"]["sha256"]
        row["distance_scan_ratio"] = repr(distances / config["n"])
    if len(checksums) != 1:
        raise ValueError("result checksums differ across engines or repeats")
    return rows


def publish_file(source, destination):
    """Publish an artifact alias only after its run-local source is complete."""
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def write_aggregates(rows, configurations, results, *, run_id=None,
                     protocol="full_serial", run_directory=None):
    if protocol not in ("full_serial", "quick_smoke_not_for_final_claims"):
        raise ValueError("unknown aggregation protocol")
    results = Path(results)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if run_directory is None:
        run_directory = results / "serial_runs" / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
    else:
        run_directory = Path(run_directory)
    csv_path = run_directory / "all.csv"
    fields = ["experiment_id", "data_kind", "base_sha256", "queries_sha256"] + CSV_FIELDS + ["distance_scan_ratio"]
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    summaries = []
    for config in configurations:
        matching = [row for row in rows if row["experiment_id"] == config["id"]]
        flat = [row for row in matching if row["engine"] == "flat"]
        flat_p50 = statistics.median(float(row["p50_us"]) for row in flat)
        for engine in ENGINES:
            group = [row for row in matching if row["engine"] == engine]
            entry = {key: config[key] for key in
                     ("id", "dataset", "data_kind", "intervention", "n", "dim", "queries", "k", "leaf")}
            entry.update({"engine": engine, "repeats": len(group), "consistent": True})
            for field in METRICS + ("distance_scan_ratio",):
                entry[field + "_median"] = statistics.median(float(row[field]) for row in group)
            entry["p50_us_min"] = min(float(row["p50_us"]) for row in group)
            entry["p50_us_max"] = max(float(row["p50_us"]) for row in group)
            entry["p50_speedup_vs_flat"] = flat_p50 / entry["p50_us_median"]
            summaries.append(entry)
    summary = {
        "run_id": run_id,
        "protocol": protocol,
        "smoke_only": protocol != "full_serial",
        "summary_definition": "Each metric is the median of per-repeat CLI rows; percentile medians are not pooled query percentiles.",
        "scan_ratio_definition": "mean_distance_evals / n; exact count of evaluated corpus points, excluding bounding-box work",
        "speedup_definition": "median(flat p50_us) / median(engine p50_us) for the identical configuration; >1 is faster",
        "build_note": "One tree build per CLI invocation, reused by its repeats; repeated build_ms entries are not independent builds.",
        "rows": summaries,
    }
    summary_path = run_directory / "summary.json"
    atomic_json(summary_path, summary)
    # A successful smoke run must never replace the formal result aliases.
    prefix = "serial" if protocol == "full_serial" else "serial_smoke"
    published_csv = results / (prefix + "_all.csv")
    published_summary = results / (prefix + "_summary.json")
    publish_file(csv_path, published_csv)
    publish_file(summary_path, published_summary)
    latest = {"run_id": run_id, "protocol": protocol,
              "csv": display_path(csv_path), "summary": display_path(summary_path)}
    atomic_json(results / "serial_latest.json", latest)
    return {"csv": display_path(csv_path), "csv_sha256": sha256_file(csv_path),
            "summary": display_path(summary_path), "summary_sha256": sha256_file(summary_path),
            "published_csv": display_path(published_csv),
            "published_summary": display_path(published_summary),
            "raw_rows": len(rows), "summary_groups": len(summaries)}


def save_run_manifest(results, run_directory, manifest):
    """Keep history and progress; promote formal/smoke manifests only on success."""
    atomic_json(Path(run_directory) / "experiment_manifest.json", manifest)
    atomic_json(Path(results) / "experiment_latest_manifest.json", manifest)
    if manifest["status"] == "completed":
        name = ("experiment_manifest.json" if manifest["protocol"] == "full_serial"
                else "experiment_smoke_manifest.json")
        atomic_json(Path(results) / name, manifest)


def regression_result_isolation():
    """Small filesystem fixture, no executable/benchmark/network invocation.

    Root cause: quick/full aggregates shared top-level names without run identity;
    a quick success replaced formal metrics, and a failed run could replace the
    last completed manifest. Keep immutable run-local artifacts and promote only
    the corresponding completed protocol. This helper verifies those boundaries.
    """
    # Keep the fixture in the explicit project root: some Windows sandboxes allow
    # creation in redirected TEMP but deny subsequent access to its children.
    with tempfile.TemporaryDirectory(prefix=".result-isolation-fixture-", dir=ROOT) as temporary:
        results = Path(temporary)
        assert results.resolve().parent == ROOT.resolve(), "fixture cleanup must stay inside the project"
        config = {"id": "fixture", "dataset": "fixture", "data_kind": "synthetic_fixture",
                  "intervention": "fixture", "n": 4, "dim": 2, "queries": 2, "k": 1,
                  "leaf": 1}

        def fixture_rows(scale):
            rows = []
            for engine in ENGINES:
                row = {field: "1" for field in CSV_FIELDS}
                row.update(dataset="fixture", n="4", dim="2", queries="2", k="1",
                           leaf="1", engine=engine, consistent="true", checksum="123",
                           p50_us=str(scale), p95_us=str(scale * 2),
                           batch_ms=str(scale * 2 / 1000), qps=str(1000000 / scale),
                           mean_distance_evals="4", distance_scan_ratio="1.0",
                           experiment_id="fixture", data_kind="synthetic_fixture",
                           base_sha256="fixture-base", queries_sha256="fixture-queries")
                rows.append(row)
            return rows

        full = write_aggregates(fixture_rows(10), [config], results, run_id="fixture-full")
        formal_names = ("serial_all.csv", "serial_summary.json")
        formal_bytes = {name: (results / name).read_bytes() for name in formal_names}
        full_directory = results / "serial_runs" / "fixture-full"
        full_manifest = {"run_id": "fixture-full", "protocol": "full_serial", "status": "completed"}
        save_run_manifest(results, full_directory, full_manifest)
        formal_manifest_bytes = (results / "experiment_manifest.json").read_bytes()

        smoke = write_aggregates(fixture_rows(1), [config], results, run_id="fixture-smoke",
                                 protocol="quick_smoke_not_for_final_claims")
        smoke_directory = results / "serial_runs" / "fixture-smoke"
        smoke_manifest = {"run_id": "fixture-smoke", "protocol": "quick_smoke_not_for_final_claims",
                          "status": "completed"}
        save_run_manifest(results, smoke_directory, smoke_manifest)
        for name, expected in formal_bytes.items():
            assert (results / name).read_bytes() == expected, "smoke replaced formal aggregate"
        assert (results / "experiment_manifest.json").read_bytes() == formal_manifest_bytes
        assert Path(full["csv"]).read_bytes() == formal_bytes["serial_all.csv"]
        assert Path(full["summary"]).read_bytes() == formal_bytes["serial_summary.json"]
        full_summary = json.loads(Path(full["summary"]).read_text(encoding="utf-8"))
        smoke_summary = json.loads(Path(smoke["summary"]).read_text(encoding="utf-8"))
        assert full_summary["run_id"] == "fixture-full" and full_summary["protocol"] == "full_serial"
        assert smoke_summary["run_id"] == "fixture-smoke" and smoke_summary["smoke_only"] is True
        assert (results / "serial_smoke_summary.json").read_bytes() == Path(smoke["summary"]).read_bytes()

        failed_directory = results / "serial_runs" / "fixture-failed"
        failed_directory.mkdir()
        for status in ("running", "failed"):
            save_run_manifest(results, failed_directory,
                              {"run_id": "fixture-failed", "protocol": "full_serial", "status": status})
            assert (results / "experiment_manifest.json").read_bytes() == formal_manifest_bytes
            for name, expected in formal_bytes.items():
                assert (results / name).read_bytes() == expected, "incomplete run replaced formal aggregate"
        assert json.loads((results / "experiment_latest_manifest.json").read_text(encoding="utf-8"))["status"] == "failed"
        assert Path(full["csv"]).exists() and Path(smoke["csv"]).exists()
    return {"status": "passed", "benchmark_invocations": 0,
            "checked": ["run-local artifacts", "summary identity", "smoke isolation", "failed-run isolation"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="smoke protocol; all 397 real queries are retained")
    parser.add_argument("--timeout-seconds", type=float, default=1800,
                        help="maximum wall-clock seconds per CLI invocation (default: 1800)")
    args = parser.parse_args(argv)
    if args.exe is None:
        args.exe = find_executable("exactsearch")
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = results / "serial_runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "run_id": run_id, "status": "running", "started_utc": utc_now(),
        "protocol": "quick_smoke_not_for_final_claims" if args.quick else "full_serial",
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count()},
        "script": {"path": display_path(__file__), "sha256": sha256_file(__file__)},
        "measurement": {"engines": list(ENGINES), "benchmark_processes_concurrent": 1,
                        "engine_order": "CLI rotates engine order by repeat; three warmup queries per engine batch",
                        "timing_scope": "search call only; reference generation, correctness check, checksum, destruction and tree build excluded",
                        "query_order": "fixed input-file order", "clock_unit": "CLI steady_clock microseconds",
                        "correctness": "every timed and warmup query compared with CLI full-sort reference; all checksums must agree",
                        "limitations": "one host/session; fixed-seed synthetic data are controlled experiments, not real descriptors"},
        "input_files": {}, "configurations": [], "commands": [],
    }

    def save_manifest():
        save_run_manifest(results, run_directory, manifest)

    save_manifest()
    stage = "preflight"
    try:
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be finite and positive")
        executable = args.exe if args.exe.is_absolute() else ROOT / args.exe
        executable = executable.resolve(strict=True)
        manifest["executable"] = {"path": display_path(executable), "sha256": sha256_file(executable)}
        save_manifest()
        print(f"Protocol={manifest['protocol']} run={run_id}; generating/validating inputs...", flush=True)
        stage = "data_generation_and_validation"
        configurations = prepare_data(args.quick, manifest, save_manifest)
        all_rows = []
        for index, config in enumerate(configurations, start=1):
            stage = "benchmark:" + config["id"]
            output = run_directory / (config["id"] + ".csv")
            command = [str(executable), "bench", "--base", str(ROOT / config["base"]["path"]),
                       "--queries", str(ROOT / config["query_input"]["path"]), "--k", str(config["k"]),
                       "--leaf", str(config["leaf"]), "--repeats", str(config["repeats"]),
                       "--out", str(output), "--dataset", config["dataset"],
                       "--limit-queries", str(config["queries"]), "--engines", ",".join(ENGINES)]
            record = {"configuration_id": config["id"], "command": command, "cwd": str(ROOT),
                      "base_sha256": config["base"]["sha256"], "queries_sha256": config["query_input"]["sha256"],
                      "started_utc": utc_now(), "status": "running", "exit_code": None,
                      "stdout": "", "stderr": "", "csv": display_path(output)}
            manifest["commands"].append(record)
            save_manifest()
            print(f"[{index}/{len(configurations)}] START {config['id']} repeats={config['repeats']}", flush=True)
            begin = time.perf_counter()
            try:
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                           encoding="utf-8", errors="replace", timeout=args.timeout_seconds,
                                           check=False)
                record.update(exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
            except subprocess.TimeoutExpired as error:
                record.update(status="timeout", stdout=error.stdout.decode("utf-8", errors="replace")
                              if isinstance(error.stdout, bytes) else (error.stdout or ""),
                              stderr=error.stderr.decode("utf-8", errors="replace")
                              if isinstance(error.stderr, bytes) else (error.stderr or ""))
                raise
            except OSError as error:
                record.update(status="launch_failed", stderr=str(error))
                raise
            finally:
                record["elapsed_wall_seconds"] = time.perf_counter() - begin
                record["finished_utc"] = utc_now()
                for channel in ("stdout", "stderr"):
                    log = run_directory / (config["id"] + "." + channel + ".txt")
                    log.write_text(record[channel], encoding="utf-8")
                    record[channel + "_log"] = display_path(log)
                save_manifest()
            if record["stdout"]:
                print(record["stdout"], end="" if record["stdout"].endswith("\n") else "\n", flush=True)
            if record["stderr"]:
                print(record["stderr"], file=sys.stderr,
                      end="" if record["stderr"].endswith("\n") else "\n", flush=True)
            if record["exit_code"] != 0:
                record["status"] = "failed"
                raise RuntimeError(f"benchmark exited {record['exit_code']}: {config['id']}")
            try:
                rows = validate_csv(output, config)
            except Exception:
                record["status"] = "csv_validation_failed"
                raise
            record.update(status="completed", csv_sha256=sha256_file(output), validated_rows=len(rows))
            all_rows.extend(rows)
            save_manifest()
            print(f"[{index}/{len(configurations)}] PASS {config['id']} wall={record['elapsed_wall_seconds']:.3f}s", flush=True)
        stage = "aggregation"
        manifest["aggregates"] = write_aggregates(all_rows, configurations, results,
                                                  run_id=run_id, protocol=manifest["protocol"],
                                                  run_directory=run_directory)
        manifest.update(status="completed", finished_utc=utc_now())
        save_manifest()
        print(f"Completed {len(configurations)} configurations, {len(all_rows)} validated rows; "
              f"manifest={run_directory / 'experiment_manifest.json'}", flush=True)
        return 0
    except Exception as error:
        manifest.update(status="failed", failed_stage=stage, finished_utc=utc_now(),
                        error_type=type(error).__name__, error=str(error))
        save_manifest()
        print(f"FAILED at {stage}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
