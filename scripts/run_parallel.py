"""Five controlled batch workloads; wall-time comparisons use one batch API."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from common import ROOT, executable


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exe=executable("exactsearch_batch")
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    result=ROOT/"results"
    folder=result/"parallel_runs"/run_id
    folder.mkdir(parents=True)
    generated=ROOT/"data"/"generated"/"serial_seed42"
    real=ROOT/"data"/"raw"/"optdigits"
    slices=ROOT/"data"/"generated"/"parallel_queries"
    slices.mkdir(parents=True,exist_ok=True)
    source=(real/"queries.fvecs").read_bytes()
    if len(source)!=397*260:
        raise RuntimeError("unexpected real query shape")
    for q in [1,8]:
        (slices/f"real_q{q}.fvecs").write_bytes(source[:q*260])
    cases=[
        ("uniform8_q128",generated/"base_n10000_d8.fvecs",generated/"queries_q128_d8.fvecs",10000,8,128),
        ("uniform128_q128",generated/"base_n10000_d128.fvecs",generated/"queries_q128_d128.fvecs",10000,128,128),
        ("optdigits_q397",real/"base.fvecs",real/"queries.fvecs",1400,64,397),
        ("optdigits_q1",real/"base.fvecs",slices/"real_q1.fvecs",1400,64,1),
        ("optdigits_q8",real/"base.fvecs",slices/"real_q8.fvecs",1400,64,8),
    ]
    manifest={"run_id":run_id,"protocol":"full_parallel","status":"running",
              "exe_sha256":sha(exe),"repeats":5,"commands":[],
              "timing":"Warm BatchExecutor.search wall time includes query copies, output allocation, task submission and future collection; excludes constructor/tree build/validation/result destruction. Startup_ms measures executor constructor only.",
              "baseline":"Same batch API at threads=1; serial CLI qps is not used as a baseline."}
    all_rows=[]
    try:
        for dataset,base,queries,n,dim,q in cases:
            out=folder/(dataset+".csv")
            command=[str(exe),str(base),str(queries),str(out),"5",dataset]
            record={"dataset":dataset,"command":command,"base_sha256":sha(base),
                    "queries_sha256":sha(queries),"status":"running"}
            manifest["commands"].append(record)
            write_json(folder/"manifest.json",manifest)
            print("START "+dataset,flush=True)
            start=time.perf_counter()
            process=subprocess.run(command,cwd=ROOT,capture_output=True,encoding="utf-8",timeout=180)
            record.update(exit=process.returncode,wall_seconds=time.perf_counter()-start)
            for channel in ["stdout","stderr"]:
                (folder/(dataset+"."+channel+".txt")).write_text(getattr(process,channel),encoding="utf-8")
            if process.returncode!=0:
                raise RuntimeError(process.stderr)
            with out.open(encoding="utf-8",newline="") as f:
                rows=list(csv.DictReader(f))
            expected={(e,t,r) for e in ["flat","kdtree"] for t in [1,2,4,8] for r in range(1,6)}
            seen=set()
            for row in rows:
                key=(row["engine"],int(row["threads"]),int(row["repeat"]))
                if key not in expected or key in seen or row["consistent"]!="true":
                    raise RuntimeError("invalid/missing/duplicate batch row")
                seen.add(key)
                if any(int(row[k])!=v for k,v in [("n",n),("dim",dim),("queries",q),("k",10),("leaf",16)]):
                    raise RuntimeError("batch row shape mismatch")
                if row["dataset"]!=dataset or any(not math.isfinite(float(row[k])) or float(row[k])<0
                    for k in ["startup_ms","batch_ms","qps"]):
                    raise RuntimeError("invalid batch numeric result")
                if float(row["batch_ms"])<=0 or float(row["qps"])<=0:
                    raise RuntimeError("timer resolution too low for this batch")
                expected_qps=q*1000/float(row["batch_ms"])
                if not math.isclose(float(row["qps"]),expected_qps,rel_tol=1e-10):
                    raise RuntimeError("batch qps unit mismatch")
            if seen!=expected or len({row["checksum"] for row in rows})!=1:
                raise RuntimeError("batch coverage or checksum inconsistency")
            record.update(status="completed",csv_sha256=sha(out))
            all_rows.extend(rows)
            print("PASS "+dataset+" (40 validated rows)",flush=True)
        with (folder/"all.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        summary=[]
        for dataset,*_ in cases:
            for engine in ["flat","kdtree"]:
                baseline=statistics.median(float(r["batch_ms"]) for r in all_rows
                    if r["dataset"]==dataset and r["engine"]==engine and int(r["threads"])==1)
                for threads in [1,2,4,8]:
                    rows=[r for r in all_rows if r["dataset"]==dataset and r["engine"]==engine and int(r["threads"])==threads]
                    entry={k:rows[0][k] for k in ["dataset","engine"]}
                    entry.update({k:int(rows[0][k]) for k in ["n","dim","queries","k","leaf","threads"]})
                    entry["repeats"]=len(rows)
                    for field in ["batch_ms","qps","startup_ms"]:
                        values=[float(r[field]) for r in rows]
                        entry[field+"_median"]=statistics.median(values)
                        entry[field+"_min"]=min(values)
                        entry[field+"_max"]=max(values)
                    entry["speedup_vs_1thread"]=baseline/entry["batch_ms_median"]
                    summary.append(entry)
        summary_obj={"run_id":run_id,"protocol":"full_parallel","rows":summary,
                     "definition":"Medians of 5 batches; speedup=median(T1 wall_ms)/median(Tp wall_ms), same dataset and engine. Not per-query latency or serial CLI call-time qps."}
        write_json(folder/"summary.json",summary_obj)
        manifest.update(status="completed",raw_rows=len(all_rows),summary_groups=len(summary))
        write_json(folder/"manifest.json",manifest)
        # Publish aliases only after every case and its consistency checks finish.
        (result/"parallel_all.csv").write_bytes((folder/"all.csv").read_bytes())
        write_json(result/"parallel_summary.json",summary_obj)
        write_json(result/"parallel_manifest.json",manifest)
        print(f"Completed {len(cases)} workloads, {len(all_rows)} validated rows",flush=True)
    except Exception as e:
        manifest.update(status="failed",error=repr(e))
        write_json(folder/"manifest.json",manifest)
        raise


if __name__=="__main__":
    main()
