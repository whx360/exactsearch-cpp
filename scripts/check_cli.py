"""Regression of CLI validation, Unicode absolute paths and input protection."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

from common import ROOT, executable
EXE = executable("exactsearch")


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    records = []

    def run(name, args, success, must_contain=None):
        result = subprocess.run([str(EXE), *map(str, args)], cwd=ROOT,
                                encoding="utf-8", capture_output=True, timeout=30)
        ok = (result.returncode == 0) == success
        if must_contain:
            ok = ok and must_contain in result.stdout
        records.append({"name": name, "args": list(map(str,args)), "exit": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr, "passed": ok})
        if not ok:
            raise RuntimeError("CLI regression failed: " + name + "\n" + result.stderr)

    with tempfile.TemporaryDirectory(prefix="路径回归_", dir=ROOT) as temporary:
        folder = Path(temporary)
        base, queries = folder/"输入数据.fvecs", folder/"查询数据.fvecs"
        with base.open("wb") as f:
            for x, y in [(0,0),(1,0),(0,1),(2,2)]:
                f.write(struct.pack("<i2f",2,x,y))
        with queries.open("wb") as f:
            f.write(struct.pack("<i2f",2,0,0))
        common = ["--base",base,"--queries",queries]
        for engine in ["sort","flat","kdtree"]:
            run("Unicode absolute query "+engine,
                ["query",*common,"--k",3,"--engine",engine],True,"0,0\n1,1\n2,1")
        output = folder/"实验输出.csv"
        run("Unicode absolute CSV output",["bench",*common,"--k",3,"--repeats",1,"--out",output],True)
        if not output.exists():
            raise RuntimeError("Unicode output not created")
        before = hashlib.sha256(base.read_bytes()).hexdigest()
        run("prevent output overwriting input",["bench",*common,"--out",base],False)
        if hashlib.sha256(base.read_bytes()).hexdigest() != before:
            raise RuntimeError("input changed by rejected output")
        run("unknown flag",["query",*common,"--kk",3],False)
        run("negative k",["query",*common,"--k",-1],False)
        run("duplicate flag",["query",*common,"--k",1,"--k",2],False)
        run("unknown engine",["query",*common,"--engine","ann"],False)
        run("out of range query",["query",*common,"--query-index",999],False)
        run("missing flag value",["query",*common,"--k"],False)
        run("zero dimension synthetic",["bench","--dim",0,"--out",output],False)
        run("empty synthetic corpus",["bench","--n",0,"--out",output],False)
        run("duplicate engines",["bench",*common,"--out",output,"--engines","flat,flat"],False)
        run("manual demo",["demo"],True,"Expected")
    report = {"passed":True,"cases":len(records),"records":records,
              "note":"Temporary Unicode fixtures removed after regression; paths recorded for diagnosis."}
    (ROOT/"results"/"cli_test_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"CLI: {len(records)} regression cases passed")


if __name__ == "__main__":
    main()
