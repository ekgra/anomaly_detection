# scripts/extract_templates.py
import json, re, sys, os, glob
from pathlib import Path
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence  # built-in

def strip(line: str) -> str:
    line = re.sub(r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,\.]\d+)?\s*", "", line)
    line = re.sub(r"^\s*(DEBUG|INFO|WARN|ERROR|TRACE)\s+", "", line, flags=re.I)
    return line.strip()

def main(in_path, out_path, state_path="drain3_state.bin"):
    tm = TemplateMiner(persistence_handler=FilePersistence(state_path))  # INI se config uth jayega
    templates = set()
    with open(in_path, "r", errors="ignore") as fin:
        for raw in fin:
            msg = strip(raw.rstrip("\n"))
            if not msg: continue
            res = tm.add_log_message(msg)
            if isinstance(res, dict) and "template_mined" in res:
                templates.add(res["template_mined"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(sorted(templates), open(out_path, "w"), indent=2)

if __name__ == "__main__":
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/templates"

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log_files = sorted(glob.glob(os.path.join(raw_dir, "*.log")))
    if not log_files:
        print(f"[warn] No .log files found in {raw_dir}")
        sys.exit(0)

    for in_path in log_files:
        name = Path(in_path).stem  # e.g., 'hdfs' from 'hdfs.log'
        out_path = os.path.join(out_dir, f"templates_{name}.json")
        print(f"[drain3] {name}: {in_path} -> {out_path}")
        try:
            main(in_path, out_path)
        except Exception as e:
            print(f"[error] Failed on {in_path}: {e}")
