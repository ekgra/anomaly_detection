#!/usr/bin/env python3
"""
Generate synthetic logs automatically.

Defaults:
- Templates dir: data/templates/  (expects files like templates_hdfs.json)
- Param pools:   data/params/*.txt (optional; e.g., ip.txt, user.txt)
- Outputs: logs/train_normal_100k.log, logs/val_normal_10k.log,
          logs/test_clean_109800.log (+ test_mixed_110k.log if --with-anomalies)

USAGE:          
  # Auto: reads templates from data/templates, params from data/params, writes logs/*
  python scripts/generate_logs.py

  # With anomalies too
  python scripts/generate_logs.py --with-anomalies

  # Override sizes / paths if needed
  python scripts/generate_logs.py --train 200000 --val 20000 --test-clean 150000 --out-dir out_logs
"""

import argparse, json, os, re, random
from pathlib import Path
from datetime import datetime, timedelta

# -------- Config --------
TS_FMT = "%Y-%m-%d %H:%M:%S,%f"  # we will trim to millis
LEVELS_DEFAULT = {"INFO": 0.80, "WARN": 0.15, "ERROR": 0.05}
PLACEHOLDER_RE = re.compile(r"<([^>]+)>")  # <*>, <IP>, <NUM>, <USER>...

# -------- Helpers --------
def parse_level_dist(s: str):
    if not s:
        return LEVELS_DEFAULT
    dist = {}
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        k, v = part.split(":")
        dist[k.strip().upper()] = float(v)
    tot = sum(dist.values())
    if tot <= 0:
        raise ValueError("level distribution must sum > 0")
    for k in dist:
        dist[k] /= tot
    return dist

def choose_level(dist):
    ks, ws = list(dist.keys()), [dist[k] for k in dist.keys()]
    return random.choices(ks, weights=ws, k=1)[0]

def load_templates_from_dir(dir_path: Path):
    arr = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            strs = [str(x) for x in data if isinstance(x, str)]
            arr.extend(strs)
            print(f"[tmpl] loaded {len(strs)} templates from {p.name}")
        except Exception as e:
            print(f"[warn] failed reading {p}: {e}")
    if not arr:
        raise FileNotFoundError(f"No templates found in {dir_path} (need *.json arrays)")
    print(f"[tmpl] total templates: {len(arr)}")
    return arr

def load_param_pools_from_dir(dir_path: Path):
    pools = {}
    generic = []
    if dir_path.exists():
        for p in sorted(dir_path.glob("*.txt")):
            vals = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            key = p.stem.lower()
            pools[key] = vals
            generic.extend(vals)
            print(f"[param] {key}: {len(vals)} values")
    pools["_generic"] = generic
    return pools

def rand_ip(pools):
    if "ip" in pools and pools["ip"]:
        return random.choice(pools["ip"])
    return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def rand_num():
    return str(random.randint(0, 10**6))

def rand_generic(pools):
    g = pools.get("_generic", [])
    if g:
        return random.choice(g)
    return random.choice(["alpha","beta","gamma","delta","svc","node","user","blk"])

def replace_placeholders(template: str, pools):
    def _repl(m):
        tag = m.group(1).strip()
        tagU = tag.upper()
        if tag == "*":
            return rand_generic(pools)
        if tagU in ("IP","ADDR","ADDRESS"):
            return rand_ip(pools)
        if tagU in ("NUM","NUMBER","INT","ID"):
            return rand_num()
        key = tag.lower()
        if key in pools and pools[key]:
            return random.choice(pools[key])
        return rand_generic(pools)

    out = PLACEHOLDER_RE.sub(_repl, template)
    out = re.sub(r"\*", lambda _: rand_generic(pools), out)  # stray * → token
    return out

def ts_next(start: datetime, idx: int, step_ms: int, jitter_ms: int, monotonic: bool, last_ts: datetime|None):
    if monotonic:
        delta = step_ms + random.randint(-jitter_ms, jitter_ms)
        if delta < 1:
            delta = 1
        base = (last_ts or start) + timedelta(milliseconds=delta)
        return base, base
    base = start + timedelta(milliseconds=idx*step_ms)
    base = base + timedelta(milliseconds=random.randint(-jitter_ms, jitter_ms))
    return base, last_ts

def make_line(template: str, pools, when: datetime, level: str):
    body = replace_placeholders(template, pools)
    ts_str = when.strftime(TS_FMT)[:-3]  # millis
    return f"{ts_str} {level} {body}"

# -------- Anomaly injection (optional) --------
def inject_anomalies(lines: list[str], count: int, types=("typo","new_template","order_violation","burst")):
    out = lines[:]  # copy
    n_each = max(1, count // len(types))
    injected = 0

    def do_typo(s): return s.replace("block", "blok", 1) if "block" in s else s.replace("INFO","INOF",1)

    def make_new_template():
        now = datetime.utcnow().strftime(TS_FMT)[:-3]
        return f"{now} CRITICAL kernel panic: code={random.randint(1,9999)} on {rand_ip({'ip':[],'_generic':[]})}"

    # 1) typos
    for i in random.sample(range(len(out)), min(n_each, len(out))):
        out[i] = out[i] + " ###ANOM### type=typo"
        out[i] = do_typo(out[i])
        injected += 1

    # 2) new templates
    for _ in range(n_each):
        out.append(make_new_template() + " ###ANOM### type=new_template")
        injected += 1

    # 3) order violation: reverse a short chunk
    if len(out) > 20:
        start = random.randint(0, len(out)-21)
        chunk = out[start:start+20]
        chunk_rev = list(reversed(chunk))
        out[start:start+20] = [ln + " ###ANOM### type=order_violation" for ln in chunk_rev]
        injected += 20

    # 4) burst: duplicate some lines
    picks = random.sample(range(len(out)), min(n_each, len(out)))
    for i in picks:
        out.append(out[i] + " ###ANOM### type=burst")
        injected += 1

    print(f"[anomaly] injected ~{injected} anomalies")
    return out

# -------- Generation core --------
def generate_file(templates, pools, out_path: Path, count: int, seed: int,
                  monotonic: bool, jitter_ms: int, step_ms: int, level_dist):
    random.seed(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.utcnow()
    last_ts = None
    with out_path.open("w") as f:
        for i in range(count):
            tpl = random.choice(templates)
            when, last_ts = ts_next(start, i, step_ms, jitter_ms, monotonic, last_ts)
            lvl = choose_level(level_dist)
            f.write(make_line(tpl, pools, when, lvl) + "\n")
    print(f"[ok] wrote {count:>7} → {out_path}")

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Auto-generate train/val/test logs from templates")
    ap.add_argument("--templates-dir", default="data/templates", help="Dir with *.json template arrays")
    ap.add_argument("--params-dir", default="data/params", help="Dir with *.txt pools (ip.txt, user.txt, ...)")
    ap.add_argument("--out-dir", default="logs", help="Output directory")
    ap.add_argument("--train", type=int, default=100_000)
    ap.add_argument("--val", type=int, default=10_000)
    ap.add_argument("--test-clean", type=int, default=109_800)
    ap.add_argument("--with-anomalies", action="store_true", help="Also create test_mixed_110k.log (approx)")
    ap.add_argument("--anomaly-count", type=int, default=200)
    ap.add_argument("--jitter-ms", type=int, default=300)
    ap.add_argument("--step-ms", type=int, default=10)
    ap.add_argument("--level-dist", default="INFO:0.8,WARN:0.15,ERROR:0.05")
    # seeds
    ap.add_argument("--seed-train", type=int, default=101)
    ap.add_argument("--seed-val", type=int, default=202)
    ap.add_argument("--seed-test", type=int, default=303)
    args = ap.parse_args()

    tdir = Path(args.templates_dir)
    pdir = Path(args.params_dir)
    odir = Path(args.out_dir)

    templates = load_templates_from_dir(tdir)
    pools = load_param_pools_from_dir(pdir)
    level_dist = parse_level_dist(args.level_dist)

    # train (monotonic)
    generate_file(templates, pools, odir / "train_normal_100k.log",
                  args.train, args.seed_train, True, args.jitter_ms, args.step_ms, level_dist)

    # val (monotonic)
    generate_file(templates, pools, odir / "val_normal_10k.log",
                  args.val, args.seed_val, True, args.jitter_ms, args.step_ms, level_dist)

    # test-clean (non-monotonic for realism)
    test_clean_path = odir / "test_clean_109800.log"
    generate_file(templates, pools, test_clean_path,
                  args.test_clean, args.seed_test, False, args.jitter_ms, args.step_ms, level_dist)

    # optionally test-mixed with anomalies
    if args.with_anomalies:
        lines = test_clean_path.read_text().splitlines()
        mixed = inject_anomalies(lines, args.anomaly_count)
        out = odir / "test_mixed_110k.log"
        out.write_text("\n".join(mixed) + "\n")
        print(f"[ok] wrote      ~{len(mixed):>7} → {out}")

if __name__ == "__main__":
    main()
