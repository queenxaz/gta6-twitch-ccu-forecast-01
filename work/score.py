#!/usr/bin/env python3
"""Local scorer for gta6-twitch-ccu-forecast-01 practice cases.

Runs a candidate main.py on the 4 practice cases and scores it with the
official pinball loss on the log scale, using practice answers reconstructed
from SullyGnome's public hourly averages (the same source the kit's cases
were cut from). Hour-1 prints verified to match the case files exactly.

Usage:
  python3 score.py <repo_dir> [--cases p00 p01] [--keep-out DIR]
Prints per-case loss, mean loss, typical miss %, coverage90.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

KIT = Path.home() / "workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit"
WORK = Path(__file__).resolve().parent
ANSWERS = json.load(open(WORK / "practice_answers.json"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def pinball(y, q, tau):
    d = math.log(max(1.0, y)) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_case(case_id, out_path):
    out = json.load(open(out_path))
    rows = {r["hour"]: r["q"] for r in out["hours"]}
    ans = ANSWERS[case_id]
    total = 0.0
    n = 0
    miss = 0.0
    cover = 0
    hours_scored = 0
    for h in range(2, 25):
        y = ans[str(h)]
        if y is None:
            continue
        q = rows[h]
        for tau, qq in zip(LEVELS, q):
            total += pinball(y, qq, tau)
            n += 1
        miss += abs(math.log(max(1.0, y)) - math.log(q[2]))
        if q[0] <= y <= q[4]:
            cover += 1
        hours_scored += 1
    return {
        "loss": total / n if n else float("nan"),
        "typical_miss_pct": 100.0 * (math.exp(miss / hours_scored) - 1.0),
        "coverage90": cover / hours_scored if hours_scored else 0.0,
        "hours": hours_scored,
    }


def main():
    repo = Path(sys.argv[1]).resolve()
    cases = sys.argv[2:] if len(sys.argv) > 2 else ["p00", "p01", "p02", "p03"]
    main_py = repo / "main.py"
    assert main_py.exists(), f"no main.py in {repo}"
    results = {}
    with tempfile.TemporaryDirectory() as td:
        in_dir = Path(td) / "in"
        out_dir = Path(td) / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        for cid in cases:
            (in_dir / f"{cid}.in").write_bytes(
                (KIT / "input" / f"{cid}.in").read_bytes()
            )
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(in_dir), MAKEFUN_OUTPUT_DIR=str(out_dir))
        t0 = __import__("time").time()
        r = subprocess.run(
            [sys.executable, str(main_py)], env=env,
            capture_output=True, text=True, timeout=120,
        )
        elapsed = __import__("time").time() - t0
        if r.returncode != 0:
            print("RUN FAILED rc=", r.returncode)
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            sys.exit(1)
        for cid in cases:
            op = out_dir / f"{cid}.out"
            if not op.exists():
                print(cid, "MISSING OUTPUT")
                results[cid] = None
                continue
            results[cid] = score_case(cid, op)
    losses = [results[c]["loss"] for c in cases if results[c]]
    print(f"runtime: {elapsed:.2f}s for {len(cases)} cases")
    for cid in cases:
        s = results[cid]
        if s:
            print(f"{cid}: loss={s['loss']:.6f} typical_miss={s['typical_miss_pct']:.1f}% coverage90={s['coverage90']:.3f} hours={s['hours']}")
    if losses:
        print(f"MEAN loss={sum(losses)/len(losses):.6f}")
    return results


if __name__ == "__main__":
    main()
