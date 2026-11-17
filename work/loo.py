#!/usr/bin/env python3
"""Leave-one-out validation on comparables: treat each comparable launch as
a pseudo-case (its own release day, p1 known, hours 2..24 as answer) and
score forecasters fit on the OTHER comparables. Guards against overfitting
to the 4 practice cases.
"""
import json, math, statistics
from pathlib import Path
from variants import (KIT, ANSWERS, LEVELS, Z, quantiles, comp_day_curves,
                      make_rows, score_forecast, _ols)

CASES = ["p00", "p01", "p02", "p03"]


def loo_score(forecaster_fn, p1_min=10000, verbose=False):
    """forecaster_fn(curves_train, p1) -> output dict. curves_train is a list
    of (name, {hour: viewers}). Scores on every complete comparable curve
    in p00's case (same 38 comps in all practice cases)."""
    case = json.load(open(KIT / "input/p00.in"))
    all_curves = comp_day_curves(case)
    losses = []
    per = []
    for name, m in all_curves:
        train = [(n, mm) for n, mm in all_curves if n != name]
        p1 = m[1]
        out = forecaster_fn(train, p1)
        # score vs this comp's actual hours 2..24
        rows = {r["hour"]: r["q"] for r in out["hours"]}
        tot, n = 0.0, 0
        for h in range(2, 25):
            y = m[h]
            q = rows[h]
            for tau, qq in zip(LEVELS, q):
                d = math.log(max(1.0, y)) - math.log(qq)
                tot += 2.0 * max(tau * d, (tau - 1.0) * d)
                n += 1
        losses.append(tot / n)
        per.append((name, tot / n))
    mean = sum(losses) / len(losses)
    if verbose:
        for name, l in sorted(per, key=lambda t: -t[1])[:8]:
            print(f"    worst {name[:30]:30s} {l:.4f}")
    return mean, per


def mk_v5(spread_mult=1.0, p1_min=10000):
    def f(train, p1):
        lp = math.log(max(1, p1))
        curves = [(n, m) for n, m in train if m[1] >= p1_min]
        if len(curves) < 5:
            curves = train
        med, spr = {}, {}
        for h in range(2, 25):
            xs = [math.log(m[1]) for _, m in curves]
            ys = [math.log(m[h] / m[1]) for _, m in curves]
            a, b = _ols(xs, ys)
            med[h] = a + b * lp
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
            sd = statistics.pstdev(resid) if len(resid) > 1 else 0.3
            spr[h] = [spread_mult * z * sd for z in Z]
        return make_rows(max(1, p1), med, spr)
    return f


def mk_v5_all(spread_mult=1.0):
    def f(train, p1):
        lp = math.log(max(1, p1))
        med, spr = {}, {}
        for h in range(2, 25):
            xs = [math.log(m[1]) for _, m in train]
            ys = [math.log(m[h] / m[1]) for _, m in train]
            a, b = _ols(xs, ys)
            med[h] = a + b * lp
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
            sd = statistics.pstdev(resid) if len(resid) > 1 else 0.3
            spr[h] = [spread_mult * z * sd for z in Z]
        return make_rows(max(1, p1), med, spr)
    return f


def mk_v3(spread_mult=1.0, p1_min=10000):
    def f(train, p1):
        lp = math.log(max(1, p1))
        curves = [(n, m) for n, m in train if m[1] >= p1_min]
        if len(curves) < 5:
            curves = train
        med, spr = {}, {}
        for h in range(2, 25):
            xs = [math.log(m[1]) for _, m in curves]
            ys = [math.log(m[h] / m[1]) for _, m in curves]
            a, b = _ols(xs, ys)
            med[h] = a + b * lp
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
            qs = quantiles(resid, LEVELS)
            spr[h] = [spread_mult * (q - qs[2]) for q in qs]
        return make_rows(max(1, p1), med, spr)
    return f


def mk_persistence(spread=0.15):
    def f(train, p1):
        med = {h: 0.0 for h in range(2, 25)}
        spr = {h: [z * spread * math.sqrt(h - 1) for z in Z] for h in range(2, 25)}
        return make_rows(max(1, p1), med, spr)
    return f


def mk_knn(k=8, spread_mult=1.0):
    """Median curve + empirical fan from the k nearest comps by log p1."""
    def f(train, p1):
        lp = math.log(max(1, p1))
        near = sorted(train, key=lambda nm: abs(math.log(nm[1][1]) - lp))[:k]
        med, spr = {}, {}
        for h in range(2, 25):
            lr = [math.log(m[h] / m[1]) for _, m in near]
            qs = quantiles(lr, LEVELS)
            med[h] = qs[2]
            spr[h] = [spread_mult * (q - qs[2]) for q in qs]
        return make_rows(max(1, p1), med, spr)
    return f


if __name__ == "__main__":
    print("LOO on 37 comparable launches (lower = better):")
    for name, fn in [
        ("persistence 0.15*sqrt", mk_persistence(0.15)),
        ("V3 reg filtered emp-fan x1.0", mk_v3(1.0)),
        ("V3 reg filtered emp-fan x1.25", mk_v3(1.25)),
        ("V5 reg filtered norm x0.8", mk_v5(0.8)),
        ("V5 reg filtered norm x1.0", mk_v5(1.0)),
        ("V5 reg filtered norm x1.25", mk_v5(1.25)),
        ("V5 reg filtered norm x1.5", mk_v5(1.5)),
        ("V5 reg ALL norm x1.0", mk_v5_all(1.0)),
        ("V5 reg ALL norm x1.25", mk_v5_all(1.25)),
        ("kNN k=8 emp-fan x1.0", mk_knn(8, 1.0)),
        ("kNN k=8 emp-fan x1.25", mk_knn(8, 1.25)),
        ("kNN k=12 emp-fan x1.0", mk_knn(12, 1.0)),
    ]:
        mean, per = loo_score(fn)
        print(f"  {name:32s} LOO mean={mean:.6f}")
