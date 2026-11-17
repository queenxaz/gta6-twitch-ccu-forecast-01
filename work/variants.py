#!/usr/bin/env python3
"""Variant testing harness: forecasters for gta6-twitch-ccu-forecast-01.

Each forecaster takes a case dict and returns {"hours": [...]}.
Scored locally against practice_answers.json (SullyGnome-sourced).
"""
import json, math, statistics
from pathlib import Path

KIT = Path.home() / "workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit"
WORK = Path(__file__).resolve().parent
ANSWERS = json.load(open(WORK / "practice_answers.json"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def pinball(y, q, tau):
    d = math.log(max(1.0, y)) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_forecast(case_id, out):
    rows = {r["hour"]: r["q"] for r in out["hours"]}
    ans = ANSWERS[case_id]
    tot, n, miss, cover, hrs = 0.0, 0, 0.0, 0, 0
    for h in range(2, 25):
        y = ans[str(h)]
        if y is None:
            continue
        q = rows[h]
        for tau, qq in zip(LEVELS, q):
            tot += pinball(y, qq, tau)
            n += 1
        miss += abs(math.log(max(1.0, y)) - math.log(q[2]))
        cover += 1 if q[0] <= y <= q[4] else 0
        hrs += 1
    return tot / n, 100.0 * (math.exp(miss / hrs) - 1.0), cover / hrs


def score_all(forecaster, verbose=True):
    losses = []
    for cid in ["p00", "p01", "p02", "p03"]:
        case = json.load(open(KIT / "input" / f"{cid}.in"))
        out = forecaster(case)
        l, tm, cov = score_forecast(cid, out)
        losses.append(l)
        if verbose:
            print(f"  {cid}: loss={l:.6f} miss={tm:.1f}% cov90={cov:.3f}")
    mean = sum(losses) / len(losses)
    print(f"  MEAN loss={mean:.6f}")
    return mean, losses


# ---------- shared helpers ----------

def quantiles(vals, levels):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return [0.0 for _ in levels]
    if n == 1:
        return [s[0] for _ in levels]
    out = []
    for tau in levels:
        pos = tau * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        f = pos - lo
        out.append(s[lo] * (1 - f) + s[hi] * f)
    return out


def comp_day_curves(case):
    """Return list of (name, {hour: viewers}) for comparables with a complete
    release day (hours 1..24 present), excluding the GTA VI category itself."""
    from datetime import datetime
    out = []
    for c in case["comparables"]:
        if "Grand Theft Auto VI" in c["name"]:
            continue
        day = c["day_start"][:10]
        m = {}
        for row in c["hourly"]:
            if row["from"][:10] == day:
                m[int(row["from"][11:13]) + 1] = row["viewers"]
        if all(h in m for h in range(1, 25)) and m[1] > 0:
            out.append((c["name"], m))
    return out


def make_rows(p1, med_logr, spread_logr):
    """med_logr[h], spread_logr[h]: median log ratio and per-level offsets.
    spread_logr[h] is a list of 5 log offsets (one per level)."""
    hours = []
    for h in range(2, 25):
        qs = []
        for k in range(5):
            v = p1 * math.exp(med_logr[h] + spread_logr[h][k])
            qs.append(max(1.0, min(1e9, v)))
        qs.sort()
        hours.append({"hour": h, "q": [round(v, 2) for v in qs]})
    return {"hours": hours}


# ---------- variants ----------

def v0_starter(case):
    p1 = max(1, case["known"][0]["viewers"])
    med = {h: 0.0 for h in range(2, 25)}
    spr = {h: [z * 0.15 * math.sqrt(h - 1) for z in Z] for h in range(2, 25)}
    return make_rows(p1, med, spr)


def v1_pooled_empirical(case):
    """Per-hour empirical quantiles of log ratio across ALL comparables."""
    p1 = max(1, case["known"][0]["viewers"])
    curves = comp_day_curves(case)
    med, spr = {}, {}
    for h in range(2, 25):
        lr = [math.log(m[h] / m[1]) for _, m in curves]
        qs = quantiles(lr, LEVELS)
        med[h] = qs[2]
        spr[h] = [q - qs[2] for q in qs]
    return make_rows(p1, med, spr)


def v2_filtered_empirical(case, p1_min=10000):
    """Per-hour empirical quantiles over comparables already underway at
    hour 1 (p1 >= p1_min)."""
    p1 = max(1, case["known"][0]["viewers"])
    curves = [(n, m) for n, m in comp_day_curves(case) if m[1] >= p1_min]
    med, spr = {}, {}
    for h in range(2, 25):
        lr = [math.log(m[h] / m[1]) for _, m in curves]
        qs = quantiles(lr, LEVELS)
        med[h] = qs[2]
        spr[h] = [q - qs[2] for q in qs]
    return make_rows(p1, med, spr)


def _ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def v3_regression_filtered(case, p1_min=10000, spread_mult=1.0):
    """Per-hour OLS: log r_h ~ a_h + b_h * log(p1), fit on filtered comps.
    Median = fitted value at the case's log p1; fan = empirical quantiles of
    the per-hour residuals."""
    p1 = max(1, case["known"][0]["viewers"])
    lp = math.log(p1)
    curves = [(n, m) for n, m in comp_day_curves(case) if m[1] >= p1_min]
    med, spr = {}, {}
    for h in range(2, 25):
        xs = [math.log(m[1]) for _, m in curves]
        ys = [math.log(m[h] / m[1]) for _, m in curves]
        a, b = _ols(xs, ys)
        med[h] = a + b * lp
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        qs = quantiles(resid, LEVELS)
        spr[h] = [spread_mult * (q - qs[2]) for q in qs]
    return make_rows(p1, med, spr)


def v4_regression_all(case, spread_mult=1.0):
    """Same OLS but fit on ALL comparables (log p1 carries the family)."""
    p1 = max(1, case["known"][0]["viewers"])
    lp = math.log(p1)
    curves = comp_day_curves(case)
    med, spr = {}, {}
    for h in range(2, 25):
        xs = [math.log(m[1]) for _, m in curves]
        ys = [math.log(m[h] / m[1]) for _, m in curves]
        a, b = _ols(xs, ys)
        med[h] = a + b * lp
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        qs = quantiles(resid, LEVELS)
        spr[h] = [spread_mult * (q - qs[2]) for q in qs]
    return make_rows(p1, med, spr)


def v5_regression_normal_spread(case, p1_min=10000, spread_mult=1.0):
    """OLS median on filtered comps; fan = normal z * residual SD."""
    p1 = max(1, case["known"][0]["viewers"])
    lp = math.log(p1)
    curves = [(n, m) for n, m in comp_day_curves(case) if m[1] >= p1_min]
    med, spr = {}, {}
    for h in range(2, 25):
        xs = [math.log(m[1]) for _, m in curves]
        ys = [math.log(m[h] / m[1]) for _, m in curves]
        a, b = _ols(xs, ys)
        med[h] = a + b * lp
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sd = statistics.pstdev(resid) if len(resid) > 1 else 0.3
        spr[h] = [spread_mult * z * sd for z in Z]
    return make_rows(p1, med, spr)


if __name__ == "__main__":
    import sys
    variants = {
        "V0 starter": v0_starter,
        "V1 pooled empirical": v1_pooled_empirical,
        "V2 filtered empirical p1>=10k": v2_filtered_empirical,
        "V3 regression filtered": v3_regression_filtered,
        "V4 regression all": v4_regression_all,
        "V5 regression normal spread": v5_regression_normal_spread,
    }
    only = sys.argv[1:] or list(variants)
    for name in only:
        print(name)
        score_all(variants[name])
