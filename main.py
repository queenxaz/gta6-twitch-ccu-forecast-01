#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun
gta6-twitch-ccu-forecast-01). Python 3.13, standard library only,
deterministic, no network.

Method -- comparable-launch ratio regression (fit at run time from the
case's own `comparables`, so the final case's 42 launches, practice
launches included, are all used):

  * Reference class. A past launch is usable only if its release day is
    fully observed (hours 1..24 all present with viewers > 0). Two filters
    then keep the launches that behave like GTA VI, a console game with a
    rolling midnight-local unlock and no early access:
      - hour-1 viewers >= 10,000: launches still locked at 00:00 UTC (a
        single global unlock later in the day, e.g. Silksong at 14:00Z)
        start near zero and then jump 100-2000x; their ratio curves are a
        different regime and poison a pooled fit.
      - no early access: launches whose `unlock` text says early access
        happened (Hogwarts, Starfield, Diablo IV, ...) are flat on release
        day because the audience already played; GTA VI has none. (A text
        saying "no early access" counts as clean.)
  * Median. For each hour h = 2..24, regress the log ratio on the log
    hour-1 print across the filtered launches:
        log(y_h / y_1) ~ a_h + b_h * log(y_1)
    and evaluate at this case's log hour-1 print. The slope is strongly
    negative (r = -0.94 for the day's peak across the 38 practice
    comparables): a launch already huge at hour 1 stays flat, a small
    hour 1 grows into the evening as time zones unlock. The hour-specific
    intercepts carry the shape every launch shares: a dip in hours 2-4,
    a step at hours 5-6 (US East midnight unlock), a midday lull, and the
    evening prime-time peak at hours 17-21.
  * Spread. The fan is normal quantiles z = (-1.645, -0.674, 0, 0.674,
    1.645) times the per-hour residual standard deviation of that
    regression, times 0.9. The residual SD grows with horizon, so the fan
    widens over the day the way the data says it should (local-volatility
    style: spread from realised dispersion, not a fixed 0.15*sqrt(h)).

Practice (local re-score against SullyGnome's hourly averages, the same
source the kit's cases were cut from; hour-1 prints verified to match):
  house persistence baseline 0.387; this entry 0.177 (p00 0.224, p01
  0.167, p02 0.137, p03 0.181 -- per-case figures in BUILD-NOTE.md).
  Leave-one-out over the 18 clean, already-underway comparable launches
  agrees this variant is the best of those tried (0.275 vs 0.284 for the
  unfiltered regression).

Fallbacks (an invalid case scores as persistence and forfeits any prize,
so the program must never crash): if fewer than 5 launches survive the
filters, drop the early-access filter, then the hour-1 filter; if the
fit still fails, fall back to the house persistence forecast.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = range(2, 25)

# A launch counts as "already underway at hour 1" only above this print.
P1_MIN = 10000
# Fan width multiplier on the residual standard deviation (1.0 is the
# honest estimate; 0.9 is the value both practice and leave-one-out
# validation support -- see module docstring).
SPREAD_MULT = 0.9
# Minimum filtered launches for the regression; below this, filters relax.
MIN_TRAIN = 5
# House persistence spread, used only by the fallback.
FALLBACK_SIGMA = 0.15


def round2(x):
    """x rounded to 2 decimals, half up (as the starter rounds it)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def has_early_access(comp):
    """True when the unlock text says early access actually happened.
    A text saying "no early access" counts as clean."""
    text = (comp.get("unlock") or "").lower()
    return "early access" in text and "no early access" not in text


def release_day_curve(comp):
    """{hour: viewers} for the comparable's release day, or None when any
    of hours 1..24 is missing or not positive."""
    day_start = comp.get("day_start") or ""
    day = day_start[:10]
    if not day:
        return None
    curve = {}
    for row in comp.get("hourly") or []:
        stamp = row.get("from") or ""
        if stamp[:10] != day:
            continue
        viewers = row.get("viewers")
        if viewers is None:
            continue
        hour = int(stamp[11:13]) + 1
        if 1 <= hour <= 24:
            curve[hour] = viewers
    if len(curve) != 24 or any(v is None or v <= 0 for v in curve.values()):
        return None
    return curve


def training_curves(case):
    """Filtered (name, curve) pairs for the regression, with fallbacks."""
    all_curves = []
    for comp in case.get("comparables") or []:
        if "Grand Theft Auto VI" in (comp.get("name") or ""):
            continue  # the category's own pre-launch history, not a launch
        curve = release_day_curve(comp)
        if curve is not None:
            all_curves.append((comp.get("name") or "", curve, comp))
    underway = [(n, c) for n, c, _ in all_curves if c[1] >= P1_MIN]
    clean = [(n, c) for n, c, comp in all_curves
             if c[1] >= P1_MIN and not has_early_access(comp)]
    if len(clean) >= MIN_TRAIN:
        return clean
    if len(underway) >= MIN_TRAIN:
        return underway
    return [(n, c) for n, c, _ in all_curves]


def ols(xs, ys):
    """Least-squares intercept and slope of ys on xs."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def pstdev(values):
    n = len(values)
    if n < 2:
        return 0.3
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    curves = training_curves(case)
    rows = []
    if len(curves) >= 2:
        log_p1 = math.log(p1)
        for h in HOURS:
            xs = [math.log(c[1]) for _, c in curves]
            ys = [math.log(c[h] / c[1]) for _, c in curves]
            a, b = ols(xs, ys)
            median_logr = a + b * log_p1
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
            sd = pstdev(resid)
            qs = []
            for z in Z:
                v = p1 * math.exp(median_logr + SPREAD_MULT * z * sd)
                qs.append(max(1.0, min(1e9, v)))
            qs.sort()
            rows.append({"hour": h, "q": [round2(v) for v in qs]})
    if len(rows) != 23:
        # Fallback: house persistence (never leave a case invalid).
        rows = []
        for h in HOURS:
            qs = []
            for z in Z:
                v = p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))
                qs.append(max(1.0, min(1e9, v)))
            qs.sort()
            rows.append({"hour": h, "q": [round2(v) for v in qs]})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
