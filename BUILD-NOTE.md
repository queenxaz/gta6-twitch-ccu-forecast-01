# BUILD NOTE — gta6-twitch-ccu-forecast-01

## Winning variant
Comparable-launch ratio regression (see main.py docstring / README.md):
per hour h = 2..24, OLS `log(y_h / y_1) ~ a_h + b_h · log(y_1)` over
filtered comparables — complete release day, hour-1 viewers ≥ 10,000,
no early access (unlock text) — evaluated at the case's log hour-1
print; fan = normal quantiles × per-hour residual SD × 0.9. Fallback to
house persistence if the fit fails. Fitted at run time from the case's
own comparables, so the final's 42 launches (practice launches
included) are all used.

## Practice losses (local re-score)
Practice answers reconstructed from SullyGnome hourly averages (same
source the kit's cases were cut from; hour-1 prints verified to match
the case files exactly). Pinball loss on the log scale, per case:

- p00 Marvel's Wolverine:       0.224041 (typical miss 43.7%, cov90 0.913)
- p01 Red Dead Redemption 2:    0.166751 (typical miss 31.2%, cov90 0.957)
- p02 God of War Ragnarök:      0.137451 (typical miss 21.3%, cov90 1.000)
- p03 Marvel's Spider-Man 2:    0.180713 (typical miss 28.2%, cov90 0.783)
- MEAN:                         0.177239

House persistence baseline (starter, verified locally): 0.387097
(p00 0.743671, p01 0.286813, p02 0.214579, p03 0.303324).

**Beats the house baseline 0.387 by 0.210 (54% lower).**

## Variants tried (practice mean loss)
- V0 starter / house persistence: 0.387097
- V1 pooled empirical quantiles, all comps: 0.323970
- V2 filtered empirical quantiles (p1 >= 10k): 0.290542
- V3 regression, filtered, empirical residual fan: 0.225151
- V4 regression, all comps, normal fan: 0.227267
- V5 regression, filtered, normal fan x1.0: 0.213482
- V5 + early-access filter, normal fan x0.9 (WINNER): 0.177239
- V5 + EA filter, fan x1.0: 0.181575; fan x0.85: 0.175555
- V5 + EA filter + blend 30% all-comps median: 0.177509
- kNN k=8 (nearest by log p1): 0.234903
- Median smoothing / pooled slope / slope shrink: no improvement
  (0.180 / 0.228 / 0.178)

Leave-one-out over the 18 clean, already-underway comparable launches:
winner 0.275, unfiltered regression 0.284, all-comps regression 0.338 —
the winner is best on LOO too, so the choice is not a 4-case overfit.
Spread 0.9 chosen as the middle of {0.85, 0.9, 1.0}, all within noise
on LOO (0.2755 / 0.2748 / 0.2742).

## Verification
- `python3 score.py <repo>` (work/score.py): mean 0.177239, runtime
  0.18 s for 4 cases (cap 8 s per case).
- Format: 23 rows, hours 2..24, 5 quantiles each, finite, > 0,
  ≤ 1e9, non-decreasing, ≤ 65,536 bytes — all pass.
- Determinism: two runs byte-identical.
- Edge cases pass: no comparables (persistence fallback), tiny p1,
  missing unlock text — all produce valid output, no crash.
- grade-local does NOT work for this competition (kit: "scorer and
  reference data are not published in the kit, so it is graded on the
  server only"); scoring above is a local re-implementation validated
  by reproducing the house baseline's stated 0.387 exactly (0.387097).

## Key finding
log(hour-1 print) predicts the day's peak log-ratio with r = -0.94
across comparables: launches already huge at hour 1 stay flat, small
hour-1 launches grow into the evening as time zones unlock. The two
regimes (rolling midnight unlock vs single global unlock later in the
day) must not be pooled — the single-unlock launches jump 100–2000x
and wreck any pooled quantile or regression.

## Files
- main.py — the entry (stdlib only)
- README.md — method + results
- work/score.py, work/variants.py, work/loo.py, work/practice_answers.json
  — local scoring harness (not part of the entry)
