[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/queenxaz/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/queenxaz/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast — gta6-twitch-ccu-forecast-01

Forecasts Grand Theft Auto VI's Twitch viewers hour by hour over its
release day (2026-11-19 UTC), from the first hour's print, as five
quantiles per hour (0.05 / 0.25 / 0.5 / 0.75 / 0.95). Python 3.13,
standard library only, deterministic, no network.

## Method — comparable-launch ratio regression

Everything is fit at run time from the case's own `comparables`, so the
final case's 42 launches (practice launches included) are all used.

1. **Reference class.** A past launch is usable only if its release day
   is fully observed (hours 1–24 all present, viewers > 0). Two filters
   keep the launches that behave like GTA VI — a console game with a
   rolling midnight-local unlock and no early access:
   - hour-1 viewers ≥ 10,000. Launches still locked at 00:00 UTC (a
     single global unlock later in the day, e.g. Silksong at 14:00Z)
     start near zero and then jump 100–2000×; their ratio curves are a
     different regime and poison a pooled fit.
   - no early access. Launches whose `unlock` text says early access
     happened (Hogwarts, Starfield, Diablo IV, …) are flat on release
     day because the audience already played; GTA VI has none. (A text
     saying "no early access" counts as clean.)
2. **Median.** For each hour h = 2..24, regress the log ratio on the log
   hour-1 print across the filtered launches:
   `log(y_h / y_1) ~ a_h + b_h · log(y_1)`, evaluated at this case's log
   hour-1 print. The slope is strongly negative (r = −0.94 for the
   day's peak across the 38 practice comparables): a launch already huge
   at hour 1 stays flat, a small hour 1 grows into the evening as time
   zones unlock. The hour-specific intercepts carry the shape every
   launch shares: a dip in hours 2–4, a step at hours 5–6 (US East
   midnight unlock), a midday lull, and the evening prime-time peak at
   hours 17–21.
3. **Spread.** Normal quantiles (−1.645, −0.674, 0, 0.674, 1.645) times
   the per-hour residual standard deviation of that regression, × 0.9.
   The residual SD grows with horizon, so the fan widens over the day
   the way the data says it should (spread from realised dispersion,
   not a fixed 0.15·√h).

If fewer than 5 launches survive the filters, the early-access filter is
dropped, then the hour-1 filter; if the fit still fails, the program
falls back to the house persistence forecast — an invalid case forfeits
any prize, so the program must never crash.

## Results (practice, local re-score)

Practice answers reconstructed from SullyGnome's hourly averages, the
same source the kit's cases were cut from (hour-1 prints verified to
match the case files exactly). Mean pinball loss on the log scale:

| variant | p00 | p01 | p02 | p03 | mean |
|---|---|---|---|---|---|
| house persistence baseline | 0.744 | 0.287 | 0.215 | 0.303 | **0.387** |
| pooled empirical quantiles (all comps) | | | | | 0.324 |
| filtered empirical quantiles (p1 ≥ 10k) | | | | | 0.291 |
| ratio regression, filtered (empirical fan) | | | | | 0.225 |
| ratio regression, filtered (normal fan) | | | | | 0.213 |
| ratio regression, filtered + no-early-access | 0.224 | 0.167 | 0.137 | 0.181 | **0.177** |

Leave-one-out over the 18 clean, already-underway comparable launches
agrees the final variant is best (0.275 vs 0.284 for the unfiltered
regression and 0.338 for a regression on all comps).

## Running

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Reads every `*.in` in the input directory and writes one `<id>.out` per
case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, …]}`.
Runtime ~0.04 s per case (cap 8 s).
