# GTA 6 Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/luckylawless1/gta6-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/luckylawless1/gta6-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecasting program for make.fun's [Forecast GTA 6 trophy rarity](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty. For each of GTA VI's 55 PlayStation 5 trophies it predicts the earn rate the trophy will show on Exophase at the target time, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95).

## How it works

The printed rate is modelled as what it physically is: an integer count of earners divided by an integer count of tracked players, refreshed in steps. The forecaster therefore works in counts, not percentages:

- It reconstructs earners and players from every read in the case history and detects the page's refresh states.
- The player pool is projected to the target with saturating growth curves fitted to the case's own refresh points; among near-tied fits the median projection is used, and the near-tie spread is carried into the quantile width — the pool at a week's horizon is genuinely uncertain, and a fraction inherits that uncertainty directly.
- Each trophy's earners follow an ensemble of five: a two-state accrual model (new players arrive, a pool of eventual earners converts over time), an earners flow regression (earners grow with elapsed days and with pool growth, fitted per trophy), a log-time trend that is allowed to fall, a power law toward a ceiling or a floor, and persistence. The weights come from a lead-weighted backtest on the case's own history blended with a structural prior. The platinum is capped just below the rarest other trophy.
- Quantile spread combines the trophy's own step volatility, the backtest error, the candidates' disagreement, the pool spread and small-count noise. Every quantile is snapped to a whole-earner count at the projected pool, so the forecast is count-consistent: the median is a rate the page could actually print.
- If a case's page never moved (a frozen page, as in the practice cases), the forecast is built from the game's measured first day instead of priors: the pool follows the measured path (638 players before release, 3,868 at +4.5 h, at least 13,334 at +27.7 h) and each trophy's earners grow at their measured day-1 rate, blended half time-driven, half pool-flow-driven. On the measured evidence the typical trophy gains almost no earners while the pool multiplies — dilution, not accrual, is the dominant dynamic of week one.

The program is deterministic: the same case file always produces byte-identical output, and a defensive fallback yields a persistence forecast rather than a crash on any malformed case.

## Run it

```
MAKEFUN_INPUT_DIR=<dir with *.in case files> MAKEFUN_OUTPUT_DIR=<output dir> node main.ts
```

Node 22, standard library only, no network. One `<id>.out` file is written per case.
