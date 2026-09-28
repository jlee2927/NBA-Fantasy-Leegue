# Walk-forward backtest

Error is RMSE per category relative to that category's own mean, averaged over categories. Lower is better.

- `walk_forward`: parameters fitted only on seasons before the target
- `tuned_on_everything`: fitted on all seasons including the target, which leaks and so flatters itself
- `last_season_only`: no projection at all, just carry last season forward

## Headline

- Seasons scored: 17 (tuning) + 1 (holdout)
- Walk-forward mean error: **0.2023**
- Tuned-on-everything mean: 0.2023 (optimistic by -0.0%)
- Carry last season forward: 0.2248
- Improvement over carrying last season: **10.0%**

## Holdout 2026

Never used to tune anything, at any stage.

| target | walk_forward | tuned_on_everything | last_season_only | aging | min_base | game_base |
|---|---|---|---|---|---|---|
| 2026 | 0.2121 | 0.2120 | 0.2331 | delta | 500 | 0.2250 |

## Every scored season

| target | walk_forward | tuned_on_everything | last_season_only | aging | min_base | game_base | tuning_seasons |
|---|---|---|---|---|---|---|---|
| 2009 | 0.2028 | 0.2035 | 0.2166 | delta | 600 | 0.2750 | 4 |
| 2010 | 0.1979 | 0.1975 | 0.2238 | delta | 600 | 0.2750 | 5 |
| 2011 | 0.1987 | 0.1989 | 0.2160 | delta | 600 | 0.2750 | 6 |
| 2012 | 0.2085 | 0.2094 | 0.2227 | delta | 600 | 0.2750 | 7 |
| 2013 | 0.1995 | 0.1997 | 0.2243 | delta | 500 | 0.2500 | 8 |
| 2014 | 0.2066 | 0.2063 | 0.2358 | delta | 500 | 0.2500 | 9 |
| 2015 | 0.2025 | 0.2031 | 0.2196 | delta | 500 | 0.2500 | 10 |
| 2016 | 0.2047 | 0.2053 | 0.2323 | delta | 500 | 0.2500 | 11 |
| 2017 | 0.2114 | 0.2105 | 0.2368 | delta | 500 | 0.2500 | 12 |
| 2018 | 0.2059 | 0.2056 | 0.2312 | delta | 600 | 0.2750 | 13 |
| 2019 | 0.2057 | 0.2054 | 0.2273 | delta | 500 | 0.2750 | 14 |
| 2020 | 0.2066 | 0.2061 | 0.2254 | delta | 500 | 0.2750 | 15 |
| 2021 | 0.2109 | 0.2106 | 0.2411 | delta | 500 | 0.2500 | 16 |
| 2022 | 0.1964 | 0.1964 | 0.2162 | delta | 500 | 0.2250 | 17 |
| 2023 | 0.2011 | 0.2011 | 0.2264 | delta | 500 | 0.2250 | 18 |
| 2024 | 0.1927 | 0.1928 | 0.2134 | delta | 500 | 0.2250 | 19 |
| 2025 | 0.1877 | 0.1877 | 0.2122 | delta | 400 | 0.2250 | 20 |
| 2026 | 0.2121 | 0.2120 | 0.2331 | delta | 500 | 0.2250 | 21 |

## Structural parameter stability

k refitted on targets through 2015 versus 2016 onward. Large moves would mean the long window is mixing eras and the structural/regime split is drawn wrong.

| parameter | early | late | relative_change | flag |
|---|---|---|---|---|
| k[pts] | 1500 | 1000 | 0.3333 |  |
| k[tpm] | 500 | 500 | 0.0000 |  |
| k[reb] | 500 | 500 | 0.0000 |  |
| k[ast] | 500 | 500 | 0.0000 |  |
| k[stl] | 2500 | 2500 | 0.0000 |  |
| k[blk] | 500 | 1000 | 0.5000 |  |
| k[fgm] | 1500 | 1000 | 0.3333 |  |
| k[fga] | 1000 | 1000 | 0.0000 |  |
| k[ftm] | 1500 | 1000 | 0.3333 |  |
| k[fta] | 1500 | 1000 | 0.3333 |  |
| k[tov] | 2500 | 1500 | 0.4000 |  |

