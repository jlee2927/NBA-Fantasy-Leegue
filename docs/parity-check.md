# Three-way data parity check

Run before switching the projection input away from Basketball Monster.
Players with 500+ minutes in a season.

## 1. hoopR vs Kaggle, full population

8,411 player-seasons, 2002-2026, matched on name+birthdate.

| field | n | agree_pct | within_1pct | median_abs_diff | mean_rel_diff_pct |
|---|---|---|---|---|---|
| g | 8,411 | 85.0 | 85.0 | 0.0 | 0.4 |
| min | 8,411 | 0.2 | 7.9 | 31.5 | 2.2 |
| pts | 8,411 | 93.5 | 95.6 | 0.0 | 0.1 |
| tpm | 8,411 | 97.0 | 97.0 | 0.0 | 0.1 |
| reb | 8,411 | 92.0 | 95.4 | 0.0 | 0.1 |
| ast | 8,411 | 94.6 | 95.9 | 0.0 | 0.1 |
| stl | 8,411 | 96.0 | 96.1 | 0.0 | 0.1 |
| blk | 8,411 | 97.1 | 97.2 | 0.0 | 0.2 |
| fgm | 8,411 | 93.9 | 95.7 | 0.0 | 0.1 |
| fga | 8,411 | 91.6 | 95.4 | 0.0 | 0.1 |
| ftm | 8,411 | 95.9 | 96.4 | 0.0 | 0.1 |
| fta | 8,411 | 95.6 | 96.3 | 0.0 | 0.1 |

## 2. Basketball Monster vs hoopR

1,058 player-seasons, 2022-2026, top 234 only.

| field | n | agree_pct | within_1pct | median_abs_diff | mean_rel_diff_pct |
|---|---|---|---|---|---|
| g | 1,057 | 93.9 | 93.9 | 0.0 | 0.1 |
| min | 1,057 | 33.1 | 96.4 | 1.6 | 0.2 |
| pts | 1,057 | 90.4 | 97.5 | 0.0 | 0.1 |
| tpm | 1,057 | 92.6 | 98.4 | 0.0 | 0.0 |
| reb | 1,057 | 80.4 | 97.1 | 0.0 | 0.1 |
| ast | 1,057 | 86.2 | 97.8 | 0.0 | 0.1 |
| stl | 1,057 | 82.4 | 87.3 | 0.0 | 0.3 |
| blk | 1,057 | 87.6 | 93.0 | 0.0 | 0.3 |
| fgm | 1,057 | 66.0 | 97.6 | 0.0 | 0.0 |
| fga | 1,057 | 83.1 | 96.7 | 0.0 | 0.1 |
| ftm | 1,057 | 67.9 | 97.5 | 0.0 | 0.1 |
| fta | 1,057 | 89.3 | 97.3 | 0.0 | 0.1 |

## 3. Basketball Monster vs Kaggle

| field | n | agree_pct | within_1pct | median_abs_diff | mean_rel_diff_pct |
|---|---|---|---|---|---|
| g | 1,057 | 92.3 | 92.3 | 0.0 | 0.2 |
| min | 1,057 | 0.0 | 32.3 | 28.1 | 1.5 |
| pts | 1,057 | 88.6 | 95.9 | 0.0 | 0.1 |
| tpm | 1,057 | 91.6 | 96.8 | 0.0 | 0.1 |
| reb | 1,057 | 78.8 | 95.6 | 0.0 | 0.1 |
| ast | 1,057 | 85.0 | 95.6 | 0.0 | 0.1 |
| stl | 1,057 | 81.4 | 86.3 | 0.0 | 0.3 |
| blk | 1,057 | 86.7 | 92.0 | 0.0 | 0.3 |
| fgm | 1,057 | 64.9 | 95.9 | 0.0 | 0.1 |
| fga | 1,057 | 81.8 | 95.5 | 0.0 | 0.1 |
| ftm | 1,057 | 67.2 | 96.6 | 0.0 | 0.1 |
| fta | 1,057 | 88.0 | 96.5 | 0.0 | 0.1 |

## 4. Who is wrong when they disagree

Counts over the rows where all three sources have a figure. When two agree and one differs, the one standing alone carries the error.

| field | n | all_agree_pct | hoopR_odd | kaggle_odd | bm_odd | no_majority |
|---|---|---|---|---|---|---|
| g | 1,057 | 88.6 | 40 | 57 | 23 | 1 |
| pts | 1,057 | 90.4 | 28 | 47 | 22 | 4 |
| tpm | 1,057 | 94.2 | 19 | 34 | 6 | 2 |
| reb | 1,057 | 80.4 | 30 | 48 | 121 | 8 |
| ast | 1,057 | 88.6 | 25 | 41 | 48 | 7 |
| stl | 1,057 | 84.0 | 19 | 30 | 120 | 0 |
| blk | 1,057 | 90.0 | 15 | 25 | 65 | 1 |
| fgm | 1,057 | 91.4 | 27 | 47 | 13 | 4 |
| fga | 1,057 | 84.9 | 30 | 49 | 75 | 6 |
| ftm | 1,057 | 93.5 | 21 | 32 | 16 | 0 |
| fta | 1,057 | 93.1 | 21 | 33 | 19 | 0 |

Name matches rejected by the age check: 2.

