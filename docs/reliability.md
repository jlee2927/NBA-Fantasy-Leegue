# Category reliability, and why it is not applied twice

A z-score treats every category as equally knowable. They are not. The
obvious correction is to shrink each category's z-score by how reliably it
is projected, so a number that is mostly noise counts for less.

Measured, tested, and **not shipped**, because it made the rankings worse in
every season tried.

## How reliable each category is

Correlation between what the engine projected and what happened, weighted by
playing time, averaged over 2010-2025.

| category | reliability |
|---|---|
| fg% | 0.799 |
| stl | 0.799 |
| ft% | 0.829 |
| tov | 0.833 |
| pts | 0.865 |
| tpm | 0.885 |
| blk | 0.902 |
| ast | 0.914 |
| reb | 0.939 |

Blocks are the trap. They have the **worst** error relative to their own mean
of any category, and one of the **best** reliabilities. Both are true: the
average is low so a small absolute miss looks enormous in relative terms,
while the gap between a rim protector and a guard is so wide that even a poor
projection sorts them correctly. Steals are the reverse - almost everyone
lands between half a steal and one and a half, so projection noise swamps the
real differences.

A draft board ranks players, so ranking is what was measured. Error relative
to the mean answers a different question and would have had this backwards.

## The test

For each season from 2016 to 2025: measure reliability on strictly earlier
seasons, project the target, then compare two predictions of what a player's
value turned out to be. Realised value is the plain average of his actual
z-scores, because a real league counts all nine categories equally.

| target | plain | reliability-weighted | delta |
|---|---|---|---|
| 2016 | 0.7932 | 0.7889 | -0.0043 |
| 2017 | 0.8110 | 0.8066 | -0.0044 |
| 2018 | 0.8193 | 0.8169 | -0.0024 |
| 2019 | 0.8182 | 0.8156 | -0.0025 |
| 2020 | 0.7842 | 0.7812 | -0.0030 |
| 2021 | 0.7789 | 0.7756 | -0.0033 |
| 2022 | 0.8034 | 0.8004 | -0.0029 |
| 2023 | 0.8183 | 0.8170 | -0.0013 |
| 2024 | 0.8255 | 0.8234 | -0.0022 |
| 2025 | 0.8459 | 0.8444 | -0.0015 |

Mean 0.8098 plain against 0.8070 weighted. Worse in 10 seasons out of 10.

## Why

The engine already does this, one step earlier. Marcel regresses each
category toward league average by a constant `k` fitted per category, and
those constants come out as very nearly the inverse of reliability:

| category | fitted k | reliability |
|---|---|---|
| stl | 2500 (most regression) | 0.799 (least reliable) |
| tov | 1500 | 0.833 |
| pts | 1000 | 0.865 |
| ast | 500 | 0.914 |
| reb | 500 (least regression) | 0.939 (most reliable) |

Steals are pulled hardest toward average and trusted least; rebounds and
assists are pulled least and trusted most. By the time a projection reaches
the z-score, the volatile categories have already been shrunk. Weighting them
down again double-counts the same correction, which is why the result is
consistently and mildly negative rather than dramatic.

The correction belongs on the rate, where it already is, fitted from data
rather than inferred from a correlation.

## What is kept

`puntfit/reliability.py` stays as a measurement tool. The numbers are worth
knowing even though the weighting is not worth applying - they are the reason
to leave blocks alone when deciding which categories to trust.
