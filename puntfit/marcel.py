"""
Marcel projections for PuntFit.

Reads one Basketball Monster season export per season
(data/seasons/Fantasy_YYYY-YYYY.xls), builds a delta-method age curve,
backtests against held-out seasons to tune k (regression) and the minutes
baseline, then projects the next season.

Usage:
    python -m puntfit.marcel --seasons data/seasons --out data

Outputs:
    marcel_projections.csv / .json   next-season projections
    marcel_backtest.csv              error by stat and method
    marcel_age_curve.csv             multiplicative aging factor by age and stat

Data notes (Basketball Monster export):
  * Stats are per game, so totals = per-game x games.
  * FGM/FTM are derived from fg% x fga, ft% x fta.
  * "Age" is age on the export date in EVERY sheet, not age during that
    season, so season ages are back-computed from EXPORT_DATE.
  * Each sheet holds only the top-ranked players (234). A player missing
    from a season may have played outside that pool, so missing seasons are
    treated as "no information" (0 weight), not as 0 minutes played.
  * No player ID and no turnovers column. Names are the join key.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

EXPORT_DATE = date(2026, 9, 19)  # when the sheets were exported
WEIGHTS = (5, 4, 3)              # Y-1, Y-2, Y-3
STATS = ["pts", "tpm", "reb", "ast", "stl", "blk", "fgm", "fga", "ftm", "fta"]
NEGATIVE_STATS = {"tov"}         # aging not applied (fewer is better)
MIN_PAIR_MINUTES = 250           # delta-method pair minimum, each season
MIN_EVAL_MINUTES = 500           # backtest: target-season minutes to be scored

K_GRID = [0, 500, 1000, 1500, 2500, 4000, 6000, 9000, 13000, 20000]
MIN_BASE_GRID = list(range(0, 1601, 100))
GAME_BASE_GRID = list(range(0, 51, 2))


# ---------------------------------------------------------------- loading
def season_end_year(label: str) -> int:
    return int(re.search(r"(\d{4})-(\d{4})", label).group(2))


def age_on_feb1(age_at_export: pd.Series, end_year: int) -> pd.Series:
    """Age on Feb 1 of the season's end year (the usual convention)."""
    shift = (date(end_year, 2, 1) - EXPORT_DATE).days / 365.25
    return age_at_export + shift


def load_season(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, engine="xlrd")
    g = raw["g"].astype(float)
    df = pd.DataFrame({
        "name": raw["Name"].str.strip(),
        "team": raw["Team"],
        "pos": raw["Pos"],
        "inj": raw["Inj"],
        "g": g,
        "min": raw["m/g"] * g,
        "pts": raw["p/g"] * g,
        "tpm": raw["3/g"] * g,
        "reb": raw["r/g"] * g,
        "ast": raw["a/g"] * g,
        "stl": raw["s/g"] * g,
        "blk": raw["b/g"] * g,
        "fga": raw["fga/g"] * g,
        "fgm": raw["fga/g"] * raw["fg%"] * g,
        "fta": raw["fta/g"] * g,
        "ftm": raw["fta/g"] * raw["ft%"] * g,
    })
    if "to/g" in raw.columns:  # picked up automatically if added to the export
        df["tov"] = raw["to/g"] * g
    end = season_end_year(path.stem)
    df["season"] = end
    df["age"] = age_on_feb1(raw["Age"], end)
    return df


def load_all(folder: Path) -> pd.DataFrame:
    # Basketball Monster names its exports "Fantasy 2025-2026.xls" with a
    # space; accept an underscore too, since the docs long said underscore.
    files = sorted(folder.glob("Fantasy[ _]*.xls*"), key=lambda p: season_end_year(p.stem))
    if not files:
        raise FileNotFoundError(f"No 'Fantasy YYYY-YYYY.xls' files in {folder}")
    df = pd.concat([load_season(f) for f in files], ignore_index=True)
    if "tov" in df.columns and "tov" not in STATS:
        STATS.append("tov")
    return df


def league_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-minute league rate for each season (sum of stat / sum of minutes)."""
    sums = df.groupby("season")[STATS + ["min"]].sum()
    return sums[STATS].div(sums["min"], axis=0)


# ---------------------------------------------------------------- aging
def age_curve(df: pd.DataFrame, max_season: int) -> pd.DataFrame:
    """Delta method on seasons <= max_season.

    For every player in consecutive seasons, compare per-minute rates.
    Pairs are weighted by the harmonic mean of the two seasons' minutes.
    The per-age factor is sum(w*rate2) / sum(w*rate1), which handles zero
    rates (bigs with no threes). log(factor) is smoothed with a weighted
    quadratic in age because 4-5 transitions leave thin age buckets.
    Returns a one-year multiplicative factor for going from age a to a+1.
    """
    d = df[df.season <= max_season]
    a = d.merge(d.assign(season=d.season - 1), on=["name", "season"], suffixes=("1", "2"))
    a = a[(a.min1 >= MIN_PAIR_MINUTES) & (a.min2 >= MIN_PAIR_MINUTES)]
    a["w"] = 2 / (1 / a.min1 + 1 / a.min2)
    a["age_bin"] = np.floor(a.age1).astype(int)

    ages = np.arange(19, 41)
    out = pd.DataFrame(index=ages)
    for s in STATS:
        if s in NEGATIVE_STATS:
            out[s] = 1.0
            continue
        r1 = a[s + "1"] / a.min1
        r2 = a[s + "2"] / a.min2
        grp = pd.DataFrame({"age": a.age_bin, "n": a.w * r2, "d": a.w * r1, "w": a.w})
        b = grp.groupby("age").sum()
        b = b[(b.d > 0) & (b.n > 0)]
        y = np.log(b.n / b.d)
        coef = np.polyfit(b.index, y, 2, w=np.sqrt(b.w))
        out[s] = np.exp(np.polyval(coef, ages))
    out.index.name = "age"
    return out


def aging_factor(curve: pd.DataFrame, stat: str, age_from: float, age_to: float) -> float:
    """Chain one-year factors from age_from to age_to (fractional ages ok)."""
    if stat in NEGATIVE_STATS:
        return 1.0
    lo, hi = curve.index.min(), curve.index.max()
    f, a = 1.0, age_from
    while a < age_to - 1e-9:
        step = min(1.0, age_to - a)
        f *= curve.loc[int(np.clip(np.floor(a), lo, hi)), stat] ** step
        a += step
    return f


def tango_factor(age: float, stat: str, ref: float = 29.0) -> float:
    if stat in NEGATIVE_STATS:
        return 1.0
    return 1 + (ref - age) * 0.006 if age < ref else 1 - (age - ref) * 0.003


# ---------------------------------------------------------------- marcel
def marcel(df: pd.DataFrame, target: int, k: dict, min_base: float, game_base: float,
           aging: str = "delta", curve: pd.DataFrame | None = None) -> pd.DataFrame:
    """Project season `target` from target-1, -2, -3."""
    yrs = [target - 1, target - 2, target - 3]
    lg = league_rates(df[df.season.isin(yrs)])
    players = df[df.season.isin(yrs)].name.unique()
    wide = df[df.season.isin(yrs)].pivot_table(index="name", columns="season",
                                                values=STATS + ["min", "g"], aggfunc="sum")
    wide = wide.reindex(players).fillna(0.0)

    wmin = sum(w * wide["min"].get(y, 0) for w, y in zip(WEIGHTS, yrs))
    # league rate weighted the same way as the player's seasons
    wl = sum(WEIGHTS)
    lg_rate = {s: sum(w * lg.loc[y, s] for w, y in zip(WEIGHTS, yrs) if y in lg.index) / wl
               for s in STATS}

    last = df[df.season.isin(yrs)].sort_values("season").groupby("name").last()
    age_now = last.age + (target - last.season)  # age on Feb 1 of target season
    age_now = age_now.reindex(players)
    # centre of the weighted inputs, used by the delta curve
    lag = sum(w * wide["min"].get(y, 0) * (target - y) for w, y in zip(WEIGHTS, yrs)) / wmin.replace(0, np.nan)

    out = pd.DataFrame(index=players)
    out["age"] = age_now
    for s in STATS:
        wstat = sum(w * wide[s].get(y, 0) for w, y in zip(WEIGHTS, yrs))
        rate = (wstat + k[s] * lg_rate[s]) / (wmin + k[s])
        if aging == "tango29":
            adj = age_now.apply(lambda a: tango_factor(a, s, 29))
        elif aging == "tango27":
            adj = age_now.apply(lambda a: tango_factor(a, s, 27))
        elif aging == "delta":
            adj = pd.Series([aging_factor(curve, s, a - l, a) for a, l in zip(age_now, lag.fillna(1))],
                            index=players)
        else:
            adj = 1.0
        out[s + "_rate"] = rate * adj

    m1, m2 = wide["min"].get(yrs[0], 0), wide["min"].get(yrs[1], 0)
    g1, g2 = wide["g"].get(yrs[0], 0), wide["g"].get(yrs[1], 0)
    out["proj_min"] = 0.5 * m1 + 0.1 * m2 + min_base
    out["proj_g"] = np.minimum(82, 0.5 * g1 + 0.1 * g2 + game_base)
    # Minutes per game is projected on its own (5/4/3 weighted by games) so
    # per-game stats are not dragged down by the games-missed regression
    # baked into proj_min. Season totals still use proj_min per the spec.
    wg = sum(w * wide["g"].get(y, 0) for w, y in zip(WEIGHTS, yrs))
    out["proj_mpg"] = wmin / wg.replace(0, np.nan)
    for s in STATS:
        out[s] = out[s + "_rate"] * out["proj_min"]
    return out


# ---------------------------------------------------------------- backtest
def _rmse(err, w):
    return float(np.sqrt(np.sum(w * err ** 2) / np.sum(w)))


def score_rates(df, target, proj):
    """Minutes-weighted RMSE of per-36 rates, Marcel vs last season only.
    Scored on players present in the target season AND in Y-1, so both
    methods are compared on the same players."""
    act = df[(df.season == target) & (df["min"] >= MIN_EVAL_MINUTES)].set_index("name")
    prev = df[df.season == target - 1].set_index("name")
    names = act.index.intersection(prev.index).intersection(proj.index)
    a, p, l = act.loc[names], proj.loc[names], prev.loc[names]
    rows = []
    for s in STATS:
        truth = a[s] / a["min"] * 36
        rows.append({"target": target, "stat": s, "n": len(names),
                     "marcel": _rmse(p[s + "_rate"] * 36 - truth, a["min"]),
                     "last_season": _rmse(l[s] / l["min"] * 36 - truth, a["min"]),
                     "mean_per36": float(np.average(truth, weights=a["min"]))})
    for made, att, lab in (("fgm", "fga", "fg%"), ("ftm", "fta", "ft%")):
        truth = a[made] / a[att]
        rows.append({"target": target, "stat": lab, "n": len(names),
                     "marcel": _rmse(p[made + "_rate"] / p[att + "_rate"] - truth, a[att]),
                     "last_season": _rmse(l[made] / l[att] - truth, a[att]),
                     "mean_per36": float(np.average(truth, weights=a[att]))})
    return rows


def score_playing_time(df, target, proj):
    act = df[df.season == target].set_index("name")
    prev = df[df.season == target - 1].set_index("name")
    names = act.index.intersection(prev.index).intersection(proj.index)
    a, p, l = act.loc[names], proj.loc[names], prev.loc[names]
    return {
        "mpg_marcel": _rmse(p.proj_mpg - a["min"] / a.g, a.g),
        "mpg_last": _rmse(l["min"] / l.g - a["min"] / a.g, a.g),
        "min_marcel": _rmse(p.proj_min - a["min"], np.ones(len(names))),
        "min_last": _rmse(l["min"] - a["min"], np.ones(len(names))),
        "g_marcel": _rmse(p.proj_g - a.g, np.ones(len(names))),
        "g_last": _rmse(l.g - a.g, np.ones(len(names))),
    }


def backtest_targets(df):
    seasons = sorted(df.season.unique())
    return [t for t in seasons if all(t - i in seasons for i in (1, 2, 3))]


def tune(df):
    """Grid search k per stat (no aging, so k is not tuned to cover aging),
    then the minutes and games baselines, then pick the aging method."""
    targets = backtest_targets(df)
    curves = {t: age_curve(df, t - 1) for t in targets}  # no leakage
    k = {}
    for s in STATS:
        best = None
        for kk in K_GRID:
            ks = {x: kk for x in STATS}
            err = 0.0
            for t in targets:
                p = marcel(df, t, ks, 0, 0, aging="none")
                err += next(r["marcel"] for r in score_rates(df, t, p) if r["stat"] == s) ** 2
            if best is None or err < best[0]:
                best = (err, kk)
        k[s] = best[1]

    def pt_err(mb, gb, key):
        return sum(score_playing_time(df, t, marcel(df, t, k, mb, gb, "none"))[key] ** 2 for t in targets)

    min_base = min(MIN_BASE_GRID, key=lambda b: pt_err(b, 0, "min_marcel"))
    game_base = min(GAME_BASE_GRID, key=lambda b: pt_err(0, b, "g_marcel"))

    aging_scores = {}
    for method in ("none", "tango29", "tango27", "delta"):
        tot = 0.0
        for t in targets:
            p = marcel(df, t, k, min_base, game_base, method, curves[t])
            # scale-free: RMSE relative to the stat's mean, averaged over cats
            tot += np.mean([r["marcel"] / r["mean_per36"] for r in score_rates(df, t, p)])
        aging_scores[method] = tot / len(targets)
    aging = min(aging_scores, key=aging_scores.get)

    report = []
    for t in targets:
        p = marcel(df, t, k, min_base, game_base, aging, curves[t])
        report += score_rates(df, t, p)
        pt = score_playing_time(df, t, p)
        report += [{"target": t, "stat": "mpg", "marcel": pt["mpg_marcel"], "last_season": pt["mpg_last"]},
                   {"target": t, "stat": "minutes", "marcel": pt["min_marcel"], "last_season": pt["min_last"]},
                   {"target": t, "stat": "games", "marcel": pt["g_marcel"], "last_season": pt["g_last"]}]
    return k, min_base, game_base, aging, aging_scores, pd.DataFrame(report)


# ---------------------------------------------------------------- main
def run(seasons_dir: Path, out_dir: Path) -> dict:
    df = load_all(seasons_dir)
    k, min_base, game_base, aging, aging_scores, report = tune(df)
    target = int(df.season.max()) + 1
    curve = age_curve(df, target - 1)
    proj = marcel(df, target, k, min_base, game_base, aging, curve)

    # keep only players who appeared last season (retired/overseas drop out)
    last = df[df.season == target - 1].set_index("name")
    proj = proj.loc[proj.index.intersection(last.index)].copy()
    proj["team"], proj["pos"], proj["inj"] = last.team, last.pos, last.inj
    per_game = proj[[c + "_rate" for c in STATS]].mul(proj.proj_mpg, axis=0)
    per_game.columns = [c + "_pg" for c in STATS]
    proj = proj.join(per_game)
    proj["fg_pct"] = proj.fgm / proj.fga
    proj["ft_pct"] = proj.ftm / proj.fta
    proj = proj.sort_values("pts_pg", ascending=False)
    proj.index.name = "name"

    out_dir.mkdir(parents=True, exist_ok=True)
    proj.round(4).to_csv(out_dir / "marcel_projections.csv")
    report.round(4).to_csv(out_dir / "marcel_backtest.csv", index=False)
    curve.round(4).to_csv(out_dir / "marcel_age_curve.csv")
    k = {s: int(v) for s, v in k.items()}
    params = {"target_season": f"{target - 1}-{target}", "k": k, "min_baseline": int(min_base),
              "games_baseline": int(game_base), "aging": aging,
              "aging_scores": {m: round(v, 4) for m, v in aging_scores.items()},
              "backtest_targets": [int(t) for t in backtest_targets(df)]}
    with open(out_dir / "marcel_projections.json", "w") as f:
        json.dump({"params": params,
                   "players": json.loads(proj.reset_index().round(4).to_json(orient="records"))}, f, indent=1)
    return params


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="data/seasons", type=Path)
    ap.add_argument("--out", default="data", type=Path)
    a = ap.parse_args()
    print(json.dumps(run(a.seasons, a.out), indent=1))
