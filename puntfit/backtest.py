"""Walk-forward backtest for the projection parameters.

The question this answers is whether the tuned parameters actually project
better, rather than whether they fit the seasons they were tuned on.

Three things keep the answer honest:

  Walk-forward. Every season is projected using only parameters fitted on
  seasons before it. Tuning on all seasons at once and then scoring those same
  seasons measures fit, not skill, and always flatters itself.

  A holdout. One season is never used to tune anything, at any stage, so the
  headline number comes from data no parameter has seen.

  A split by what each parameter describes. The k constants and the aging
  method describe how basketball works - how much a season of evidence is
  worth, how players decline - and get every prior season. The minutes and
  games baselines describe how the league is behaving now: load management and
  schedule density move them, so a long window averages across eras that no
  longer apply. They get a short recent window instead.

Everything configurable lives in config.toml.

Usage:
    python -m puntfit.backtest --config config.toml --out docs/backtest.md
"""
from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import marcel as M

AGING_METHODS = ("none", "tango29", "tango27", "delta")


@dataclass(frozen=True)
class Config:
    source: Path
    min_history: int
    holdout: int
    min_prior_targets: int
    structural_window: int
    regime_window: int
    recency_halflife: float
    early_through: int
    late_from: int
    flag_relative_change: float

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = tomllib.loads(Path(path).read_text())
        return cls(
            source=Path(raw["data"]["source"]),
            min_history=raw["backtest"]["min_history"],
            holdout=raw["backtest"]["holdout"],
            min_prior_targets=raw["backtest"]["min_prior_targets"],
            structural_window=raw["structural"]["window"],
            regime_window=raw["regime"]["window"],
            recency_halflife=raw["regime"]["recency_halflife"],
            early_through=raw["stability"]["early_through"],
            late_from=raw["stability"]["late_from"],
            flag_relative_change=raw["stability"]["flag_relative_change"],
        )


# ------------------------------------------------------------------ scoring
def season_error(rows: list[dict]) -> float:
    """One number per season: RMSE relative to each category's own mean,
    averaged over categories. Scale-free, so categories on different units
    contribute equally."""
    return float(np.mean([r["marcel"] / r["mean_per36"] for r in rows]))


def naive_error(rows: list[dict]) -> float:
    """Same measure for 'just use last season', the thing to beat."""
    return float(np.mean([r["last_season"] / r["mean_per36"] for r in rows]))


def tuning_window(targets: list[int], target: int, holdout: int) -> list[int]:
    """Seasons a given target is allowed to learn from.

    Strictly earlier than the target, and never the holdout. This is the whole
    no-leakage guarantee, kept in one place so it can be tested directly.
    """
    return [t for t in targets if t < target and t != holdout]


def weights_for(window: list[int], target: int, halflife: float) -> np.ndarray:
    if halflife <= 0:
        return np.ones(len(window))
    return np.array([0.5 ** ((target - t) / halflife) for t in window])


# ------------------------------------------------- precomputed error tables
def rate_errors(df, targets, curves, stats) -> dict:
    """Per-category error for every (target, uniform k). Aging is off here so
    that k is not tuned to absorb an aging effect.

    Cached because the projection for a given (target, k) is the same whichever
    later season is being tuned, which is what keeps walk-forward affordable.
    """
    out = {}
    for t in targets:
        for kk in M.K_GRID:
            proj = M.marcel(df, t, {s: kk for s in stats}, 0, 0, aging="none")
            rows = M.score_rates(df, t, proj)
            out[(t, kk)] = {r["stat"]: r["marcel"] for r in rows}
    return out


def playing_time_errors(df, targets) -> tuple[dict, dict]:
    """Minutes and games baselines move proj_min and proj_g only, and neither
    depends on k or on the aging method, so both grids are swept once."""
    mins, games = {}, {}
    zero = {s: 0 for s in M.stats_in(df)}
    for t in targets:
        for mb in M.MIN_BASE_GRID:
            p = M.marcel(df, t, zero, mb, 0, aging="none")
            mins[(t, mb)] = M.score_playing_time(df, t, p)["min_marcel"]
        for gb in M.GAME_BASE_GRID:
            p = M.marcel(df, t, zero, 0, gb, aging="none")
            games[(t, gb)] = M.score_playing_time(df, t, p)["g_marcel"]
    return mins, games


# ---------------------------------------------------------------- tuning
def tune_k(rate_err: dict, window: list[int], stats) -> dict:
    k = {}
    for s in stats:
        k[s] = min(M.K_GRID,
                   key=lambda kk: sum(rate_err[(t, kk)][s] ** 2 for t in window))
    return k


def tune_baselines(mins, games, window, target, halflife) -> tuple[float, float]:
    w = weights_for(window, target, halflife)
    mb = min(M.MIN_BASE_GRID,
             key=lambda b: float(np.sum(w * np.array([mins[(t, b)] for t in window]) ** 2)))
    gb = min(M.GAME_BASE_GRID,
             key=lambda b: float(np.sum(w * np.array([games[(t, b)] for t in window]) ** 2)))
    return mb, gb


def tune_aging(df, window, curves, k, mb, gb) -> str:
    scores = {}
    for method in AGING_METHODS:
        scores[method] = np.mean([
            season_error(M.score_rates(df, t, M.marcel(df, t, k, mb, gb, method, curves[t])))
            for t in window])
    return min(scores, key=scores.get)


def fit(df, window, target, curves, rate_err, mins, games, cfg, stats) -> dict:
    """Every parameter for one target, using only the seasons in `window`."""
    structural = window if cfg.structural_window <= 0 else window[-cfg.structural_window:]
    regime = window[-cfg.regime_window:]
    k = tune_k(rate_err, structural, stats)
    mb, gb = tune_baselines(mins, games, regime, target, cfg.recency_halflife)
    aging = tune_aging(df, structural, curves, k, mb, gb)
    return {"k": k, "min_base": mb, "game_base": gb, "aging": aging,
            "structural_seasons": len(structural), "regime_seasons": len(regime)}


# ------------------------------------------------------------ walk forward
def walk_forward(df, cfg: Config) -> tuple[pd.DataFrame, dict]:
    stats = M.stats_in(df)
    targets = [t for t in M.backtest_targets(df)]
    curves = {t: M.age_curve(df, t - 1) for t in targets}
    rate_err = rate_errors(df, targets, curves, stats)
    mins, games = playing_time_errors(df, targets)

    rows, params = [], {}
    for target in targets:
        window = tuning_window(targets, target, cfg.holdout)
        if len(window) < cfg.min_prior_targets:
            continue
        p = fit(df, window, target, curves, rate_err, mins, games, cfg, stats)
        params[target] = p

        scored = M.score_rates(df, target, M.marcel(
            df, target, p["k"], p["min_base"], p["game_base"], p["aging"], curves[target]))
        # the leaky comparison: tune on every season including this one
        allp = fit(df, targets, target, curves, rate_err, mins, games, cfg, stats)
        leaky = M.score_rates(df, target, M.marcel(
            df, target, allp["k"], allp["min_base"], allp["game_base"],
            allp["aging"], curves[target]))

        rows.append({"target": target,
                     "walk_forward": season_error(scored),
                     "tuned_on_everything": season_error(leaky),
                     "last_season_only": naive_error(scored),
                     "aging": p["aging"], "min_base": p["min_base"],
                     "game_base": p["game_base"],
                     "tuning_seasons": len(window)})
    return pd.DataFrame(rows), params


def stability(df, cfg: Config) -> pd.DataFrame:
    """Refit the structural parameters on an early and a late window."""
    stats = M.stats_in(df)
    targets = M.backtest_targets(df)
    curves = {t: M.age_curve(df, t - 1) for t in targets}
    rate_err = rate_errors(df, targets, curves, stats)

    early = [t for t in targets if t <= cfg.early_through]
    late = [t for t in targets if cfg.late_from <= t and t != cfg.holdout]
    ke, kl = tune_k(rate_err, early, stats), tune_k(rate_err, late, stats)

    rows = []
    for s in stats:
        a, b = ke[s], kl[s]
        denom = max(a, b, 1)
        rows.append({"parameter": f"k[{s}]", "early": a, "late": b,
                     "relative_change": abs(a - b) / denom})
    out = pd.DataFrame(rows)
    out["flag"] = np.where(out.relative_change > cfg.flag_relative_change, "MOVED", "")
    return out


# ----------------------------------------------------------------- report
def _table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                              for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def write_report(scores, params, stab, cfg, out: Path) -> None:
    held = scores[scores.target == cfg.holdout]
    tuned = scores[scores.target != cfg.holdout]
    lines = [
        "# Walk-forward backtest", "",
        "Error is RMSE per category relative to that category's own mean, "
        "averaged over categories. Lower is better.", "",
        "- `walk_forward`: parameters fitted only on seasons before the target",
        "- `tuned_on_everything`: fitted on all seasons including the target, "
        "which leaks and so flatters itself",
        "- `last_season_only`: no projection at all, just carry last season forward",
        "",
        "## Headline", "",
        f"- Seasons scored: {len(tuned)} (tuning) + {len(held)} (holdout)",
        f"- Walk-forward mean error: **{tuned.walk_forward.mean():.4f}**",
        f"- Tuned-on-everything mean: {tuned.tuned_on_everything.mean():.4f} "
        f"(optimistic by {100 * (1 - tuned.tuned_on_everything.mean() / tuned.walk_forward.mean()):.1f}%)",
        f"- Carry last season forward: {tuned.last_season_only.mean():.4f}",
        f"- Improvement over carrying last season: "
        f"**{100 * (1 - tuned.walk_forward.mean() / tuned.last_season_only.mean()):.1f}%**",
        "",
        f"## Holdout {cfg.holdout}", "",
        "Never used to tune anything, at any stage.", "",
        _table(held.drop(columns=["tuning_seasons"])) if len(held) else "_not reached_",
        "", "## Every scored season", "", _table(scores), "",
        "## Structural parameter stability", "",
        f"k refitted on targets through {cfg.early_through} versus "
        f"{cfg.late_from} onward. Large moves would mean the long window is "
        "mixing eras and the structural/regime split is drawn wrong.", "",
        _table(stab), "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml", type=Path)
    ap.add_argument("--out", default="docs/backtest.md", type=Path)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    df = M.load_hoopr(cfg.source)
    scores, params = walk_forward(df, cfg)
    stab = stability(df, cfg)
    write_report(scores, params, stab, cfg, a.out)

    tuned = scores[scores.target != cfg.holdout]
    print(scores.to_string(index=False))
    print()
    print(f"walk-forward       {tuned.walk_forward.mean():.4f}")
    print(f"tuned on all       {tuned.tuned_on_everything.mean():.4f}")
    print(f"last season only   {tuned.last_season_only.mean():.4f}")
    print(f"\nwrote {a.out}")
