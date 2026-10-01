"""Category definitions for the supported scoring formats."""
from __future__ import annotations

# Ratio categories, mapped to the projected made/attempted columns they are
# built from. Never valued off a regressed percentage.
RATIO = {"fg": ("fgm_pg", "fga_pg"), "ft": ("ftm_pg", "fta_pg")}

# Categories where fewer is better, so the z-score is negated before summing.
NEGATIVE = frozenset({"tov"})

# What each category looks like as a projected statistic rather than a z-score,
# as (column, heading, decimal places). Percentages are shown as the rate the
# player is projected to shoot; everything else is per game.
STAT_VIEW = {
    "fg":  ("fg_pct",  "fg%",  3),
    "ft":  ("ft_pct",  "ft%",  3),
    "tpm": ("tpm_pg",  "3/g",  1),
    "pts": ("pts_pg",  "p/g",  1),
    "reb": ("reb_pg",  "r/g",  1),
    "ast": ("ast_pg",  "a/g",  1),
    "stl": ("stl_pg",  "s/g",  1),
    "blk": ("blk_pg",  "b/g",  1),
    "tov": ("tov_pg",  "to/g", 1),
}

NINE_CAT = ("fg", "ft", "tpm", "pts", "reb", "ast", "stl", "blk", "tov")
EIGHT_CAT = tuple(c for c in NINE_CAT if c != "tov")

FORMATS = {"9cat": NINE_CAT, "8cat": EIGHT_CAT}
