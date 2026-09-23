"""Category definitions for the supported scoring formats."""
from __future__ import annotations

# Ratio categories, mapped to the projected made/attempted columns they are
# built from. Never valued off a regressed percentage.
RATIO = {"fg": ("fgm_pg", "fga_pg"), "ft": ("ftm_pg", "fta_pg")}

# Categories where fewer is better, so the z-score is negated before summing.
NEGATIVE = frozenset({"tov"})

NINE_CAT = ("fg", "ft", "tpm", "pts", "reb", "ast", "stl", "blk", "tov")
EIGHT_CAT = tuple(c for c in NINE_CAT if c != "tov")

FORMATS = {"9cat": NINE_CAT, "8cat": EIGHT_CAT}
