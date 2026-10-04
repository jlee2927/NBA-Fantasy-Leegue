
from pathlib import Path

import pytest

from puntfit.valuation import load_projections


@pytest.fixture(scope="session")
def projections():
    df, _ = load_projections(
        Path(__file__).parent / "data" / "marcel_projections.json")
    return df
