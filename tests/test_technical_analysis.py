import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from technical_analysis import moving_average, rsi, macd


def test_moving_average():
    prices = pd.Series([1, 2, 3, 4, 5])
    ma = moving_average(prices, 3)
    assert ma.iloc[-1] == pytest.approx(4)


def test_rsi_basic():
    prices = pd.Series(range(1, 16))
    r = rsi(prices, 14)
    assert not r.isna().all()
    assert 0 <= r.iloc[-1] <= 100


def test_macd_shapes():
    prices = pd.Series(range(1, 50))
    macd_line, signal_line, hist = macd(prices)
    assert len(macd_line) == len(prices)
    assert len(signal_line) == len(prices)
    assert len(hist) == len(prices)
