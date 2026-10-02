from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.clients import prices as price_client
from app.config import Settings, settings

START = date(2026, 9, 28)
END = date(2026, 10, 1)


def make_frame(rows: list[tuple]) -> pd.DataFrame:
    # The same structure yfinance returned for a real call (AAPL, auto_adjust=False,
    # actions=False): a DatetimeIndex named "Date", tz-aware US Eastern at midnight, float64
    # price columns and an int64 Volume
    index = pd.DatetimeIndex(
        [pd.Timestamp(day) for day, *_ in rows], tz="America/New_York", name="Date"
    )
    return pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
            "Adj Close": [row[5] for row in rows],
            "Volume": pd.Series([row[6] for row in rows], dtype="int64").to_numpy(),
        },
        index=index,
    )


# float32 noise on purpose: 189.3 stored as a float32 and read back as a float64
NOISY = float(np.float32(189.3))

THREE_DAYS = [
    ("2026-09-28", 100.0, 102.0, 99.0, 101.0, 100.5, 1000),
    # A row with a missing price
    ("2026-09-29", float("nan"), 103.0, 100.0, 102.0, 101.5, 2000),
    ("2026-09-30", NOISY, 190.123456, 188.0, 189.2999877929688, 188.9999, 3000),
]


class FakeTicker:
    # Records how yfinance was called, and returns the frame the test prepared
    calls: list[dict] = []
    frame: pd.DataFrame | Exception = pd.DataFrame()

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, **kwargs: object) -> pd.DataFrame:
        FakeTicker.calls.append({"ticker": self.ticker, **kwargs})
        if isinstance(FakeTicker.frame, Exception):
            raise FakeTicker.frame
        return FakeTicker.frame


@pytest.fixture(autouse=True)
def fake_yfinance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeTicker.calls = []
    FakeTicker.frame = make_frame(THREE_DAYS)
    monkeypatch.setattr(price_client.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))


def test_columns_are_mapped_and_the_index_becomes_plain_dates() -> None:
    bars = price_client.YFinanceProvider().get_daily_bars("AAPL", START, END)

    first = bars[0]
    assert first == {
        "trade_date": date(2026, 9, 28),
        "open": Decimal("100"),
        "high": Decimal("102"),
        "low": Decimal("99"),
        "close": Decimal("101"),
        "adj_close": Decimal("100.5"),
        "volume": 1000,
    }
    # A plain date, not a timestamp
    assert type(first["trade_date"]) is date


def test_a_row_with_a_nan_price_is_dropped() -> None:
    bars = price_client.YFinanceProvider().get_daily_bars("AAPL", START, END)

    assert [bar["trade_date"] for bar in bars] == [date(2026, 9, 28), date(2026, 9, 30)]


def test_prices_are_rounded_to_four_decimals_and_returned_as_decimal() -> None:
    bars = price_client.YFinanceProvider().get_daily_bars("AAPL", START, END)

    noisy_bar = bars[1]
    # 189.3000030517578 and 189.2999877929688 are both float32 noise around 189.3
    assert noisy_bar["open"] == Decimal("189.3")
    assert noisy_bar["close"] == Decimal("189.3")
    assert noisy_bar["high"] == Decimal("190.1235")
    for field in ("open", "high", "low", "close", "adj_close"):
        assert type(noisy_bar[field]) is Decimal
        assert noisy_bar[field].as_tuple().exponent >= -4


def test_volume_is_a_plain_int() -> None:
    bars = price_client.YFinanceProvider().get_daily_bars("AAPL", START, END)

    assert type(bars[0]["volume"]) is int
    assert bars[1]["volume"] == 3000


def test_bars_are_sorted_oldest_first() -> None:
    # The frame is given newest first on purpose
    FakeTicker.frame = make_frame(list(reversed(THREE_DAYS)))

    bars = price_client.YFinanceProvider().get_daily_bars("AAPL", START, END)

    assert [bar["trade_date"] for bar in bars] == sorted(bar["trade_date"] for bar in bars)


def test_yfinance_gets_an_exclusive_end_one_day_later_and_no_auto_adjust() -> None:
    price_client.YFinanceProvider().get_daily_bars("AAPL", START, END)

    assert len(FakeTicker.calls) == 1
    call = FakeTicker.calls[0]
    assert call["ticker"] == "AAPL"
    assert call["start"] == START
    # `end` is inclusive for us, exclusive for yfinance
    assert call["end"] == END + timedelta(days=1)
    assert call["auto_adjust"] is False
    assert call["actions"] is False


def test_the_raw_frame_is_saved_as_csv_before_converting(tmp_path: Path) -> None:
    price_client.YFinanceProvider().get_daily_bars("AAPL", START, END)

    raw_file = tmp_path / "prices" / "yfinance" / "AAPL.csv"
    assert raw_file.exists()
    lines = raw_file.read_text().splitlines()
    assert lines[0] == "Date,Open,High,Low,Close,Adj Close,Volume"
    # The raw file keeps every row, including the one with the NaN
    assert len(lines) == 1 + len(THREE_DAYS)


def test_an_empty_frame_raises_a_provider_error() -> None:
    FakeTicker.frame = pd.DataFrame()

    with pytest.raises(price_client.PriceProviderError) as error:
        price_client.YFinanceProvider().get_daily_bars("ZZZZ", START, END)

    assert "ZZZZ" in str(error.value)


def test_an_exception_from_yfinance_is_wrapped() -> None:
    FakeTicker.frame = RuntimeError("rate limited")

    with pytest.raises(price_client.PriceProviderError) as error:
        price_client.YFinanceProvider().get_daily_bars("AAPL", START, END)

    assert "AAPL" in str(error.value)
    assert "rate limited" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_get_price_provider_returns_yfinance_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "PRICE_PROVIDER", "yfinance")

    assert isinstance(price_client.get_price_provider(), price_client.YFinanceProvider)


def test_get_price_provider_rejects_an_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "PRICE_PROVIDER", "nonsense")

    with pytest.raises(ValueError, match="yfinance"):
        price_client.get_price_provider()


def test_the_default_provider_setting_is_yfinance() -> None:
    assert Settings.model_fields["PRICE_PROVIDER"].default == "yfinance"
    assert Settings.model_fields["PRICES_LOOKBACK_YEARS"].default == 5
