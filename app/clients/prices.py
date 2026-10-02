import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import yfinance as yf

from app.config import settings

logger = logging.getLogger(__name__)


class PriceProviderError(Exception):
    pass


class PriceProvider(ABC):
    # The interface every price source implements. To add a provider: write one new subclass
    # and add one branch in get_price_provider() below.
    @abstractmethod
    def get_daily_bars(self, ticker: str, start: date, end: date) -> list[dict]:
        """Daily bars from `start` to `end`, both INCLUSIVE, sorted oldest first.

        Each bar is a dict with trade_date (date), open, high, low, close, adj_close (Decimal,
        4 decimal places) and volume (int). close is split-adjusted; adj_close is adjusted for
        splits and dividends. Raises PriceProviderError if no bars could be fetched.
        """


class YFinanceProvider(PriceProvider):
    # yfinance is an unofficial Yahoo Finance wrapper: free, no API key, fine for a personal
    # learning project but not for commercial use.
    def get_daily_bars(self, ticker: str, start: date, end: date) -> list[dict]:
        # yfinance treats `end` as exclusive, hence the extra day. auto_adjust=False keeps
        # Close (split-adjusted) and Adj Close (splits and dividends) as separate columns
        try:
            frame = yf.Ticker(ticker).history(
                start=start, end=end + timedelta(days=1), auto_adjust=False, actions=False
            )
        except Exception as error:
            # The one place a broad except is allowed: this is the library boundary, and
            # yfinance can fail in many ways (network, rate limit, parsing, changed Yahoo pages)
            raise PriceProviderError(f"{ticker}: yfinance failed ({error})") from error

        # yfinance often returns an empty frame instead of raising (unknown ticker, rate limit)
        if frame.empty:
            raise PriceProviderError(f"{ticker}: yfinance returned no data")

        # Save the raw answer before converting it, so it can be re-processed later
        raw_file = Path(settings.RAW_DATA_DIR) / "prices" / "yfinance" / f"{ticker}.csv"
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(raw_file)

        # Rows with a missing price are unusable
        price_columns = ["Open", "High", "Low", "Close", "Adj Close"]
        complete = frame.dropna(subset=price_columns)
        if len(complete) < len(frame):
            logger.warning(
                "%s: dropped %d rows with missing prices", ticker, len(frame) - len(complete)
            )

        bars = []
        for timestamp, row in complete.iterrows():
            # yfinance prices carry float32 noise (189.2999877929688), so round first.
            # Decimal(str(...)) keeps the rounded value exact. The index is tz-aware midnight in
            # US Eastern time, so .date() is the trading date
            bars.append(
                {
                    "trade_date": timestamp.date(),
                    "open": Decimal(str(round(float(row["Open"]), 4))),
                    "high": Decimal(str(round(float(row["High"]), 4))),
                    "low": Decimal(str(round(float(row["Low"]), 4))),
                    "close": Decimal(str(round(float(row["Close"]), 4))),
                    "adj_close": Decimal(str(round(float(row["Adj Close"]), 4))),
                    "volume": int(row["Volume"]),
                }
            )

        # The frame is already oldest first, but the interface promises it, so make sure
        bars.sort(key=lambda bar: bar["trade_date"])
        return bars


def get_price_provider() -> PriceProvider:
    if settings.PRICE_PROVIDER == "yfinance":
        return YFinanceProvider()
    raise ValueError(
        f"Unknown PRICE_PROVIDER '{settings.PRICE_PROVIDER}'. Supported values: yfinance"
    )
