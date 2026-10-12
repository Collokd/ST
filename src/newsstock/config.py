"""Project configuration: stock universe, date range, paths and split settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Small universe of heavily covered U.S. stocks. Each ticker maps to the search
# terms used to query GDELT and the regex used to confirm a headline really is
# about that company (reduces ambiguous matches such as "apple" the fruit).
DEFAULT_UNIVERSE: dict[str, dict[str, str]] = {
    "AAPL": {"query": '("Apple Inc" OR "Apple" OR "iPhone" OR AAPL)', "pattern": r"\b(?:apple|iphone|aapl|tim cook)\b"},
    "MSFT": {"query": '("Microsoft" OR MSFT)', "pattern": r"\b(?:microsoft|msft|satya nadella)\b"},
    "AMZN": {"query": '("Amazon" OR AMZN)', "pattern": r"\b(?:amazon|amzn|aws|andy jassy)\b"},
    "TSLA": {"query": '("Tesla" OR TSLA)', "pattern": r"\b(?:tesla|tsla|elon musk)\b"},
    "NVDA": {"query": '("Nvidia" OR NVDA)', "pattern": r"\b(?:nvidia|nvda|jensen huang)\b"},
}

MARKET_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16  # 4:00 pm ET regular-session close


@dataclass
class Config:
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    start: str = "2022-01-01"
    end: str = "2024-12-31"
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    # Chronological split fractions (by trading date); test = remainder.
    train_frac: float = 0.6
    val_frac: float = 0.2
    random_state: int = 42
    # Sentiment model on the Hugging Face hub.
    finbert_model: str = "ProsusAI/finbert"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)
