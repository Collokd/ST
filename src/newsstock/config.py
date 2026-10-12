"""Settings for the project: which stocks we study, the date range, and where files go."""

from dataclasses import dataclass, field
from pathlib import Path

# We picked five large U.S. companies that get a lot of news coverage.
# "query" is what we search for in GDELT. "pattern" is a regular expression
# we use afterwards to make sure the headline is really about the company
# (for example, so a headline about apple pie doesn't count as Apple news).
COMPANIES = {
    "AAPL": {"query": '("Apple Inc" OR "Apple" OR "iPhone" OR AAPL)', "pattern": r"\b(?:apple|iphone|aapl|tim cook)\b"},
    "MSFT": {"query": '("Microsoft" OR MSFT)', "pattern": r"\b(?:microsoft|msft|satya nadella)\b"},
    "AMZN": {"query": '("Amazon" OR AMZN)', "pattern": r"\b(?:amazon|amzn|aws|andy jassy)\b"},
    "TSLA": {"query": '("Tesla" OR TSLA)', "pattern": r"\b(?:tesla|tsla|elon musk)\b"},
    "NVDA": {"query": '("Nvidia" OR NVDA)', "pattern": r"\b(?:nvidia|nvda|jensen huang)\b"},
}

# The U.S. market closes at 4:00 pm New York time.
MARKET_TIMEZONE = "America/New_York"
MARKET_CLOSE_HOUR = 16


@dataclass
class Config:
    tickers: list = field(default_factory=lambda: list(COMPANIES))
    start: str = "2022-01-01"
    end: str = "2024-12-31"
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    # We split the data by date: the first 60% of trading days for training,
    # the next 20% for validation (tuning), and the last 20% for testing.
    train_frac: float = 0.6
    val_frac: float = 0.2

    random_state: int = 42
    finbert_model: str = "ProsusAI/finbert"

    @property
    def raw_dir(self):
        return self.data_dir / "raw"

    @property
    def processed_dir(self):
        return self.data_dir / "processed"

    def make_folders(self):
        for folder in [self.raw_dir, self.processed_dir, self.output_dir]:
            folder.mkdir(parents=True, exist_ok=True)
