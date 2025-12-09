"""
Data Downloader Module
======================
Binance multi-timeframe data downloader for Crypto Swing Trading.
Supports 1h, 1d timeframes using CCXT library.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Literal

import pandas as pd
import numpy as np
import ccxt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()
logger = logging.getLogger(__name__)


TimeframeType = Literal["1h", "1d"]


@dataclass
class DownloadMetadata:
    """Metadata for downloaded data."""
    ticker: str
    start_date: str
    end_date: str
    timeframes: list[str]
    base_timeframe: str
    rows: int
    created_at: str
    exchange: str = "binance"


class BinanceDownloader:
    """
    Multi-timeframe data downloader for Binance using CCXT.

    Timeframes supported for Crypto Swing Trading:
    - 1h (hourly): Base timeframe for swing trading
    - 1d (daily): Higher timeframe for context

    Features:
    - Downloads 2 timeframes (1h, 1d) optimized for swing trading
    - Aligns higher timeframes to hourly base
    - Saves in parquet format with train/eval split
    - No API key required for historical data
    - Rate limiting handled automatically
    """

    TIMEFRAMES = ['1h', '1d']

    # CCXT timeframe mapping
    CCXT_TIMEFRAMES = {
        '1h': '1h',
        '1d': '1d',
    }

    # Milliseconds per timeframe
    MS_PER_CANDLE = {
        '1h': 60 * 60 * 1000,      # 1 hour
        '1d': 24 * 60 * 60 * 1000,  # 1 day
    }

    def __init__(
        self,
        data_dir: str | Path = "data",
        exchange_id: str = "binance",
    ):
        """
        Initialize the downloader.

        Args:
            data_dir: Directory for processed data
            exchange_id: CCXT exchange ID (default: binance)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.exchange_id = exchange_id

        # Initialize CCXT exchange (no API key needed for public data)
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for CCXT.

        Accepts: BTC/USDT, BTCUSDT, btc/usdt, btcusdt
        Returns: BTC/USDT (CCXT format)
        """
        symbol = symbol.upper().strip()

        # If already in correct format
        if '/' in symbol:
            return symbol

        # Try to parse common patterns (BTCUSDT -> BTC/USDT)
        common_quotes = ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB']
        for quote in common_quotes:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                if base:
                    return f"{base}/{quote}"

        # Default: assume last 4 chars are quote currency
        if len(symbol) > 4:
            return f"{symbol[:-4]}/{symbol[-4:]}"

        return symbol

    def _download_timeframe(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        until: int,
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a single timeframe from Binance.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: '1h' or '1d'
            since: Start timestamp in milliseconds
            until: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            ccxt_tf = self.CCXT_TIMEFRAMES[timeframe]
            ms_per_candle = self.MS_PER_CANDLE[timeframe]

            all_candles = []
            current_since = since
            limit = 1000  # Binance max limit per request

            while current_since < until:
                try:
                    # Fetch OHLCV data
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol,
                        timeframe=ccxt_tf,
                        since=current_since,
                        limit=limit,
                    )

                    if not ohlcv:
                        break

                    all_candles.extend(ohlcv)

                    # Move to next batch
                    last_timestamp = ohlcv[-1][0]
                    current_since = last_timestamp + ms_per_candle

                    # Small delay to respect rate limits
                    time.sleep(self.exchange.rateLimit / 1000)

                    # Progress indicator
                    if len(all_candles) % 5000 == 0:
                        logger.debug(f"Downloaded {len(all_candles)} candles for {symbol} {timeframe}")

                except ccxt.NetworkError as e:
                    logger.warning(f"Network error, retrying: {e}")
                    time.sleep(5)
                    continue
                except ccxt.ExchangeError as e:
                    logger.error(f"Exchange error: {e}")
                    break

            if not all_candles:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(
                all_candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            # Remove timezone for consistency
            df.index = df.index.tz_localize(None)

            # Sort and remove duplicates
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]

            # Filter to requested range
            start_dt = pd.Timestamp(since, unit='ms')
            end_dt = pd.Timestamp(until, unit='ms')
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]

            return df

        except Exception as e:
            logger.error(f"Failed to download {symbol} {timeframe}: {e}")
            return None

    def _align_timeframes(
        self,
        df_1h: pd.DataFrame,
        df_1d: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align 2 timeframes by repeating higher timeframe values WITH LAG.

        CRITICAL: To avoid data leakage, daily data is shifted by 1 period.
        This ensures that at hour H, the agent only sees daily data that has
        ALREADY CLOSED, not the current incomplete day.

        Each row of the result corresponds to an hourly bar (base timeframe).
        Daily values are from the PREVIOUS completed day.

        Args:
            df_1h: Hourly OHLCV data (base)
            df_1d: Daily OHLCV data

        Returns:
            Aligned DataFrame with suffixed columns (no data leakage)
        """
        # Rename columns with suffixes
        df_1h_renamed = df_1h.add_suffix('_1h')
        df_1d_renamed = df_1d.add_suffix('_1d')

        # CRITICAL: Shift daily data by 1 period to prevent data leakage
        # This ensures we only see data from days that have already closed
        df_1d_shifted = df_1d_renamed.shift(1)

        # Drop NaN rows created by shift at the beginning
        df_1d_shifted = df_1d_shifted.dropna()

        # Start with hourly data as base
        aligned = df_1h_renamed.copy()

        # Create alignment key: date (without time) for joining daily data
        aligned['_date'] = aligned.index.normalize()

        # Prepare daily data - use date as key
        df_1d_for_merge = df_1d_shifted.copy()
        df_1d_for_merge['_date'] = df_1d_for_merge.index.normalize()

        # Remove duplicates keeping the first occurrence for each day
        df_1d_for_merge = df_1d_for_merge.drop_duplicates(subset=['_date'], keep='first')
        df_1d_for_merge = df_1d_for_merge.set_index('_date')

        # Join daily data using the date key
        aligned = aligned.join(df_1d_for_merge, on='_date', how='left')

        # Drop temporary alignment column
        aligned = aligned.drop(columns=['_date'], errors='ignore')

        # Forward fill any gaps (for hours where daily data not yet available)
        aligned = aligned.ffill()

        # Drop rows at the beginning where we don't have lagged data (no bfill to avoid leakage)
        aligned = aligned.dropna()

        return aligned

    def download_multi_timeframe(
        self,
        symbol: str,
        days: int = 365 * 2,  # 2 years default for crypto
        show_progress: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Download and align data for 2 timeframes (1h, 1d).

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
            days: Days of history to download (default 2 years)
            show_progress: Show progress indicator

        Returns:
            Aligned DataFrame with all timeframes, or None if failed
        """
        # Normalize symbol
        symbol = self._normalize_symbol(symbol)

        console.print(f"\n[bold cyan]═══ Crypto Data Download ═══[/bold cyan]")
        console.print(f"[bold]Downloading {symbol} multi-timeframe data from Binance[/bold]")
        console.print(f"  Timeframes: 1h (hourly), 1d (daily)")
        console.print(f"  History: {days} days (~{days/365:.1f} years)")

        # Calculate timestamps
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        since_ms = int(start_time.timestamp() * 1000)
        until_ms = int(end_time.timestamp() * 1000)

        df_1h = None
        df_1d = None

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                # Download hourly data
                task = progress.add_task("[cyan]Downloading 1h data...", total=None)
                df_1h = self._download_timeframe(symbol, "1h", since_ms, until_ms)
                if df_1h is not None:
                    progress.update(task, description=f"[green]✓ 1h data: {len(df_1h):,} candles")
                else:
                    progress.update(task, description="[red]✗ 1h data failed")

                # Download daily data
                task = progress.add_task("[cyan]Downloading 1d data...", total=None)
                df_1d = self._download_timeframe(symbol, "1d", since_ms, until_ms)
                if df_1d is not None:
                    progress.update(task, description=f"[green]✓ 1d data: {len(df_1d):,} candles")
                else:
                    progress.update(task, description="[red]✗ 1d data failed")
        else:
            console.print("  Downloading 1h data...")
            df_1h = self._download_timeframe(symbol, "1h", since_ms, until_ms)

            console.print("  Downloading 1d data...")
            df_1d = self._download_timeframe(symbol, "1d", since_ms, until_ms)

        # Validate downloads
        if df_1h is None or df_1h.empty:
            console.print("[red]Failed to download hourly data[/red]")
            return None

        if df_1d is None or df_1d.empty:
            console.print("[red]Failed to download daily data[/red]")
            return None

        console.print(f"\n[green]Downloaded data:[/green]")
        console.print(f"  1h:  {len(df_1h):,} rows ({df_1h.index[0]} to {df_1h.index[-1]})")
        console.print(f"  1d:  {len(df_1d):,} rows ({df_1d.index[0]} to {df_1d.index[-1]})")

        # Align timeframes
        console.print("\n[cyan]Aligning timeframes (with lag to prevent data leakage)...[/cyan]")
        aligned = self._align_timeframes(df_1h, df_1d)

        # Remove rows with NaN (start of data where alignment fails)
        initial_len = len(aligned)
        aligned = aligned.dropna()
        dropped = initial_len - len(aligned)

        if dropped > 0:
            console.print(f"  [yellow]Dropped {dropped} rows with incomplete data[/yellow]")

        console.print(f"[green]Aligned data: {len(aligned):,} rows[/green]")
        console.print(f"  Date range: {aligned.index[0]} to {aligned.index[-1]}")

        return aligned

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        train_ratio: float = 0.8,
    ) -> Path:
        """
        Save aligned DataFrame with STRICT CHRONOLOGICAL train/eval split.

        IMPORTANT: No random splitting for time series data!
        The split is strictly chronological:
        - Train: 80% OLDEST data (first 80% chronologically)
        - Eval: 20% NEWEST data (last 20% chronologically)

        This prevents data leakage and ensures realistic backtesting.

        Directory structure: data/{symbol}/multi_tf/{date_range}/
            - train.parquet (80% oldest)
            - eval.parquet (20% newest)
            - metadata.json

        Args:
            df: Aligned DataFrame (must be sorted by date)
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            train_ratio: Ratio for training split (chronological, NOT random)

        Returns:
            Path to saved directory
        """
        # Ensure data is sorted chronologically
        df = df.sort_index()

        # Format symbol for directory name (replace / with _)
        symbol_safe = symbol.replace('/', '_')

        # Format date range
        start_date = df.index[0].strftime("%Y_%m_%d")
        end_date = df.index[-1].strftime("%Y_%m_%d")

        # Create directory
        save_dir = self.data_dir / symbol_safe / "multi_tf" / f"{start_date}-{end_date}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # STRICT CHRONOLOGICAL SPLIT: oldest data for training, newest for evaluation
        # This is CRITICAL for time series to avoid look-ahead bias
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()  # First 80% (oldest)
        eval_df = df.iloc[split_idx:].copy()   # Last 20% (newest)

        # Validate no overlap
        if len(train_df) > 0 and len(eval_df) > 0:
            train_end = train_df.index[-1]
            eval_start = eval_df.index[0]
            if train_end >= eval_start:
                raise ValueError(f"Data leakage detected! Train end ({train_end}) >= Eval start ({eval_start})")

        # Save parquet files
        train_path = save_dir / "train.parquet"
        eval_path = save_dir / "eval.parquet"

        train_df.to_parquet(train_path, engine='pyarrow', compression='snappy')
        eval_df.to_parquet(eval_path, engine='pyarrow', compression='snappy')

        # Save metadata with split dates for verification
        train_start_date = train_df.index[0].strftime("%Y-%m-%d %H:%M") if len(train_df) > 0 else "N/A"
        train_end_date = train_df.index[-1].strftime("%Y-%m-%d %H:%M") if len(train_df) > 0 else "N/A"
        eval_start_date = eval_df.index[0].strftime("%Y-%m-%d %H:%M") if len(eval_df) > 0 else "N/A"
        eval_end_date = eval_df.index[-1].strftime("%Y-%m-%d %H:%M") if len(eval_df) > 0 else "N/A"

        metadata = DownloadMetadata(
            ticker=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframes=["1h", "1d"],
            base_timeframe="1h",
            rows=len(df),
            created_at=datetime.now().isoformat(),
            exchange=self.exchange_id,
        )

        # Extended metadata with split info
        metadata_dict = asdict(metadata)
        metadata_dict["split_info"] = {
            "method": "chronological_strict",
            "train_ratio": train_ratio,
            "train_rows": len(train_df),
            "train_date_range": f"{train_start_date} to {train_end_date}",
            "eval_rows": len(eval_df),
            "eval_date_range": f"{eval_start_date} to {eval_end_date}",
        }

        metadata_path = save_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata_dict, indent=2))

        console.print(f"\n[green]Saved to: {save_dir}/[/green]")
        console.print(f"  [cyan]train.parquet[/cyan]: {len(train_df):,} rows ({train_ratio*100:.0f}%)")
        console.print(f"    Date range: {train_start_date} to {train_end_date}")
        console.print(f"  [cyan]eval.parquet[/cyan]:  {len(eval_df):,} rows ({(1-train_ratio)*100:.0f}%)")
        console.print(f"    Date range: {eval_start_date} to {eval_end_date}")
        console.print(f"  [cyan]metadata.json[/cyan]")
        console.print(f"  [yellow]Split method: CHRONOLOGICAL (no data leakage)[/yellow]")

        return save_dir

    def download_and_save(
        self,
        symbol: str,
        days: int = 365 * 2,
        train_ratio: float = 0.8,
    ) -> Optional[Path]:
        """
        Download multi-timeframe crypto data and save in one step.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            days: Days of history
            train_ratio: Train/eval split ratio

        Returns:
            Path to saved directory or None if failed
        """
        df = self.download_multi_timeframe(symbol, days)
        if df is not None:
            return self.save(df, symbol, train_ratio)
        return None

    def list_available_symbols(self, quote: str = "USDT") -> list[str]:
        """
        List available trading pairs on Binance.

        Args:
            quote: Quote currency to filter by (default: USDT)

        Returns:
            List of available symbols
        """
        try:
            self.exchange.load_markets()
            symbols = [
                s for s in self.exchange.symbols
                if s.endswith(f"/{quote}") and self.exchange.markets[s].get('spot', False)
            ]
            return sorted(symbols)
        except Exception as e:
            logger.error(f"Failed to list symbols: {e}")
            return []


def load_multi_tf_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load saved multi-timeframe data.

    Args:
        data_path: Path to data directory containing train.parquet, eval.parquet, metadata.json

    Returns:
        Tuple of (train_df, eval_df, metadata)
    """
    data_path = Path(data_path)

    train_df = pd.read_parquet(data_path / "train.parquet")
    eval_df = pd.read_parquet(data_path / "eval.parquet")

    with open(data_path / "metadata.json") as f:
        metadata = json.load(f)

    console.print(f"[green]Loaded crypto data from: {data_path}[/green]")
    console.print(f"  Train: {len(train_df):,} rows")
    console.print(f"  Eval:  {len(eval_df):,} rows")
    console.print(f"  Timeframes: {metadata.get('timeframes', ['1h', '1d'])}")

    return train_df, eval_df, metadata


# Keep YahooDownloader as alias for backwards compatibility (deprecated)
class YahooDownloader(BinanceDownloader):
    """
    DEPRECATED: Use BinanceDownloader instead.
    This class is kept for backwards compatibility only.
    """
    def __init__(self, *args, **kwargs):
        console.print("[yellow]⚠ YahooDownloader is DEPRECATED. Use BinanceDownloader for crypto data.[/yellow]")
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    # Test download
    console.print("[bold cyan]═══ Testing Binance Multi-Timeframe Download ═══[/bold cyan]\n")

    downloader = BinanceDownloader()

    # List available symbols
    console.print("[cyan]Available USDT pairs (sample):[/cyan]")
    symbols = downloader.list_available_symbols("USDT")[:10]
    for s in symbols:
        console.print(f"  - {s}")

    # Test with BTC/USDT
    console.print("\n[cyan]Testing with BTC/USDT (30 days)...[/cyan]\n")

    try:
        df = downloader.download_multi_timeframe(
            symbol="BTC/USDT",
            days=30,  # Short test
            show_progress=True,
        )

        if df is not None:
            console.print(f"\n[bold green]Success![/bold green]")
            console.print(f"Shape: {df.shape}")
            console.print(f"\nColumns:")
            for col in df.columns:
                console.print(f"  - {col}")

            console.print(f"\nFirst 3 rows:")
            console.print(df.head(3).to_string())

            console.print(f"\nLast 3 rows:")
            console.print(df.tail(3).to_string())

            # Test save
            console.print("\n[cyan]Testing save...[/cyan]")
            save_path = downloader.save(df, "BTC/USDT", train_ratio=0.8)

            # Test load
            console.print("\n[cyan]Testing load...[/cyan]")
            train_df, eval_df, metadata = load_multi_tf_data(save_path)
            console.print(f"Metadata: {json.dumps(metadata, indent=2)}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
