"""
Data Downloader Module
======================
Yahoo Finance multi-timeframe data downloader.
Supports 1d, 1wk, 1mo timeframes with 10+ years of history.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Literal

import pandas as pd
import numpy as np
import yfinance as yf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()
logger = logging.getLogger(__name__)


TimeframeType = Literal["1d", "1wk", "1mo"]


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


class YahooDownloader:
    """
    Multi-timeframe data downloader using Yahoo Finance.

    Timeframes supported (all with 10+ years of history):
    - 1d (daily): Base timeframe
    - 1wk (weekly): Aggregated weekly data
    - 1mo (monthly): Aggregated monthly data

    Features:
    - Downloads 3 timeframes in parallel
    - Aligns higher timeframes to daily base
    - Saves in parquet format with train/eval split
    - No API key required
    """

    TIMEFRAMES = ['1d', '1wk', '1mo']

    def __init__(
        self,
        data_dir: str | Path = "data",
    ):
        """
        Initialize the downloader.

        Args:
            data_dir: Directory for processed data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _download_timeframe(
        self,
        ticker: str,
        interval: str,
        years: int,
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a single timeframe.

        Args:
            ticker: Stock/crypto symbol
            interval: "1d", "1wk", or "1mo"
            years: Years of history to download

        Returns:
            DataFrame with OHLCV data or None
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)

            # Download using yfinance
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,  # Adjust for splits/dividends
            )

            if df.empty:
                logger.warning(f"No data returned for {ticker} {interval}")
                return None

            # Ensure column names are lowercase
            df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]

            # Keep only OHLCV columns
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in ohlcv_cols if c in df.columns]
            df = df[available_cols]

            # Sort by date
            df = df.sort_index()

            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]

            return df

        except Exception as e:
            logger.error(f"Failed to download {ticker} {interval}: {e}")
            return None

    def _align_timeframes(
        self,
        df_1d: pd.DataFrame,
        df_1wk: pd.DataFrame,
        df_1mo: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align 3 timeframes by repeating higher timeframe values.

        Each row of the result corresponds to a daily bar (base timeframe).
        Weekly values are repeated ~5 times per week.
        Monthly values are repeated ~20-22 times per month.

        Args:
            df_1d: Daily OHLCV data (base)
            df_1wk: Weekly OHLCV data
            df_1mo: Monthly OHLCV data

        Returns:
            Aligned DataFrame with suffixed columns
        """
        # Rename columns with suffixes
        df_1d_renamed = df_1d.add_suffix('_1d')
        df_1wk_renamed = df_1wk.add_suffix('_1wk')
        df_1mo_renamed = df_1mo.add_suffix('_1mo')

        # Start with daily data as base
        aligned = df_1d_renamed.copy()

        # Create alignment keys for daily data
        aligned['_week_start'] = aligned.index.to_period('W-SUN').start_time
        aligned['_month_start'] = aligned.index.to_period('M').start_time

        # Prepare weekly data - use week start as key
        df_1wk_for_merge = df_1wk_renamed.copy()
        df_1wk_for_merge['_week_start'] = df_1wk_for_merge.index.to_period('W-SUN').start_time
        # Remove duplicates keeping the first occurrence for each week
        df_1wk_for_merge = df_1wk_for_merge.drop_duplicates(subset=['_week_start'], keep='first')
        df_1wk_for_merge = df_1wk_for_merge.set_index('_week_start')

        # Prepare monthly data - use month start as key
        df_1mo_for_merge = df_1mo_renamed.copy()
        df_1mo_for_merge['_month_start'] = df_1mo_for_merge.index.to_period('M').start_time
        # Remove duplicates keeping the first occurrence for each month
        df_1mo_for_merge = df_1mo_for_merge.drop_duplicates(subset=['_month_start'], keep='first')
        df_1mo_for_merge = df_1mo_for_merge.set_index('_month_start')

        # Join weekly data using the week_start key
        aligned = aligned.join(df_1wk_for_merge, on='_week_start', how='left')

        # Join monthly data using the month_start key
        aligned = aligned.join(df_1mo_for_merge, on='_month_start', how='left')

        # Drop temporary alignment columns
        aligned = aligned.drop(columns=['_week_start', '_month_start'], errors='ignore')

        # Forward fill any gaps (for initial period)
        aligned = aligned.ffill()

        # Backward fill remaining NaNs at the start
        aligned = aligned.bfill()

        return aligned

    def download_multi_timeframe(
        self,
        ticker: str,
        years: int = 10,
        show_progress: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Download and align data for 3 timeframes.

        Args:
            ticker: Stock/crypto symbol (e.g., 'AAPL', 'BTC-USD')
            years: Years of history to download
            show_progress: Show progress indicator

        Returns:
            Aligned DataFrame with all timeframes, or None if failed
        """
        console.print(f"\n[bold]Downloading {ticker} multi-timeframe data[/bold]")
        console.print(f"  Timeframes: 1d (daily), 1wk (weekly), 1mo (monthly)")
        console.print(f"  History: {years} years")

        df_1d = None
        df_1wk = None
        df_1mo = None

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                # Download daily data
                task = progress.add_task("[cyan]Downloading 1d data...", total=None)
                df_1d = self._download_timeframe(ticker, "1d", years)
                progress.update(task, description="[green]✓ 1d data downloaded")

                # Download weekly data
                task = progress.add_task("[cyan]Downloading 1wk data...", total=None)
                df_1wk = self._download_timeframe(ticker, "1wk", years)
                progress.update(task, description="[green]✓ 1wk data downloaded")

                # Download monthly data
                task = progress.add_task("[cyan]Downloading 1mo data...", total=None)
                df_1mo = self._download_timeframe(ticker, "1mo", years)
                progress.update(task, description="[green]✓ 1mo data downloaded")
        else:
            console.print("  Downloading 1d data...")
            df_1d = self._download_timeframe(ticker, "1d", years)

            console.print("  Downloading 1wk data...")
            df_1wk = self._download_timeframe(ticker, "1wk", years)

            console.print("  Downloading 1mo data...")
            df_1mo = self._download_timeframe(ticker, "1mo", years)

        # Validate downloads
        if df_1d is None or df_1d.empty:
            console.print("[red]Failed to download daily data[/red]")
            return None

        if df_1wk is None or df_1wk.empty:
            console.print("[red]Failed to download weekly data[/red]")
            return None

        if df_1mo is None or df_1mo.empty:
            console.print("[red]Failed to download monthly data[/red]")
            return None

        console.print(f"\n[green]Downloaded data:[/green]")
        console.print(f"  1d:  {len(df_1d):,} rows ({df_1d.index[0].date()} to {df_1d.index[-1].date()})")
        console.print(f"  1wk: {len(df_1wk):,} rows ({df_1wk.index[0].date()} to {df_1wk.index[-1].date()})")
        console.print(f"  1mo: {len(df_1mo):,} rows ({df_1mo.index[0].date()} to {df_1mo.index[-1].date()})")

        # Align timeframes
        console.print("\n[cyan]Aligning timeframes...[/cyan]")
        aligned = self._align_timeframes(df_1d, df_1wk, df_1mo)

        # Remove rows with NaN (start of data where alignment fails)
        initial_len = len(aligned)
        aligned = aligned.dropna()
        dropped = initial_len - len(aligned)

        if dropped > 0:
            console.print(f"  [yellow]Dropped {dropped} rows with incomplete data[/yellow]")

        console.print(f"[green]Aligned data: {len(aligned):,} rows[/green]")
        console.print(f"  Date range: {aligned.index[0].date()} to {aligned.index[-1].date()}")

        return aligned

    def save(
        self,
        df: pd.DataFrame,
        ticker: str,
        train_ratio: float = 0.8,
    ) -> Path:
        """
        Save aligned DataFrame with train/eval split.

        Directory structure: data/{ticker}/multi_tf/{date_range}/
            - train.parquet (80% oldest)
            - eval.parquet (20% newest)
            - metadata.json

        Args:
            df: Aligned DataFrame
            ticker: Ticker symbol
            train_ratio: Ratio for training split

        Returns:
            Path to saved directory
        """
        # Format date range
        start_date = df.index[0].strftime("%Y_%m_%d")
        end_date = df.index[-1].strftime("%Y_%m_%d")

        # Create directory
        save_dir = self.data_dir / ticker / "multi_tf" / f"{start_date}-{end_date}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Split temporally (older data for training, newer for evaluation)
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]

        # Save parquet files
        train_path = save_dir / "train.parquet"
        eval_path = save_dir / "eval.parquet"

        train_df.to_parquet(train_path, engine='pyarrow', compression='snappy')
        eval_df.to_parquet(eval_path, engine='pyarrow', compression='snappy')

        # Save metadata
        metadata = DownloadMetadata(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            timeframes=["1d", "1wk", "1mo"],
            base_timeframe="1d",
            rows=len(df),
            created_at=datetime.now().isoformat(),
        )
        metadata_path = save_dir / "metadata.json"
        metadata_path.write_text(json.dumps(asdict(metadata), indent=2))

        console.print(f"\n[green]Saved to: {save_dir}/[/green]")
        console.print(f"  [cyan]train.parquet[/cyan]: {len(train_df):,} rows ({train_ratio*100:.0f}%)")
        console.print(f"  [cyan]eval.parquet[/cyan]:  {len(eval_df):,} rows ({(1-train_ratio)*100:.0f}%)")
        console.print(f"  [cyan]metadata.json[/cyan]")

        return save_dir

    def download_and_save(
        self,
        ticker: str,
        years: int = 10,
        train_ratio: float = 0.8,
    ) -> Optional[Path]:
        """
        Download multi-timeframe data and save in one step.

        Args:
            ticker: Asset ticker symbol
            years: Years of history
            train_ratio: Train/eval split ratio

        Returns:
            Path to saved directory or None if failed
        """
        df = self.download_multi_timeframe(ticker, years)
        if df is not None:
            return self.save(df, ticker, train_ratio)
        return None


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

    console.print(f"[green]Loaded data from: {data_path}[/green]")
    console.print(f"  Train: {len(train_df):,} rows")
    console.print(f"  Eval:  {len(eval_df):,} rows")

    return train_df, eval_df, metadata


if __name__ == "__main__":
    # Test download
    console.print("[bold]Testing Yahoo Finance Multi-Timeframe Download[/bold]\n")

    downloader = YahooDownloader()

    # Test with AAPL
    console.print("[cyan]Testing with AAPL (5 years)...[/cyan]\n")

    try:
        df = downloader.download_multi_timeframe(
            ticker="AAPL",
            years=5,
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
            save_path = downloader.save(df, "AAPL_test", train_ratio=0.8)

            # Test load
            console.print("\n[cyan]Testing load...[/cyan]")
            train_df, eval_df, metadata = load_multi_tf_data(save_path)
            console.print(f"Metadata: {metadata}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
