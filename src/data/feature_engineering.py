"""
Feature Engineering Module
==========================
Technical indicators and feature generation for crypto swing trading.
Supports multi-timeframe data (1h, 1d).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


# Default features to use (17 features per timeframe)
# ALL FEATURES ARE STATIONARY OR NORMALIZED - NO ABSOLUTE PRICES
DEFAULT_FEATURES = [
    # Log returns (stationary) - 3
    'log_return',
    'log_return_high',
    'log_return_low',
    # Normalized indicators - 5
    'rsi_norm',
    'macd_norm',
    'macd_hist_norm',
    'bb_position',
    'atr_pct',  # Changed from atr_norm: ATR as percentage of price
    # Volume - 2
    'volume_norm',
    'volume_ratio',
    # Momentum - 3
    'return_5',
    'return_10',
    'return_20',
    # Candle - 2
    'candle_body_ratio',
    'candle_direction',
    # Market Regime (NEW) - 2
    'dist_to_sma200',  # Distance to SMA 200 (trend context)
    'volatility_regime',  # Volatility regime indicator
]


def add_technical_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV DataFrame.

    Indicators added:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - ATR (Average True Range)
    - Log Returns
    - Volume normalized

    Args:
        df: DataFrame with OHLCV columns
        rsi_period: RSI calculation period
        macd_fast: MACD fast EMA period
        macd_slow: MACD slow EMA period
        macd_signal: MACD signal line period
        bb_period: Bollinger Bands period
        bb_std: Bollinger Bands standard deviation
        atr_period: ATR calculation period

    Returns:
        DataFrame with added indicators
    """
    df = df.copy()

    # Ensure column names are lowercase
    df.columns = [col.lower() for col in df.columns]

    # Validate required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # =========================================================================
    # LOGARITHMIC RETURNS (Critical for stationarity)
    # =========================================================================
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_high'] = np.log(df['high'] / df['high'].shift(1))
    df['log_return_low'] = np.log(df['low'] / df['low'].shift(1))

    # =========================================================================
    # RSI - Relative Strength Index
    # =========================================================================
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    # Normalize RSI to [-1, 1] for neural network
    df['rsi_norm'] = (df['rsi'] - 50) / 50

    # =========================================================================
    # MACD - Moving Average Convergence Divergence
    # =========================================================================
    macd = ta.macd(
        df['close'],
        fast=macd_fast,
        slow=macd_slow,
        signal=macd_signal,
    )
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]  # MACD line
        df['macd_signal'] = macd.iloc[:, 1]  # Signal line
        df['macd_hist'] = macd.iloc[:, 2]  # Histogram

        # Normalize MACD relative to price
        df['macd_norm'] = df['macd'] / df['close'] * 100
        df['macd_hist_norm'] = df['macd_hist'] / df['close'] * 100

    # =========================================================================
    # BOLLINGER BANDS
    # =========================================================================
    bbands = ta.bbands(df['close'], length=bb_period, std=bb_std)
    if bbands is not None:
        df['bb_lower'] = bbands.iloc[:, 0]
        df['bb_mid'] = bbands.iloc[:, 1]
        df['bb_upper'] = bbands.iloc[:, 2]
        df['bb_bandwidth'] = bbands.iloc[:, 3] if bbands.shape[1] > 3 else None
        df['bb_percent'] = bbands.iloc[:, 4] if bbands.shape[1] > 4 else None

        # Calculate BB position normalized [-1, 1]
        # -1 = at lower band, 0 = at middle, 1 = at upper band
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (
            2 * (df['close'] - df['bb_lower']) / bb_range - 1
        ).clip(-2, 2)

    # =========================================================================
    # ATR - Average True Range (Volatility)
    # =========================================================================
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    # ATR as percentage of price (better for normalization)
    df['atr_pct'] = df['atr'] / df['close']
    # Keep atr_norm for backward compatibility
    df['atr_norm'] = df['atr_pct']

    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================
    # Volume normalized by 50-period SMA (log scale for stability)
    volume_sma_50 = df['volume'].rolling(window=50).mean()
    df['volume_norm'] = np.log1p(
        df['volume'] / volume_sma_50.replace(0, np.nan)
    ).clip(-3, 3)

    # Volume SMA ratio (20-period)
    volume_sma_20 = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = (df['volume'] / volume_sma_20).clip(0, 5)

    # =========================================================================
    # PRICE MOMENTUM
    # =========================================================================
    # Returns over different periods
    for period in [5, 10, 20]:
        df[f'return_{period}'] = (
            df['close'] / df['close'].shift(period) - 1
        ).clip(-0.5, 0.5)

    # =========================================================================
    # CANDLE PATTERNS
    # =========================================================================
    # Body size relative to range
    candle_range = df['high'] - df['low']
    candle_body = abs(df['close'] - df['open'])
    df['candle_body_ratio'] = (candle_body / candle_range.replace(0, np.nan)).clip(0, 1)

    # Direction (bullish/bearish)
    df['candle_direction'] = np.sign(df['close'] - df['open'])

    # =========================================================================
    # MARKET REGIME INDICATORS (NEW - Critical for context)
    # =========================================================================

    # SMA 200 - Long-term trend indicator
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # Distance to SMA 200 - Normalized relative position
    # Positive = above SMA (bullish), Negative = below SMA (bearish)
    # Formula: (close / sma_200) - 1
    df['dist_to_sma200'] = (df['close'] / df['sma_200'] - 1).clip(-0.5, 0.5)

    # Volatility Regime - Compare current ATR to historical ATR
    # High volatility = 1, Low volatility = -1, Normal = 0
    atr_sma_50 = df['atr'].rolling(window=50).mean()
    df['volatility_regime'] = ((df['atr'] / atr_sma_50) - 1).clip(-1, 1)

    # SMA 50 for shorter-term trend context
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['dist_to_sma50'] = (df['close'] / df['sma_50'] - 1).clip(-0.3, 0.3)

    # Trend strength: SMA 50 vs SMA 200 crossover
    # Positive = bullish (golden cross), Negative = bearish (death cross)
    df['trend_strength'] = (df['sma_50'] / df['sma_200'] - 1).clip(-0.2, 0.2)

    return df


def add_technical_indicators_multi_tf(
    df: pd.DataFrame,
    timeframes: list[str] = ['1h', '1d'],
) -> pd.DataFrame:
    """
    Add technical indicators for multi-timeframe aligned crypto data.

    Default timeframes for crypto swing trading:
    - 1h (hourly): Base timeframe for entries/exits
    - 1d (daily): Higher timeframe context

    Expects DataFrame with columns suffixed by timeframe:
    - open_1h, high_1h, low_1h, close_1h, volume_1h
    - open_1d, high_1d, low_1d, close_1d, volume_1d

    Args:
        df: Aligned multi-timeframe DataFrame
        timeframes: List of timeframe suffixes (default: ['1h', '1d'])

    Returns:
        DataFrame with technical indicators for each timeframe
    """
    df = df.copy()

    console.print(f"[cyan]Adding technical indicators for {len(timeframes)} timeframes...[/cyan]")

    for tf in timeframes:
        console.print(f"  Processing {tf}...")

        # Extract OHLCV for this timeframe
        ohlcv_cols = {
            'open': f'open_{tf}',
            'high': f'high_{tf}',
            'low': f'low_{tf}',
            'close': f'close_{tf}',
            'volume': f'volume_{tf}',
        }

        # Check columns exist
        missing = [v for v in ohlcv_cols.values() if v not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for {tf}: {missing}")

        # Create temporary DataFrame with standard column names
        tf_df = pd.DataFrame({
            'open': df[ohlcv_cols['open']],
            'high': df[ohlcv_cols['high']],
            'low': df[ohlcv_cols['low']],
            'close': df[ohlcv_cols['close']],
            'volume': df[ohlcv_cols['volume']],
        }, index=df.index)

        # Add technical indicators
        tf_df = add_technical_indicators(tf_df)

        # Copy indicators back with timeframe suffix
        # ALL features are stationary/normalized - no absolute prices
        indicator_cols = [
            # Log returns (stationary)
            'log_return', 'log_return_high', 'log_return_low',
            # Normalized oscillators
            'rsi_norm', 'macd_norm', 'macd_hist_norm',
            'bb_position',
            # Volatility (percentage-based)
            'atr_pct', 'atr_norm',
            # Volume (normalized)
            'volume_norm', 'volume_ratio',
            # Momentum (percentage returns)
            'return_5', 'return_10', 'return_20',
            # Candle patterns
            'candle_body_ratio', 'candle_direction',
            # Market Regime (NEW)
            'dist_to_sma200', 'volatility_regime',
            'dist_to_sma50', 'trend_strength',
            # Raw ATR for dynamic risk management (optional)
            'atr',
        ]

        for col in indicator_cols:
            if col in tf_df.columns:
                df[f'{col}_{tf}'] = tf_df[col]

    n_indicators = len(timeframes) * len(DEFAULT_FEATURES)
    console.print(f"[green]Added {n_indicators} technical features ({len(timeframes)} timeframes x {len(DEFAULT_FEATURES)} features)[/green]")

    return df


def prepare_features_for_env(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare and select features for the trading environment (single timeframe).

    Args:
        df: DataFrame with technical indicators
        feature_columns: Specific columns to use (None = auto-select)
        dropna: Whether to drop rows with NaN values

    Returns:
        Tuple of (processed DataFrame, list of feature column names)
    """
    df = df.copy()

    # Default feature selection (stationary/normalized features)
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURES

    # Filter to available columns
    available = [col for col in feature_columns if col in df.columns]
    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        logger.warning(f"Missing features: {missing}")

    if not available:
        raise ValueError("No features available for training")

    # Select features + OHLCV (needed for environment)
    ohlcv = ['open', 'high', 'low', 'close', 'volume']
    all_cols = ohlcv + available
    df = df[[col for col in all_cols if col in df.columns]]

    # Drop NaN rows
    if dropna:
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            console.print(
                f"[yellow]Dropped {dropped} rows with NaN values[/yellow]"
            )

    # Replace any infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    console.print(f"[green]Prepared {len(available)} features, {len(df)} samples[/green]")

    return df, available


def prepare_features_for_env_multi_tf(
    df: pd.DataFrame,
    timeframes: list[str] = ['1h', '1d'],
    feature_columns: Optional[list[str]] = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Prepare and select features for crypto multi-timeframe trading environment.

    Default: 1h (base) + 1d (context) for crypto swing trading.

    Args:
        df: DataFrame with multi-timeframe technical indicators
        timeframes: List of timeframe suffixes (default: ['1h', '1d'])
        feature_columns: Base feature names (will be suffixed with timeframe)
        dropna: Whether to drop rows with NaN values

    Returns:
        Tuple of (processed DataFrame, dict mapping timeframe to feature columns)
    """
    df = df.copy()

    # Default feature selection
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURES

    console.print(f"[cyan]Preparing features for {len(timeframes)} timeframes...[/cyan]")

    # Build feature map: {timeframe: [feature columns]}
    feature_map: dict[str, list[str]] = {}

    for tf in timeframes:
        tf_features = []
        for base_feat in feature_columns:
            col_name = f'{base_feat}_{tf}'
            if col_name in df.columns:
                tf_features.append(col_name)
            else:
                logger.warning(f"Missing feature: {col_name}")
        feature_map[tf] = tf_features
        console.print(f"  {tf}: {len(tf_features)} features")

    # Collect all feature columns
    all_feature_cols = []
    for tf_features in feature_map.values():
        all_feature_cols.extend(tf_features)

    # Collect OHLCV columns for base timeframe (needed for trading)
    base_tf = timeframes[0]  # Use first timeframe as base
    ohlcv_cols = [
        f'open_{base_tf}', f'high_{base_tf}', f'low_{base_tf}',
        f'close_{base_tf}', f'volume_{base_tf}',
    ]

    # Also include OHLCV for all timeframes (for reference)
    all_ohlcv = []
    for tf in timeframes:
        all_ohlcv.extend([
            f'open_{tf}', f'high_{tf}', f'low_{tf}',
            f'close_{tf}', f'volume_{tf}',
        ])

    # Select columns
    keep_cols = [col for col in all_ohlcv + all_feature_cols if col in df.columns]
    df = df[keep_cols]

    # Drop NaN rows
    if dropna:
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            console.print(
                f"[yellow]Dropped {dropped} rows with NaN values (warmup period)[/yellow]"
            )

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    total_features = sum(len(feats) for feats in feature_map.values())
    console.print(f"[green]Prepared {total_features} features total, {len(df)} samples[/green]")

    return df, feature_map


def get_feature_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Get statistics for features (for debugging/validation).

    Args:
        df: DataFrame with features
        features: List of feature column names

    Returns:
        DataFrame with feature statistics
    """
    stats = []
    for feat in features:
        if feat in df.columns:
            col = df[feat]
            stats.append({
                'feature': feat,
                'mean': col.mean(),
                'std': col.std(),
                'min': col.min(),
                'max': col.max(),
                'nan_count': col.isna().sum(),
            })

    return pd.DataFrame(stats)


def get_feature_stats_multi_tf(
    df: pd.DataFrame,
    feature_map: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """
    Get statistics for multi-timeframe features.

    Args:
        df: DataFrame with features
        feature_map: Dict mapping timeframe to feature columns

    Returns:
        Dict mapping timeframe to stats DataFrame
    """
    stats_by_tf = {}
    for tf, features in feature_map.items():
        stats_by_tf[tf] = get_feature_stats(df, features)
    return stats_by_tf


if __name__ == "__main__":
    # Test with sample crypto data (1h, 1d timeframes)
    console.print("[bold]Testing Crypto Feature Engineering (1h, 1d)[/bold]")

    # Create sample multi-timeframe crypto data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="h")  # Hourly frequency

    # Simulate BTC-like price
    price = 50000 + np.cumsum(np.random.randn(n) * 100)

    # Create aligned data (1h base + 1d context)
    df = pd.DataFrame(index=dates)

    for tf in ['1h', '1d']:
        df[f'open_{tf}'] = price + np.random.randn(n) * 10
        df[f'high_{tf}'] = price + abs(np.random.randn(n) * 50)
        df[f'low_{tf}'] = price - abs(np.random.randn(n) * 50)
        df[f'close_{tf}'] = price + np.random.randn(n) * 10
        df[f'volume_{tf}'] = np.random.randint(100, 1000, n)

    console.print(f"Raw crypto data shape: {df.shape}")

    # Add multi-timeframe indicators
    df_features = add_technical_indicators_multi_tf(df)
    console.print(f"With features shape: {df_features.shape}")

    # Prepare for environment
    df_prepared, feature_map = prepare_features_for_env_multi_tf(df_features)
    console.print(f"Prepared shape: {df_prepared.shape}")

    console.print("\n[bold]Feature map:[/bold]")
    for tf, features in feature_map.items():
        console.print(f"  {tf}: {len(features)} features")

    # Show stats for hourly timeframe
    stats = get_feature_stats_multi_tf(df_prepared, feature_map)
    console.print("\n[bold]Feature Statistics (1h):[/bold]")
    console.print(stats['1h'].to_string())

    console.print("\n[green]Crypto feature engineering test complete![/green]")
