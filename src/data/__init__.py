"""
Data Module
============
Data downloading and feature engineering for multi-timeframe trading.
"""

from src.data.data_downloader import YahooDownloader, load_multi_tf_data
from src.data.feature_engineering import (
    add_technical_indicators,
    add_technical_indicators_multi_tf,
    prepare_features_for_env,
    prepare_features_for_env_multi_tf,
    get_feature_stats,
    get_feature_stats_multi_tf,
)

__all__ = [
    "YahooDownloader",
    "load_multi_tf_data",
    "add_technical_indicators",
    "add_technical_indicators_multi_tf",
    "prepare_features_for_env",
    "prepare_features_for_env_multi_tf",
    "get_feature_stats",
    "get_feature_stats_multi_tf",
]
