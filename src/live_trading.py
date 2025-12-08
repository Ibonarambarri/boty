"""
Live Trading Module
===================
Connects to a cryptocurrency exchange and executes trades based on the trained agent's decisions.

**DISCLAIMER**: This is a placeholder module. Live trading is extremely risky.
Do not use this with real money without extensive testing and understanding the risks.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from src.feature_engineering import add_technical_indicators_multi_tf, prepare_features_for_env_multi_tf

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    A class to run a trained DRL agent in a live market.
    """

    def __init__(
        self,
        model_path: str,
        ticker: str,
        exchange_id: str = 'binance',
        timeframe: str = '1h',
        trade_amount: float = 100.0, # Amount in quote currency (e.g., USD)
    ) -> None:
        """
        Initialize the LiveTrader.

        Args:
            model_path: Path to the trained RecurrentPPO model.
            ticker: The market symbol (e.g., 'BTC/USDT').
            exchange_id: The ID of the exchange to connect to (e.g., 'binance', 'coinbasepro').
            timeframe: The primary timeframe for fetching data (e.g., '1h', '4h', '1d').
            trade_amount: The amount of quote currency to use for each trade.
        """
        self.model_path = model_path
        self.ticker = ticker
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.trade_amount = trade_amount

        # --- Exchange setup ---
        self.exchange = self._init_exchange()
        if not self.exchange:
            raise ConnectionError("Failed to initialize exchange.")

        # --- Model loading ---
        self.model = RecurrentPPO.load(model_path)

        # --- State variables ---
        self.lstm_states: Optional[np.ndarray] = None
        self.current_position: str = 'NONE' # 'NONE', 'LONG', 'SHORT'

    def _init_exchange(self) -> Optional[ccxt.Exchange]:
        """Initialize the CCXT exchange instance with API keys."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'apiKey': os.getenv('EXCHANGE_API_KEY'),
                'secret': os.getenv('EXCHANGE_SECRET_KEY'),
                # 'password': os.getenv('EXCHANGE_PASSWORD'), # For some exchanges
                'options': {
                    'defaultType': 'spot',
                },
            })
            # Use sandbox if available
            if exchange.has['sandbox']:
                exchange.set_sandbox_mode(True)
                logger.info(f"Using {self.exchange_id} sandbox environment.")
            
            exchange.load_markets()
            logger.info(f"Successfully connected to {self.exchange_id}.")
            return exchange
        except (AttributeError, ccxt.BaseError) as e:
            logger.error(f"Error initializing exchange: {e}")
            logger.error("Please ensure your API keys are set as environment variables (EXCHANGE_API_KEY, EXCHANGE_SECRET_KEY).")
            return None

    def _fetch_realtime_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch the latest OHLCV data for all required timeframes.
        
        This is a simplified implementation. A robust system would use websockets
        and a proper data pipeline to construct multi-timeframe data.
        """
        try:
            logger.info(f"Fetching latest data for {self.ticker}...")
            # For this example, we fetch a fixed number of candles for each timeframe
            # and re-align them. This is inefficient for live trading.
            
            # TODO: Implement a more robust data fetching and alignment strategy.
            # This placeholder fetches the last 200 candles for each timeframe and
            # assumes this is enough to generate the required features.
            
            ohlcv_1d = self.exchange.fetch_ohlcv(self.ticker, '1d', limit=200)
            ohlcv_1wk = self.exchange.fetch_ohlcv(self.ticker, '1w', limit=200)
            ohlcv_1mo = self.exchange.fetch_ohlcv(self.ticker, '1M', limit=200)

            # Convert to DataFrame (this is a complex task not fully implemented here)
            # A real implementation would need to align these timeframes correctly.
            logger.warning("Data fetching is a placeholder. Multi-timeframe alignment is not implemented.")
            
            # Using only the primary timeframe for this placeholder
            ohlcv = self.exchange.fetch_ohlcv(self.ticker, self.timeframe, limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # This is a major simplification. We are creating fake multi-timeframe data.
            # A real implementation is much more complex.
            df_multi = pd.DataFrame(index=df.index)
            for tf_suffix in ['1d', '1wk', '1mo']:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_multi[f'{col}_{tf_suffix}'] = df[col]

            return df_multi

        except ccxt.BaseError as e:
            logger.error(f"Error fetching data: {e}")
            return None

    def _preprocess_data(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Preprocess the data to create features for the model."""
        try:
            df = add_technical_indicators_multi_tf(df)
            df, feature_map = prepare_features_for_env_multi_tf(df)
            
            # Get the latest feature set
            latest_features = df.iloc[-1]

            # The environment expects a specific shape, which we mimic here
            # In a real scenario, the VecNormalize stats would need to be loaded and applied.
            logger.warning("Preprocessing is a placeholder. Normalization stats are not being used.")

            obs_parts = []
            for tf in ['1d', '1wk', '1mo']:
                tf_cols = [c for c in df.columns if c.endswith(f'_{tf}') and c in feature_map[tf]]
                obs_parts.append(latest_features[tf_cols].values)

            obs = np.concatenate(obs_parts).astype(np.float32)
            # The model expects a batch dimension
            return obs.reshape(1, -1)

        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None

    def _execute_trade(self, action: int) -> None:
        """Execute a trade based on the model's action."""
        # This is a placeholder for trade execution logic.
        # A real implementation needs to handle order types, amounts, and error checking.
        
        # Action mapping: 0=HOLD, 1=LONG, 2=SHORT
        if action == 1 and self.current_position != 'LONG':
            logger.info("Decision: Close SHORT and go LONG")
            # TODO: Close existing short position
            # TODO: Open new long position
            logger.info(f"Executing BUY for {self.trade_amount} {self.ticker.split('/')[1]}")
            self.current_position = 'LONG'

        elif action == 2 and self.current_position != 'SHORT':
            logger.info("Decision: Close LONG and go SHORT")
            # TODO: Close existing long position
            # TODO: Open new short position
            logger.info(f"Executing SELL for {self.trade_amount} {self.ticker.split('/')[1]}")
            self.current_position = 'SHORT'
            
        else:
            logger.info("Decision: HOLD")

    def run(self) -> None:
        """The main live trading loop."""
        logger.info("Starting live trading session. Press Ctrl+C to stop.")
        logger.warning("This is a simplified demo. DO NOT use with real funds.")

        while True:
            try:
                # 1. Fetch Data
                df = self._fetch_realtime_data()
                if df is None:
                    time.sleep(60) # Wait before retrying
                    continue

                # 2. Preprocess Data to get latest observation
                obs = self._preprocess_data(df)
                if obs is None:
                    time.sleep(60)
                    continue

                # 3. Get Action from Model
                episode_starts = np.ones((1,), dtype=bool) if self.lstm_states is None else np.zeros((1,), dtype=bool)
                action, self.lstm_states = self.model.predict(
                    obs,
                    state=self.lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )

                # 4. Execute Trade
                self._execute_trade(action.item())
                
                # Wait for the next candle
                # This simple loop is not robust. A real system would use scheduled tasks.
                logger.info(f"Waiting for next {self.timeframe} candle...")
                time.sleep(self.exchange.parse_timeframe(self.timeframe) * 1000)

            except KeyboardInterrupt:
                logger.info("Stopping live trading session.")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                time.sleep(60)
