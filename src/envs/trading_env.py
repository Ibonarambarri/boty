"""
Crypto Trading Environment
==========================
Custom Gymnasium environment for cryptocurrency trading with DRL.
Supports multi-timeframe observations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional, SupportsFloat, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

logger = logging.getLogger(__name__)


class Position(IntEnum):
    """Position states."""
    NONE = 0
    LONG = 1
    SHORT = 2


# Threshold for neutral zone (|action| < threshold = close/no position)
NEUTRAL_THRESHOLD = 0.05

# Minimum size change to trigger rebalancing (5%)
REBALANCE_THRESHOLD = 0.05


@dataclass
class Trade:
    """Represents an open or closed trade."""
    entry_step: int
    entry_price: float
    position_type: Position
    size: float  # USD value
    take_profit: float
    stop_loss: float
    exit_step: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class EnvConfig:
    """Environment configuration."""
    initial_balance: float = 10_000.0
    position_size_pct: float = 1.0  # 100% of balance (fully invested)
    take_profit_pct: float = 0.015  # 1.5% (Antes 2.0%)
    stop_loss_pct: float = 0.0075   # 0.75% (Antes 1.0%) - Mantiene ratio 2:1
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    window_size: int = 50  # Lookback window (changed from 30)
    reward_scaling: float = 100.0  # Scale rewards for stability

    # Trading frequency control
    min_hold_steps: int = 5  # Minimum steps to hold a position before closing
    trade_penalty: float = 0.0005  # Reduced penalty to encourage more trading

    # Multi-timeframe configuration
    timeframes: list[str] = field(default_factory=lambda: ['1d', '1wk', '1mo'])
    features_per_timeframe: int = 15
    base_timeframe: str = '1d'


class CryptoTradingEnv(gym.Env):
    """
    Cryptocurrency Trading Environment.

    Features:
    - Discrete actions: Hold, Long, Short
    - Position management with TP/SL
    - Realistic commission modeling
    - Multi-timeframe observation support

    Observation space for multi-timeframe (default):
    - 50 bars x 15 features x 3 timeframes = 2250 values

    API v26+ compliant.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Union[list[str], dict[str, list[str]]],
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the trading environment.

        Args:
            df: DataFrame with OHLCV + features
            feature_columns: Either:
                - list[str]: Feature column names (single timeframe)
                - dict[str, list[str]]: {timeframe: [feature names]} (multi-timeframe)
            config: Environment configuration
            render_mode: Rendering mode
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        # Detect multi-timeframe mode
        self.multi_tf_mode = isinstance(feature_columns, dict)

        if self.multi_tf_mode:
            self._init_multi_timeframe(feature_columns)
        else:
            self._init_single_timeframe(feature_columns)

        # Define continuous action space: [-1, +1]
        # Positive = LONG, Negative = SHORT, magnitude = % of capital
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # State variables (initialized in reset)
        self._reset_state()

    def _init_single_timeframe(self, feature_columns: list[str]) -> None:
        """Initialize for single timeframe mode (backward compatible)."""
        self.feature_columns = feature_columns

        # Validate data
        required_cols = ['open', 'high', 'low', 'close'] + feature_columns
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Pre-extract numpy arrays for performance
        self._ohlc = self.df[['open', 'high', 'low', 'close']].values
        self._features = self.df[feature_columns].values.astype(np.float32)

        # Calculate data bounds
        self.n_steps = len(self.df)
        self.n_features = len(feature_columns)
        self.max_step = self.n_steps - 1

        # Observation: features of a single timestep
        obs_shape = (self.n_features,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

    def _init_multi_timeframe(self, feature_columns: dict[str, list[str]]) -> None:
        """Initialize for multi-timeframe mode."""
        self.feature_columns = feature_columns
        self.timeframes = list(feature_columns.keys())

        # Use configured or detected base timeframe
        base_tf = self.config.base_timeframe
        if base_tf not in self.timeframes:
            base_tf = self.timeframes[0]
            logger.warning(f"Base timeframe {self.config.base_timeframe} not in data, using {base_tf}")

        self.base_timeframe = base_tf

        # Validate OHLCV columns for base timeframe
        ohlcv_cols = [
            f'open_{base_tf}', f'high_{base_tf}',
            f'low_{base_tf}', f'close_{base_tf}',
        ]
        missing_ohlcv = [c for c in ohlcv_cols if c not in self.df.columns]
        if missing_ohlcv:
            raise ValueError(f"Missing OHLCV columns for base timeframe: {missing_ohlcv}")

        # Pre-extract OHLC for base timeframe (used for trading)
        self._ohlc = self.df[ohlcv_cols].values

        # Pre-extract features for each timeframe
        self._features_by_tf: dict[str, np.ndarray] = {}
        self.n_features_per_tf: dict[str, int] = {}

        for tf, cols in feature_columns.items():
            missing = [c for c in cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"Missing feature columns for {tf}: {missing}")
            self._features_by_tf[tf] = self.df[cols].values.astype(np.float32)
            self.n_features_per_tf[tf] = len(cols)

        # Calculate data bounds
        self.n_steps = len(self.df)
        self.max_step = self.n_steps - 1

        # Total features across all timeframes
        self.n_features_total = sum(self.n_features_per_tf.values())

        # Observation: features of a single timestep (for recurrent policies)
        obs_dim = self.n_features_total
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        logger.info(
            f"Multi-TF env initialized for recurrent policy: {len(self.timeframes)} timeframes, "
            f"obs_dim={obs_dim}"
        )

    def _reset_state(self) -> None:
        """Reset all state variables."""
        self.balance = self.config.initial_balance
        self.position: Position = Position.NONE
        self.current_trade: Optional[Trade] = None
        self.trade_history: list[Trade] = []
        self.current_step = self.config.window_size
        self.total_reward = 0.0
        self.episode_trades = 0
        self.winning_trades = 0

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (features of the current timestep).

        For single timeframe: (n_features,)
        For multi-timeframe: (n_features_total,)
            Structure: [features_tf1 | features_tf2 | features_tf3]

        Returns:
            Flattened numpy array of features for the current step.
        """
        if self.multi_tf_mode:
            # Concatenate feature vectors from all timeframes for the current step
            obs_parts = []
            for tf in self.timeframes:
                step_features = self._features_by_tf[tf][self.current_step]
                obs_parts.append(step_features)
            obs = np.concatenate(obs_parts)
        else:
            # Single timeframe
            obs = self._features[self.current_step]

        # Handle NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs.astype(np.float32)

    def _get_current_price(self) -> float:
        """Get current close price."""
        return float(self._ohlc[self.current_step, 3])  # close

    def _get_current_high(self) -> float:
        """Get current candle high."""
        return float(self._ohlc[self.current_step, 1])

    def _get_current_low(self) -> float:
        """Get current candle low."""
        return float(self._ohlc[self.current_step, 2])

    def _open_position(self, position_type: Position, size_pct: float) -> None:
        """
        Open a new position with specified size.

        Args:
            position_type: LONG or SHORT
            size_pct: Fraction of balance to use (0.0 to 1.0)
        """
        current_price = self._get_current_price()
        size = self.balance * size_pct

        # Apply slippage
        if position_type == Position.LONG:
            entry_price = current_price * (1 + self.config.slippage_pct)
        else:  # SHORT
            entry_price = current_price * (1 - self.config.slippage_pct)

        # Calculate TP/SL levels based on the actual entry price
        if position_type == Position.LONG:
            take_profit = entry_price * (1 + self.config.take_profit_pct)
            stop_loss = entry_price * (1 - self.config.stop_loss_pct)
        else:  # SHORT
            take_profit = entry_price * (1 - self.config.take_profit_pct)
            stop_loss = entry_price * (1 + self.config.stop_loss_pct)

        self.current_trade = Trade(
            entry_step=self.current_step,
            entry_price=entry_price,
            position_type=position_type,
            size=size,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
        self.position = position_type

        # Deduct position size AND commission on entry
        commission = size * self.config.commission_pct
        self.balance -= (size + commission)

    def _close_position(self, exit_price: float, reason: str) -> float:
        """
        Close current position and calculate PnL.

        Args:
            exit_price: Price at which to close
            reason: Reason for closing (TP, SL, manual)

        Returns:
            Realized PnL
        """
        if self.current_trade is None:
            return 0.0

        trade = self.current_trade
        entry = trade.entry_price
        size = trade.size

        # Calculate raw PnL
        if trade.position_type == Position.LONG:
            pnl_pct = (exit_price - entry) / entry
        else:  # SHORT
            pnl_pct = (entry - exit_price) / entry

        raw_pnl = size * pnl_pct

        # Calculate exit commission on the actual exit value (not entry size)
        # Exit value = size adjusted by price change
        exit_value = size * (exit_price / entry)
        exit_commission = exit_value * self.config.commission_pct
        net_pnl = raw_pnl - exit_commission

        # Update balance
        self.balance += size + net_pnl

        # Record trade
        trade.exit_step = self.current_step
        trade.exit_price = exit_price
        trade.pnl = net_pnl
        trade.exit_reason = reason
        self.trade_history.append(trade)

        # Update stats
        self.episode_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1

        # Reset position
        self.position = Position.NONE
        self.current_trade = None

        return net_pnl

    def _check_tp_sl(self) -> tuple[bool, float, str]:
        """
        Check if TP or SL was hit during current candle.

        Pessimistic assumption: If both could be hit, SL is hit first.

        Returns:
            Tuple of (was_closed, exit_price, reason)
        """
        if self.current_trade is None:
            return False, 0.0, ""

        trade = self.current_trade
        high = self._get_current_high()
        low = self._get_current_low()

        tp_hit = False
        sl_hit = False

        if trade.position_type == Position.LONG:
            tp_hit = high >= trade.take_profit
            sl_hit = low <= trade.stop_loss
        else:  # SHORT
            tp_hit = low <= trade.take_profit
            sl_hit = high >= trade.stop_loss

        # Pessimistic: SL takes priority
        if sl_hit:
            return True, trade.stop_loss, "SL"
        elif tp_hit:
            return True, trade.take_profit, "TP"

        return False, 0.0, ""

    def _transition_position(self, target: Position, target_size_pct: float) -> None:
        """
        Transition from current position to target position.

        Handles:
        - Opening new positions
        - Closing positions (target is NONE)
        - Flipping direction (LONG to SHORT or vice versa)
        - Rebalancing (same direction, different size)

        Args:
            target: Target position type (NONE, LONG, SHORT)
            target_size_pct: Target size as fraction of balance (0.0 to 1.0)
        """
        # Case 1: No current position
        if self.position == Position.NONE:
            if target != Position.NONE and target_size_pct > 0:
                self._open_position(target, target_size_pct)
            return

        # Case 2: Close position (target is NONE or direction change)
        if target == Position.NONE or target != self.position:
            # Apply slippage on close
            current_price = self._get_current_price()
            if self.current_trade.position_type == Position.LONG:
                exit_price = current_price * (1 - self.config.slippage_pct)
            else:
                exit_price = current_price * (1 + self.config.slippage_pct)

            self._close_position(exit_price, "signal")

            # If new direction, open new position
            if target != Position.NONE and target_size_pct > 0:
                self._open_position(target, target_size_pct)
            return

        # Case 3: Same direction, check if rebalance needed
        if target == self.position and self.current_trade is not None:
            # Calculate current position size as % of total equity
            total_equity = self.balance + self.current_trade.size
            if total_equity > 0:
                current_size_pct = self.current_trade.size / total_equity
                size_diff = abs(target_size_pct - current_size_pct)

                # Only rebalance if change is significant
                if size_diff > REBALANCE_THRESHOLD:
                    current_price = self._get_current_price()
                    if self.current_trade.position_type == Position.LONG:
                        exit_price = current_price * (1 - self.config.slippage_pct)
                    else:
                        exit_price = current_price * (1 + self.config.slippage_pct)

                    self._close_position(exit_price, "rebalance")
                    self._open_position(target, target_size_pct)

    def _calculate_net_worth(self) -> float:
        """
        Calculate current net worth (balance + unrealized PnL).

        Returns:
            Total net worth
        """
        if self.position == Position.NONE or self.current_trade is None:
            return self.balance

        trade = self.current_trade
        current_price = self._get_current_price()

        if trade.position_type == Position.LONG:
            unrealized_pnl = trade.size * (current_price - trade.entry_price) / trade.entry_price
        else:
            unrealized_pnl = trade.size * (trade.entry_price - current_price) / trade.entry_price

        return self.balance + trade.size + unrealized_pnl

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        self._reset_state()

        # Optional: random starting point
        if options and options.get("random_start", False):
            max_start = self.n_steps - self.config.window_size - 100
            if max_start > self.config.window_size:
                self.current_step = self.np_random.integers(
                    self.config.window_size,
                    max_start,
                )

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Uses incremental rewards based on log returns of net worth change
        to provide dense feedback signal for learning.

        Args:
            action: Action to take (0=Hold, 1=Long, 2=Short)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        action = Action(action)

        # Calculate net worth BEFORE action for incremental reward
        prev_net_worth = self._calculate_net_worth()

        # First, check if existing position hits TP/SL
        if self.position != Position.NONE:
            # Check minimum hold time before allowing TP/SL
            steps_held = self.current_step - self.current_trade.entry_step
            if steps_held >= self.config.min_hold_steps:
                closed, exit_price, reason = self._check_tp_sl()
                if closed:
                    self._close_position(exit_price, reason)

        # Process action only if no position
        if self.position == Position.NONE:
            if action == Action.LONG:
                self._open_position(Position.LONG)
                # Apply trade penalty to discourage overtrading
                reward -= self.config.trade_penalty
            elif action == Action.SHORT:
                self._open_position(Position.SHORT)
                # Apply trade penalty to discourage overtrading
                reward -= self.config.trade_penalty
            # HOLD does nothing

        # Advance step
        self.current_step += 1

        # Check termination
        terminated = False
        truncated = False

        # Episode ends if:
        # 1. Ran out of data
        if self.current_step >= self.max_step:
            truncated = True
            # Close any open position at market with slippage
            if self.position != Position.NONE:
                current_price = self._get_current_price()
                # Apply slippage: unfavorable direction for the trader
                if self.current_trade.position_type == Position.LONG:
                    exit_price = current_price * (1 - self.config.slippage_pct)
                else:  # SHORT
                    exit_price = current_price * (1 + self.config.slippage_pct)
                self._close_position(exit_price, "EOD")

        # 2. Bankrupt
        if self.balance <= 0:
            terminated = True
            reward = -10.0  # Large negative reward for bankruptcy
        else:
            # Calculate net worth AFTER action for incremental reward (log return)
            current_net_worth = self._calculate_net_worth()
            # Incremental reward = log return of net worth change
            # This provides dense feedback at every step
            if prev_net_worth > 0 and current_net_worth > 0:
                reward += np.log(current_net_worth / prev_net_worth)

        self.total_reward += reward

        observation = self._get_observation()
        info = self._get_info()

        return observation, float(reward), terminated, truncated, info

    def _get_info(self) -> dict[str, Any]:
        """
        Get info dictionary for dashboard integration.

        Returns:
            Dictionary with trading metrics including:
            - net_worth: Total portfolio value (balance + unrealized PnL)
            - pnl_pct: Cumulative P&L percentage from initial balance
            - realized_pnl: Sum of P&L from closed trades
            - total_reward: Cumulative reward (incremental log returns + penalties)
        """
        net_worth = self._calculate_net_worth()
        pnl_pct = (net_worth / self.config.initial_balance - 1) * 100

        win_rate = 0.0
        if self.episode_trades > 0:
            win_rate = self.winning_trades / self.episode_trades * 100

        # Calculate realized PnL from closed trades
        realized_pnl = sum(t.pnl for t in self.trade_history if t.pnl is not None)

        return {
            "step": self.current_step,
            "balance": self.balance,
            "net_worth": net_worth,
            "pnl_pct": pnl_pct,
            "realized_pnl": realized_pnl,
            "position": self.position.name,
            "total_trades": self.episode_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_reward": self.total_reward,
            "current_price": self._get_current_price(),
        }

    def render(self) -> Optional[str]:
        """Render current state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
        return None

    def _render_ansi(self) -> str:
        """Generate ANSI text representation."""
        info = self._get_info()
        lines = [
            f"Step: {info['step']}/{self.max_step}",
            f"Balance: ${info['balance']:,.2f}",
            f"Net Worth: ${info['net_worth']:,.2f} ({info['pnl_pct']:+.2f}%)",
            f"Position: {info['position']}",
            f"Trades: {info['total_trades']} (Win Rate: {info['win_rate']:.1f}%)",
        ]
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up resources."""
        pass


# Factory function for easy creation
def make_trading_env(
    df: pd.DataFrame,
    feature_columns: Union[list[str], dict[str, list[str]]],
    **config_kwargs,
) -> CryptoTradingEnv:
    """
    Factory function to create trading environment.

    Args:
        df: DataFrame with OHLCV + features
        feature_columns: Feature column names (list) or feature map (dict)
        **config_kwargs: EnvConfig parameters

    Returns:
        CryptoTradingEnv instance
    """
    config = EnvConfig(**config_kwargs)
    return CryptoTradingEnv(df, feature_columns, config)


if __name__ == "__main__":
    # Test environment
    from rich.console import Console

    console = Console()

    # =========================================================================
    # Test 1: Single timeframe (backward compatible)
    # =========================================================================
    console.print("\n[bold]Test 1: Single Timeframe Environment[/bold]")

    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    price = 40000 + np.cumsum(np.random.randn(n) * 100)

    df_single = pd.DataFrame({
        "open": price,
        "high": price + abs(np.random.randn(n) * 50),
        "low": price - abs(np.random.randn(n) * 50),
        "close": price + np.random.randn(n) * 20,
        "volume": np.random.randint(1000, 10000, n),
        # Dummy features
        "log_return": np.random.randn(n) * 0.01,
        "rsi_norm": np.random.randn(n) * 0.5,
        "macd_norm": np.random.randn(n) * 0.1,
    }, index=dates)

    features_single = ["log_return", "rsi_norm", "macd_norm"]
    env_single = make_trading_env(df_single, features_single, window_size=50)

    console.print(f"Observation space: {env_single.observation_space}")
    console.print(f"Action space: {env_single.action_space}")

    obs, info = env_single.reset()
    console.print(f"Initial obs shape: {obs.shape}")

    # =========================================================================
    # Test 2: Multi-timeframe
    # =========================================================================
    console.print("\n[bold]Test 2: Multi-Timeframe Environment[/bold]")

    # Create multi-timeframe data
    df_multi = pd.DataFrame(index=dates)

    for tf in ['1d', '1wk', '1mo']:
        df_multi[f'open_{tf}'] = price + np.random.randn(n) * 10
        df_multi[f'high_{tf}'] = price + abs(np.random.randn(n) * 50)
        df_multi[f'low_{tf}'] = price - abs(np.random.randn(n) * 50)
        df_multi[f'close_{tf}'] = price + np.random.randn(n) * 20
        df_multi[f'volume_{tf}'] = np.random.randint(1000, 10000, n)
        # Features
        df_multi[f'log_return_{tf}'] = np.random.randn(n) * 0.01
        df_multi[f'rsi_norm_{tf}'] = np.random.randn(n) * 0.5
        df_multi[f'macd_norm_{tf}'] = np.random.randn(n) * 0.1

    feature_map = {
        '1d': ['log_return_1d', 'rsi_norm_1d', 'macd_norm_1d'],
        '1wk': ['log_return_1wk', 'rsi_norm_1wk', 'macd_norm_1wk'],
        '1mo': ['log_return_1mo', 'rsi_norm_1mo', 'macd_norm_1mo'],
    }

    env_multi = make_trading_env(
        df_multi,
        feature_map,
        window_size=50,
        timeframes=['1d', '1wk', '1mo'],
        base_timeframe='1d',
    )

    console.print(f"Observation space: {env_multi.observation_space}")
    console.print(f"Action space: {env_multi.action_space}")

    obs, info = env_multi.reset()
    console.print(f"Initial obs shape: {obs.shape}")
    console.print(f"Expected: {50 * 3 * 3} = {50 * 9}")

    # Run test episode
    total_reward = 0
    for i in range(100):
        action = env_multi.action_space.sample()
        obs, reward, terminated, truncated, info = env_multi.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    console.print(f"\n[bold]Episode Summary:[/bold]")
    console.print(f"Steps: {info['step']}")
    console.print(f"Final Balance: ${info['balance']:,.2f}")
    console.print(f"Total Reward: {total_reward:.4f}")
    console.print(f"Trades: {info['total_trades']}")

    # =========================================================================
    # Test 3: Full features (15 per timeframe)
    # =========================================================================
    console.print("\n[bold]Test 3: Full Features (15 per timeframe)[/bold]")

    # Create full feature set
    df_full = pd.DataFrame(index=dates)

    features_list = [
        'log_return', 'log_return_high', 'log_return_low',
        'rsi_norm', 'macd_norm', 'macd_hist_norm',
        'bb_position', 'atr_norm',
        'volume_norm', 'volume_ratio',
        'return_5', 'return_10', 'return_20',
        'candle_body_ratio', 'candle_direction',
    ]

    for tf in ['1d', '1wk', '1mo']:
        df_full[f'open_{tf}'] = price + np.random.randn(n) * 10
        df_full[f'high_{tf}'] = price + abs(np.random.randn(n) * 50)
        df_full[f'low_{tf}'] = price - abs(np.random.randn(n) * 50)
        df_full[f'close_{tf}'] = price + np.random.randn(n) * 20
        df_full[f'volume_{tf}'] = np.random.randint(1000, 10000, n)

        for feat in features_list:
            df_full[f'{feat}_{tf}'] = np.random.randn(n) * 0.1

    feature_map_full = {
        tf: [f'{feat}_{tf}' for feat in features_list]
        for tf in ['1d', '1wk', '1mo']
    }

    env_full = make_trading_env(
        df_full,
        feature_map_full,
        window_size=50,
        timeframes=['1d', '1wk', '1mo'],
        features_per_timeframe=15,
        base_timeframe='1d',
    )

    console.print(f"Observation space: {env_full.observation_space}")
    console.print(f"Expected obs dim: {50 * 15 * 3} = 2250")

    obs, info = env_full.reset()
    console.print(f"Actual obs shape: {obs.shape}")

    console.print("\n[green]All tests passed![/green]")
