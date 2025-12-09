"""
Training Module
===============
PPO training pipeline with TUI dashboard visualization.
Supports multi-timeframe observations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.live import Live
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.callbacks import TUIDashboardCallback, ModelCheckpointCallback, create_callback_list
from src.dashboard import TradingDashboard
from src.envs.trading_env import CryptoTradingEnv, EnvConfig

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # PPO Hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005  # Reduced for exploitation over exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture
    policy: str = "MlpLstmPolicy"
    net_arch: list = None  # Will use default [256, 256]
    # LSTM-specific
    n_lstm_layers: int = 1
    lstm_hidden_size: int = 256  # Increased from 64 for more capacity
    enable_critic_lstm: bool = True


    # Training settings
    total_timesteps: int = 100_000
    save_freq: int = 10_000
    update_freq: int = 100

    # Environment settings
    initial_balance: float = 10_000.0
    position_size_pct: float = 0.95
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.01
    slippage_pct: float = 0.0005
    window_size: int = 50  # Changed from 30 for multi-TF

    # Multi-timeframe settings
    timeframes: Optional[list[str]] = None
    features_per_timeframe: Optional[int] = None
    base_timeframe: Optional[str] = None

    # Paths
    model_dir: str = "models"
    log_dir: str = "logs"

    # CSV Logging
    enable_csv_logging: bool = True  # Enabled by default for Gemini analysis
    csv_buffer_size: int = 1000

    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [256, 256]  # Increased from [64, 64] for more capacity
        if self.timeframes is None:
            self.timeframes = ['1d', '1wk']  # Removed '1mo' to reduce noise
        if self.features_per_timeframe is None:
            self.features_per_timeframe = 15
        if self.base_timeframe is None:
            self.base_timeframe = '1d'


def create_env(
    df: pd.DataFrame,
    feature_columns: Union[list[str], dict[str, list[str]]],
    config: TrainingConfig,
) -> CryptoTradingEnv:
    """
    Create a trading environment.

    Args:
        df: DataFrame with OHLCV + features
        feature_columns: Feature column names (list) or feature map (dict for multi-TF)
        config: Training configuration

    Returns:
        CryptoTradingEnv instance
    """
    env_config = EnvConfig(
        initial_balance=config.initial_balance,
        position_size_pct=config.position_size_pct,
        take_profit_pct=config.take_profit_pct,
        stop_loss_pct=config.stop_loss_pct,
        slippage_pct=config.slippage_pct,
        window_size=config.window_size,
        timeframes=config.timeframes,
        features_per_timeframe=config.features_per_timeframe,
        base_timeframe=config.base_timeframe,
    )

    return CryptoTradingEnv(df, feature_columns, env_config)


def create_vec_env(
    df: pd.DataFrame,
    feature_columns: Union[list[str], dict[str, list[str]]],
    config: TrainingConfig,
    normalize: bool = True,
) -> DummyVecEnv | VecNormalize:
    """
    Create vectorized environment with optional normalization.

    Args:
        df: DataFrame with OHLCV + features
        feature_columns: Feature column names or feature map
        config: Training configuration
        normalize: Whether to apply VecNormalize

    Returns:
        Vectorized environment
    """
    def make_env():
        return create_env(df, feature_columns, config)

    vec_env = DummyVecEnv([make_env])

    if normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,  # Disabled: log returns are already small-scale
            clip_obs=10.0,
            clip_reward=10.0,
        )

    return vec_env


def train(
    df: pd.DataFrame,
    feature_columns: Union[list[str], dict[str, list[str]]],
    config: Optional[TrainingConfig] = None,
    model_save_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> tuple[RecurrentPPO, Path]:
    """
    Train a PPO agent on the trading environment.

    Args:
        df: DataFrame with OHLCV + features
        feature_columns: Feature column names (list) or feature map (dict for multi-TF)
        config: Training configuration
        model_save_dir: Directory to save model
        model_name: Name for the model (used if model_save_dir not provided)
        resume_from: Path to existing model to resume training

    Returns:
        Tuple of (trained model, path to model directory)
    """
    config = config or TrainingConfig()

    # Detect multi-timeframe mode
    is_multi_tf = isinstance(feature_columns, dict)

    if is_multi_tf:
        n_timeframes = len(feature_columns)
        n_features = sum(len(f) for f in feature_columns.values())
        console.print(f"[cyan]Multi-timeframe mode: {n_timeframes} timeframes, {n_features} total features[/cyan]")
    else:
        console.print(f"[cyan]Single timeframe mode: {len(feature_columns)} features[/cyan]")

    # Create model directory
    if model_save_dir:
        model_dir = Path(model_save_dir)
    elif model_name:
        model_dir = Path(config.model_dir) / model_name
    else:
        model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    console.print("[cyan]Creating trading environment...[/cyan]")
    vec_env = create_vec_env(df, feature_columns, config, normalize=True)

    # Log observation space
    obs_dim = vec_env.observation_space.shape[0]
    console.print(f"[cyan]Observation space: ({obs_dim},)[/cyan]")

    # Create or load model
    if resume_from:
        console.print(f"[yellow]Resuming from: {resume_from}[/yellow]")
        model = RecurrentPPO.load(resume_from, env=vec_env)
    else:
        console.print("[cyan]Creating RecurrentPPO (LSTM) model...[/cyan]")

        # Policy kwargs for network architecture
        policy_kwargs = {
            "net_arch": dict(pi=config.net_arch, vf=config.net_arch),
            "n_lstm_layers": config.n_lstm_layers,
            "lstm_hidden_size": config.lstm_hidden_size,
            "enable_critic_lstm": config.enable_critic_lstm,
        }

        model = RecurrentPPO(
            policy=config.policy,
            env=vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=0,  # Silent for TUI
            tensorboard_log=config.log_dir,
        )

    # Create dashboard
    dashboard = TradingDashboard(
        total_steps=config.total_timesteps,
        initial_balance=config.initial_balance,
        title="Crypto DRL Trading System" + (" (Multi-TF)" if is_multi_tf else ""),
    )

    # Create callbacks
    callbacks = create_callback_list(
        dashboard=dashboard,
        save_path=model_dir,
        save_freq=config.save_freq,
        update_freq=config.update_freq,
        enable_csv_logging=config.enable_csv_logging,
        csv_buffer_size=config.csv_buffer_size,
    )

    # Training with Live dashboard
    console.print("[bold green]Starting training...[/bold green]\n")

    try:
        with Live(
            dashboard.get_renderable(),
            refresh_per_second=4,
            console=console,
            screen=True,
        ) as live:
            # Connect dashboard to live context
            dashboard.set_live(live)

            # Train!
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=callbacks,
                progress_bar=False,  # We have our own
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")

    # Save final model with standardized names
    model_path = model_dir / "model"
    model.save(str(model_path))
    console.print(f"\n[green]Model saved to: {model_path}.zip[/green]")

    # Save VecNormalize stats
    if isinstance(vec_env, VecNormalize):
        vec_norm_path = model_dir / "vecnorm.pkl"
        vec_env.save(str(vec_norm_path))
        console.print(f"[green]VecNormalize saved to: {vec_norm_path}[/green]")

    # Print summary
    _print_training_summary(dashboard)

    return model, model_dir


def _print_training_summary(dashboard: TradingDashboard) -> None:
    """Print training summary statistics."""
    m = dashboard.metrics

    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Training Summary[/bold cyan]")
    console.print("=" * 60)

    console.print(f"Total Steps: {m.current_step:,}")
    console.print(f"Episodes: {m.episode}")
    console.print(f"Final Balance: ${m.balance:,.2f}")
    console.print(f"PnL: {m.pnl_pct:+.2f}%")
    console.print(f"Total Trades: {m.total_trades}")
    console.print(f"Win Rate: {m.win_rate:.1f}%")
    console.print(f"Best Episode Reward: {m.best_reward:.4f}")
    console.print(f"Mean Episode Reward: {m.mean_reward:.4f}")
    console.print("=" * 60)


def evaluate(
    model: RecurrentPPO,
    df: pd.DataFrame,
    feature_columns: Union[list[str], dict[str, list[str]]],
    config: Optional[TrainingConfig] = None,
    n_episodes: int = 10,
    vec_normalize_path: Optional[str] = None,
) -> dict:
    """
    Evaluate a trained model.

    Args:
        model: Trained PPO model
        df: DataFrame with OHLCV + features
        feature_columns: Feature column names or feature map
        config: Training configuration
        n_episodes: Number of evaluation episodes
        vec_normalize_path: Path to VecNormalize stats

    Returns:
        Dictionary with evaluation metrics
    """
    config = config or TrainingConfig()

    # Create evaluation environment
    vec_env = create_vec_env(df, feature_columns, config, normalize=True)

    # Load VecNormalize stats if available
    if vec_normalize_path and isinstance(vec_env, VecNormalize):
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    results = {
        "episode_rewards": [],
        "episode_lengths": [],
        "final_balances": [],
        "pnl_pcts": [],
        "total_trades": [],
        "win_rates": [],
    }

    console.print(f"[cyan]Evaluating over {n_episodes} episodes...[/cyan]")

    for ep in range(n_episodes):
        lstm_states = None
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=np.array([done]),
                deterministic=True,
            )
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]

            if done[0]:
                ep_info = info[0]
                results["episode_rewards"].append(episode_reward)
                results["episode_lengths"].append(ep_info.get("step", 0))
                results["final_balances"].append(ep_info.get("balance", 0))
                results["pnl_pcts"].append(ep_info.get("pnl_pct", 0))
                results["total_trades"].append(ep_info.get("total_trades", 0))
                results["win_rates"].append(ep_info.get("win_rate", 0))

    # Compute summary stats
    results["mean_reward"] = np.mean(results["episode_rewards"])
    results["std_reward"] = np.std(results["episode_rewards"])
    results["mean_balance"] = np.mean(results["final_balances"])
    results["mean_pnl"] = np.mean(results["pnl_pcts"])
    results["mean_win_rate"] = np.mean(results["win_rates"])

    console.print("\n[bold]Evaluation Results:[/bold]")
    console.print(f"  Mean Reward: {results['mean_reward']:.4f} +/- {results['std_reward']:.4f}")
    console.print(f"  Mean Balance: ${results['mean_balance']:,.2f}")
    console.print(f"  Mean PnL: {results['mean_pnl']:.2f}%")
    console.print(f"  Mean Win Rate: {results['mean_win_rate']:.1f}%")

    return results


if __name__ == "__main__":
    # Test training with multi-timeframe dummy data
    np.random.seed(42)

    # Create dummy multi-timeframe data
    n = 5000
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame()

    # Create columns for each timeframe (removed 1mo to reduce noise)
    for tf in ['1d', '1wk']:
        df[f'open_{tf}'] = price + np.random.randn(n) * 0.1
        df[f'high_{tf}'] = price + abs(np.random.randn(n) * 0.5)
        df[f'low_{tf}'] = price - abs(np.random.randn(n) * 0.5)
        df[f'close_{tf}'] = price + np.random.randn(n) * 0.1
        df[f'volume_{tf}'] = np.random.randint(1000, 10000, n)

        # Features (3 per timeframe for testing)
        df[f'log_return_{tf}'] = np.random.randn(n) * 0.01
        df[f'rsi_norm_{tf}'] = np.random.randn(n) * 0.5
        df[f'macd_norm_{tf}'] = np.random.randn(n) * 0.1

    # Feature map for multi-timeframe
    feature_map = {
        '1d': ['log_return_1d', 'rsi_norm_1d', 'macd_norm_1d'],
        '1wk': ['log_return_1wk', 'rsi_norm_1wk', 'macd_norm_1wk'],
    }

    # Quick training config for testing
    config = TrainingConfig(
        total_timesteps=10000,
        save_freq=5000,
        update_freq=50,
        n_steps=512,
        batch_size=32,
        window_size=50,
        timeframes=['1d', '1wk'],
        features_per_timeframe=3,  # Only 3 for test
        base_timeframe='1d',
    )

    # Train
    model, path = train(df, feature_map, config, model_name="test_multi_tf")

    console.print("\n[bold green]Multi-timeframe training test complete![/bold green]")
