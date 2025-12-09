"""
Training Callbacks Module
=========================
Custom callbacks for Stable-Baselines3 to connect with the TUI dashboard.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from src.dashboard import TradingDashboard

# Action name mapping for CSV logging
ACTION_NAMES = {0: "HOLD", 1: "LONG", 2: "SHORT"}


class TUIDashboardCallback(BaseCallback):
    """
    Callback to update the Rich TUI dashboard during training.

    Extracts metrics from the environment's info dict and sends
    them to the dashboard for visualization.
    """

    def __init__(
        self,
        dashboard: TradingDashboard,
        update_freq: int = 100,
        verbose: int = 0,
    ) -> None:
        """
        Initialize the callback.

        Args:
            dashboard: TradingDashboard instance
            update_freq: How often to update the dashboard (in steps)
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.dashboard = dashboard
        self.update_freq = update_freq

        # Tracking variables
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.current_episode_reward: float = 0.0
        self.current_episode_length: int = 0
        self.episode_count: int = 0

        # FPS calculation
        self.start_time: float = 0.0
        self.last_time: float = 0.0
        self.last_step: int = 0

        # Recent rewards for mean calculation (using deque for O(1) removal)
        self.max_recent: int = 100
        self.recent_rewards: deque[float] = deque(maxlen=self.max_recent)

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step = 0
        self.dashboard.start()

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Returns:
            True to continue training, False to stop
        """
        # Extract info from environment
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])
        rewards = self.locals.get("rewards", [0.0])

        # Get info from first (or only) environment
        info = infos[0] if infos else {}
        done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
        reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards

        # Accumulate episode stats
        self.current_episode_reward += float(reward)
        self.current_episode_length += 1

        # Episode finished
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Update recent rewards (deque auto-removes oldest when maxlen is reached)
            self.recent_rewards.append(self.current_episode_reward)

            # Reset for next episode
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        # Update dashboard periodically
        if self.num_timesteps % self.update_freq == 0:
            self._update_dashboard(info)

        return True

    def _update_dashboard(self, info: dict[str, Any]) -> None:
        """
        Update the dashboard with current metrics.

        Args:
            info: Info dict from environment
        """
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_time
        steps_delta = self.num_timesteps - self.last_step

        if elapsed > 0:
            fps = steps_delta / elapsed
        else:
            fps = 0.0

        self.last_time = current_time
        self.last_step = self.num_timesteps

        # Calculate mean reward
        mean_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0

        # Get latest episode reward (or current accumulated)
        latest_ep_reward = (
            self.episode_rewards[-1]
            if self.episode_rewards
            else self.current_episode_reward
        )

        # Update dashboard
        self.dashboard.update(
            step=self.num_timesteps,
            episode=self.episode_count,
            episode_reward=latest_ep_reward,
            episode_length=self.current_episode_length,
            balance=info.get("balance", 10000.0),
            net_worth=info.get("net_worth", 10000.0),
            pnl_pct=info.get("pnl_pct", 0.0),
            total_trades=info.get("total_trades", 0),
            win_rate=info.get("win_rate", 0.0),
            fps=fps,
            mean_reward=float(mean_reward),
            position=info.get("position", "NONE"),
            current_price=info.get("current_price", 0.0),
        )

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        total_time = time.time() - self.start_time

        if self.verbose > 0:
            print(f"\nTraining completed!")
            print(f"Total time: {total_time:.2f}s")
            print(f"Total episodes: {self.episode_count}")
            print(f"Total steps: {self.num_timesteps}")

            if self.episode_rewards:
                print(f"Mean episode reward: {np.mean(self.episode_rewards):.4f}")
                print(f"Max episode reward: {np.max(self.episode_rewards):.4f}")


class ModelCheckpointCallback(CheckpointCallback):
    """
    Extended checkpoint callback with additional logging.

    Saves model at regular intervals with informative naming.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str | Path,
        name_prefix: str = "ppo_trading",
        dashboard: Optional[TradingDashboard] = None,
        verbose: int = 0,
    ) -> None:
        """
        Initialize checkpoint callback.

        Args:
            save_freq: Save every N steps
            save_path: Directory to save models
            name_prefix: Prefix for saved model files
            dashboard: Optional dashboard for notifications
            verbose: Verbosity level
        """
        super().__init__(
            save_freq=save_freq,
            save_path=str(save_path),
            name_prefix=name_prefix,
            verbose=verbose,
        )
        self.dashboard = dashboard

    def _on_step(self) -> bool:
        """Called at each step."""
        result = super()._on_step()

        # Log checkpoint saves (handled by parent, but we could add dashboard notification)
        return result


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback based on reward threshold or no improvement.

    Stops training if:
    - Mean reward exceeds target
    - No improvement for N episodes
    """

    def __init__(
        self,
        reward_threshold: Optional[float] = None,
        patience: int = 50,
        min_episodes: int = 100,
        verbose: int = 0,
    ) -> None:
        """
        Initialize early stopping callback.

        Args:
            reward_threshold: Stop if mean reward exceeds this
            patience: Episodes without improvement before stopping
            min_episodes: Minimum episodes before stopping is allowed
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.reward_threshold = reward_threshold
        self.patience = patience
        self.min_episodes = min_episodes

        self.best_mean_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.episode_rewards: list[float] = []
        self.current_episode_reward: float = 0.0

    def _on_step(self) -> bool:
        """
        Check stopping conditions.

        Returns:
            True to continue, False to stop
        """
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
        done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones

        self.current_episode_reward += float(reward)

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

            # Check conditions after minimum episodes
            if len(self.episode_rewards) >= self.min_episodes:
                # Calculate recent mean reward
                recent_rewards = self.episode_rewards[-100:]
                mean_reward = np.mean(recent_rewards)

                # Check reward threshold
                if self.reward_threshold is not None:
                    if mean_reward >= self.reward_threshold:
                        if self.verbose > 0:
                            print(f"\nReward threshold reached: {mean_reward:.4f}")
                        return False

                # Check improvement
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.episodes_without_improvement = 0
                else:
                    self.episodes_without_improvement += 1

                if self.episodes_without_improvement >= self.patience:
                    if self.verbose > 0:
                        print(f"\nNo improvement for {self.patience} episodes")
                    return False

        return True


class CSVLoggerCallback(BaseCallback):
    """
    Callback to log detailed per-step metrics to CSV.

    Logs every training step with all available metrics for
    post-training analysis (e.g., with Gemini or other LLMs).
    """

    CSV_COLUMNS = [
        "global_step", "episode", "env_step", "action", "action_name",
        "reward", "balance", "net_worth", "pnl_pct", "realized_pnl",
        "position", "total_trades", "winning_trades", "win_rate",
        "current_price", "done", "timestamp"
    ]

    def __init__(
        self,
        log_dir: str | Path,
        filename: str = "training_log",
        include_timestamp: bool = True,
        buffer_size: int = 1000,
        flush_on_episode: bool = True,
        verbose: int = 0,
    ) -> None:
        """
        Initialize CSV logger.

        Args:
            log_dir: Directory to save CSV file
            filename: Base filename (without extension)
            include_timestamp: Add timestamp to filename
            buffer_size: Rows to buffer before writing to disk
            flush_on_episode: Flush buffer at end of each episode
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.log_dir = Path(log_dir)
        self.filename = filename
        self.include_timestamp = include_timestamp
        self.buffer_size = buffer_size
        self.flush_on_episode = flush_on_episode

        # State
        self.buffer: list[dict] = []
        self.csv_path: Optional[Path] = None
        self.episode_count: int = 0
        self.header_written: bool = False

    def _on_training_start(self) -> None:
        """Initialize CSV file on training start."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if self.include_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{self.filename}_{ts}.csv"
        else:
            fname = f"{self.filename}.csv"

        self.csv_path = self.log_dir / fname
        self.header_written = False
        self.buffer = []
        self.episode_count = 0

        if self.verbose > 0:
            print(f"CSV logging to: {self.csv_path}")

    def _on_step(self) -> bool:
        """Log metrics for each step."""
        # Extract data from locals
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        actions = self.locals.get("actions", [0])

        info = infos[0] if infos else {}
        reward = float(rewards[0]) if isinstance(rewards, (list, np.ndarray)) else float(rewards)
        done = bool(dones[0]) if isinstance(dones, (list, np.ndarray)) else bool(dones)
        action = int(actions[0]) if isinstance(actions, (list, np.ndarray)) else int(actions)

        # Track episode
        if done:
            self.episode_count += 1

        # Build row
        row = {
            "global_step": self.num_timesteps,
            "episode": self.episode_count,
            "env_step": info.get("step", 0),
            "action": action,
            "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
            "reward": reward,
            "balance": info.get("balance", 0.0),
            "net_worth": info.get("net_worth", 0.0),
            "pnl_pct": info.get("pnl_pct", 0.0),
            "realized_pnl": info.get("realized_pnl", 0.0),
            "position": info.get("position", "NONE"),
            "total_trades": info.get("total_trades", 0),
            "winning_trades": info.get("winning_trades", 0),
            "win_rate": info.get("win_rate", 0.0),
            "current_price": info.get("current_price", 0.0),
            "done": done,
            "timestamp": datetime.now().isoformat(),
        }

        self.buffer.append(row)

        # Flush conditions
        should_flush = (
            len(self.buffer) >= self.buffer_size or
            (self.flush_on_episode and done)
        )

        if should_flush:
            self._flush_buffer()

        return True

    def _flush_buffer(self) -> None:
        """Write buffer to CSV file."""
        if not self.buffer or self.csv_path is None:
            return

        mode = "a" if self.header_written else "w"

        with open(self.csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)

            if not self.header_written:
                writer.writeheader()
                self.header_written = True

            writer.writerows(self.buffer)

        self.buffer = []

    def _on_training_end(self) -> None:
        """Flush remaining buffer on training end."""
        self._flush_buffer()

        if self.verbose > 0 and self.csv_path:
            print(f"CSV log saved: {self.csv_path}")


def create_callback_list(
    dashboard: TradingDashboard,
    save_path: str | Path = "models",
    save_freq: int = 10000,
    update_freq: int = 100,
    early_stopping: bool = False,
    reward_threshold: Optional[float] = None,
    enable_csv_logging: bool = False,
    csv_buffer_size: int = 1000,
) -> list[BaseCallback]:
    """
    Create a list of callbacks for training.

    Args:
        dashboard: TradingDashboard instance
        save_path: Directory for model checkpoints
        save_freq: Checkpoint frequency
        update_freq: Dashboard update frequency
        early_stopping: Enable early stopping
        reward_threshold: Early stopping reward threshold
        enable_csv_logging: Enable detailed CSV logging per step
        csv_buffer_size: Buffer size for CSV logger

    Returns:
        List of callbacks
    """
    callbacks = [
        TUIDashboardCallback(dashboard, update_freq=update_freq, verbose=0),
        ModelCheckpointCallback(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix="ppo_crypto",
            dashboard=dashboard,
            verbose=0,
        ),
    ]

    if early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                reward_threshold=reward_threshold,
                patience=50,
                min_episodes=100,
                verbose=1,
            )
        )

    if enable_csv_logging:
        callbacks.append(
            CSVLoggerCallback(
                log_dir=save_path,
                filename="training_log",
                include_timestamp=True,
                buffer_size=csv_buffer_size,
                flush_on_episode=True,
                verbose=1,
            )
        )

    return callbacks
