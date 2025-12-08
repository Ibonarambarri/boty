"""
Training Callbacks Module
=========================
Custom callbacks for Stable-Baselines3 to connect with the TUI dashboard.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from src.dashboard import TradingDashboard


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


def create_callback_list(
    dashboard: TradingDashboard,
    save_path: str | Path = "models",
    save_freq: int = 10000,
    update_freq: int = 100,
    early_stopping: bool = False,
    reward_threshold: Optional[float] = None,
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

    return callbacks
