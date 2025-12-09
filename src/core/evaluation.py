"""
Evaluation Module
=================
Comprehensive backtesting and visualization for trained agents.
Includes benchmarks, performance metrics, and charts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.trading_env import CryptoTradingEnv, EnvConfig

console = Console()
logger = logging.getLogger(__name__)

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'agent': '#2ecc71',      # Green
    'buy_hold': '#3498db',   # Blue
    'random': '#e74c3c',     # Red
    'drawdown': '#e74c3c',   # Red
    'long': '#27ae60',       # Dark green
    'short': '#c0392b',      # Dark red
    'neutral': '#7f8c8d',    # Gray
}


@dataclass
class TradeRecord:
    """Record of a single trade."""
    entry_step: int
    exit_step: int
    entry_price: float
    exit_price: float
    position_type: str  # 'LONG' or 'SHORT'
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    # Equity curves
    agent_equity: np.ndarray
    buy_hold_equity: np.ndarray
    random_equity: Optional[np.ndarray] = None

    # Trade history
    trades: list[TradeRecord] = field(default_factory=list)

    # Performance metrics
    total_return: float = 0.0
    buy_hold_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # Position sizing metrics (continuous actions)
    avg_position_size: float = 0.0
    position_size_std: float = 0.0

    # Time series data
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: list = field(default_factory=list)


class AgentEvaluator:
    """
    Comprehensive agent evaluator with benchmarks and visualizations.
    """

    def __init__(
        self,
        model: Union[PPO, RecurrentPPO],
        df: pd.DataFrame,
        feature_columns: list[str] | dict[str, list[str]],
        initial_balance: float = 10_000.0,
        vec_normalize_path: Optional[str] = None,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            model: Trained PPO or RecurrentPPO model
            df: DataFrame with OHLCV + features
            feature_columns: Feature column names (list) or feature map (dict for multi-tf)
            initial_balance: Starting balance
            vec_normalize_path: Path to VecNormalize stats
        """
        self.model = model
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.vec_normalize_path = vec_normalize_path

        # Detect multi-timeframe mode
        self.multi_tf_mode = isinstance(feature_columns, dict)

        # Extract price data (use base timeframe for multi-tf)
        if self.multi_tf_mode:
            # Get base timeframe (first in dict)
            self.base_tf = list(feature_columns.keys())[0]
            self.prices = df[f'close_{self.base_tf}'].values
            self.highs = df[f'high_{self.base_tf}'].values
            self.lows = df[f'low_{self.base_tf}'].values
        else:
            self.prices = df['close'].values
            self.highs = df['high'].values
            self.lows = df['low'].values

    def _create_env(self) -> DummyVecEnv | VecNormalize:
        """Create evaluation environment."""
        if self.multi_tf_mode:
            timeframes = list(self.feature_columns.keys())
            config = EnvConfig(
                initial_balance=self.initial_balance,
                position_size_pct=0.95,
                window_size=50,
                timeframes=timeframes,
                base_timeframe=self.base_tf,
            )
        else:
            config = EnvConfig(
                initial_balance=self.initial_balance,
                position_size_pct=0.95,
            )

        def make_env():
            return CryptoTradingEnv(self.df, self.feature_columns, config)

        vec_env = DummyVecEnv([make_env])

        if self.vec_normalize_path and Path(self.vec_normalize_path).exists():
            vec_env = VecNormalize.load(self.vec_normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        return vec_env

    def run_agent_episode(self) -> tuple[np.ndarray, list[TradeRecord], dict]:
        """
        Run a single episode with the trained agent.

        Returns:
            Tuple of (equity_curve, trades, final_info)
        """
        env = self._create_env()
        obs = env.reset()
        
        is_recurrent = isinstance(self.model, RecurrentPPO)
        
        if is_recurrent:
            lstm_states = None
            # Set episode_start to True for the first step
            episode_starts = np.ones((env.num_envs,), dtype=bool)

        equity_curve = [self.initial_balance]
        trades = []
        position_sizes = []  # Track position sizes for continuous actions
        done = False

        while not done:
            if is_recurrent:
                action, lstm_states = self.model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                episode_starts = np.zeros((env.num_envs,), dtype=bool)
            else:
                action, _ = self.model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            info = info[0]
            equity_curve.append(info['net_worth'])

            # Track position size when in a position
            if info.get('position', 'NONE') != 'NONE':
                position_sizes.append(info.get('position_size_pct', 0.0))

        # Add position sizing stats to final info
        if position_sizes:
            info['avg_position_size'] = float(np.mean(position_sizes))
            info['position_size_std'] = float(np.std(position_sizes))
        else:
            info['avg_position_size'] = 0.0
            info['position_size_std'] = 0.0

        return np.array(equity_curve), trades, info

    def run_buy_hold(self) -> np.ndarray:
        """
        Calculate buy & hold equity curve.

        Returns:
            Equity curve array
        """
        # Start from window_size to align with agent
        window = 50 if self.multi_tf_mode else 30
        start_price = self.prices[window]

        # Calculate equity: initial_balance * (current_price / start_price)
        equity = self.initial_balance * (self.prices[window:] / start_price)

        # Prepend initial balance for alignment
        return np.concatenate([[self.initial_balance], equity])

    def run_random_agent(self, n_runs: int = 5) -> np.ndarray:
        """
        Run random agent baseline (average of multiple runs).

        Args:
            n_runs: Number of random runs to average

        Returns:
            Average equity curve
        """
        all_curves = []

        for _ in range(n_runs):
            env = self._create_env()
            obs = env.reset()
            equity = [self.initial_balance]
            done = False

            while not done:
                action = env.action_space.sample()
                obs, _, done, info = env.step(np.array([action]))
                equity.append(info[0]['net_worth'])

            all_curves.append(equity)

        # Pad to same length and average
        max_len = max(len(c) for c in all_curves)
        padded = []
        for c in all_curves:
            if len(c) < max_len:
                c = list(c) + [c[-1]] * (max_len - len(c))
            padded.append(c)

        return np.mean(padded, axis=0)

    def calculate_metrics(self, equity_curve: np.ndarray) -> dict:
        """
        Calculate performance metrics from equity curve.

        Args:
            equity_curve: Array of portfolio values

        Returns:
            Dictionary of metrics
        """
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[~np.isnan(returns)]  # Remove NaN

        # Total return
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100

        # Sharpe Ratio (annualized)
        # TODO: Make annualization factor dynamic based on data frequency.
        # Current hardcoded value assumes hourly data (24 * 365 = 8760 periods/year).
        # For daily data use 252, for 4h data use 6 * 365 = 2190, etc.
        annualization_factor = 24 * 365  # Hourly data assumption
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor)
        else:
            sharpe = 0.0

        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = np.mean(returns) / downside_std * np.sqrt(annualization_factor) if downside_std > 0 else 0.0
        else:
            sortino = float('inf') if np.mean(returns) > 0 else 0.0

        # Maximum Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd = np.min(drawdown) * 100

        # Max Drawdown Duration
        underwater = drawdown < 0
        if np.any(underwater):
            # Find longest underwater period
            changes = np.diff(np.concatenate([[False], underwater, [False]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            durations = ends - starts
            max_dd_duration = np.max(durations) if len(durations) > 0 else 0
        else:
            max_dd_duration = 0

        # Calmar Ratio
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_dd_duration': max_dd_duration,
            'calmar_ratio': calmar,
            'volatility': np.std(returns) * np.sqrt(24 * 365) * 100,
        }

    def evaluate(self, include_random: bool = True) -> EvaluationResult:
        """
        Run complete evaluation.

        Args:
            include_random: Whether to include random agent baseline

        Returns:
            EvaluationResult with all metrics and curves
        """
        console.print("[cyan]Running agent evaluation...[/cyan]")

        # Run agent
        agent_equity, trades, final_info = self.run_agent_episode()

        # Run buy & hold
        console.print("[cyan]Calculating Buy & Hold benchmark...[/cyan]")
        buy_hold_equity = self.run_buy_hold()

        # Align lengths
        min_len = min(len(agent_equity), len(buy_hold_equity))
        agent_equity = agent_equity[:min_len]
        buy_hold_equity = buy_hold_equity[:min_len]

        # Run random agent
        random_equity = None
        if include_random:
            console.print("[cyan]Running random agent baseline...[/cyan]")
            random_equity = self.run_random_agent(n_runs=5)
            random_equity = random_equity[:min_len]

        # Calculate metrics
        agent_metrics = self.calculate_metrics(agent_equity)
        buy_hold_metrics = self.calculate_metrics(buy_hold_equity)

        # Build result
        window = 50 if self.multi_tf_mode else 30
        # Ensure prices array matches equity array lengths
        prices_aligned = self.prices[window:]
        prices_aligned = prices_aligned[:min_len]

        result = EvaluationResult(
            agent_equity=agent_equity,
            buy_hold_equity=buy_hold_equity,
            random_equity=random_equity,
            trades=trades,
            prices=prices_aligned,
            total_return=agent_metrics['total_return'],
            buy_hold_return=buy_hold_metrics['total_return'],
            sharpe_ratio=agent_metrics['sharpe_ratio'],
            sortino_ratio=agent_metrics['sortino_ratio'],
            max_drawdown=agent_metrics['max_drawdown'],
            max_drawdown_duration=agent_metrics['max_dd_duration'],
            win_rate=final_info.get('win_rate', 0),
            total_trades=final_info.get('total_trades', 0),
            avg_position_size=final_info.get('avg_position_size', 0.0),
            position_size_std=final_info.get('position_size_std', 0.0),
        )

        return result

    def plot_results(
        self,
        result: EvaluationResult,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Create comprehensive visualization of results.

        Args:
            result: EvaluationResult from evaluate()
            save_path: Path to save figure
            show: Whether to display figure
        """
        fig = plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

        # =================================================================
        # 1. EQUITY CURVES COMPARISON (Top left, spans 2 columns)
        # =================================================================
        ax1 = fig.add_subplot(gs[0, :])

        steps = np.arange(len(result.agent_equity))

        ax1.plot(steps, result.agent_equity,
                 color=COLORS['agent'], linewidth=2, label='DRL Agent')
        ax1.plot(steps, result.buy_hold_equity,
                 color=COLORS['buy_hold'], linewidth=2, label='Buy & Hold', linestyle='--')

        if result.random_equity is not None:
            ax1.plot(steps, result.random_equity,
                     color=COLORS['random'], linewidth=1.5, label='Random Agent',
                     linestyle=':', alpha=0.7)

        ax1.axhline(y=self.initial_balance, color='gray', linestyle='-',
                    alpha=0.3, label='Initial Balance')

        ax1.set_xlabel('Steps', fontsize=11)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax1.set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Format y-axis
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # =================================================================
        # 2. DRAWDOWN CHART (Middle left)
        # =================================================================
        ax2 = fig.add_subplot(gs[1, 0])

        # Calculate drawdowns
        agent_peak = np.maximum.accumulate(result.agent_equity)
        agent_dd = (result.agent_equity - agent_peak) / agent_peak * 100

        bh_peak = np.maximum.accumulate(result.buy_hold_equity)
        bh_dd = (result.buy_hold_equity - bh_peak) / bh_peak * 100

        ax2.fill_between(steps, agent_dd, 0,
                         color=COLORS['agent'], alpha=0.3, label='Agent DD')
        ax2.plot(steps, agent_dd, color=COLORS['agent'], linewidth=1.5)

        ax2.fill_between(steps, bh_dd, 0,
                         color=COLORS['buy_hold'], alpha=0.2, label='Buy & Hold DD')
        ax2.plot(steps, bh_dd, color=COLORS['buy_hold'], linewidth=1, linestyle='--')

        ax2.set_xlabel('Steps', fontsize=11)
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # =================================================================
        # 3. RETURNS DISTRIBUTION (Middle right)
        # =================================================================
        ax3 = fig.add_subplot(gs[1, 1])

        agent_returns = np.diff(result.agent_equity) / result.agent_equity[:-1] * 100
        bh_returns = np.diff(result.buy_hold_equity) / result.buy_hold_equity[:-1] * 100

        # Remove outliers for better visualization
        agent_returns_clipped = np.clip(agent_returns, -5, 5)
        bh_returns_clipped = np.clip(bh_returns, -5, 5)

        ax3.hist(agent_returns_clipped, bins=50, alpha=0.6,
                 color=COLORS['agent'], label=f'Agent (std={np.std(agent_returns):.3f}%)')
        ax3.hist(bh_returns_clipped, bins=50, alpha=0.4,
                 color=COLORS['buy_hold'], label=f'Buy & Hold (std={np.std(bh_returns):.3f}%)')

        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax3.axvline(x=np.mean(agent_returns), color=COLORS['agent'],
                    linestyle='--', linewidth=2, label=f'Agent Mean: {np.mean(agent_returns):.4f}%')

        ax3.set_xlabel('Return (%)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # =================================================================
        # 4. CUMULATIVE RETURNS (Bottom left)
        # =================================================================
        ax4 = fig.add_subplot(gs[2, 0])

        agent_cum_ret = (result.agent_equity / self.initial_balance - 1) * 100
        bh_cum_ret = (result.buy_hold_equity / self.initial_balance - 1) * 100

        ax4.plot(steps, agent_cum_ret, color=COLORS['agent'],
                 linewidth=2, label='Agent')
        ax4.plot(steps, bh_cum_ret, color=COLORS['buy_hold'],
                 linewidth=2, label='Buy & Hold', linestyle='--')

        # Shade outperformance regions
        outperform = agent_cum_ret > bh_cum_ret
        ax4.fill_between(steps, agent_cum_ret, bh_cum_ret,
                         where=outperform, color=COLORS['agent'], alpha=0.2)
        ax4.fill_between(steps, agent_cum_ret, bh_cum_ret,
                         where=~outperform, color=COLORS['buy_hold'], alpha=0.2)

        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Steps', fontsize=11)
        ax4.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax4.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3)

        # =================================================================
        # 5. PERFORMANCE METRICS TABLE (Bottom right)
        # =================================================================
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        # Calculate additional metrics
        agent_metrics = self.calculate_metrics(result.agent_equity)
        bh_metrics = self.calculate_metrics(result.buy_hold_equity)

        # Outperformance
        outperformance = result.total_return - result.buy_hold_return

        metrics_data = [
            ['Metric', 'Agent', 'Buy & Hold', 'Difference'],
            ['Total Return', f'{result.total_return:.2f}%',
             f'{result.buy_hold_return:.2f}%',
             f'{outperformance:+.2f}%'],
            ['Sharpe Ratio', f'{agent_metrics["sharpe_ratio"]:.2f}',
             f'{bh_metrics["sharpe_ratio"]:.2f}',
             f'{agent_metrics["sharpe_ratio"] - bh_metrics["sharpe_ratio"]:+.2f}'],
            ['Sortino Ratio', f'{agent_metrics["sortino_ratio"]:.2f}',
             f'{bh_metrics["sortino_ratio"]:.2f}',
             f'{agent_metrics["sortino_ratio"] - bh_metrics["sortino_ratio"]:+.2f}'],
            ['Max Drawdown', f'{agent_metrics["max_drawdown"]:.2f}%',
             f'{bh_metrics["max_drawdown"]:.2f}%',
             f'{agent_metrics["max_drawdown"] - bh_metrics["max_drawdown"]:+.2f}%'],
            ['Volatility', f'{agent_metrics["volatility"]:.2f}%',
             f'{bh_metrics["volatility"]:.2f}%',
             f'{agent_metrics["volatility"] - bh_metrics["volatility"]:+.2f}%'],
            ['Calmar Ratio', f'{agent_metrics["calmar_ratio"]:.2f}',
             f'{bh_metrics["calmar_ratio"]:.2f}',
             f'{agent_metrics["calmar_ratio"] - bh_metrics["calmar_ratio"]:+.2f}'],
            ['Win Rate', f'{result.win_rate:.1f}%', 'N/A', ''],
            ['Total Trades', f'{result.total_trades}', 'N/A', ''],
        ]

        # Create table
        table = ax5.table(
            cellText=metrics_data[1:],
            colLabels=metrics_data[0],
            cellLoc='center',
            loc='center',
            colWidths=[0.3, 0.23, 0.23, 0.24],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        # Color code differences
        for row in range(1, len(metrics_data)):
            diff_cell = table[(row, 3)]
            diff_text = metrics_data[row][3]
            if diff_text and diff_text != '':
                if '+' in diff_text:
                    diff_cell.set_facecolor('#d5f4e6')
                elif '-' in diff_text:
                    diff_cell.set_facecolor('#fadbd8')

        ax5.set_title('Performance Metrics Comparison', fontsize=12,
                      fontweight='bold', pad=20)

        # =================================================================
        # MAIN TITLE
        # =================================================================
        fig.suptitle('DRL Trading Agent - Comprehensive Evaluation Report',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            console.print(f"[green]Figure saved to: {save_path}[/green]")

        if show:
            plt.show()
        else:
            plt.close()

    def print_report(self, result: EvaluationResult) -> None:
        """
        Print detailed evaluation report to console.

        Args:
            result: EvaluationResult from evaluate()
        """
        # Header
        console.print("\n")
        console.print(Panel(
            "[bold cyan]DRL Trading Agent - Evaluation Report[/bold cyan]",
            border_style="cyan",
        ))

        # Performance comparison table
        table = Table(title="Performance Comparison", show_header=True,
                      header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Agent", justify="right", width=15)
        table.add_column("Buy & Hold", justify="right", width=15)
        table.add_column("Winner", justify="center", width=10)

        # Calculate metrics
        agent_metrics = self.calculate_metrics(result.agent_equity)
        bh_metrics = self.calculate_metrics(result.buy_hold_equity)

        # Compare and add rows
        comparisons = [
            ("Total Return", result.total_return, result.buy_hold_return, True),
            ("Sharpe Ratio", agent_metrics['sharpe_ratio'],
             bh_metrics['sharpe_ratio'], True),
            ("Sortino Ratio", agent_metrics['sortino_ratio'],
             bh_metrics['sortino_ratio'], True),
            ("Max Drawdown", agent_metrics['max_drawdown'],
             bh_metrics['max_drawdown'], False),  # Lower is better
            ("Volatility", agent_metrics['volatility'],
             bh_metrics['volatility'], False),  # Lower is better
            ("Calmar Ratio", agent_metrics['calmar_ratio'],
             bh_metrics['calmar_ratio'], True),
        ]

        for name, agent_val, bh_val, higher_better in comparisons:
            if higher_better:
                winner = "[green]Agent[/green]" if agent_val > bh_val else "[blue]B&H[/blue]"
            else:
                # For metrics where lower is better (drawdown, volatility)
                winner = "[green]Agent[/green]" if abs(agent_val) < abs(bh_val) else "[blue]B&H[/blue]"

            # Format values
            if "Return" in name or "Drawdown" in name or "Volatility" in name:
                agent_str = f"{agent_val:.2f}%"
                bh_str = f"{bh_val:.2f}%"
            else:
                agent_str = f"{agent_val:.3f}"
                bh_str = f"{bh_val:.3f}"

            table.add_row(name, agent_str, bh_str, winner)

        console.print(table)

        # Agent-specific stats
        console.print("\n")
        agent_table = Table(title="Agent Trading Statistics",
                           show_header=True, header_style="bold green")
        agent_table.add_column("Statistic", style="cyan", width=25)
        agent_table.add_column("Value", justify="right", width=20)

        agent_table.add_row("Total Trades", str(result.total_trades))
        agent_table.add_row("Win Rate", f"{result.win_rate:.1f}%")
        agent_table.add_row("Avg Position Size", f"{result.avg_position_size*100:.1f}%")
        agent_table.add_row("Position Size Std", f"{result.position_size_std*100:.1f}%")
        agent_table.add_row("Max DD Duration", f"{result.max_drawdown_duration} steps")
        agent_table.add_row("Final Balance", f"${result.agent_equity[-1]:,.2f}")
        agent_table.add_row("B&H Final Value", f"${result.buy_hold_equity[-1]:,.2f}")

        # Outperformance
        outperf = result.total_return - result.buy_hold_return
        outperf_color = "green" if outperf > 0 else "red"
        agent_table.add_row(
            "Outperformance vs B&H",
            f"[{outperf_color}]{outperf:+.2f}%[/{outperf_color}]"
        )

        console.print(agent_table)

        # Verdict
        console.print("\n")
        if outperf > 5:
            verdict = "[bold green]AGENT SIGNIFICANTLY OUTPERFORMS[/bold green]"
        elif outperf > 0:
            verdict = "[bold yellow]AGENT SLIGHTLY OUTPERFORMS[/bold yellow]"
        elif outperf > -5:
            verdict = "[bold yellow]AGENT SLIGHTLY UNDERPERFORMS[/bold yellow]"
        else:
            verdict = "[bold red]AGENT SIGNIFICANTLY UNDERPERFORMS[/bold red]"

        console.print(Panel(
            f"Verdict: {verdict}\n\n"
            f"The agent achieved {result.total_return:.2f}% return vs "
            f"{result.buy_hold_return:.2f}% for Buy & Hold.\n"
            f"Risk-adjusted performance (Sharpe): {agent_metrics['sharpe_ratio']:.2f} vs "
            f"{bh_metrics['sharpe_ratio']:.2f}",
            title="[bold]Summary[/bold]",
            border_style="white",
        ))


def run_full_evaluation(
    model_path: str,
    data_path: str,
    feature_columns: list[str],
    output_dir: str = "evaluation",
    vec_normalize_path: Optional[str] = None,
    recurrent: bool = False,
) -> EvaluationResult:
    """
    Run complete evaluation pipeline.

    Args:
        model_path: Path to trained model
        data_path: Path to data CSV
        feature_columns: Feature column names
        output_dir: Directory for outputs
        vec_normalize_path: Path to VecNormalize stats
        recurrent: Whether the model is a recurrent one

    Returns:
        EvaluationResult
    """
    from src.data.feature_engineering import add_technical_indicators, prepare_features_for_env

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    console.print(f"[cyan]Loading model: {model_path}[/cyan]")
    if recurrent:
        model = RecurrentPPO.load(model_path)
    else:
        model = PPO.load(model_path)

    # Load and prepare data
    console.print(f"[cyan]Loading data: {data_path}[/cyan]")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = add_technical_indicators(df)
    df, feature_columns = prepare_features_for_env(df)

    # Create evaluator
    evaluator = AgentEvaluator(
        model=model,
        df=df,
        feature_columns=feature_columns,
        initial_balance=10_000.0,
        vec_normalize_path=vec_normalize_path,
    )

    # Run evaluation
    result = evaluator.evaluate(include_random=True)

    # Print report
    evaluator.print_report(result)

    # Save plots
    plot_path = output_path / "evaluation_report.png"
    evaluator.plot_results(result, save_path=str(plot_path), show=True)

    return result


if __name__ == "__main__":
    # Test with existing model
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/ppo_btc_usd_final"

    result = run_full_evaluation(
        model_path=model_path,
        data_path="data/BTC_USD/1h/BTC-USD_1h_20251208_163042.csv",
        feature_columns=[],  # Will be auto-detected
        output_dir="evaluation",
    )
