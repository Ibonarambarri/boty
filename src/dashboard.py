"""
Trading Dashboard Module
========================
Professional TUI dashboard using Rich library for training visualization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


# ASCII chart characters (sorted by height)
SPARK_CHARS = " ▁▂▃▄▅▆▇█"


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    current_step: int = 0
    total_steps: int = 0
    episode: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    balance: float = 10000.0
    net_worth: float = 10000.0
    pnl_pct: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    fps: float = 0.0
    mean_reward: float = 0.0
    best_reward: float = float('-inf')
    current_position: str = "NONE"
    current_price: float = 0.0

    # History for charts
    reward_history: deque = field(default_factory=lambda: deque(maxlen=50))
    balance_history: deque = field(default_factory=lambda: deque(maxlen=50))


def create_sparkline(values: list[float], width: int = 40) -> str:
    """
    Create ASCII sparkline chart from values.

    Args:
        values: List of numeric values
        width: Chart width in characters

    Returns:
        Sparkline string
    """
    if not values:
        return " " * width

    # Resample if needed
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    elif len(values) < width:
        values = list(values) + [values[-1]] * (width - len(values))

    # Normalize to 0-8 range for spark chars
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1

    normalized = [
        int((v - min_val) / val_range * 8)
        for v in values
    ]

    return "".join(SPARK_CHARS[min(8, max(0, n))] for n in normalized)


class TradingDashboard:
    """
    Professional trading dashboard with Rich TUI.

    Features:
    - Real-time metrics table
    - ASCII sparkline charts
    - Progress bars
    - Color-coded indicators
    """

    def __init__(
        self,
        total_steps: int,
        initial_balance: float = 10000.0,
        title: str = "Crypto DRL Trading System",
    ) -> None:
        """
        Initialize the dashboard.

        Args:
            total_steps: Total training steps
            initial_balance: Starting balance
            title: Dashboard title
        """
        self.total_steps = total_steps
        self.initial_balance = initial_balance
        self.title = title
        self.start_time = datetime.now()

        # Metrics
        self.metrics = TrainingMetrics(
            total_steps=total_steps,
            balance=initial_balance,
            net_worth=initial_balance,
        )

        # Progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
        )
        self.task_id: Optional[TaskID] = None

        # Live context reference (set during training)
        self._live: Optional[Live] = None

    def start(self) -> None:
        """Start the progress tracking."""
        self.task_id = self.progress.add_task(
            "Training",
            total=self.total_steps,
        )
        self.start_time = datetime.now()

    def _create_header(self) -> Panel:
        """Create header panel."""
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=2)
        grid.add_column(justify="right", ratio=1)

        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]

        grid.add_row(
            Text("v1.0.0", style="dim"),
            Text(self.title, style="bold magenta"),
            Text(f"Elapsed: {elapsed_str}", style="dim"),
        )

        return Panel(grid, style="bold white on dark_blue")

    def _create_metrics_table(self) -> Panel:
        """Create metrics table panel."""
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Metric", style="dim", width=15)
        table.add_column("Value", justify="right")
        table.add_column("Metric", style="dim", width=15)
        table.add_column("Value", justify="right")

        m = self.metrics

        # Color coding for balance
        balance_color = "green" if m.balance >= self.initial_balance else "red"
        pnl_color = "green" if m.pnl_pct >= 0 else "red"

        # Row 1: Step info and Balance
        table.add_row(
            "Step",
            f"{m.current_step:,}/{m.total_steps:,}",
            "Balance",
            f"[{balance_color}]${m.balance:,.2f}[/{balance_color}]",
        )

        # Row 2: Episode and Net Worth
        table.add_row(
            "Episode",
            str(m.episode),
            "Net Worth",
            f"[{pnl_color}]${m.net_worth:,.2f}[/{pnl_color}]",
        )

        # Row 3: FPS and PnL %
        table.add_row(
            "FPS",
            f"{m.fps:.1f}",
            "PnL",
            f"[{pnl_color}]{m.pnl_pct:+.2f}%[/{pnl_color}]",
        )

        # Row 4: Trades and Win Rate
        win_color = "green" if m.win_rate >= 50 else "yellow" if m.win_rate >= 30 else "red"
        table.add_row(
            "Trades",
            str(m.total_trades),
            "Win Rate",
            f"[{win_color}]{m.win_rate:.1f}%[/{win_color}]",
        )

        # Row 5: Rewards
        table.add_row(
            "Episode Reward",
            f"{m.episode_reward:.4f}",
            "Mean Reward",
            f"{m.mean_reward:.4f}",
        )

        # Row 6: Position and Price
        pos_color = {
            "NONE": "dim",
            "LONG": "green",
            "SHORT": "red",
        }.get(m.current_position, "dim")

        table.add_row(
            "Position",
            f"[{pos_color}]{m.current_position}[/{pos_color}]",
            "Price",
            f"${m.current_price:,.2f}",
        )

        return Panel(table, title="[bold]Trading Metrics[/bold]", border_style="cyan")

    def _create_charts_panel(self) -> Panel:
        """Create ASCII charts panel."""
        content_lines = []

        # Reward sparkline
        reward_spark = create_sparkline(list(self.metrics.reward_history), width=50)
        content_lines.append(
            Text.assemble(
                ("Reward History: ", "bold yellow"),
                (reward_spark, "yellow"),
            )
        )

        # Add min/max labels for reward
        if self.metrics.reward_history:
            min_r = min(self.metrics.reward_history)
            max_r = max(self.metrics.reward_history)
            content_lines.append(
                Text(f"  Min: {min_r:.4f}  Max: {max_r:.4f}", style="dim")
            )
        else:
            content_lines.append(Text("  No data yet", style="dim"))

        content_lines.append(Text(""))

        # Balance sparkline
        balance_spark = create_sparkline(list(self.metrics.balance_history), width=50)
        balance_color = "green" if self.metrics.balance >= self.initial_balance else "red"
        content_lines.append(
            Text.assemble(
                ("Balance History: ", f"bold {balance_color}"),
                (balance_spark, balance_color),
            )
        )

        # Add min/max labels for balance
        if self.metrics.balance_history:
            min_b = min(self.metrics.balance_history)
            max_b = max(self.metrics.balance_history)
            content_lines.append(
                Text(f"  Min: ${min_b:,.2f}  Max: ${max_b:,.2f}", style="dim")
            )
        else:
            content_lines.append(Text("  No data yet", style="dim"))

        return Panel(
            Group(*content_lines),
            title="[bold]Performance Charts[/bold]",
            border_style="green",
        )

    def _create_progress_panel(self) -> Panel:
        """Create progress bar panel."""
        return Panel(
            self.progress,
            title="[bold]Training Progress[/bold]",
            border_style="blue",
        )

    def _create_footer(self) -> Panel:
        """Create footer panel."""
        footer_text = Text.assemble(
            ("Actions: ", "bold"),
            ("Ctrl+C", "bold red"),
            (" to stop  |  ", "dim"),
            ("Model saves to ", "dim"),
            ("models/", "cyan"),
            ("  |  ", "dim"),
            (f"Best Reward: {self.metrics.best_reward:.4f}", "green"),
        )
        return Panel(footer_text, style="dim")

    def get_renderable(self) -> Layout:
        """
        Get the complete dashboard layout.

        Returns:
            Rich Layout object
        """
        layout = Layout()

        # Create structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        # Body split
        layout["body"].split_column(
            Layout(name="metrics", size=12),
            Layout(name="charts", size=8),
            Layout(name="progress", size=5),
        )

        # Assign panels
        layout["header"].update(self._create_header())
        layout["metrics"].update(self._create_metrics_table())
        layout["charts"].update(self._create_charts_panel())
        layout["progress"].update(self._create_progress_panel())
        layout["footer"].update(self._create_footer())

        return layout

    def update(
        self,
        step: Optional[int] = None,
        episode: Optional[int] = None,
        episode_reward: Optional[float] = None,
        episode_length: Optional[int] = None,
        balance: Optional[float] = None,
        net_worth: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        total_trades: Optional[int] = None,
        win_rate: Optional[float] = None,
        fps: Optional[float] = None,
        mean_reward: Optional[float] = None,
        position: Optional[str] = None,
        current_price: Optional[float] = None,
    ) -> None:
        """
        Update dashboard metrics.

        Args:
            step: Current training step
            episode: Current episode number
            episode_reward: Total reward for current episode
            episode_length: Length of current episode
            balance: Current balance
            net_worth: Current net worth
            pnl_pct: PnL percentage
            total_trades: Total trades in episode
            win_rate: Win rate percentage
            fps: Steps per second
            mean_reward: Mean reward over recent episodes
            position: Current position (NONE, LONG, SHORT)
            current_price: Current asset price
        """
        m = self.metrics

        if step is not None:
            m.current_step = step
            if self.task_id is not None:
                self.progress.update(self.task_id, completed=step)

        if episode is not None:
            m.episode = episode

        if episode_reward is not None:
            m.episode_reward = episode_reward
            m.reward_history.append(episode_reward)
            if episode_reward > m.best_reward:
                m.best_reward = episode_reward

        if episode_length is not None:
            m.episode_length = episode_length

        if balance is not None:
            m.balance = balance
            m.balance_history.append(balance)

        if net_worth is not None:
            m.net_worth = net_worth

        if pnl_pct is not None:
            m.pnl_pct = pnl_pct

        if total_trades is not None:
            m.total_trades = total_trades

        if win_rate is not None:
            m.win_rate = win_rate

        if fps is not None:
            m.fps = fps

        if mean_reward is not None:
            m.mean_reward = mean_reward

        if position is not None:
            m.current_position = position

        if current_price is not None:
            m.current_price = current_price

        # Refresh live display if available
        if self._live is not None:
            self._live.update(self.get_renderable())

    def set_live(self, live: Live) -> None:
        """Set the Live context for automatic updates."""
        self._live = live


if __name__ == "__main__":
    # Demo the dashboard
    import time
    import random

    console = Console()

    dashboard = TradingDashboard(total_steps=10000, initial_balance=10000.0)
    dashboard.start()

    with Live(dashboard.get_renderable(), refresh_per_second=4, console=console) as live:
        dashboard.set_live(live)

        for step in range(1, 101):
            # Simulate metrics
            balance = 10000 + random.uniform(-500, 500)
            reward = random.uniform(-0.1, 0.15)

            dashboard.update(
                step=step * 100,
                episode=step // 10,
                episode_reward=reward,
                balance=balance,
                net_worth=balance + random.uniform(-100, 100),
                pnl_pct=(balance - 10000) / 100,
                total_trades=step,
                win_rate=random.uniform(40, 60),
                fps=random.uniform(50, 100),
                mean_reward=random.uniform(-0.05, 0.05),
                position=random.choice(["NONE", "LONG", "SHORT"]),
                current_price=random.uniform(40000, 45000),
            )

            time.sleep(0.1)

    console.print("[bold green]Dashboard demo complete![/bold green]")
