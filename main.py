#!/usr/bin/env python3
"""
Crypto DRL Trading System
=========================
End-to-end Deep Reinforcement Learning system for cryptocurrency trading.
Supports multi-timeframe observations (1d, 1wk, 1mo).

Usage:
    python main.py download --ticker AAPL --years 10
    python main.py train --ticker AAPL --timesteps 100000
    python main.py evaluate --model models/AAPL/multi_tf/...
    python main.py demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Project imports
from src.data_downloader import YahooDownloader
from src.feature_engineering import (
    add_technical_indicators,
    add_technical_indicators_multi_tf,
    prepare_features_for_env,
    prepare_features_for_env_multi_tf,
    get_feature_stats,
)
from src.train import train, evaluate, TrainingConfig
from src.optimize import run_optimization

console = Console()


def print_banner() -> None:
    """Print application banner."""
    console.print(Panel(
        "[bold cyan]CRYPTO DRL[/bold cyan]",
        title="[bold white]Deep Reinforcement Learning Trading System[/bold white]",
        subtitle="[dim]v2.0.0 - Multi-Timeframe (1d/1wk/1mo)[/dim]",
        border_style="cyan",
    ))


def cmd_download(args: argparse.Namespace) -> None:
    """Download multi-timeframe historical data using Yahoo Finance."""
    console.print(f"\n[bold]Downloading {args.ticker} multi-timeframe data...[/bold]\n")

    downloader = YahooDownloader(data_dir=args.data_dir)

    # Download and save
    save_dir = downloader.download_and_save(
        ticker=args.ticker,
        years=args.years,
        train_ratio=args.train_ratio,
    )

    if save_dir is not None:
        # Show summary
        table = Table(title="Download Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Ticker", args.ticker)
        table.add_row("Years", str(args.years))
        table.add_row("Timeframes", "1d (daily), 1wk (weekly), 1mo (monthly)")
        table.add_row("Directory", str(save_dir))

        console.print(table)
    else:
        console.print("[red]Download failed[/red]")


def cmd_train(args: argparse.Namespace) -> None:
    """Train PPO agent with multi-timeframe data."""
    console.print("\n[bold]Preparing training data...[/bold]\n")

    # Find latest data directory
    data_dir = find_latest_data(args.ticker, args.data_dir)

    if data_dir is None:
        console.print(f"[red]No data found for {args.ticker}[/red]")
        console.print("[yellow]Run 'python main.py download' first[/yellow]")
        sys.exit(1)

    # Check for parquet or csv
    train_path = data_dir / "train.parquet"
    is_multi_tf = True

    if not train_path.exists():
        # Try legacy CSV format
        train_path = data_dir / "train.csv"
        is_multi_tf = False
        if not train_path.exists():
            console.print(f"[red]train.parquet/csv not found in {data_dir}[/red]")
            sys.exit(1)

    console.print(f"[cyan]Loading training data from: {train_path}[/cyan]")

    # Load data
    if train_path.suffix == ".parquet":
        df = pd.read_parquet(train_path)
    else:
        df = pd.read_csv(train_path, index_col=0, parse_dates=True)

    console.print(f"[green]Loaded {len(df):,} rows for training[/green]")

    # Process features based on data type
    if is_multi_tf:
        # Multi-timeframe data
        df = add_technical_indicators_multi_tf(df)
        df, feature_map = prepare_features_for_env_multi_tf(df)
        feature_columns = feature_map  # dict for multi-tf

        console.print(f"[green]Prepared multi-timeframe features:[/green]")
        for tf, features in feature_map.items():
            console.print(f"  {tf}: {len(features)} features")
    else:
        # Single timeframe data (legacy)
        df = add_technical_indicators(df)
        df, feature_columns = prepare_features_for_env(df)
        console.print(f"[green]Prepared {len(feature_columns)} features[/green]")

    # Show feature stats
    if args.verbose:
        if isinstance(feature_columns, dict):
            for tf, features in feature_columns.items():
                stats = get_feature_stats(df, features)
                console.print(f"\n[bold]Feature Statistics ({tf}):[/bold]")
                console.print(stats.head(5).to_string())
        else:
            stats = get_feature_stats(df, feature_columns)
            console.print("\n[bold]Feature Statistics:[/bold]")
            console.print(stats.to_string())

    # Training config
    config = TrainingConfig(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        initial_balance=args.balance,
        save_freq=args.save_freq,
        window_size=args.window,
        # Multi-timeframe settings
        timeframes=['1d', '1wk', '1mo'] if is_multi_tf else None,
        features_per_timeframe=15 if is_multi_tf else None,
        base_timeframe='1d' if is_multi_tf else None,
    )

    # Model save directory
    date_range = data_dir.name
    model_save_dir = Path("models") / args.ticker / "multi_tf" / date_range

    # Train
    model, model_dir = train(
        df=df,
        feature_columns=feature_columns,
        config=config,
        model_save_dir=model_save_dir,
        resume_from=args.resume,
    )

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"[green]Model saved to: {model_dir}/[/green]")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate trained model with comprehensive benchmarks and charts."""
    import json
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from src.evaluation import AgentEvaluator

    model_dir = Path(args.model)

    console.print(f"\n[bold]Comprehensive Evaluation[/bold]\n")
    console.print(f"[cyan]Model directory: {model_dir}[/cyan]")

    # Find model.zip in the directory
    model_path = model_dir / "model.zip"
    if not model_path.exists():
        model_path = model_dir / "model"
        if not model_path.exists() and not Path(f"{model_path}.zip").exists():
            console.print(f"[red]model.zip not found in {model_dir}[/red]")
            sys.exit(1)

    # Find eval data
    # Model path: models/{ticker}/multi_tf/{date_range}/
    # Data path: data/{ticker}/multi_tf/{date_range}/
    try:
        date_range = model_dir.name
        data_type = model_dir.parent.name  # "multi_tf" or interval
        ticker = model_dir.parent.parent.name
        eval_path = Path(args.data_dir) / ticker / data_type / date_range / "eval.parquet"

        if not eval_path.exists():
            # Try CSV format
            eval_path = eval_path.with_suffix(".csv")
    except Exception:
        console.print(f"[red]Could not determine data path from model directory[/red]")
        sys.exit(1)

    if not eval_path.exists():
        console.print(f"[red]eval data not found: {eval_path}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Loading evaluation data from: {eval_path}[/cyan]")

    # Load data
    if eval_path.suffix == ".parquet":
        df = pd.read_parquet(eval_path)
        is_multi_tf = True
    else:
        df = pd.read_csv(eval_path, index_col=0, parse_dates=True)
        is_multi_tf = False

    # Process features
    if is_multi_tf:
        df = add_technical_indicators_multi_tf(df)
        df, feature_columns = prepare_features_for_env_multi_tf(df)
    else:
        df = add_technical_indicators(df)
        df, feature_columns = prepare_features_for_env(df)

    console.print(f"[green]Eval data prepared: {len(df)} samples[/green]")

    # Load model
    console.print(f"[cyan]Loading model: {model_path}[/cyan]")
    if args.recurrent:
        console.print("[yellow]Using RecurrentPPO (LSTM) model[/yellow]")
        model = RecurrentPPO.load(str(model_path).replace(".zip", ""))
    else:
        model = PPO.load(str(model_path).replace(".zip", ""))

    # Check for VecNormalize
    vec_norm_path = model_dir / "vecnorm.pkl"
    if not vec_norm_path.exists():
        vec_norm_path = None
    else:
        console.print(f"[cyan]Loading VecNormalize: {vec_norm_path}[/cyan]")
        vec_norm_path = str(vec_norm_path)

    # Create evaluator
    evaluator = AgentEvaluator(
        model=model,
        df=df,
        feature_columns=feature_columns,
        initial_balance=args.balance,
        vec_normalize_path=vec_norm_path,
    )

    # Run evaluation
    result = evaluator.evaluate(include_random=not args.no_random)

    # Print report
    evaluator.print_report(result)

    # Save evaluation results
    eval_output_dir = model_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Save plot
    if not args.no_plot:
        plot_path = eval_output_dir / "report.png"
        evaluator.plot_results(
            result,
            save_path=str(plot_path),
            show=not args.save_only,
        )

    # Save metrics as JSON
    metrics = {
        "total_return": result.total_return,
        "buy_hold_return": result.buy_hold_return,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "outperformance": result.total_return - result.buy_hold_return,
        "avg_position_size": result.avg_position_size,
        "position_size_std": result.position_size_std,
    }

    metrics_path = eval_output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"\n[green]Evaluation saved to: {eval_output_dir}/[/green]")


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run hyperparameter optimization for the PPO agent."""
    console.print("\n[bold]Starting Hyperparameter Optimization...[/bold]\n")

    # Find latest data directory
    data_dir = find_latest_data(args.ticker, args.data_dir)
    if data_dir is None:
        console.print(f"[red]No data found for {args.ticker}[/red]")
        console.print("[yellow]Run 'python main.py download' first[/yellow]")
        sys.exit(1)

    train_path = data_dir / "train.parquet"
    eval_path = data_dir / "eval.parquet"

    if not train_path.exists() or not eval_path.exists():
        console.print(f"[red]train.parquet or eval.parquet not found in {data_dir}[/red]")
        console.print("[yellow]Optimization requires multi-timeframe .parquet data.[/yellow]")
        sys.exit(1)

    study_name = f"ppo-{args.ticker}-study"
    storage_url = f"sqlite:///{study_name}.db"

    console.print(f"[cyan]Study Name: {study_name}[/cyan]")
    console.print(f"[cyan]Storage: {storage_url}[/cyan]")
    console.print(f"[cyan]Training data: {train_path}[/cyan]")
    console.print(f"[cyan]Evaluation data: {eval_path}[/cyan]")

    run_optimization(
        study_name=study_name,
        storage_url=storage_url,
        train_data_path=str(train_path),
        eval_data_path=str(eval_path),
        n_trials=args.trials,
    )

    console.print(f"\n[bold green]Optimization complete![/bold green]")
    console.print(f"Database with results saved to: {storage_url}")


def cmd_live(args: argparse.Namespace) -> None:
    """Run the live trading bot."""
    from src.live_trading import LiveTrader
    
    console.print("\n[bold red]-- LIVE TRADING MODE --[/bold red]")
    console.print("[yellow]DISCLAIMER: Live trading is extremely risky. This is a demo implementation.[/yellow]")
    console.print("[yellow]Do NOT use this with real money without extensive testing and understanding the risks.[/yellow]\n")
    
    trader = LiveTrader(
        model_path=args.model,
        ticker=args.ticker,
        exchange_id=args.exchange,
        trade_amount=args.amount,
    )
    trader.run()


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a quick demo with synthetic multi-timeframe data."""
    import numpy as np

    console.print("\n[bold]Running Demo Mode (Multi-Timeframe)[/bold]\n")
    console.print("[yellow]Using synthetic data for demonstration[/yellow]\n")

    # Generate synthetic price data
    np.random.seed(42)
    n = 3000
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)

    # Create multi-timeframe data
    df = pd.DataFrame()

    for tf in ['1d', '1wk', '1mo']:
        df[f'open_{tf}'] = price + np.random.randn(n) * 0.1
        df[f'high_{tf}'] = price + abs(np.random.randn(n) * 0.5)
        df[f'low_{tf}'] = price - abs(np.random.randn(n) * 0.5)
        df[f'close_{tf}'] = price + np.random.randn(n) * 0.1
        df[f'volume_{tf}'] = np.random.randint(1000, 10000, n)

    # Add features
    df = add_technical_indicators_multi_tf(df)
    df, feature_map = prepare_features_for_env_multi_tf(df)

    total_features = sum(len(f) for f in feature_map.values())
    console.print(f"[green]Generated {len(df)} samples with {total_features} features (3 timeframes)[/green]\n")

    # Quick training
    config = TrainingConfig(
        total_timesteps=args.timesteps,
        save_freq=args.timesteps // 2,
        update_freq=50,
        n_steps=512,
        batch_size=64,
        window_size=50,
        timeframes=['1d', '1wk', '1mo'],
        features_per_timeframe=15,
        base_timeframe='1d',
    )

    model, path = train(
        df=df,
        feature_columns=feature_map,
        config=config,
        model_name="demo_multi_tf",
    )

    console.print("\n[bold green]Demo complete![/bold green]")


def find_latest_data(
    ticker: str,
    data_dir: str = "data",
) -> Optional[Path]:
    """
    Find the most recent data directory for a ticker.

    Checks for multi_tf format first, then legacy interval format.
    """
    base_path = Path(data_dir) / ticker

    if not base_path.exists():
        return None

    # Check multi_tf first
    multi_tf_path = base_path / "multi_tf"
    if multi_tf_path.exists():
        date_dirs = [d for d in multi_tf_path.iterdir() if d.is_dir()]
        if date_dirs:
            return max(date_dirs, key=lambda p: p.stat().st_mtime)

    # Check for legacy interval directories
    for interval_dir in base_path.iterdir():
        if interval_dir.is_dir() and interval_dir.name != "multi_tf":
            date_dirs = [d for d in interval_dir.iterdir() if d.is_dir()]
            if date_dirs:
                return max(date_dirs, key=lambda p: p.stat().st_mtime)

    return None


def main() -> None:
    """Main entry point."""
    print_banner()

    parser = argparse.ArgumentParser(
        description="Crypto DRL Trading System (Multi-Timeframe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download multi-timeframe historical data")
    dl_parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (e.g., AAPL, BTC-USD, MSFT)")
    dl_parser.add_argument("--years", type=int, default=10, help="Years of history to download")
    dl_parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/eval split ratio")
    dl_parser.add_argument("--data-dir", default="data", help="Data directory")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train PPO agent with multi-timeframe data")
    train_parser.add_argument("--ticker", default="AAPL", help="Ticker symbol")
    train_parser.add_argument("--timesteps", type=int, default=100_000, help="Training steps")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--balance", type=float, default=10_000, help="Initial balance")
    train_parser.add_argument("--window", type=int, default=50, help="Observation window (default: 50)")
    train_parser.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint frequency")
    train_parser.add_argument("--data-dir", default="data", help="Data directory")
    train_parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    train_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--model", required=True, help="Path to model directory")
    eval_parser.add_argument("--balance", type=float, default=10_000, help="Initial balance")
    eval_parser.add_argument("--data-dir", default="data", help="Data directory")
    eval_parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    eval_parser.add_argument("--no-random", action="store_true", help="Skip random agent baseline")
    eval_parser.add_argument("--save-only", action="store_true", help="Save plot without displaying")
    eval_parser.add_argument("--recurrent", action="store_true", help="Use RecurrentPPO (LSTM) model")

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Run hyperparameter optimization")
    opt_parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to optimize for")
    opt_parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    opt_parser.add_argument("--data-dir", default="data", help="Data directory")

    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live trading bot")
    live_parser.add_argument("--model", required=True, help="Path to trained model file")
    live_parser.add_argument("--ticker", default="BTC/USDT", help="Ticker symbol for trading")
    live_parser.add_argument("--exchange", default="binance", help="Exchange ID (e.g., binance, coinbasepro)")
    live_parser.add_argument("--amount", type=float, default=100.0, help="Amount in quote currency for trades")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quick demo with synthetic data")
    demo_parser.add_argument("--timesteps", type=int, default=5_000, help="Demo training steps")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        console.print("\n[yellow]Usage examples:[/yellow]")
        console.print("  python main.py demo                         # Quick demo with synthetic data")
        console.print("  python main.py download --ticker AAPL       # Download 10 years of data")
        console.print("  python main.py download --ticker BTC-USD    # Download crypto data")
        console.print("  python main.py train --ticker AAPL --timesteps 50000")
        console.print("  python main.py evaluate --model models/AAPL/multi_tf/...")
        console.print("\n[cyan]Multi-timeframe: The agent sees 50 bars of 1d, 1wk, and 1mo data simultaneously[/cyan]")
        console.print("[cyan]Context: From ~2.5 months (daily) to ~4 years (monthly) of market history[/cyan]")
        sys.exit(0)

    # Execute command
    commands = {
        "download": cmd_download,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "optimize": cmd_optimize,
        "live": cmd_live,
        "demo": cmd_demo,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if "--verbose" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
