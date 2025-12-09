#!/usr/bin/env python3
"""
Crypto DRL Swing Trading System
===============================
End-to-end Deep Reinforcement Learning system for CRYPTOCURRENCY swing trading.
Uses Binance data via CCXT with multi-timeframe observations (1h, 1d).

Usage:
    python main.py download --ticker BTC/USDT --days 730
    python main.py train --ticker BTC/USDT --timesteps 100000
    python main.py evaluate --model models/BTC_USDT/multi_tf/...
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
from src.data.data_downloader import BinanceDownloader
from src.data.feature_engineering import (
    add_technical_indicators,
    add_technical_indicators_multi_tf,
    prepare_features_for_env,
    prepare_features_for_env_multi_tf,
    get_feature_stats,
)
from src.core.train import train, evaluate, TrainingConfig
from src.core.optimize import run_optimization

console = Console()


def print_banner() -> None:
    """Print application banner."""
    console.print(Panel(
        "[bold cyan]CRYPTO DRL SWING TRADER[/bold cyan]",
        title="[bold white]Deep Reinforcement Learning for Crypto[/bold white]",
        subtitle="[dim]v3.0.0 - Multi-Timeframe (1h/1d) - Binance[/dim]",
        border_style="cyan",
    ))


def cmd_download(args: argparse.Namespace) -> None:
    """Download multi-timeframe historical crypto data from Binance."""
    console.print(f"\n[bold]Downloading {args.ticker} crypto data from Binance...[/bold]\n")

    downloader = BinanceDownloader(data_dir=args.data_dir)

    # Download and save
    save_dir = downloader.download_and_save(
        symbol=args.ticker,
        days=args.days,
        train_ratio=args.train_ratio,
    )

    if save_dir is not None:
        # Show summary
        table = Table(title="Crypto Data Download Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Symbol", args.ticker)
        table.add_row("Days", str(args.days))
        table.add_row("Timeframes", "1h (hourly), 1d (daily)")
        table.add_row("Exchange", "Binance")
        table.add_row("Directory", str(save_dir))

        console.print(table)
    else:
        console.print("[red]Download failed - check symbol format (e.g., BTC/USDT)[/red]")


def cmd_train(args: argparse.Namespace) -> None:
    """Train PPO agent with multi-timeframe crypto data."""
    console.print("\n[bold]Preparing crypto training data...[/bold]\n")

    # Normalize ticker for file path (BTC/USDT -> BTC_USDT)
    ticker_safe = args.ticker.replace('/', '_')

    # Find latest data directory
    data_dir = find_latest_data(ticker_safe, args.data_dir)

    if data_dir is None:
        console.print(f"[red]No data found for {args.ticker}[/red]")
        console.print("[yellow]Run 'python main.py download --ticker {args.ticker}' first[/yellow]")
        sys.exit(1)

    # Check for parquet
    train_path = data_dir / "train.parquet"
    is_multi_tf = True

    if not train_path.exists():
        console.print(f"[red]train.parquet not found in {data_dir}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Loading crypto training data from: {train_path}[/cyan]")

    # Load data
    df = pd.read_parquet(train_path)

    console.print(f"[green]Loaded {len(df):,} hourly candles for training[/green]")

    # Process features (always multi-timeframe for crypto)
    df = add_technical_indicators_multi_tf(df)
    df, feature_map = prepare_features_for_env_multi_tf(df)
    feature_columns = feature_map  # dict for multi-tf

    console.print(f"[green]Prepared multi-timeframe features:[/green]")
    for tf, features in feature_map.items():
        console.print(f"  {tf}: {len(features)} features")

    # Show feature stats
    if args.verbose:
        for tf, features in feature_columns.items():
            stats = get_feature_stats(df, features)
            console.print(f"\n[bold]Feature Statistics ({tf}):[/bold]")
            console.print(stats.head(5).to_string())

    # Training config for crypto swing trading
    config = TrainingConfig(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        initial_balance=args.balance,
        save_freq=args.save_freq,
        window_size=args.window,
        # Crypto swing trading: 1h base + 1d context
        timeframes=['1h', '1d'],
        features_per_timeframe=17,
        base_timeframe='1h',
    )

    # Model save directory
    date_range = data_dir.name
    model_save_dir = Path("models") / ticker_safe / "multi_tf" / date_range

    # Train
    model, model_dir = train(
        df=df,
        feature_columns=feature_columns,
        config=config,
        model_save_dir=model_save_dir,
        resume_from=args.resume,
    )

    console.print(f"\n[bold green]Crypto swing trading model trained![/bold green]")
    console.print(f"[green]Model saved to: {model_dir}/[/green]")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate trained model with comprehensive benchmarks and charts."""
    import json
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from src.core.evaluation import AgentEvaluator

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
    from src.core.live_trading import LiveTrader
    
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
    """Run a quick demo with synthetic crypto multi-timeframe data."""
    import numpy as np

    console.print("\n[bold]Running Crypto Swing Trading Demo[/bold]\n")
    console.print("[yellow]Using synthetic BTC-like data for demonstration[/yellow]\n")

    # Generate synthetic BTC price data
    np.random.seed(42)
    n = 3000
    price = 50000 + np.cumsum(np.random.randn(n) * 100)  # BTC-like volatility

    # Create multi-timeframe data (1h, 1d for crypto swing)
    df = pd.DataFrame()

    for tf in ['1h', '1d']:
        df[f'open_{tf}'] = price + np.random.randn(n) * 10
        df[f'high_{tf}'] = price + abs(np.random.randn(n) * 50)
        df[f'low_{tf}'] = price - abs(np.random.randn(n) * 50)
        df[f'close_{tf}'] = price + np.random.randn(n) * 10
        df[f'volume_{tf}'] = np.random.randint(100, 1000, n)

    # Add features
    df = add_technical_indicators_multi_tf(df)
    df, feature_map = prepare_features_for_env_multi_tf(df)

    total_features = sum(len(f) for f in feature_map.values())
    console.print(f"[green]Generated {len(df)} hourly samples with {total_features} features (2 timeframes)[/green]\n")

    # Quick training config for crypto
    config = TrainingConfig(
        total_timesteps=args.timesteps,
        save_freq=args.timesteps // 2,
        update_freq=50,
        n_steps=512,
        batch_size=64,
        window_size=50,
        timeframes=['1h', '1d'],
        features_per_timeframe=17,
        base_timeframe='1h',
    )

    model, path = train(
        df=df,
        feature_columns=feature_map,
        config=config,
        model_name="demo_crypto_swing",
    )

    console.print("\n[bold green]Crypto swing trading demo complete![/bold green]")


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
        description="Crypto DRL Swing Trading System (Multi-Timeframe 1h/1d)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command - CRYPTO ONLY via Binance
    dl_parser = subparsers.add_parser("download", help="Download crypto data from Binance")
    dl_parser.add_argument("--ticker", default="BTC/USDT", help="Crypto pair (e.g., BTC/USDT, ETH/USDT)")
    dl_parser.add_argument("--days", type=int, default=730, help="Days of history (default: 730 = 2 years)")
    dl_parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/eval split ratio")
    dl_parser.add_argument("--data-dir", default="data", help="Data directory")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train PPO agent for crypto swing trading")
    train_parser.add_argument("--ticker", default="BTC/USDT", help="Crypto pair (e.g., BTC/USDT)")
    train_parser.add_argument("--timesteps", type=int, default=100_000, help="Training steps")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--balance", type=float, default=10_000, help="Initial balance (USDT)")
    train_parser.add_argument("--window", type=int, default=50, help="Observation window (default: 50)")
    train_parser.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint frequency")
    train_parser.add_argument("--data-dir", default="data", help="Data directory")
    train_parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    train_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained crypto model")
    eval_parser.add_argument("--model", required=True, help="Path to model directory")
    eval_parser.add_argument("--balance", type=float, default=10_000, help="Initial balance (USDT)")
    eval_parser.add_argument("--data-dir", default="data", help="Data directory")
    eval_parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    eval_parser.add_argument("--no-random", action="store_true", help="Skip random agent baseline")
    eval_parser.add_argument("--save-only", action="store_true", help="Save plot without displaying")
    eval_parser.add_argument("--recurrent", action="store_true", help="Use RecurrentPPO (LSTM) model")

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Run hyperparameter optimization")
    opt_parser.add_argument("--ticker", default="BTC/USDT", help="Crypto pair to optimize for")
    opt_parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    opt_parser.add_argument("--data-dir", default="data", help="Data directory")

    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live crypto trading bot")
    live_parser.add_argument("--model", required=True, help="Path to trained model file")
    live_parser.add_argument("--ticker", default="BTC/USDT", help="Crypto pair for trading")
    live_parser.add_argument("--exchange", default="binance", help="Exchange ID (e.g., binance)")
    live_parser.add_argument("--amount", type=float, default=100.0, help="Amount in USDT per trade")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quick demo with synthetic crypto data")
    demo_parser.add_argument("--timesteps", type=int, default=5_000, help="Demo training steps")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        console.print("\n[yellow]Usage examples (Crypto Only):[/yellow]")
        console.print("  python main.py demo                              # Quick demo with synthetic BTC data")
        console.print("  python main.py download --ticker BTC/USDT        # Download 2 years of BTC hourly data")
        console.print("  python main.py download --ticker ETH/USDT --days 365")
        console.print("  python main.py train --ticker BTC/USDT --timesteps 50000")
        console.print("  python main.py evaluate --model models/BTC_USDT/multi_tf/...")
        console.print("\n[cyan]Crypto Swing Trading: 1h (base) + 1d (context) timeframes[/cyan]")
        console.print("[cyan]Agent sees 50 bars of hourly data with daily context for swing trades[/cyan]")
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
