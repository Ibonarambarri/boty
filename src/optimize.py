"""
Hyperparameter Optimization Module
================================
Uses Optuna to find the best hyperparameters for the PPO agent and trading environment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import optuna
import pandas as pd
from sb3_contrib import RecurrentPPO

from src.train import TrainingConfig, create_vec_env
from src.evaluation import AgentEvaluator
from src.feature_engineering import (
    add_technical_indicators_multi_tf,
    prepare_features_for_env_multi_tf,
)

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.trial.Trial,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_map: dict[str, list[str]],
) -> float:
    """
    Optuna objective function to train and evaluate a model.

    Args:
        trial: An Optuna trial object.
        train_df: DataFrame for training.
        eval_df: DataFrame for evaluation.
        feature_map: Dictionary of features for multi-timeframe.

    Returns:
        The metric to optimize (e.g., Sharpe ratio).
    """
    # 1. Suggest Hyperparameters
    # PPO
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)

    # Environment
    take_profit_pct = trial.suggest_float("take_profit_pct", 0.01, 0.1, log=True)
    stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.01, 0.1, log=True)
    slippage_pct = trial.suggest_float("slippage_pct", 1e-5, 1e-3, log=True)

    # Architecture
    net_arch_size = trial.suggest_categorical("net_arch_size", [32, 64, 128, 256])
    net_arch = [net_arch_size, net_arch_size]
    # LSTM-specific
    n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2])
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128, 256])


    config = TrainingConfig(
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        net_arch=net_arch,
        n_lstm_layers=n_lstm_layers,
        lstm_hidden_size=lstm_hidden_size,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        slippage_pct=slippage_pct,
        total_timesteps=50_000,  # Shorter training for optimization
        save_freq=1_000_000, # Don't save intermediate models
        update_freq=1000, # Less frequent updates to speed up
    )

    # 2. Train the Model
    try:
        vec_env = create_vec_env(train_df, feature_map, config, normalize=True)
        
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
            verbose=0,
        )
        
        # Disable TUI dashboard for optimization
        model.learn(total_timesteps=config.total_timesteps, progress_bar=False)

        # Save vecnorm stats
        vecnorm_path = Path("./vecnorm_trial.pkl")
        vec_env.save(str(vecnorm_path))
        vec_env.close()
        del vec_env

    except (AssertionError, ValueError) as e:
        # Prune trial if it fails (e.g. due to invalid hyperparam combination)
        logger.warning(f"Trial failed with error: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    # 3. Evaluate the Model
    evaluator = AgentEvaluator(
        model=model,
        df=eval_df,
        feature_columns=feature_map,
        vec_normalize_path=str(vecnorm_path),
    )
    
    agent_equity, _, _ = evaluator.run_agent_episode()
    metrics = evaluator.calculate_metrics(agent_equity)
    
    # Clean up
    vecnorm_path.unlink(missing_ok=True)
    del model

    # 4. Return Objective Value (maximize Sharpe ratio)
    sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
    
    # Handle cases where Sharpe is NaN or infinity
    if sharpe_ratio != sharpe_ratio or abs(sharpe_ratio) == float('inf'):
        return -1.0

    return sharpe_ratio


def run_optimization(
    study_name: str,
    storage_url: str,
    train_data_path: str,
    eval_data_path: str,
    n_trials: int = 100,
) -> None:
    """
    Run hyperparameter optimization using Optuna.

    Args:
        study_name: Name for the Optuna study.
        storage_url: Database URL for Optuna storage.
        train_data_path: Path to training data (.parquet).
        eval_data_path: Path to evaluation data (.parquet).
        n_trials: Number of optimization trials to run.
    """
    # Load and prepare data
    train_df = pd.read_parquet(train_data_path)
    eval_df = pd.read_parquet(eval_data_path)

    # This assumes multi-timeframe data is already prepared
    feature_map = {
        '1d': [c for c in train_df.columns if c.endswith('_1d')],
        '1wk': [c for c in train_df.columns if c.endswith('_1wk')],
        '1mo': [c for c in train_df.columns if c.endswith('_1mo')],
    }

    # Validate feature map
    for tf, features in feature_map.items():
        if not features:
            raise ValueError(
                f"No features found for timeframe '{tf}'. "
                f"Data must have columns ending with '_{tf}' suffix."
            )

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, train_df, eval_df, feature_map),
        n_trials=n_trials,
        n_jobs=1,  # Set to -1 to use all available CPUs, but can be unstable
        show_progress_bar=True,
    )

    # Print results
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f} (Sharpe Ratio)")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
