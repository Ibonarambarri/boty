"""
Core Module
===========
Training, evaluation, optimization, and live trading components.
"""

from src.core.train import train, evaluate, TrainingConfig, create_env, create_vec_env
from src.core.evaluation import AgentEvaluator, EvaluationResult, run_full_evaluation
from src.core.optimize import run_optimization
from src.core.live_trading import LiveTrader

__all__ = [
    "train",
    "evaluate",
    "TrainingConfig",
    "create_env",
    "create_vec_env",
    "AgentEvaluator",
    "EvaluationResult",
    "run_full_evaluation",
    "run_optimization",
    "LiveTrader",
]
