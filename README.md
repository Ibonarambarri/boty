# Crypto DRL Trading System

Sistema de trading con Deep Reinforcement Learning (PPO) multi-temporalidad.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# 1. Descargar datos (10 años)
python main.py download --ticker AAPL

# 2. Entrenar modelo
python main.py train --ticker AAPL --timesteps 100000

# 3. Evaluar modelo
python main.py evaluate --model models/AAPL/multi_tf/<fecha>
```
