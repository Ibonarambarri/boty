# Crypto DRL Trading System

Sistema de trading con Deep Reinforcement Learning (PPO) usando observaciones multi-temporalidad.

## Características

- **Multi-temporalidad**: El agente ve simultáneamente 50 barras de 3 temporalidades (1d, 1wk, 1mo)
- **Contexto amplio**: Desde ~2.5 meses (diario) hasta ~4 años (mensual) de histórico
- **15 indicadores técnicos** por temporalidad (45 features totales por barra)
- **Observation space**: 2,250 valores (50 barras × 15 features × 3 timeframes)
- **Acciones**: Hold, Long, Short con TP/SL automático

## Requisitos

- Python 3.10+
- ~4GB RAM mínimo
- GPU opcional (acelera entrenamiento)

## Instalación

### 1. Clonar/Descargar el proyecto

```bash
cd /ruta/a/boty
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso Paso a Paso

### Paso 1: Descargar datos históricos

Descarga 10 años de datos multi-temporalidad desde Yahoo Finance:

```bash
# Acciones
python main.py download --ticker AAPL --years 10

# Crypto
python main.py download --ticker BTC-USD --years 10

# Otros ejemplos
python main.py download --ticker MSFT --years 10
python main.py download --ticker ETH-USD --years 5
```

**Output esperado:**
```
Downloading AAPL multi-timeframe data...
  Timeframes: 1d (daily), 1wk (weekly), 1mo (monthly)
  History: 10 years

Downloaded data:
  1d:  2,517 rows (2015-01-02 to 2025-01-06)
  1wk: 523 rows (2015-01-05 to 2025-01-06)
  1mo: 121 rows (2015-01-01 to 2025-01-01)

Aligning timeframes...
Aligned data: 2,517 rows

Saved to: data/AAPL/multi_tf/2015_01_02-2025_01_06/
  train.parquet: 2,013 rows (80%)
  eval.parquet:  504 rows (20%)
```

Los datos se guardan en:
```
data/
└── AAPL/
    └── multi_tf/
        └── 2015_01_02-2025_01_06/
            ├── train.parquet    # 80% para entrenamiento
            ├── eval.parquet     # 20% para evaluación
            └── metadata.json
```

### Paso 2: Entrenar el modelo

```bash
# Entrenamiento básico (100k steps, ~30-60 min CPU)
python main.py train --ticker AAPL --timesteps 100000

# Entrenamiento más largo (recomendado para producción)
python main.py train --ticker AAPL --timesteps 500000

# Con parámetros personalizados
python main.py train --ticker AAPL \
    --timesteps 200000 \
    --lr 0.0003 \
    --balance 10000 \
    --window 50 \
    --save-freq 20000
```

**Parámetros de entrenamiento:**
| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `--timesteps` | 100,000 | Pasos totales de entrenamiento |
| `--lr` | 3e-4 | Learning rate |
| `--balance` | 10,000 | Balance inicial simulado |
| `--window` | 50 | Barras de histórico por temporalidad |
| `--save-freq` | 10,000 | Guardar checkpoint cada N pasos |
| `--resume` | None | Ruta a modelo para continuar entrenamiento |

Durante el entrenamiento verás un dashboard en tiempo real:
```
╭─────────────────── Crypto DRL Trading System (Multi-TF) ───────────────────╮
│ Step: 45,000/100,000 [████████████░░░░░░░░░░░░] 45%                        │
│                                                                             │
│ Episode: 127    Balance: $10,847    PnL: +8.47%                            │
│ Trades: 312     Win Rate: 54.2%     Best Reward: 0.0892                    │
│                                                                             │
│ Recent: L +$42 | S -$18 | L +$31 | S +$27 | L -$15                         │
╰─────────────────────────────────────────────────────────────────────────────╯
```

El modelo se guarda en:
```
models/
└── AAPL/
    └── multi_tf/
        └── 2015_01_02-2025_01_06/
            ├── model.zip        # Modelo entrenado
            ├── vecnorm.pkl      # Normalización de observaciones
            └── checkpoints/     # Checkpoints intermedios
```

### Paso 3: Evaluar el modelo

```bash
python main.py evaluate --model models/AAPL/multi_tf/2015_01_02-2025_01_06
```

**Output esperado:**
```
╭─────────────────── Evaluation Report ───────────────────╮
│                                                          │
│  Total Return:     +23.45%                               │
│  Buy & Hold:       +18.32%                               │
│  Outperformance:   +5.13%                                │
│                                                          │
│  Sharpe Ratio:     1.42                                  │
│  Sortino Ratio:    1.87                                  │
│  Max Drawdown:     -12.3%                                │
│                                                          │
│  Total Trades:     89                                    │
│  Win Rate:         56.2%                                 │
│  Avg Win:          +2.1%                                 │
│  Avg Loss:         -1.0%                                 │
│                                                          │
╰──────────────────────────────────────────────────────────╯
```

Los resultados se guardan en:
```
models/AAPL/multi_tf/2015_01_02-2025_01_06/evaluation/
├── report.png      # Gráfico de equity curve
└── metrics.json    # Métricas en JSON
```

### Paso 4 (Opcional): Demo rápido

Para probar el sistema sin descargar datos:

```bash
# Demo con datos sintéticos (5k steps, ~2 min)
python main.py demo --timesteps 5000
```

## Configuración del Trading

Los parámetros de trading están en `src/envs/trading_env.py`:

```python
@dataclass
class EnvConfig:
    initial_balance: float = 10_000.0    # Balance inicial
    position_size_pct: float = 0.10      # 10% del balance por trade
    take_profit_pct: float = 0.02        # +2% TP
    stop_loss_pct: float = 0.01          # -1% SL
    commission_pct: float = 0.001        # 0.1% comisión
    window_size: int = 50                # Barras de histórico
```

**Nota**: El `position_size_pct` usa el **balance actual**, no el inicial. Si ganas, arriesgas más en valor absoluto; si pierdes, arriesgas menos.

## Indicadores Técnicos (15 por temporalidad)

| Categoría | Indicadores |
|-----------|-------------|
| Returns | log_return, log_return_high, log_return_low |
| Momentum | RSI normalizado, MACD normalizado, MACD histogram |
| Volatilidad | BB position, ATR normalizado |
| Volumen | volume_norm, volume_ratio |
| Price Action | return_5, return_10, return_20 |
| Candles | candle_body_ratio, candle_direction |

## Estructura del Proyecto

```
boty/
├── main.py                    # CLI principal
├── requirements.txt           # Dependencias
├── README.md                  # Este archivo
├── src/
│   ├── data_downloader.py     # Descarga de Yahoo Finance
│   ├── feature_engineering.py # Indicadores técnicos
│   ├── train.py               # Pipeline de entrenamiento PPO
│   ├── evaluation.py          # Evaluación y métricas
│   ├── callbacks.py           # Callbacks para dashboard
│   ├── dashboard.py           # TUI dashboard
│   └── envs/
│       └── trading_env.py     # Gymnasium environment
├── data/                      # Datos descargados
└── models/                    # Modelos entrenados
```

## Troubleshooting

### Error: No data found for TICKER
```bash
# Primero descarga los datos
python main.py download --ticker AAPL
```

### Error: CUDA out of memory
```bash
# Usa CPU
CUDA_VISIBLE_DEVICES="" python main.py train --ticker AAPL
```

### Entrenamiento muy lento
- Reduce `--timesteps` para pruebas iniciales
- Usa GPU si está disponible
- Reduce `--window` (afecta calidad del modelo)

### El modelo no aprende (reward no mejora)
- Aumenta `--timesteps` (mínimo 200k para resultados decentes)
- Prueba diferentes `--lr` (3e-4, 1e-4, 1e-3)
- Verifica que los datos tienen suficiente volatilidad

## Tips para Mejor Rendimiento

1. **Más datos = mejor**: Usa `--years 10` para máximo histórico
2. **Más steps = mejor**: Mínimo 200k para resultados serios, 500k+ para producción
3. **Evalúa regularmente**: Los checkpoints permiten comparar diferentes puntos del entrenamiento
4. **Prueba diferentes activos**: Algunos activos son más predecibles que otros
5. **No sobreoptimices**: Un modelo que funciona en eval pero no en live está sobreajustado

## Licencia

MIT
