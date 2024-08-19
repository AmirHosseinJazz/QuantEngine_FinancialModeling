# QuantEngine_FinancialModeling

QuantEngine_FinancialModeling is the financial modeling and analysis component of the QuantEngine project. This repository focuses on developing, testing, and optimizing trading strategies using advanced machine learning models and statistical techniques, specifically targeting Bitcoin (BTC) trading.

## Features

- **Machine Learning Models:** Implements and trains time series prediction models (e.g., XGBoost, Random Forest) with hyperparameter tuning via Optuna.
- **MLflow Integration:** Tracks and manages machine learning experiments, records performance metrics, and promotes successful models.
- **Backtesting:** Uses Backtrader for strategy testing, with results stored in TimescaleDB and visualized in Grafana.
- **Dashboarding:** Grafana dashboards provide insights into data quality, model performance, and strategy effectiveness.

## Project Structure

## Getting Started

### Services

- Docker and Docker Compose
- MLflow
- Grafana
- Backtrader

### Installation

Access services: - **MLflow:** `http://localhost:5000` - **Grafana:** `http://localhost:3000`

### Usage

1. **Model Training:**

   - Train and tune models using the `train.py` script in the `scripts/` directory.
   - Track experiments and model performance in MLflow.

2. **Backtesting:**
   - Use the `backtest.py` script to test trading strategies.
   - Analyze strategy performance through Grafana dashboards.

### Future Enhancements

- **Advanced Trading Strategies:** Develop and implement more sophisticated trading strategies, including portfolio optimization.
- **Reinforcement Learning:** Explore reinforcement learning for dynamic trading strategy optimization.
- **Trade Execution:** Integrate real-time trade execution based on model predictions.
