import numpy as np
import pandas as pd
from scipy.optimize import minimize


def minimum_variance(returns: pd.DataFrame) -> np.ndarray:
    n = returns.shape[1]
    cov = returns.cov().values

    def portfolio_variance(w):
        return w @ cov @ w

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(portfolio_variance, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else w0


def risk_parity(returns: pd.DataFrame) -> np.ndarray:
    n = returns.shape[1]
    cov = returns.cov().values

    def risk_contribution_diff(w):
        portfolio_var = w @ cov @ w
        marginal_risk = cov @ w
        risk_contrib  = w * marginal_risk / (portfolio_var + 1e-9)
        target        = np.ones(n) / n
        return np.sum((risk_contrib - target) ** 2)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0.01, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(risk_contribution_diff, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = result.x if result.success else w0
    return w / w.sum()


def momentum(prices: pd.DataFrame, lookback: int = 60, top_k: int = 10) -> np.ndarray:
    n = prices.shape[1]
    if len(prices) < lookback:
        return np.ones(n) / n

    recent_return = prices.iloc[-1] / prices.iloc[-lookback] - 1
    top_idx = recent_return.nlargest(top_k).index
    weights = np.zeros(n)

    for i, col in enumerate(prices.columns):
        if col in top_idx:
            weights[i] = 1.0 / top_k
    return weights


def market_cap_weight(prices: pd.DataFrame) -> np.ndarray:
    current_prices = prices.iloc[-1].values
    weights = current_prices / current_prices.sum()
    return weights


def backtest_baseline(
    strategy: str,
    prices_arr: np.ndarray,
    returns_arr: np.ndarray,
    prices_df: pd.DataFrame,
    initial_cash: float = 10_000_000,
    rebalance_freq: int = 20,
    transaction_cost: float = 0.001,
    lookback: int = 60,
    top_k: int = 10,
) -> np.ndarray:
    T, N = prices_arr.shape
    portfolio_val = initial_cash
    weights = np.ones(N) / N
    history = [portfolio_val]

    for t in range(1, T):
        daily_return = prices_arr[t] / prices_arr[t-1] - 1
        portfolio_val *= (1 + np.dot(weights, daily_return))

        if t % rebalance_freq == 0 and t >= lookback:
            old_weights  = weights.copy()
            ret_window   = pd.DataFrame(returns_arr[max(0, t-lookback):t], columns=prices_df.columns)
            price_window = prices_df.iloc[max(0, t-lookback):t]

            if strategy == "mv":
                new_weights = minimum_variance(ret_window)
            elif strategy == "rp":
                new_weights = risk_parity(ret_window)
            elif strategy == "momentum":
                new_weights = momentum(price_window, lookback, top_k)
            elif strategy == "market_cap":
                new_weights = market_cap_weight(price_window)
            else:
                new_weights = np.ones(N) / N

            turnover = np.abs(new_weights - old_weights).sum() / 2
            cost = turnover * transaction_cost * portfolio_val
            portfolio_val -= cost
            weights = new_weights

        history.append(portfolio_val)

    return np.array(history)


def compute_metrics(history: np.ndarray, rf_rate: float = 0.05) -> dict:
    returns      = np.diff(history) / history[:-1]
    total_return = (history[-1] / history[0] - 1) * 100
    ann_return   = (1 + total_return / 100) ** (252 / max(len(returns), 1)) - 1
    ann_vol      = returns.std() * np.sqrt(252)
    sharpe       = (ann_return - rf_rate) / (ann_vol + 1e-9)
    peak         = np.maximum.accumulate(history)
    mdd          = ((history - peak) / peak).min() * 100
    calmar       = ann_return / (abs(mdd / 100) + 1e-9)

    return {
        "Total Return (%)":  round(total_return, 2),
        "Ann. Return (%)":   round(ann_return * 100, 2),
        "Ann. Volatility":   round(ann_vol, 4),
        "Sharpe Ratio":      round(sharpe, 3),
        "MDD (%)":           round(mdd, 2),
        "Calmar Ratio":      round(calmar, 3),
    }


def run_all_baselines(prices_df: pd.DataFrame, test_start_idx: int, initial_cash: float = 10_000_000) -> pd.DataFrame:
    test_prices_df  = prices_df.iloc[test_start_idx:].copy()
    test_prices_arr = test_prices_df.values
    test_returns    = test_prices_df.pct_change().fillna(0).values

    strategies = {
        "Equal Weight": "equal_weight",
        "Min Variance": "mv",
        "Risk Parity":  "rp",
        "Momentum":     "momentum",
        "Market Cap":   "market_cap",
    }

    rows = []
    for name, strategy in strategies.items():
        print(f"[baseline] Running {name}...")
        history = backtest_baseline(
            strategy=strategy,
            prices_arr=test_prices_arr,
            returns_arr=test_returns,
            prices_df=test_prices_df,
            initial_cash=initial_cash,
        )
        metrics = compute_metrics(history)
        metrics["Strategy"] = name
        rows.append(metrics)
        print(f"  Return: {metrics['Total Return (%)']:.2f}% | Sharpe: {metrics['Sharpe Ratio']:.3f}")

    df = pd.DataFrame(rows).set_index("Strategy")
    return df


if __name__ == "__main__":
    from data import fetch_price_data, TICKERS

    print("[test] Loading price data...")
    prices_df = fetch_price_data(TICKERS)

    T = len(prices_df)
    split = int(T * 0.8)

    print(f"[test] Train: {split} | Test: {T - split}")
    print("[test] Running all baselines...\n")

    results = run_all_baselines(prices_df, split)

    print("\n" + "=" * 60)
    print("  BASELINE RESULTS")
    print("=" * 60)
    print(results.to_string())
    print("=" * 60)
