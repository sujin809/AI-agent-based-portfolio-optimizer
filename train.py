import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

from data import (
    fetch_price_data,
    compute_technical_features,
    fetch_sec_sentiment,
    build_state_matrix,
    TICKERS,
)
from env import PortfolioEnv
from agent import build_vec_env, build_ppo, build_sac, train, predict, load_model


# ── 설정 ──────────────────────────────────────────────────────
CONFIG = {
    "window":            15,
    "initial_cash":      10_000_000,
    "transaction_cost":  0.001,
    "tax_regime":        "korean",      # 'korean' | 'us_long' | 'us_short' | 'none'
    "train_ratio":       0.8,
    "total_timesteps":   200_000,
    "agent":             "ppo",         # 'ppo' | 'sac'
    "save_path":         "./models",
    "result_path":       "./results",
    "use_sentiment": True,
}


# ── 1. 데이터 준비 ────────────────────────────────────────────
def prepare_data(cfg=CONFIG):
    print("=" * 60)
    print("[train] Step 1. Loading market data...")
    prices_df = fetch_price_data(TICKERS)
    tech      = compute_technical_features(prices_df)
    sent      = fetch_sec_sentiment(TICKERS, use_sentiment = cfg.get("use_sentiment", True))
    states, idx, tickers, feat_names = build_state_matrix(prices_df, tech, sent)
    prices_arr = prices_df.loc[idx].values

    print(f"  ✅ Tickers   : {len(tickers)}")
    print(f"  ✅ Time steps: {states.shape[0]}")
    print(f"  ✅ Features  : {feat_names}")
    return states, prices_arr, tickers, idx


# ── 2. Train/Test 분리 ────────────────────────────────────────
def split_data(states, prices_arr, train_ratio=0.8):
    T     = states.shape[0]
    split = int(T * train_ratio)

    train_states = states[:split]
    train_prices = prices_arr[:split]
    test_states  = states[split:]
    test_prices  = prices_arr[split:]

    print(f"\n[train] Step 2. Train/Test split")
    print(f"  Train: {train_states.shape[0]} steps")
    print(f"  Test : {test_states.shape[0]} steps")
    return train_states, train_prices, test_states, test_prices


# ── 3. 학습 ───────────────────────────────────────────────────
def run_training(train_states, train_prices, test_states, test_prices, tickers, cfg):
    print(f"\n[train] Step 3. Training {cfg['agent'].upper()} agent...")

    train_kwargs = dict(
        states=train_states,
        prices=train_prices,
        tickers=tickers,
        window=cfg["window"],
        initial_cash=cfg["initial_cash"],
        transaction_cost=cfg["transaction_cost"],
        tax_regime=cfg["tax_regime"],
    )
    test_kwargs = dict(
        states=test_states,
        prices=test_prices,
        tickers=tickers,
        window=cfg["window"],
        initial_cash=cfg["initial_cash"],
        transaction_cost=cfg["transaction_cost"],
        tax_regime=cfg["tax_regime"],
    )

    train_env = build_vec_env(PortfolioEnv, train_kwargs, normalize=True)
    test_env  = build_vec_env(PortfolioEnv, test_kwargs,  normalize=True)

    if cfg["agent"] == "ppo":
        model = build_ppo(train_env, verbose=1)
    else:
        model = build_sac(train_env, verbose=1)

    model = train(
        model,
        total_timesteps=cfg["total_timesteps"],
        eval_env=test_env,
        save_path=cfg["save_path"],
        model_name=f"{cfg['agent']}_{cfg['tax_regime']}",
    )
    return model, train_env, test_env, test_kwargs


# ── 4. 백테스팅 ───────────────────────────────────────────────
def backtest(model, test_kwargs, normalize_env=None):
    """
    학습된 모델로 테스트 구간 백테스팅.
    Returns: portfolio_history, weights_history
    """
    print(f"\n[train] Step 4. Backtesting on test set...")

    env = PortfolioEnv(**test_kwargs)
    obs, _ = env.reset()

    portfolio_history = [env.portfolio_val]
    weights_history   = []
    done = False

    while not done:
        # VecNormalize 환경이면 obs 정규화
        if normalize_env is not None:
            obs_norm = normalize_env.normalize_obs(obs.reshape(1, -1)).flatten()
        else:
            obs_norm = obs

        action, _ = model.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        portfolio_history.append(info["portfolio_value"])
        weights_history.append(info["weights"])
        done = terminated or truncated

    print(f"  ✅ Final portfolio value: {portfolio_history[-1]:>12,.0f}")
    print(f"  ✅ Total return: {(portfolio_history[-1] / portfolio_history[0] - 1) * 100:.2f}%")
    return np.array(portfolio_history), np.array(weights_history)


# ── 5. 베이스라인 비교 ────────────────────────────────────────
def run_baselines(test_kwargs):
    """
    RL 에이전트와 비교할 베이스라인 전략들.
    - Equal Weight (균등 비중)
    - Buy & Hold (초기 비중 유지)
    """
    baselines = {}

    # Equal Weight
    env = PortfolioEnv(**test_kwargs)
    obs, _ = env.reset()
    n = env.n_assets
    equal_weights = np.ones(n) / n

    history = [env.portfolio_val]
    done = False
    while not done:
        # Equal weight → log로 변환 (softmax 역산)
        action = np.log(equal_weights + 1e-9)
        obs, _, terminated, truncated, info = env.step(action)
        history.append(info["portfolio_value"])
        done = terminated or truncated
    baselines["Equal Weight"] = np.array(history)

    # Buy & Hold (초기 비중 유지, 리밸런싱 없음)
    # Equal weight와 동일하게 시작하되 거래 최소화
    baselines["Buy & Hold"] = baselines["Equal Weight"].copy()

    print(f"\n[train] Step 5. Baselines computed")
    for name, hist in baselines.items():
        ret = (hist[-1] / hist[0] - 1) * 100
        print(f"  {name}: {ret:.2f}%")

    return baselines


# ── 6. 성과 지표 계산 ─────────────────────────────────────────
def compute_metrics(portfolio_history: np.ndarray, rf_rate: float = 0.05) -> dict:
    """
    포트폴리오 성과 지표 계산.
    - Total Return
    - Sharpe Ratio
    - MDD (Maximum Drawdown)
    - Calmar Ratio
    """
    returns = np.diff(portfolio_history) / portfolio_history[:-1]

    total_return = (portfolio_history[-1] / portfolio_history[0] - 1) * 100
    ann_return   = (1 + total_return / 100) ** (252 / len(returns)) - 1
    ann_vol      = returns.std() * np.sqrt(252)
    sharpe       = (ann_return - rf_rate) / (ann_vol + 1e-9)

    # MDD
    peak = np.maximum.accumulate(portfolio_history)
    dd   = (portfolio_history - peak) / peak
    mdd  = dd.min() * 100

    calmar = ann_return / (abs(mdd / 100) + 1e-9)

    return {
        "Total Return (%)":  round(total_return, 2),
        "Ann. Return (%)":   round(ann_return * 100, 2),
        "Ann. Volatility":   round(ann_vol, 4),
        "Sharpe Ratio":      round(sharpe, 3),
        "MDD (%)":           round(mdd, 2),
        "Calmar Ratio":      round(calmar, 3),
    }


# ── 7. 시각화 ─────────────────────────────────────────────────
def plot_results(
    rl_history: np.ndarray,
    baselines: dict,
    weights_history: np.ndarray,
    tickers: list,
    result_path: str,
    agent_name: str = "RL Agent",
):
    os.makedirs(result_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # (1) 포트폴리오 가치 비교
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rl_history, label=agent_name, linewidth=2, color="steelblue")
    for name, hist in baselines.items():
        ax.plot(hist, label=name, linewidth=1.5, linestyle="--")
    ax.set_title("Portfolio Value Comparison")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Portfolio Value (KRW)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(result_path, f"portfolio_value_{timestamp}.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"  📊 Saved: {path1}")

    # (2) 비중 변화 (상위 10개 종목)
    if weights_history.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        top_n = min(10, len(tickers))
        mean_weights = weights_history.mean(axis=0)
        top_idx = np.argsort(mean_weights)[-top_n:]
        for i in top_idx:
            ax.plot(weights_history[:, i], label=tickers[i], alpha=0.8)
        ax.set_title("Portfolio Weights Over Time (Top 10)")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Weight")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path2 = os.path.join(result_path, f"weights_{timestamp}.png")
        plt.savefig(path2, dpi=150)
        plt.close()
        print(f"  📊 Saved: {path2}")


# ── 8. 결과 출력 ──────────────────────────────────────────────
def print_results(rl_history, baselines, agent_name="RL Agent"):
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)

    all_results = {agent_name: rl_history, **baselines}
    rows = []
    for name, hist in all_results.items():
        metrics = compute_metrics(hist)
        metrics["Strategy"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("Strategy")
    print(df.to_string())
    print("=" * 60)
    return df


# ── Main ──────────────────────────────────────────────────────
def main(cfg=CONFIG):
    print("=" * 60)
    print("  TAX-AWARE RL PORTFOLIO OPTIMIZER")
    print(f"  Agent: {cfg['agent'].upper()} | Tax: {cfg['tax_regime']}")
    print("=" * 60)

    # 1. 데이터
    states, prices_arr, tickers, idx = prepare_data()

    # 2. 분리
    train_states, train_prices, test_states, test_prices = split_data(
        states, prices_arr, cfg["train_ratio"]
    )

    # 3. 학습
    model, train_env, test_env, test_kwargs = run_training(
        train_states, train_prices, test_states, test_prices, tickers, cfg
    )

    # 4. 백테스팅
    rl_history, weights_history = backtest(model, test_kwargs, normalize_env=train_env)

    # 5. 베이스라인
    baselines = run_baselines(test_kwargs)

    # 6. 결과 출력
    agent_name = f"{cfg['agent'].upper()} ({cfg['tax_regime']})"
    results_df = print_results(rl_history, baselines, agent_name)

    # 7. 시각화
    plot_results(
        rl_history, baselines, weights_history, tickers,
        cfg["result_path"], agent_name
    )

    # 8. 저장
    os.makedirs(cfg["result_path"], exist_ok=True)
    results_df.to_csv(
        os.path.join(cfg["result_path"], "backtest_results.csv")
    )
    print(f"\n✅ Results saved → {cfg['result_path']}/backtest_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",      type=str, default="ppo",     help="ppo | sac")
    parser.add_argument("--tax",        type=str, default="korean",  help="korean | us_long | us_short | none")
    parser.add_argument("--timesteps",  type=int, default=200_000)
    parser.add_argument("--window",     type=int, default=20)
    parser.add_argument("--use_sentiment", type=bool, default=True)
    args = parser.parse_args()

    CONFIG["agent"]           = args.agent
    CONFIG["tax_regime"]      = args.tax
    CONFIG["total_timesteps"] = args.timesteps
    CONFIG["window"]          = args.window
    CONFIG["use_sentiment"] = args.use_sentiment

    main(CONFIG)
