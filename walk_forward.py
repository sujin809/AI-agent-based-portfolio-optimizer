import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

from data import fetch_price_data, compute_technical_features, fetch_sec_sentiment, build_state_matrix, TICKERS
from env import PortfolioEnv
from agent import build_vec_env, build_ppo, build_sac, train, predict


# ── 설정 ──────────────────────────────────────────────────────
WF_CONFIG = {
    "train_window":   504,   # 학습 윈도우 (약 2년 거래일)
    "test_window":    126,   # 테스트 윈도우 (약 6개월)
    "step_size":      63,    # 슬라이딩 간격 (약 3개월)
    "total_timesteps": 50_000,  # 각 fold 학습 스텝 (빠르게)
    "agent":          "ppo",
    "tax_regime":     "korean",
    "window":         20,
    "initial_cash":   10_000_000,
    "transaction_cost": 0.001,
    "result_path":    "./results/walk_forward",
}


# ── 1. Fold 생성 ──────────────────────────────────────────────
def generate_folds(T: int, train_window: int, test_window: int, step_size: int) -> list:
    """
    Walk-forward fold 생성.

    Returns: list of (train_start, train_end, test_start, test_end)
    """
    folds = []
    train_start = 0

    while True:
        train_end  = train_start + train_window
        test_start = train_end
        test_end   = test_start + test_window

        if test_end > T:
            break

        folds.append((train_start, train_end, test_start, test_end))
        train_start += step_size

    print(f"[walk_forward] Total folds: {len(folds)}")
    for i, (ts, te, vs, ve) in enumerate(folds):
        print(f"  Fold {i+1}: Train [{ts}:{te}] | Test [{vs}:{ve}]")

    return folds


# ── 2. 단일 Fold 학습 + 테스트 ────────────────────────────────
def run_fold(
    fold_idx: int,
    states: np.ndarray,
    prices: np.ndarray,
    tickers: list,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    cfg: dict,
) -> dict:
    """
    단일 fold에서 학습 + 백테스팅 수행.

    Returns: dict with portfolio_history, metrics
    """
    print(f"\n[walk_forward] ── Fold {fold_idx + 1} ──────────────────")

    train_states = states[train_start:train_end]
    train_prices = prices[train_start:train_end]
    test_states  = states[test_start:test_end]
    test_prices  = prices[test_start:test_end]

    env_kwargs_train = dict(
        states=train_states,
        prices=train_prices,
        tickers=tickers,
        window=cfg["window"],
        initial_cash=cfg["initial_cash"],
        transaction_cost=cfg["transaction_cost"],
        tax_regime=cfg["tax_regime"],
    )
    env_kwargs_test = dict(
        states=test_states,
        prices=test_prices,
        tickers=tickers,
        window=cfg["window"],
        initial_cash=cfg["initial_cash"],
        transaction_cost=cfg["transaction_cost"],
        tax_regime=cfg["tax_regime"],
    )

    # 학습
    train_env = build_vec_env(PortfolioEnv, env_kwargs_train, normalize=True)
    test_env  = build_vec_env(PortfolioEnv, env_kwargs_test,  normalize=True)

    if cfg["agent"] == "ppo":
        model = build_ppo(train_env, verbose=0)
    else:
        model = build_sac(train_env, verbose=0)

    model = train(
        model,
        total_timesteps=cfg["total_timesteps"],
        eval_env=test_env,
        save_path=f"./models/fold_{fold_idx+1}",
        model_name=f"{cfg['agent']}_fold{fold_idx+1}",
        log_freq=10000,
    )

    # 백테스팅
    env = PortfolioEnv(**env_kwargs_test)
    obs, _ = env.reset()
    portfolio_history = [env.portfolio_val]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        portfolio_history.append(info["portfolio_value"])
        done = terminated or truncated

    # Equal Weight 베이스라인
    env_eq = PortfolioEnv(**env_kwargs_test)
    obs_eq, _ = env_eq.reset()
    eq_history = [env_eq.portfolio_val]
    done = False
    n = env_eq.n_assets
    equal_action = np.log(np.ones(n) / n + 1e-9)

    while not done:
        obs_eq, _, terminated, truncated, info_eq = env_eq.step(equal_action)
        eq_history.append(info_eq["portfolio_value"])
        done = terminated or truncated

    # 성과 지표
    rl_metrics = compute_fold_metrics(np.array(portfolio_history))
    eq_metrics = compute_fold_metrics(np.array(eq_history))

    print(f"  RL  → Return: {rl_metrics['total_return']:.2f}% | Sharpe: {rl_metrics['sharpe']:.3f}")
    print(f"  EW  → Return: {eq_metrics['total_return']:.2f}% | Sharpe: {eq_metrics['sharpe']:.3f}")

    return {
        "fold": fold_idx + 1,
        "rl_history":  np.array(portfolio_history),
        "eq_history":  np.array(eq_history),
        "rl_metrics":  rl_metrics,
        "eq_metrics":  eq_metrics,
    }


# ── 3. 성과 지표 ──────────────────────────────────────────────
def compute_fold_metrics(history: np.ndarray, rf_rate: float = 0.05) -> dict:
    returns      = np.diff(history) / history[:-1]
    total_return = (history[-1] / history[0] - 1) * 100
    ann_return   = (1 + total_return / 100) ** (252 / max(len(returns), 1)) - 1
    ann_vol      = returns.std() * np.sqrt(252)
    sharpe       = (ann_return - rf_rate) / (ann_vol + 1e-9)
    peak         = np.maximum.accumulate(history)
    mdd          = ((history - peak) / peak).min() * 100

    return {
        "total_return": round(total_return, 2),
        "sharpe":       round(sharpe, 3),
        "mdd":          round(mdd, 2),
        "ann_vol":      round(ann_vol, 4),
    }


# ── 4. 전체 Walk-forward 실행 ─────────────────────────────────
def run_walk_forward(cfg=WF_CONFIG):
    print("=" * 60)
    print("  WALK-FORWARD VALIDATION")
    print(f"  Agent: {cfg['agent'].upper()} | Tax: {cfg['tax_regime']}")
    print("=" * 60)

    # 데이터 로드
    print("[walk_forward] Loading data...")
    prices_df = fetch_price_data(TICKERS)
    tech      = compute_technical_features(prices_df)
    sent      = fetch_sec_sentiment(TICKERS)
    states, idx, tickers, _ = build_state_matrix(prices_df, tech, sent)
    prices_arr = prices_df.loc[idx].values

    T = states.shape[0]
    print(f"[walk_forward] Total time steps: {T}")

    # Fold 생성
    folds = generate_folds(
        T,
        cfg["train_window"],
        cfg["test_window"],
        cfg["step_size"],
    )

    if not folds:
        print("[walk_forward] Not enough data for walk-forward validation!")
        return

    # 각 fold 실행
    results = []
    for i, (ts, te, vs, ve) in enumerate(folds):
        result = run_fold(i, states, prices_arr, tickers, ts, te, vs, ve, cfg)
        results.append(result)

    # 결과 집계
    summarize_results(results, cfg)


# ── 5. 결과 집계 + 시각화 ─────────────────────────────────────
def summarize_results(results: list, cfg: dict):
    print("\n" + "=" * 60)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 60)

    rows = []
    for r in results:
        rows.append({
            "Fold":          r["fold"],
            "RL Return (%)": r["rl_metrics"]["total_return"],
            "EW Return (%)": r["eq_metrics"]["total_return"],
            "RL Sharpe":     r["rl_metrics"]["sharpe"],
            "EW Sharpe":     r["eq_metrics"]["sharpe"],
            "RL MDD (%)":    r["rl_metrics"]["mdd"],
            "EW MDD (%)":    r["eq_metrics"]["mdd"],
            "RL > EW":       r["rl_metrics"]["total_return"] > r["eq_metrics"]["total_return"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    rl_win_rate = df["RL > EW"].mean() * 100
    print(f"\n  RL 승률 (vs Equal Weight): {rl_win_rate:.1f}%")
    print(f"  평균 RL Return:  {df['RL Return (%)'].mean():.2f}%")
    print(f"  평균 EW Return:  {df['EW Return (%)'].mean():.2f}%")
    print(f"  평균 RL Sharpe:  {df['RL Sharpe'].mean():.3f}")
    print(f"  평균 EW Sharpe:  {df['EW Sharpe'].mean():.3f}")

    # 저장
    os.makedirs(cfg["result_path"], exist_ok=True)
    df.to_csv(os.path.join(cfg["result_path"], "walk_forward_results.csv"), index=False)

    # 시각화
    plot_walk_forward(results, cfg)


def plot_walk_forward(results: list, cfg: dict):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # (1) fold별 수익률 비교
    ax = axes[0]
    folds     = [r["fold"] for r in results]
    rl_rets   = [r["rl_metrics"]["total_return"] for r in results]
    eq_rets   = [r["eq_metrics"]["total_return"] for r in results]

    x = np.arange(len(folds))
    ax.bar(x - 0.2, rl_rets, 0.4, label="RL Agent", color="steelblue")
    ax.bar(x + 0.2, eq_rets, 0.4, label="Equal Weight", color="lightcoral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Walk-Forward: Return by Fold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)

    # (2) fold별 샤프 비교
    ax = axes[1]
    rl_sharpes = [r["rl_metrics"]["sharpe"] for r in results]
    eq_sharpes = [r["eq_metrics"]["sharpe"] for r in results]

    ax.plot(folds, rl_sharpes, marker="o", label="RL Agent", color="steelblue")
    ax.plot(folds, eq_sharpes, marker="s", label="Equal Weight", color="lightcoral", linestyle="--")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Walk-Forward: Sharpe Ratio by Fold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(cfg["result_path"], f"walk_forward_{timestamp}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  📊 Saved: {path}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",      type=str, default="ppo",    help="ppo | sac")
    parser.add_argument("--tax",        type=str, default="korean", help="korean | us_long | none")
    parser.add_argument("--timesteps",  type=int, default=50_000)
    args = parser.parse_args()

    WF_CONFIG["agent"]           = args.agent
    WF_CONFIG["tax_regime"]      = args.tax
    WF_CONFIG["total_timesteps"] = args.timesteps

    run_walk_forward(WF_CONFIG)
