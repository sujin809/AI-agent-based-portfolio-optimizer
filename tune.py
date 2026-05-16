import optuna
import numpy as np
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from data import fetch_price_data, compute_technical_features, fetch_sec_sentiment, build_state_matrix, TICKERS
from env import PortfolioEnv
from agent import build_vec_env, build_ppo, build_sac, train


# ── 데이터 전역 로드 ──────────────────────────────────────────
print("[tune] Loading data...")
prices_df = fetch_price_data(TICKERS)
tech      = compute_technical_features(prices_df)

# SEC 감성 캐시 사용
from sec_sentiment import build_sentiment_matrix
start = prices_df.index[0].strftime("%Y-%m-%d")
end   = prices_df.index[-1].strftime("%Y-%m-%d")
sent  = build_sentiment_matrix(
    tickers=list(prices_df.columns),
    date_index=prices_df.index,
    start_date=start,
    end_date=end,
    cache_path="./cache/sentiment_full.parquet",
)

states, idx, tickers, feat_names = build_state_matrix(prices_df, tech, sent)
prices_arr = prices_df.loc[idx].values

T     = states.shape[0]
split = int(T * 0.8)

train_states = states[:split]
train_prices = prices_arr[:split]
val_states   = states[split:]
val_prices   = prices_arr[split:]

print(f"[tune] Train: {split} | Val: {T - split}")


def make_objective(agent: str):
    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma         = trial.suggest_float("gamma", 0.95, 0.999)
        window        = trial.suggest_int("window", 10, 40, step=5)
        batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        if agent == "ppo":
            n_steps    = trial.suggest_categorical("n_steps", [256, 512, 1024])
            n_epochs   = trial.suggest_int("n_epochs", 5, 20)
            ent_coef   = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
            clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        else:
            buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000])
            tau         = trial.suggest_float("tau", 0.001, 0.02)

        train_kwargs = dict(
            states=train_states, prices=train_prices, tickers=tickers,
            window=window, initial_cash=10_000_000,
            transaction_cost=0.001, tax_regime="korean",
        )
        val_kwargs = dict(
            states=val_states, prices=val_prices, tickers=tickers,
            window=window, initial_cash=10_000_000,
            transaction_cost=0.001, tax_regime="korean",
        )

        try:
            train_env = build_vec_env(PortfolioEnv, train_kwargs, normalize=True)

            if agent == "ppo":
                model = build_ppo(
                    train_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    ent_coef=ent_coef,
                    clip_range=clip_range,
                    verbose=0,
                )
            else:
                model = build_sac(
                    train_env,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    gamma=gamma,
                    tau=tau,
                    verbose=0,
                )

            model = train(
                model,
                total_timesteps=50_000,
                save_path=f"./models/tune/trial_{trial.number}",
                model_name=f"{agent}_trial{trial.number}",
                log_freq=100000,
            )

            env = PortfolioEnv(**val_kwargs)
            obs, _ = env.reset()
            history = [env.portfolio_val]
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                history.append(info["portfolio_value"])
                done = terminated or truncated

            history  = np.array(history)
            returns  = np.diff(history) / history[:-1]
            ann_ret  = (history[-1] / history[0]) ** (252 / max(len(returns), 1)) - 1
            ann_vol  = returns.std() * np.sqrt(252) + 1e-9
            sharpe   = (ann_ret - 0.05) / ann_vol

            print(f"  Trial {trial.number}: Sharpe={sharpe:.3f} | LR={learning_rate:.2e} | window={window}")
            return sharpe

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            return -999.0

    return objective


def run_tuning(n_trials: int = 30, agent: str = "ppo"):
    os.makedirs("./models/tune", exist_ok=True)
    os.makedirs("./results/tune", exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"portfolio_{agent}_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(make_objective(agent), n_trials=n_trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print(f"  TUNING RESULTS ({agent.upper()})")
    print("=" * 60)
    print(f"  Best Sharpe: {study.best_value:.4f}")
    print(f"  Best Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    df = study.trials_dataframe()
    df.to_csv(f"./results/tune/optuna_{agent}_results.csv", index=False)
    print(f"\n  Results saved → ./results/tune/optuna_{agent}_results.csv")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--agent",    type=str, default="ppo", help="ppo | sac")
    args = parser.parse_args()

    study = run_tuning(n_trials=args.n_trials, agent=args.agent)
