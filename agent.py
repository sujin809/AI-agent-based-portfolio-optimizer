import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import os


# ── 1. 환경 생성 헬퍼 ─────────────────────────────────────────
def make_env(env_class, env_kwargs: dict):
    """DummyVecEnv용 환경 생성 함수."""
    def _init():
        env = env_class(**env_kwargs)
        env = Monitor(env)
        return env
    return _init


def build_vec_env(env_class, env_kwargs: dict, normalize: bool = True):
    """
    Vectorized + Normalized 환경 생성.
    normalize=True → observation/reward 정규화 (학습 안정성 향상)
    """
    vec_env = DummyVecEnv([make_env(env_class, env_kwargs)])
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


# ── 2. PPO 에이전트 ───────────────────────────────────────────
def build_ppo(
    vec_env,
    learning_rate: float = 3e-4,
    n_steps: int = 512,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    verbose: int = 1,
    seed: int = 42,
    device: str = "auto",
) -> PPO:
    """
    PPO (Proximal Policy Optimization) 에이전트 생성.

    포트폴리오 환경에 맞게 튜닝된 하이퍼파라미터:
    - ent_coef=0.01: 탐색 장려 (다양한 비중 시도)
    - gae_lambda=0.95: 분산-편향 트레이드오프
    - clip_range=0.2: 정책 업데이트 안정성
    """
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
        device=device,
    )
    return model


# ── 3. SAC 에이전트 ───────────────────────────────────────────
def build_sac(
    vec_env,
    learning_rate: float = 3e-4,
    buffer_size: int = 100_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    ent_coef: str = "auto",
    verbose: int = 1,
    seed: int = 42,
    device: str = "auto",
) -> SAC:
    """
    SAC (Soft Actor-Critic) 에이전트 생성.

    연속 행동공간에서 PPO보다 샘플 효율이 높음.
    ent_coef='auto': 엔트로피 계수 자동 조정
    """
    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,
    )

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
        device=device,
    )
    return model


# ── 4. 커스텀 콜백: 학습 로그 ────────────────────────────────
class PortfolioLogCallback(BaseCallback):
    """
    학습 중 포트폴리오 가치를 주기적으로 출력하는 콜백.
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean(
                    [ep["r"] for ep in self.model.ep_info_buffer]
                )
                print(
                    f"  [callback] Step {self.n_calls:6d} | "
                    f"Mean Episode Reward: {mean_reward:.4f}"
                )
        return True


# ── 5. 학습 함수 ──────────────────────────────────────────────
def train(
    model,
    total_timesteps: int = 100_000,
    eval_env=None,
    save_path: str = "./models",
    model_name: str = "portfolio_agent",
    log_freq: int = 1000,
):
    """
    에이전트 학습.

    Args:
        model: PPO or SAC 모델
        total_timesteps: 총 학습 스텝
        eval_env: 평가용 환경 (없으면 eval 콜백 스킵)
        save_path: 모델 저장 경로
        model_name: 저장 파일명
        log_freq: 로그 출력 주기
    """
    os.makedirs(save_path, exist_ok=True)

    callbacks = [PortfolioLogCallback(log_freq=log_freq)]

    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_path, "best"),
            log_path=os.path.join(save_path, "logs"),
            eval_freq=5000,
            n_eval_episodes=3,
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_callback)

    print(f"\n[agent] Training {model.__class__.__name__} for {total_timesteps:,} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    save_file = os.path.join(save_path, model_name)
    model.save(save_file)
    print(f"[agent] Model saved → {save_file}.zip")
    return model


# ── 6. 추론 함수 ──────────────────────────────────────────────
def predict(model, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
    """
    학습된 모델로 행동(비중) 예측.

    Returns:
        weights: np.ndarray [N] — Softmax 적용된 종목별 비중
    """
    action, _ = model.predict(obs, deterministic=deterministic)
    weights = np.exp(action) / np.exp(action).sum()  # Softmax
    return weights


def load_model(model_type: str, path: str, vec_env):
    """
    저장된 모델 로드.

    Args:
        model_type: 'ppo' or 'sac'
        path: 모델 파일 경로 (.zip 제외)
    """
    if model_type.lower() == "ppo":
        model = PPO.load(path, env=vec_env)
    elif model_type.lower() == "sac":
        model = SAC.load(path, env=vec_env)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    print(f"[agent] Model loaded ← {path}.zip")
    return model


# ── 테스트 ────────────────────────────────────────────────────
if __name__ == "__main__":
    from data import (
        fetch_price_data,
        compute_technical_features,
        fetch_sec_sentiment,
        build_state_matrix,
    )
    from env import PortfolioEnv

    print("[test] Loading data...")
    prices_df = fetch_price_data()
    tech      = compute_technical_features(prices_df)
    sent      = fetch_sec_sentiment()
    states, idx, tickers, feat_names = build_state_matrix(prices_df, tech, sent)
    prices_arr = prices_df.loc[idx].values

    env_kwargs = dict(
        states=states,
        prices=prices_arr,
        tickers=tickers,
        window=20,
        tax_regime="korean",
    )

    # Train/Test split (80/20)
    T = states.shape[0]
    split = int(T * 0.8)

    train_kwargs = dict(**env_kwargs)
    train_kwargs["states"] = states[:split]
    train_kwargs["prices"] = prices_arr[:split]

    test_kwargs = dict(**env_kwargs)
    test_kwargs["states"] = states[split:]
    test_kwargs["prices"] = prices_arr[split:]

    print("[test] Building environments...")
    train_env = build_vec_env(PortfolioEnv, train_kwargs, normalize=True)
    test_env  = build_vec_env(PortfolioEnv, test_kwargs,  normalize=False)

    print("[test] Building PPO agent...")
    model = build_ppo(train_env, verbose=1)
    print(f"\n✅ PPO model created")
    print(f"   Policy: {model.policy}")
    print(f"   Device: {model.device}")

    # 짧은 학습 테스트 (1000 스텝)
    print("\n[test] Quick training test (1,000 steps)...")
    model = train(model, total_timesteps=1_000, save_path="./models", model_name="test_ppo")
    print("\n✅ agent.py test passed!")
