import numpy as np
import gymnasium as gym
from gymnasium import spaces
from data import compute_tax


class PortfolioEnv(gym.Env):
    """
    Tax-aware Portfolio Optimization Environment.

    State:  [T x N x F] 중 현재 시점의 슬라이딩 윈도우
    Action: 각 종목 비중 (N,) — Softmax로 합=1 보장
    Reward: 세후 수익률 - 거래비용
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        states: np.ndarray,       # [T x N x F]
        prices: np.ndarray,       # [T x N] 일별 종가
        tickers: list,
        window: int = 20,         # 관측 윈도우 (일)
        initial_cash: float = 10_000_000,   # 초기 자본 (10백만원)
        transaction_cost: float = 0.001,    # 거래비용 0.1%
        tax_regime: str = "korean",         # 세금 체계
        mode: str = "train",                # 'train' or 'test'
    ):
        super().__init__()

        self.states           = states      # [T, N, F]
        self.prices           = prices      # [T, N]
        self.tickers          = tickers
        self.window           = window
        self.initial_cash     = initial_cash
        self.transaction_cost = transaction_cost
        self.tax_regime       = tax_regime
        self.mode             = mode

        self.n_steps   = states.shape[0]
        self.n_assets  = states.shape[1]
        self.n_features= states.shape[2]

        # ── Action Space: 각 종목 비중 [-1, 1] → Softmax로 정규화
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # ── Observation Space: 윈도우 x 종목 x feature + 현재 포트폴리오 비중
        obs_size = self.window * self.n_assets * self.n_features + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.reset()

    # ── Reset ────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step  = self.window
        self.portfolio_val = self.initial_cash
        self.cash          = self.initial_cash
        self.holdings      = np.zeros(self.n_assets)   # 주식 보유 수량
        self.weights       = np.ones(self.n_assets) / self.n_assets  # 초기 균등 비중
        self.cost_basis    = np.zeros(self.n_assets)   # 평균 매수 단가 (세금 계산용)
        self.portfolio_history = [self.initial_cash]

        obs = self._get_obs()
        return obs, {}

    # ── Step ─────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        # 1. Action → 비중 (Softmax)
        new_weights = self._softmax(action)

        # 2. 현재 가격
        current_prices = self.prices[self.current_step]   # [N]
        prev_prices    = self.prices[self.current_step - 1]

        # 3. 리밸런싱 → 거래비용 + 세금 계산
        reward, tax_paid = self._rebalance(new_weights, current_prices)

        # 4. 다음 날 수익률 반영
        if self.current_step + 1 < self.n_steps:
            next_prices = self.prices[self.current_step + 1]
            price_return = next_prices / current_prices - 1  # [N]
            self.portfolio_val = self.portfolio_val * (
                1 + np.dot(self.weights, price_return)
            )
        self.portfolio_history.append(self.portfolio_val)

        # 5. 다음 스텝
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = False

        obs  = self._get_obs()
        info = {
            "portfolio_value": self.portfolio_val,
            "weights": self.weights,
            "tax_paid": tax_paid,
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    # ── Rebalance ────────────────────────────────────────────
    def _rebalance(self, new_weights: np.ndarray, current_prices: np.ndarray):
        """
        리밸런싱 시 거래비용과 세금을 계산하고 Reward를 반환.
        """
        old_weights = self.weights
        weight_diff = np.abs(new_weights - old_weights)
        turnover    = weight_diff.sum() / 2  # 전체 회전율

        # 거래비용
        transaction_cost = turnover * self.transaction_cost * self.portfolio_val

        # 세금: 매도 시 양도차익에 대해 계산
        tax_paid = 0.0
        for i in range(self.n_assets):
            if new_weights[i] < old_weights[i]:  # 매도 발생
                sold_ratio    = old_weights[i] - new_weights[i]
                sold_value    = sold_ratio * self.portfolio_val
                capital_gain  = sold_value * (
                    1 - self.cost_basis[i] / (current_prices[i] + 1e-9)
                ) if self.cost_basis[i] > 0 else sold_value * 0.2
                tax_paid += compute_tax(capital_gain, self.tax_regime)

        # 순 수익률 계산 (Reward)
        net_cost = transaction_cost + tax_paid
        
        if len(self.portfolio_history) >= 20:
            recent = np.array(self.portfolio_history[-20:])
            returns = np.diff(recent) / recent[:-1]
            mean_r  = returns.mean()
            std_r   = returns.std() + 1e-9
            sharpe_reward = mean_r / std_r

            peak = np.maximum.accumulate(recent)
            mdd = ((recent - peak) / peak).min()
            mdd_penalty = abs(mdd) * 0.5 # 페널티 강도 조절
        else:
            sharpe_reward = 0.0
            mdd_penalty = 0.0

        reward = sharpe_reward - net_cost / (self.portfolio_val + 1e-9)

        # 비중 업데이트
        self.weights = new_weights

        return reward, tax_paid

    # ── Observation ──────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """
        현재 윈도우의 state + 현재 포트폴리오 비중을 flatten해서 반환.
        """
        start = self.current_step - self.window
        end   = self.current_step
        window_states = self.states[start:end]  # [window, N, F]
        flat_states   = window_states.flatten().astype(np.float32)
        flat_weights  = self.weights.astype(np.float32)
        obs = np.concatenate([flat_states, flat_weights])
        return obs

    # ── Utils ────────────────────────────────────────────────
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_portfolio_history(self):
        return np.array(self.portfolio_history)

    def render(self):
        print(
            f"Step {self.current_step:4d} | "
            f"Value: {self.portfolio_val:>12,.0f} | "
            f"Weights: {np.round(self.weights, 3)}"
        )


# ── 테스트 ────────────────────────────────────────────────────
if __name__ == "__main__":
    from data import (
        fetch_price_data,
        compute_technical_features,
        fetch_sec_sentiment,
        build_state_matrix,
    )

    print("[test] Loading data...")
    prices_df = fetch_price_data()
    tech      = compute_technical_features(prices_df)
    sent      = fetch_sec_sentiment()
    states, idx, tickers, feat_names = build_state_matrix(prices_df, tech, sent)
    prices_arr = prices_df.loc[idx].values  # [T x N]

    print("[test] Creating environment...")
    env = PortfolioEnv(
        states=states,
        prices=prices_arr,
        tickers=tickers,
        window=20,
        tax_regime="korean",
    )

    obs, _ = env.reset()
    print(f"\n✅ Observation shape: {obs.shape}")
    print(f"✅ Action space:      {env.action_space}")

    # 랜덤 액션으로 몇 스텝 테스트
    total_reward = 0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated:
            break

    print(f"\n✅ Total reward (5 steps): {total_reward:.6f}")
    print(f"✅ Final portfolio value:  {info['portfolio_value']:,.0f}")
