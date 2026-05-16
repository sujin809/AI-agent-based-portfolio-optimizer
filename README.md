# RL Portfolio Optimizer

> **Tax-Aware Dynamic Portfolio Optimization via Reinforcement Learning with SEC Filing-Based Text Signals**

세금 구조와 SEC 공시 기반 텍스트 신호를 통합한 RL 기반 동적 포트폴리오 최적화 프레임워크입니다.  
매일 27개 미국 주식의 비중을 RL 에이전트가 동적으로 결정하며, 세후 실질 수익을 최대화하는 전략을 학습합니다.

---

## Why This Project / 왜 이 프로젝트인가

기존 RL 포트폴리오 최적화 연구들은 두 가지 중요한 요소를 무시해왔습니다:

**1. 세금 구조 무시**
대부분의 논문이 세금을 단순화하거나 완전히 무시합니다. 하지만 실제 투자자의 경우 양도세는 실질 수익에 큰 영향을 미칩니다. 이 프로젝트는 세금을 Reward에 직접 통합한 **일반화된 Tax-aware Reward 프레임워크**를 제안하며, 한국 투자자(양도세 22%)를 케이스 스터디로 적용합니다.

**2. 공식 텍스트 데이터 미활용**
기존 연구들은 뉴스나 트위터 감성 분석을 사용하지만, 신뢰도와 look-ahead bias 문제가 있습니다. 이 프로젝트는 **SEC 8-K 공시**를 text-based signal로 활용합니다. 공식 문서라 신뢰도가 높고, 공시 날짜가 명확해 look-ahead bias를 통제하기 쉽습니다.

---

## Pipeline / 파이프라인

```
시장 데이터 수집 (yfinance, 2019~현재)
        +
SEC 8-K 공시 수집 (EDGAR API)
        ↓
Groq LLM으로 감성 점수 추출 (-1 ~ +1)
        +
기술적 지표 계산 (RSI, MACD, 볼린저 밴드 등)
        ↓
State 행렬 구성 [T x N x F]
        ↓
RL 환경 (Tax-aware Reward + Sharpe 기반 + MDD 패널티)
        ↓
PPO / SAC 에이전트 학습 (Optuna 하이퍼파라미터 튜닝)
        ↓
백테스팅 & 전통 전략과 비교
```

---

## Getting Started / 실행 방법

```bash
git clone https://github.com/sujin809/rl-portfolio.git
cd rl-portfolio
pip install -r requirements.txt
```

`.env` 파일 생성:
```
GROQ_API_KEY=your_groq_api_key_here
```

학습 실행:
```bash
# PPO + 한국 세금
python train.py --agent ppo --tax korean --timesteps 200000

# SAC + 한국 세금
python train.py --agent sac --tax korean --timesteps 200000
```

하이퍼파라미터 튜닝:
```bash
python tune.py --agent ppo --n_trials 30
python tune.py --agent sac --n_trials 30
```

Walk-forward Validation:
```bash
python walk_forward.py --agent ppo --tax korean --timesteps 200000
```

베이스라인 비교:
```bash
python baseline.py
```

---

## Methodology / 방법론

### 1. State Space
| 구분 | Feature | 설명 |
|------|---------|------|
| 시장 데이터 | ret_1d, ret_5d, ret_20d | 1/5/20일 수익률 |
| 기술적 지표 | ma20_ratio, ma60_ratio | 이동평균 대비 가격 비율 |
| 기술적 지표 | rsi | RSI (14일) |
| 기술적 지표 | vol_20d | 20일 변동성 |
| 기술적 지표 | macd | MACD |
| 기술적 지표 | bb_upper, bb_lower | 볼린저 밴드 |
| 텍스트 신호 | sentiment | SEC 8-K 감성 점수 (-1~+1) |

### 2. Action Space
27개 종목에 대한 비중 벡터. [-1, 1] 범위의 숫자 → Softmax → 합=1 보장.

### 3. Tax-aware Reward
세금 구조를 Reward에 직접 통합한 일반화 프레임워크:

```
Reward = Sharpe(최근 20일) - 거래비용 - 세금 - MDD 패널티
```

| Tax Regime | 세율 |
|------------|------|
| Korean | 양도차익의 22% (기본공제 250만원) |
| US Long | 양도차익의 20% |
| US Short | 양도차익의 37% |
| None | 세금 없음 |

### 4. 에이전트 및 최적 하이퍼파라미터 (Optuna 튜닝)

**PPO:**
| 파라미터 | 값 |
|---------|-----|
| learning_rate | 7.37e-05 |
| n_steps | 512 |
| batch_size | 32 |
| n_epochs | 8 |
| gamma | 0.9502 |
| clip_range | 0.233 |
| ent_coef | 0.000247 |
| window | 35 |

**SAC:**
| 파라미터 | 값 |
|---------|-----|
| learning_rate | 1.55e-04 |
| batch_size | 64 |
| buffer_size | 50,000 |
| gamma | 0.9654 |
| tau | 0.00977 |
| window | 15 |

### 5. 투자 유니버스 (27개 종목)
| 섹터 | 종목 |
|------|------|
| Technology | AAPL, MSFT, GOOGL, META, NVDA, TSLA |
| Consumer | AMZN, HD, MCD, NKE, COST |
| Healthcare | JNJ, UNH, PFE, ABBV, LLY |
| Finance | JPM, BAC, GS, BRK-B |
| Energy | XOM, CVX |
| Industrial | CAT, HON, UPS |

---

## Results / 실험 결과

> 백테스팅 기간: 2019~2026 | 초기 자본: ₩10,000,000 | 거래비용: 0.1%

### RL 에이전트 vs 전통 전략 전체 비교

| 전략 | 수익률 | 샤프 비율 | MDD |
|------|--------|-----------|-----|
| **SAC (korean) 튜닝** | **37.26%** | **1.225** | -18.21% |
| PPO (korean) 튜닝 | 31.69% | 1.009 | -18.63% |
| SAC (none) | 27.41% | 0.937 | -18.16% |
| Equal Weight | 25.75% | 0.830 | -17.61% |
| Market Cap | 20.47% | 0.521 | -17.25% |
| Risk Parity | 18.05% | 0.495 | -15.81% |
| PPO (none) | 17.07% | 0.437 | -19.57% |
| Momentum | 16.77% | 0.406 | -18.38% |
| SAC (korean) 디폴트 | 16.62% | 0.409 | -20.78% |
| Min Variance | 9.21% | 0.075 | -17.61% |

### Tax-aware Reward 효과

| | SAC (korean) | SAC (none) | 차이 |
|--|-------------|------------|------|
| 수익률 | **37.26%** | 27.41% | **+9.85%** |
| 샤프 비율 | **1.225** | 0.937 | **+0.288** |

| | PPO (korean) | PPO (none) | 차이 |
|--|-------------|------------|------|
| 수익률 | **31.69%** | 17.07% | **+14.62%** |
| 샤프 비율 | **1.009** | 0.437 | **+0.572** |

> 세금을 Reward에 반영했을 때 PPO +14.62%, SAC +9.85% 수익률 개선. **Tax-aware Reward가 실질 수익 개선에 유의미한 영향을 미침.**

### 하이퍼파라미터 튜닝 효과

| | 디폴트 | 튜닝 후 | 개선 |
|--|--------|---------|------|
| PPO (korean) | 28.52% | **31.69%** | +3.17% |
| SAC (korean) | 16.62% | **37.26%** | +20.64% |

### Walk-forward Validation (PPO korean, 200,000 steps)

| | RL Agent | Equal Weight |
|--|---------|-------------|
| 평균 수익률 | 6.94% | 7.44% |
| 평균 샤프 | 1.257 | 1.428 |
| **승률** | **47.4%** | - |

---

## Key Insights / 핵심 인사이트

**1. Tax-aware Reward의 효과**
PPO와 SAC 모두 세금 반영 시 수익률과 샤프 비율이 크게 개선됨. 에이전트가 세금을 최소화하는 방향으로 리밸런싱 전략을 학습한 결과.

**2. 하이퍼파라미터 튜닝의 중요성**
Optuna 튜닝으로 SAC가 16.62% → 37.26%로 대폭 개선. 디폴트 파라미터로는 RL의 잠재력을 충분히 발휘하지 못함.

**3. PPO vs SAC**
튜닝 후 SAC가 PPO보다 우수한 성과. 충분한 튜닝이 이루어졌을 때 SAC의 높은 샘플 효율이 장점으로 작용.

**4. RL vs 전통 전략**
튜닝된 SAC (korean)이 모든 전통 전략 대비 최고 수익률과 샤프 비율 달성.

---

## Limitations & Future Work

**한계점:**
- Non-stationarity: 주식시장의 구조적 변화에 RL이 완전히 적응하지 못할 수 있음
- Walk-forward 승률 47.4%로 일관된 우위 미확보

**Future Work:**
- Ablation study: text-based signal 기여도 분석
- Regime-aware RL (시장 상태별 다른 에이전트)
- Meta-learning으로 non-stationarity 완화
- 더 많은 종목 및 글로벌 시장으로 확장

---

## File Structure

```
rl-portfolio/
├── data.py             # 데이터 수집 + 기술적 지표
├── env.py              # gym 환경 (Tax-aware Reward)
├── agent.py            # PPO/SAC 에이전트
├── train.py            # 학습 + 백테스팅
├── sec_sentiment.py    # SEC 8-K + Groq LLM 감성 분석
├── walk_forward.py     # Walk-forward Validation
├── baseline.py         # 전통 전략 베이스라인
├── tune.py             # Optuna 하이퍼파라미터 튜닝
├── .env                # API 키 (미업로드)
├── models/             # 학습된 모델
├── results/            # 백테스팅 결과 및 차트
└── cache/              # SEC 감성 캐시
```

---

## Author

**정수진 (Sujin Jeong)**  
Industrial Engineering + Biomedical Engineering (Minor), UNIST  
Founder, FIC (Finance Investment Club — UNIST, KAIST, POSTECH, DGIST, GIST)  
GitHub: [@sujin809](https://github.com/sujin809)
