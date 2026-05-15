# RL Portfolio Optimizer

> **Tax-Aware Dynamic Portfolio Optimization via Reinforcement Learning with SEC Filing-Based Text Signals**

세금 구조와 SEC 공시 기반 텍스트 신호를 통합한 RL 기반 동적 포트폴리오 최적화 프레임워크입니다.  
매일 27개 미국 주식의 비중을 RL 에이전트가 동적으로 결정하며, 세후 실질 수익을 최대화하는 전략을 학습합니다.

---

## Why This Project / 왜 이 프로젝트인가

기존 RL 포트폴리오 최적화 연구들은 두 가지 중요한 요소를 무시해왔습니다:

**1. 세금 구조 무시**
대부분의 논문이 세금을 단순화하거나 완전히 무시합니다. 하지만 한국 투자자의 경우 해외주식 양도세 22%는 실질 수익에 큰 영향을 미칩니다. 이 프로젝트는 세금을 Reward에 직접 통합한 **일반화된 Tax-aware Reward 프레임워크**를 제안합니다.

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
RL 환경 (Tax-aware Reward + Sharpe 기반)
        ↓
PPO / SAC 에이전트 학습
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

# SAC + 세금 없음
python train.py --agent sac --tax none --timesteps 200000
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

### 4. 에이전트
| 에이전트 | 특징 |
|---------|------|
| PPO | 온폴리시, 안정적, clip_range=0.2로 급격한 정책 변화 방지 |
| SAC | 오프폴리시, 샘플 효율 높음, 엔트로피 최대화로 탐색 장려 |

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

### RL 에이전트 vs 전통 전략 비교

| 전략 | 수익률 | 샤프 비율 | MDD |
|------|--------|-----------|-----|
| **PPO (korean)** | **28.52%** | **0.973** | -16.14% |
| SAC (none) | 27.41% | 0.937 | -18.16% |
| Equal Weight | 25.75% | 0.830 | -17.61% |
| Market Cap | 20.47% | 0.521 | -17.25% |
| Risk Parity | 18.05% | 0.495 | -15.81% |
| Momentum | 16.77% | 0.406 | -18.38% |
| PPO (none) | 17.07% | 0.437 | -19.57% |
| SAC (korean) | 16.62% | 0.409 | -20.78% |
| Min Variance | 9.21% | 0.075 | -17.61% |

### Tax-aware Reward 효과

| | PPO (korean) | PPO (none) | 차이 |
|--|-------------|------------|------|
| 수익률 | **28.52%** | 17.07% | **+11.45%** |
| 샤프 비율 | **0.973** | 0.437 | **+0.536** |
| MDD | **-16.14%** | -19.57% | **개선** |

> PPO에서 세금 반영 시 수익률 +11.45%, 샤프 +0.536 개선. **세금을 Reward에 통합하는 것이 실질 수익 개선에 유의미한 영향을 미침.**

### Walk-forward Validation (PPO korean, 200,000 steps)

| | RL Agent | Equal Weight |
|--|---------|-------------|
| 평균 수익률 | 6.94% | 7.44% |
| 평균 샤프 | 1.257 | 1.428 |
| **승률** | **47.4%** | - |

> 19개 fold 중 9개에서 Equal Weight 초과 성과. 특정 시장 환경(상승장)에서 유의미한 우위.

---

## Key Insights / 핵심 인사이트

**1. Tax-aware Reward의 효과**
PPO에서 세금 반영 시 수익률과 샤프 비율이 크게 개선됨. 에이전트가 세금을 최소화하는 방향으로 리밸런싱 전략을 학습한 결과.

**2. PPO vs SAC**
PPO가 전반적으로 우수한 성과. 금융 데이터처럼 노이즈가 많고 샘플이 제한적인 환경에서는 안정적인 PPO가 적합.

**3. RL vs 전통 전략**
PPO (korean)이 모든 전통 전략 중 최고 수익률과 샤프 비율 달성. Equal Weight 대비 수익률 +2.77%, 샤프 +0.143.

---

## Limitations & Future Work

**한계점:**
- Non-stationarity: 주식시장의 구조적 변화에 RL이 완전히 적응하지 못할 수 있음
- Walk-forward 승률 47.4%로 일관된 우위 미확보

**Future Work:**
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
