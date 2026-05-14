import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ── 종목 유니버스 ──────────────────────────────────────────────
TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA",
    # Consumer
    "AMZN", "HD", "MCD", "NKE", "COST",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "LLY",
    # Finance
    "JPM", "BAC", "GS", "BRK-B",
    # Energy
    "XOM", "CVX",
    # Industrial
    "CAT", "HON", "UPS",
]

START_DATE = "2019-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")


# ── 1. 주가 데이터 수집 ────────────────────────────────────────
def fetch_price_data(tickers=TICKERS, start=START_DATE, end=END_DATE):
    """
    yfinance로 일별 종가(Adj Close) 수집.
    Returns: DataFrame [날짜 x 종목]
    """
    print(f"[data] Fetching price data for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all")
    prices = prices.ffill().dropna()
    print(f"[data] Price data shape: {prices.shape}")
    return prices


# ── 2. 기술적 지표 계산 ────────────────────────────────────────
def compute_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    각 종목에 대해 기술적 지표를 계산.
    - 수익률 (1일, 5일, 20일)
    - 이동평균 (MA20, MA60)
    - RSI (14일)
    - 변동성 (20일 rolling std)

    Returns: MultiIndex DataFrame [날짜 x (지표, 종목)]
    """
    features = {}

    for ticker in prices.columns:
        p = prices[ticker]
        df = pd.DataFrame()

        # 수익률
        df["ret_1d"]  = p.pct_change(1)
        df["ret_5d"]  = p.pct_change(5)
        df["ret_20d"] = p.pct_change(20)

        # 이동평균 대비 가격 비율
        df["ma20_ratio"] = p / p.rolling(20).mean() - 1
        df["ma60_ratio"] = p / p.rolling(60).mean() - 1

        # RSI
        df["rsi"] = _compute_rsi(p, window=14)

        # 변동성
        df["vol_20d"] = df["ret_1d"].rolling(20).std()

        # MACD
        ema12 = p.ewm(span = 12).mean()
        ema26 = p.ewm(span = 26).mean()
        df["macd"] = ema12 -ema26

        #볼린저 밴드
        ma20 = p.rolling(20).mean()
        std20 = p.rolling(20).std()
        df["bb_upper"] = (p - (ma20 + 2 * std20)) / p
        df["bb_lower"] = (p - (ma20 - 2 * std20)) / p

        features[ticker] = df

    combined = pd.concat(features, axis=1)  # MultiIndex: (ticker, feature)
    combined = combined.swaplevel(axis=1).sort_index(axis=1)  # (feature, ticker)
    combined = combined.dropna()
    return combined


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ── 3. Text-based Signal (SEC 공시) ──────────────────────────
def fetch_sec_sentiment(tickers=TICKERS, lookback_days: int = 30) -> pd.DataFrame:
    from sec_sentiment import build_sentiment_matrix
    import yfinance as yf

    prices = fetch_price_data(tickers)
    start  = prices.index[0].strftime("%Y-%m-%d")
    end    = prices.index[-1].strftime("%Y-%m-%d")

    sentiment = build_sentiment_matrix(
        tickers=tickers,
        date_index=prices.index,
        start_date=start,
        end_date=end,
        cache_path="./cache/sentiment_full.parquet",
    )
    return sentiment


# ── 4. 최종 State 행렬 구성 ───────────────────────────────────
def build_state_matrix(
    prices: pd.DataFrame,
    tech_features: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> np.ndarray:
    """
    State = [기술적 지표 + 감성 점수] 를 하나의 행렬로 합침.

    Returns:
        states: np.ndarray [T x N x F]
            T = 시간 스텝
            N = 종목 수
            F = feature 수
    """
    tickers = prices.columns.tolist()
    feature_names = tech_features.columns.get_level_values(0).unique().tolist()

    # 공통 날짜 인덱스
    common_idx = tech_features.index.intersection(sentiment.index)
    tech_features = tech_features.loc[common_idx]
    sentiment     = sentiment.loc[common_idx]

    T = len(common_idx)
    N = len(tickers)
    F = len(feature_names) + 1  # 기술적 지표 + 감성 점수

    states = np.zeros((T, N, F))

    for i, ticker in enumerate(tickers):
        for j, feat in enumerate(feature_names):
            if (feat, ticker) in tech_features.columns:
                states[:, i, j] = tech_features[(feat, ticker)].values
        # 마지막 feature = 감성 점수
        if ticker in sentiment.columns:
            states[:, i, -1] = sentiment[ticker].values

    return states, common_idx, tickers, feature_names + ["sentiment"]


# ── 5. 세금 계산 유틸 ─────────────────────────────────────────
def compute_tax(
    capital_gain: float,
    tax_regime: str = "korean",
    exemption: float = 2_500_000,
) -> float:
    """
    세금 계산 (일반화된 프레임워크).

    Args:
        capital_gain: 양도차익 (KRW 기준)
        tax_regime: 'korean' | 'us_short' | 'us_long' | 'none'
        exemption: 기본 공제액 (한국: 250만원)

    Returns:
        tax: 납부 세금
    """
    if tax_regime == "korean":
        taxable = max(0, capital_gain - exemption)
        return taxable * 0.22  # 양도세 22% (지방세 포함)

    elif tax_regime == "us_short":
        return max(0, capital_gain) * 0.37  # 단기 양도세

    elif tax_regime == "us_long":
        return max(0, capital_gain) * 0.20  # 장기 양도세

    elif tax_regime == "none":
        return 0.0

    else:
        raise ValueError(f"Unknown tax_regime: {tax_regime}")


# ── 테스트 ────────────────────────────────────────────────────
if __name__ == "__main__":
    prices = fetch_price_data()
    tech   = compute_technical_features(prices)
    sent   = fetch_sec_sentiment()
    states, idx, tickers, feat_names = build_state_matrix(prices, tech, sent)

    print(f"\n✅ State matrix shape: {states.shape}")
    print(f"   시간 스텝: {states.shape[0]}")
    print(f"   종목 수:   {states.shape[1]}")
    print(f"   feature 수:{states.shape[2]}")
    print(f"   features:  {feat_names}")
    print(f"\n✅ Tax 계산 테스트:")
    print(f"   한국 (500만원 이익): {compute_tax(5_000_000, 'korean'):,.0f}원")
    print(f"   미국 장기 (500만원): {compute_tax(5_000_000, 'us_long'):,.0f}원")
