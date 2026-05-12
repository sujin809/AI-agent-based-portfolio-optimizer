import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# SEC EDGAR 헤더 (이메일 필수)
HEADERS = {
    "User-Agent": "carus0809@unist.ac.kr",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_BASE = "https://data.sec.gov"
FULLTEXT_BASE = "https://efts.sec.gov/LATEST/search-index"


# ── 1. 티커 → CIK 변환 ───────────────────────────────────────
def get_cik(ticker: str) -> str | None:
    """
    티커 심볼을 SEC CIK 번호로 변환.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        data = resp.json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception as e:
        print(f"[sec] CIK lookup failed for {ticker}: {e}")
    return None


# ── 2. 8-K 공시 목록 가져오기 ─────────────────────────────────
def get_8k_filings(cik: str, start_date: str, end_date: str) -> list:
    """
    특정 기간의 8-K 공시 목록 반환.

    Returns: list of dicts with keys: accessionNumber, filingDate, primaryDocument
    """
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        data = resp.json()

        filings = data.get("filings", {}).get("recent", {})
        forms       = filings.get("form", [])
        dates       = filings.get("filingDate", [])
        accessions  = filings.get("accessionNumber", [])
        documents   = filings.get("primaryDocument", [])

        results = []
        for form, date, acc, doc in zip(forms, dates, accessions, documents):
            if form == "8-K" and start_date <= date <= end_date:
                results.append({
                    "accessionNumber": acc.replace("-", ""),
                    "filingDate": date,
                    "primaryDocument": doc,
                })
        return results

    except Exception as e:
        print(f"[sec] Filing fetch failed for CIK {cik}: {e}")
        return []


# ── 3. 8-K 본문 텍스트 가져오기 ──────────────────────────────
def get_filing_text(cik: str, accession: str, document: str) -> str:
    """
    8-K 공시 본문 텍스트 추출 (첫 3000자만 사용).
    """
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession}/{document}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        text = resp.text

        # HTML 태그 간단히 제거
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:3000]  # LLM 토큰 절약

    except Exception as e:
        print(f"[sec] Text fetch failed: {e}")
        return ""


# ── 4. Groq LLM으로 감성 점수 추출 ───────────────────────────
def analyze_sentiment(text: str, ticker: str) -> float:
    """
    Groq LLM으로 8-K 공시 텍스트의 감성 점수 추출.

    Returns: float in [-1, 1]
        -1: 매우 부정적
         0: 중립
        +1: 매우 긍정적
    """
    if not text.strip():
        return 0.0

    prompt = f"""You are a financial analyst. Analyze the following SEC 8-K filing excerpt for {ticker} and provide a sentiment score.

Filing text:
{text}

Respond with ONLY a single number between -1.0 and 1.0:
- (-1.0): Very negative (bankruptcy, major loss, fraud, lawsuit)
- (-0.5): Negative (earnings miss, guidance cut, key executive departure)
- (0.0): Neutral (routine disclosure, no major news)
- (0.5): Positive (earnings beat, new contract, strategic partnership)
- (1.0): Very positive (major acquisition, record revenue, breakthrough)

Score:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        score_str = response.choices[0].message.content.strip()
        score = float(score_str)
        return max(-1.0, min(1.0, score))  # clamp

    except Exception as e:
        print(f"[sec] Sentiment analysis failed: {e}")
        return 0.0


# ── 5. 종목별 감성 점수 시계열 생성 ──────────────────────────
def fetch_ticker_sentiment(
    ticker: str,
    start_date: str,
    end_date: str,
    date_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    특정 종목의 8-K 공시 감성 점수를 날짜별로 정리.
    공시가 없는 날은 0 (중립), 공시 이후 신호는 다음 거래일까지 유지.

    Returns: pd.Series [날짜 index, 감성 점수]
    """
    sentiment = pd.Series(0.0, index=date_index)

    cik = get_cik(ticker)
    if cik is None:
        print(f"[sec] CIK not found for {ticker}, using 0")
        return sentiment

    filings = get_8k_filings(cik, start_date, end_date)
    print(f"[sec] {ticker}: {len(filings)} 8-K filings found")

    for filing in filings:
        date     = filing["filingDate"]
        accession= filing["accessionNumber"]
        document = filing["primaryDocument"]

        text  = get_filing_text(cik, accession, document)
        score = analyze_sentiment(text, ticker)

        # 해당 날짜 이후 5 거래일 동안 신호 유지 (decay)
        filing_dt = pd.Timestamp(date)
        for i, idx_date in enumerate(date_index):
            if filing_dt <= idx_date <= filing_dt + pd.Timedelta(days=7):
                decay = 1.0 - (i * 0.15)  # 시간이 지날수록 신호 감쇠
                sentiment[idx_date] = score * max(0.1, decay)

        time.sleep(0.5)  # SEC rate limit 준수

    return sentiment


# ── 6. 전체 유니버스 감성 점수 생성 ──────────────────────────
def build_sentiment_matrix(
    tickers: list,
    date_index: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
    cache_path: str = "./cache/sentiment.parquet",
) -> pd.DataFrame:
    """
    모든 종목의 감성 점수 행렬 생성 및 캐싱.

    Returns: DataFrame [날짜 x 종목] 감성 점수
    """
    # 캐시 확인
    if os.path.exists(cache_path):
        print(f"[sec] Loading sentiment from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        # 날짜 인덱스 맞추기
        df = df.reindex(date_index).fillna(0.0)
        return df

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    sentiment_dict = {}

    for i, ticker in enumerate(tickers):
        print(f"[sec] Processing {ticker} ({i+1}/{len(tickers)})...")
        s = fetch_ticker_sentiment(ticker, start_date, end_date, date_index)
        sentiment_dict[ticker] = s
        time.sleep(1.0)  # API rate limit

    df = pd.DataFrame(sentiment_dict, index=date_index).fillna(0.0)

    # 캐시 저장
    df.to_parquet(cache_path)
    print(f"[sec] Sentiment matrix saved → {cache_path}")
    return df


# ── 테스트 ────────────────────────────────────────────────────
if __name__ == "__main__":
    import yfinance as yf
    from data import TICKERS

    # 테스트용 짧은 기간 + 소수 종목
    TEST_TICKERS = ["AAPL", "MSFT", "GOOGL"]
    START = "2024-01-01"
    END   = "2024-03-31"

    # 날짜 인덱스 생성 (거래일 기준)
    prices = yf.download(TEST_TICKERS, start=START, end=END,
                         auto_adjust=True, progress=False)["Close"]
    date_index = prices.index

    print(f"[test] Date range: {date_index[0]} ~ {date_index[-1]}")
    print(f"[test] Trading days: {len(date_index)}")

    # CIK 테스트
    for ticker in TEST_TICKERS:
        cik = get_cik(ticker)
        print(f"  {ticker} → CIK: {cik}")

    # 감성 행렬 생성 (캐시 사용)
    sentiment_df = build_sentiment_matrix(
        TEST_TICKERS, date_index, START, END,
        cache_path="./cache/test_sentiment.parquet",
    )

    print(f"\n✅ Sentiment matrix shape: {sentiment_df.shape}")
    print(f"\n샘플 (처음 5행):")
    print(sentiment_df.head())
    print(f"\n비중립 점수 수: {(sentiment_df != 0).sum().sum()}")
