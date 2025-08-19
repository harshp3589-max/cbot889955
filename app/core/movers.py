import pandas as pd
from typing import Tuple

from ..indicators import ema, macd, rsi, bollinger_bands, adx, obv

def donchian_high(series: pd.Series, period: int=20) -> pd.Series:
    return series.rolling(window=period).max()

def compute_mover_score(df1h: pd.DataFrame, df4h: pd.DataFrame) -> Tuple[int, dict, str]:
    """0-100 score using trend/momentum/volatility/participation/breakout confluence."""
    ema200_1h = df1h["close"].ewm(span=200, adjust=False).mean()
    macd_line_1h = df1h["close"].ewm(span=12, adjust=False).mean() - df1h["close"].ewm(span=26, adjust=False).mean()
    signal_1h = macd_line_1h.ewm(span=9, adjust=False).mean()
    rsi_1h = rsi(df1h["close"], 14)
    upper, mid, lower = bollinger_bands(df1h["close"], 20, 2.0)
    bb_width = upper - lower
    bb_width_ma = bb_width.rolling(20).mean()
    vol_1h = df1h["volume"]
    vol_ma_1h = vol_1h.rolling(20).mean()
    obv_1h = obv(df1h["close"], df1h["volume"])
    don_hi_1h = donchian_high(df1h["high"], 20)

    ema50_4h = df4h["close"].ewm(span=50, adjust=False).mean()
    ema200_4h = df4h["close"].ewm(span=200, adjust=False).mean()

    price1h = df1h["close"].iloc[-1]
    trend_1h = 1 if price1h > ema200_1h.iloc[-1] else 0
    trend_4h = 1 if (df4h["close"].iloc[-1] > ema200_4h.iloc[-1] or ema50_4h.iloc[-1] > ema200_4h.iloc[-1]) else 0
    macd_up = 1 if (macd_line_1h.iloc[-2] <= signal_1h.iloc[-2] and macd_line_1h.iloc[-1] > signal_1h.iloc[-1]) else 0
    rsi_gate = 1 if (rsi_1h.iloc[-2] < 50 and rsi_1h.iloc[-1] >= 50) else 0
    bb_expand = 1 if bb_width.iloc[-1] > 1.3 * (bb_width_ma.iloc[-1] + 1e-9) else 0
    vol_spike = 1 if vol_1h.iloc[-1] >= 2.0 * (vol_ma_1h.iloc[-1] + 1e-9) else 0
    obv_up = 1 if obv_1h.iloc[-1] > obv_1h.iloc[-5] else 0
    breakout = 1 if price1h >= don_hi_1h.iloc[-2] else 0

    weights = {"trend_1h":12,"trend_4h":18,"macd_up":15,"rsi_gate":8,"bb_expand":15,"vol_spike":20,"obv_up":6,"breakout":6}
    components = {k:(weights[k] if v else 0) for k,v in {"trend_1h":trend_1h,"trend_4h":trend_4h,"macd_up":macd_up,"rsi_gate":rsi_gate,"bb_expand":bb_expand,"vol_spike":vol_spike,"obv_up":obv_up,"breakout":breakout}.items()}
    score = int(sum(components.values()))

    parts = []
    if trend_1h: parts.append("1h>EMA200")
    if trend_4h: parts.append("4h trend")
    if macd_up: parts.append("MACD bull (1h)")
    if rsi_gate: parts.append("RSI>50 (1h)")
    if bb_expand: parts.append("BB expand (1h)")
    if vol_spike: parts.append("Vol≥2×20 (1h)")
    if obv_up: parts.append("OBV up (1h)")
    if breakout: parts.append("Donchian-20 breakout")
    reason_text = " | ".join(parts) if parts else "No strong confluence"
    return score, components, reason_text

def dollar_volume(df: pd.DataFrame, bars: int = 24) -> float:
    sub = df.tail(max(1, bars))
    return float((sub['close'] * sub['volume']).sum())


def compute_early_mover(df_fast: pd.DataFrame, df_slow: pd.DataFrame,
                        min_ret_fast: float = 0.01, vol_spike_x: float = 1.5) -> Tuple[bool, str]:
    """Detects early momentum before classic breakout.
    Conditions (boolean AND/OR gates):
      - Fast TF (e.g., 5m) cumulative return over last 3 bars >= min_ret_fast
      - Fast TF last bar volume >= vol_spike_x × 20-bar MA
      - Slow TF (e.g., 15m) MACD bullish cross OR EMA20>EMA50
      - Optional: Avoid if RSI on slow TF > 75 to reduce chasing
    Returns (triggered, reason_text)
    """
    if len(df_fast) < 60 or len(df_slow) < 60:
        return False, "insufficient data"

    close_f = df_fast["close"]
    vol_f = df_fast["volume"]
    # 5m cumulative ret over last 3 bars
    ret3 = (close_f.iloc[-1] / close_f.iloc[-4]) - 1.0

    vol_ma20 = vol_f.rolling(20).mean()
    vol_spike = (vol_f.iloc[-1] >= vol_ma20.iloc[-1] * vol_spike_x)

    # Slow TF momentum
    ema20_s = df_slow["close"].ewm(span=20, adjust=False).mean()
    ema50_s = df_slow["close"].ewm(span=50, adjust=False).mean()
    macd_line_s = df_slow["close"].ewm(span=12, adjust=False).mean() - df_slow["close"].ewm(span=26, adjust=False).mean()
    signal_s = macd_line_s.ewm(span=9, adjust=False).mean()
    macd_cross = (macd_line_s.iloc[-2] <= signal_s.iloc[-2] and macd_line_s.iloc[-1] > signal_s.iloc[-1])
    ema_trend = ema20_s.iloc[-1] > ema50_s.iloc[-1]

    # RSI gate to avoid already-hot parabolic
    from .engine import rsi as _rsi  # local import to avoid circulars during type checking
    rsi_s = _rsi(df_slow["close"], 14)
    rsi_ok = rsi_s.iloc[-1] < 75

    triggered = (ret3 >= min_ret_fast) and vol_spike and (macd_cross or ema_trend) and rsi_ok

    parts = []
    if ret3 >= min_ret_fast: parts.append(f"{int(min_ret_fast*100)}%+ in 15m")
    if vol_spike: parts.append(f"Vol≥{vol_spike_x}×20 (fast)")
    if macd_cross: parts.append("MACD cross (slow)")
    if ema_trend and not macd_cross: parts.append("EMA20>EMA50 (slow)")
    if not rsi_ok: parts.append("RSI hot")
    reason = " | ".join(parts) if parts else "no confluence"
    return bool(triggered), reason
