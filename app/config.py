import os
from dotenv import load_dotenv

load_dotenv()

def as_bool(v: str, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

class Settings:
    ENABLE_TELEGRAM = as_bool(os.getenv("ENABLE_TELEGRAM","true"), True)
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","")

    EXCHANGE = os.getenv("EXCHANGE","coinbase")
    EXCHANGE_FALLBACKS = [e.strip() for e in os.getenv("EXCHANGE_FALLBACKS","kucoin,bybit,binanceus").split(",") if e.strip()]
    QUOTE_ASSET = os.getenv("QUOTE_ASSET","USDC")
    POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS","60"))

    ENABLE_BTC = as_bool(os.getenv("ENABLE_BTC","true"), True)
    ENABLE_ETH = as_bool(os.getenv("ENABLE_ETH","true"), True)
    ENABLE_XRP = as_bool(os.getenv("ENABLE_XRP","true"), True)
    ENABLE_SOL = as_bool(os.getenv("ENABLE_SOL","true"), True)
    ENABLE_DOGE = as_bool(os.getenv("ENABLE_DOGE","true"), True)

    # Movers
    ENABLE_MOVERS = as_bool(os.getenv("ENABLE_MOVERS","true"), True)
    MOVERS_UNIVERSE_LIMIT = int(os.getenv("MOVERS_UNIVERSE_LIMIT","50"))
    MOVERS_TF_FAST = os.getenv("MOVERS_TF_FAST","1h")
    MOVERS_TF_SLOW = os.getenv("MOVERS_TF_SLOW","4h")
    MOVERS_SCORE_THRESHOLD = int(os.getenv("MOVERS_SCORE_THRESHOLD","60"))
    MOVERS_MIN_DOLLAR_VOL = float(os.getenv("MOVERS_MIN_DOLLAR_VOL","2000000"))
    MOVERS_COOLDOWN_MINUTES = int(os.getenv("MOVERS_COOLDOWN_MINUTES","360"))

    # Early Movers
    ENABLE_EARLY_MOVERS = as_bool(os.getenv("ENABLE_EARLY_MOVERS","true"), True)
    EARLY_MOVERS_TF_FAST = os.getenv("EARLY_MOVERS_TF_FAST","5m")
    EARLY_MOVERS_TF_SLOW = os.getenv("EARLY_MOVERS_TF_SLOW","15m")
    EARLY_MOVERS_MIN_RET_5M = float(os.getenv("EARLY_MOVERS_MIN_RET_5M","0.01"))
    EARLY_MOVERS_VOL_SPIKE_X = float(os.getenv("EARLY_MOVERS_VOL_SPIKE_X","1.5"))
    EARLY_MOVERS_COOLDOWN_MINUTES = int(os.getenv("EARLY_MOVERS_COOLDOWN_MINUTES","120"))
    EARLY_MOVERS_MIN_DOLLAR_VOL = float(os.getenv("EARLY_MOVERS_MIN_DOLLAR_VOL","1000000"))

    # Sentiment (Fear & Greed)
    ENABLE_FNG_FILTER = as_bool(os.getenv("ENABLE_FNG_FILTER","true"), True)
    FNG_GREED_THRESHOLD = int(os.getenv("FNG_GREED_THRESHOLD","70"))
    FNG_FEAR_THRESHOLD = int(os.getenv("FNG_FEAR_THRESHOLD","30"))
    FNG_CACHE_MINUTES = int(os.getenv("FNG_CACHE_MINUTES","15"))

    # Alert hygiene
    ALWAYS_USE_CLOSED_CANDLE = as_bool(os.getenv("ALWAYS_USE_CLOSED_CANDLE","true"), True)
    MIN_ALERT_GAP_MINUTES = int(os.getenv("MIN_ALERT_GAP_MINUTES","30"))

    # Hedging
    ENABLE_HEDGING = as_bool(os.getenv("ENABLE_HEDGING", "true"), True)
    INVERSE_ETF_SYMBOL = os.getenv("INVERSE_ETF_SYMBOL", "BITI")

    # Auto Trading
    ENABLE_AUTO_TRADING = as_bool(os.getenv("ENABLE_AUTO_TRADING", "false"), False)
    TRADE_SIZE_USD = float(os.getenv("TRADE_SIZE_USD", "100"))

    LOG_LEVEL = os.getenv("LOG_LEVEL","INFO")
    HOST = os.getenv("HOST","0.0.0.0")
    PORT = int(os.getenv("PORT","8000"))