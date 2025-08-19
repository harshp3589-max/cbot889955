# Crypto Signals Telegram (v2.6 Pro — Early Movers FIX3)

Flask + Socket.IO UI that runs multiple technical strategies (BTC/ETH/XRP/SOL/DOGE) and a **Top Movers** scan, and sends alerts to **Telegram**. Public market data only (no exchange API keys required).

## Features
- **Strategies:** pre-built logic per coin/timeframe (configurable).
- **Top Movers (1H/4H):** confluence score with liquidity & cooldown filters.
- **Sentiment gate:** optional Fear & Greed filter with thresholds.
- **Closed-candle evaluation** and **alert cooldowns** to reduce noise.
- **Web UI**: start/stop, live logs, test Telegram, reset/force-send, run Movers scan.
- **Health endpoint**: `GET /health` for Docker healthchecks.

## Quick Start (Docker)
```bash
cp .env.example .env                    # edit with your Telegram token & chat id
docker compose up -d --build
# open http://localhost:8000
```

## Environment (.env)
```env
ENABLE_TELEGRAM=true
TELEGRAM_BOT_TOKEN=replace_me
TELEGRAM_CHAT_ID=replace_me

EXCHANGE=coinbase
EXCHANGE_FALLBACKS=kucoin,bybit,binance
QUOTE_ASSET=USDC

HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

> **Note:** Do not commit real tokens. `.env.example` contains placeholders.

## Useful Endpoints
- `GET /` — UI
- `GET /status` — JSON status
- `POST /start` / `POST /stop`
- `POST /telegram/test`
- `POST /signals/force` / `POST /signals/reset`
- `POST /movers/scan`
- `GET /health`

## Troubleshooting
- If the UI loads but buttons fail, check container logs: `docker logs -f crypto_signals_telegram_v2_6`
- Verify `.env` values and that Telegram token/chat id are valid.
- If rate-limited by data sources, increase `POLL_INTERVAL_SECONDS`.
- On first run, `logs/` and `data/` are created in your working directory.
