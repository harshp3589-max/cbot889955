import requests
try:
    from app.config import Settings
except Exception:
    from ..config import Settings

def send_telegram(text: str):
    s = Settings()
    if not s.ENABLE_TELEGRAM or not s.TELEGRAM_BOT_TOKEN or not s.TELEGRAM_CHAT_ID:
        return {"ok": False, "reason": "telegram disabled or not configured"}
    try:
        url = f"https://api.telegram.org/bot{s.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": s.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        r = requests.post(url, json=payload, timeout=8)
        return {"ok": r.status_code == 200, "status": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}
