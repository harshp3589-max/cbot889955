import os
import logging
import sys
from typing import Any, Dict
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

from app.config import Settings
from app.core.engine import SignalEngine
from app.core.telegram import send_telegram

# ---------- logging ----------
def setup_logging(log_level: str) -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger("signals")
    logger.propagate = False
    if not logger.handlers:
        # Stream handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(sh)

        # File handler
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler("logs/signals.log")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    return logger

# ---------- app setup ----------
settings = Settings()
logger = setup_logging(settings.LOG_LEVEL)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "a_very_secret_key_that_should_be_changed")
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", path='/socket.io')

# ---------- socket helpers ----------
def emit_log(msg: str):
    try:
        socketio.emit("log", {"line": msg})
    except Exception as e:
        logger.warning(f"Failed to emit log via socket: {e}")

def emit_status(asset: str, status: Dict[str, Any]):
    try:
        socketio.emit("status", {"asset": asset, "status": status})
    except Exception as e:
        logger.warning(f"Failed to emit status via socket: {e}")

engine = SignalEngine(log_emit=emit_log, status_emit=emit_status)

# ---------- routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/status', methods=['GET'])
def status():
    data = {
        'running': engine.is_running(),
        'exchange': getattr(engine, 'exchange_name', None),
        'quote': getattr(engine, 'quote', None),
        'last_state': getattr(engine, 'last_state', {})
    }
    return jsonify(data), 200

@app.route('/start', methods=['POST'])
def start():
    if engine.is_running():
        return jsonify({"ok": True, "running": True, "note": "already running"}), 200
    try:
        engine.start()
        return jsonify({"ok": True, "running": True}), 200
    except Exception as e:
        logger.exception("Error in /start")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop():
    try:
        engine.stop()
        return jsonify({"ok": True, "running": False}), 200
    except Exception as e:
        logger.exception("Error in /stop")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/logs/clear', methods=['POST'])
def logs_clear():
    try:
        with open("logs/signals.log", "w") as f:
            f.truncate()
        emit_log("== LOGS CLEARED ==")
        return jsonify({"ok": True}), 200
    except Exception as e:
        logger.exception("Error clearing logs")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/telegram/test', methods=['POST'])
def telegram_test():
    try:
        r = send_telegram("✅ Test message from Crypto Signals bot")
        return jsonify(r), (200 if r.get("ok") else 400)
    except Exception as e:
        logger.exception("Error in telegram test")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/signals/force', methods=['POST'])
def signals_force():
    try:
        sent = engine.force_send_all()
        return jsonify({"ok": True, "sent": sent}), 200
    except Exception as e:
        logger.exception("Error in /signals/force")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/signals/reset', methods=['POST'])
def signals_reset():
    try:
        engine.reset_states()
        return jsonify({"ok": True}), 200
    except Exception as e:
        logger.exception("Error in /signals/reset")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/movers/scan', methods=['POST'])
def movers_scan():
    try:
        res = engine.scan_movers_once()
        return jsonify({"ok": True, "found": res}), 200
    except Exception as e:
        logger.exception("Error in /movers/scan")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/logs', methods=['GET'])
def logs_tail():
    try:
        lines = int(request.args.get('lines', 300))
        lines = max(1, min(lines, 5000))  # Clamp line count
        
        path = "logs/signals.log"
        data = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.readlines()[-lines:]
        
        data = [ln.rstrip("\n") for ln in data]
        return jsonify({"ok": True, "lines": data}), 200
    except Exception as e:
        logger.exception("Error in /logs")
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host=settings.HOST, port=settings.PORT, use_reloader=False)