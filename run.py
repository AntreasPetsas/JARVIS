"""Launch Jarvis: start the web server and open the HUD in your browser.

    python run.py
"""
from __future__ import annotations

import threading
import webbrowser

import uvicorn

from jarvis.config import load_config
from jarvis.server import create_app


def main() -> None:
    cfg = load_config()
    host = cfg.get("server.host", "127.0.0.1")
    port = int(cfg.get("server.port", 8765))
    app = create_app(cfg)

    if cfg.get("server.open_browser", True):
        url = f"http://{host}:{port}/"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"\n  J.A.R.V.I.S online  ->  http://{host}:{port}/\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
