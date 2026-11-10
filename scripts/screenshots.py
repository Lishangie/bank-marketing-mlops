"""Скриншоты панели после каждой порции (ПР5, пункт 9).

Для каждого снимка базы data/system/snapshots/after_batch_0N.db запускается
streamlit (BANK_DB → снимок), страница открывается в Chromium без окна
(Playwright) и сохраняется целиком в report_assets/pr5/img/рис_3_N.png.

    uv run python scripts/screenshots.py
Нужен браузер: `uv run playwright install chromium` (или установленный Chrome).
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

from bank_marketing.config import ASSETS_DIR, ROOT

SNAP = ROOT / "data" / "system" / "snapshots"
OUT = ASSETS_DIR / "pr5" / "img"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(url: str, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url)
            return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(url)


def launch(p):
    for kw in ({"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(**kw)
        except Exception:  # noqa: BLE001 — пробуем следующий вариант браузера
            continue
    raise RuntimeError("нет браузера: uv run playwright install chromium")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snaps = sorted(SNAP.glob("after_batch_*.db"))
    with sync_playwright() as p:
        browser = launch(p)
        for i, db in enumerate(snaps, start=1):
            port = free_port()
            env = {**os.environ, "BANK_DB": str(db)}
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "app/dashboard.py",
                    "--server.headless",
                    "true",
                    "--server.port",
                    str(port),
                    "--browser.gatherUsageStats",
                    "false",
                    "--theme.base",
                    "light",
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                url = f"http://127.0.0.1:{port}"
                wait(url)
                page = browser.new_page(
                    viewport={"width": 1400, "height": 900}, device_scale_factor=1.5
                )
                page.goto(url)
                page.wait_for_selector("text=Список обзвона", timeout=60000)
                page.wait_for_timeout(2500)  # графики Vega дорисовываются
                # служебная шапка Streamlit не нужна на скриншоте
                page.add_style_tag(
                    content="header, [data-testid='stToolbar'],"
                    "[data-testid='stDecoration'] {display: none !important;}"
                )
                # содержимое прокручивается внутри контейнера Streamlit: окно растягивается
                # на всю высоту, чтобы скриншот захватил панель целиком
                height = page.evaluate(
                    "Math.max(...[...document.querySelectorAll('[data-testid=stMain],"
                    " [data-testid=stAppViewContainer], .block-container')]"
                    ".map(e => e.scrollHeight))"
                )
                page.set_viewport_size({"width": 1400, "height": int(height) + 20})
                page.wait_for_timeout(1500)
                page.screenshot(path=str(OUT / f"рис_3_{i}.png"), full_page=True)
                page.close()
                print(f"{db.name} -> рис_3_{i}.png")
            finally:
                proc.terminate()
                proc.wait()
        browser.close()


if __name__ == "__main__":
    main()
