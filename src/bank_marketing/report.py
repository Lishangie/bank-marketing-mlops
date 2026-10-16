"""Текстовые выводы для отчёта."""

import sys
from pathlib import Path


def emit(lines: list[str], path: Path) -> None:
    """Пишет строки в файл (UTF-8) и печатает их, не падая на кодировке консоли."""
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc), end="")
