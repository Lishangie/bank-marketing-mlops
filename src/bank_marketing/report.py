"""Текстовые выводы для отчёта."""

import sys
from pathlib import Path

WRAP = 90


def emit(lines: list[str], path: Path) -> None:
    """Пишет строки в файл (UTF-8) и печатает их, не падая на кодировке консоли."""
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc), end="")


def wrap_items(
    items: list[str], indent: str = "  ", cont: str | None = None, width: int = WRAP
) -> list[str]:
    """Склеивает элементы через «; », перенося строку только между элементами."""
    cont = indent if cont is None else cont
    lines, cur = [], indent
    for i, item in enumerate(items):
        piece = item + (";" if i < len(items) - 1 else "")
        if cur.strip() and len(cur) + 1 + len(piece) > width:
            lines.append(cur.rstrip())
            cur = cont
        cur += ("" if not cur.strip() else " ") + piece
    lines.append(cur.rstrip())
    return lines
