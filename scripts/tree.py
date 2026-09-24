"""Печатает дерево репозитория (├── └──) и сохраняет листинг для отчёта.

Пропускаются .git, .venv, __pycache__, кеши инструментов и .dvc/cache;
глубина не больше 3 уровней.
"""

from pathlib import Path

from bank_marketing.config import ASSETS_CODE_DIR, ROOT

MAX_DEPTH = 3
SKIP_NAMES = {".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache", "mlruns"}
SKIP_PATHS = {Path(".dvc/cache"), Path(".dvc/tmp"), Path("PROMPT.md")}
OUT = ASSETS_CODE_DIR / "листинг_2_2_структура.txt"


def skipped(path: Path) -> bool:
    return path.name in SKIP_NAMES or path.relative_to(ROOT) in SKIP_PATHS


def walk(folder: Path, prefix: str = "", depth: int = 1) -> list[str]:
    entries = sorted(
        (p for p in folder.iterdir() if not skipped(p)),
        key=lambda p: (p.is_file(), p.name.lower()),
    )
    lines = []
    for i, path in enumerate(entries):
        last = i == len(entries) - 1
        name = path.name + ("/" if path.is_dir() else "")
        lines.append(f"{prefix}{'└── ' if last else '├── '}{name}")
        if path.is_dir() and depth < MAX_DEPTH:
            lines.extend(walk(path, prefix + ("    " if last else "│   "), depth + 1))
    return lines


def main() -> None:
    lines = [f"{ROOT.name}/", *walk(ROOT)]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
