"""Рендер диаграмм ПР5 (контейнеры, развёртывание) в report_assets/pr5/img.

Java и plantuml.jar — как для scripts/render_diagrams.py (см. README).
"""

import os
import shutil
import subprocess

from render_diagrams import JAR, OUT_DIR, SRC_DIR, find_dot, find_java  # noqa: I001

from bank_marketing.config import ASSETS_DIR

FIGURES = {"pr5_container": "рис_1_1.png", "pr5_deployment": "рис_1_2.png"}


def main() -> None:
    env = dict(os.environ)
    if dot := find_dot():
        env["GRAPHVIZ_DOT"] = dot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [find_java(), "-jar", str(JAR), "-tpng", "-charset", "UTF-8", "-o", str(OUT_DIR)]
    subprocess.run([*cmd, *(str(SRC_DIR / f"{n}.puml") for n in FIGURES)], check=True, env=env)
    dest = ASSETS_DIR / "pr5" / "img"
    dest.mkdir(parents=True, exist_ok=True)
    for name, figure in FIGURES.items():
        shutil.copyfile(OUT_DIR / f"{name}.png", dest / figure)
        print(f"{name}.puml -> report_assets/pr5/img/{figure}")


if __name__ == "__main__":
    main()
