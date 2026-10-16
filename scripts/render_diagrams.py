"""Рендер диаграмм PlantUML (docs/architecture/*.puml) в PNG для отчёта.

Java ищется так: переносная JDK в tools/jdk-*, затем JAVA_HOME, затем PATH;
нужна Java 11+. PNG пишутся в docs/architecture/png/ и копируются
в report_assets/pr2/img/ и reports/figures/pr2/ под именами рисунков.
Печатается размер каждого PNG и ширина на странице при 200 dpi.
"""

import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

from bank_marketing.config import PR2_FIGURES_DIR, PR2_IMG_DIR, ROOT

SRC_DIR = ROOT / "docs" / "architecture"
OUT_DIR = SRC_DIR / "png"
JAR = ROOT / "tools" / "plantuml.jar"
DPI = 200
PAGE_CM = 16  # ширина рисунка в отчёте
FIGURES = {
    "c4_context": "рис_2_1.png",
    "c4_container": "рис_2_2.png",
    "use_cases": "рис_2_3.png",
    "sequence_batch": "рис_2_4.png",
    "activity_pipeline": "рис_2_5.png",
}


def java_version(java: str) -> int:
    res = subprocess.run([java, "-version"], capture_output=True, text=True)
    m = re.search(r'version "(\d+)(?:\.(\d+))?', res.stderr)
    if not m:
        return 0
    major = int(m.group(1))
    return int(m.group(2)) if major == 1 else major  # "1.8" → 8


def find_java() -> str:
    exe = "java.exe" if os.name == "nt" else "java"
    candidates = sorted((ROOT / "tools").glob(f"jdk-*/bin/{exe}"), reverse=True)
    if os.environ.get("JAVA_HOME"):
        candidates.append(Path(os.environ["JAVA_HOME"]) / "bin" / exe)
    if shutil.which("java"):
        candidates.append(Path(shutil.which("java")))
    for java in candidates:
        if java.exists() and java_version(str(java)) >= 11:
            return str(java)
    raise RuntimeError("нужна Java 11+: см. README, раздел «Диаграммы»")


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def main() -> None:
    if not JAR.exists():
        raise FileNotFoundError(f"{JAR} не найден: см. README, раздел «Диаграммы»")
    java = find_java()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = [SRC_DIR / f"{name}.puml" for name in FIGURES]
    cmd = [java, "-jar", str(JAR), "-tpng", "-charset", "UTF-8", "-o", str(OUT_DIR)]
    subprocess.run([*cmd, *map(str, sources)], check=True)

    version = subprocess.run(
        [java, "-jar", str(JAR), "-version"], capture_output=True, text=True
    ).stdout.splitlines()[0]
    print(f"{version}; Java {java_version(java)}")
    for name, figure in FIGURES.items():
        png = OUT_DIR / f"{name}.png"
        for folder in (PR2_IMG_DIR, PR2_FIGURES_DIR):
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(png, folder / figure)
        w, h = png_size(png)
        natural_cm = w / DPI * 2.54
        scale = min(1.0, PAGE_CM / natural_cm)
        print(
            f"{figure}: {w} x {h} px; при {DPI} dpi {natural_cm:.1f} x {h / DPI * 2.54:.1f} см; "
            f"масштаб при ширине {PAGE_CM} см: {scale:.2f}"
        )


if __name__ == "__main__":
    main()
