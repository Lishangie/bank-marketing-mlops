"""Скачивает UCI Bank Marketing и извлекает bank-additional-full.csv в data/raw/.

Внешний архив bank+marketing.zip содержит вложенный bank-additional.zip,
внутри которого лежит нужный CSV.
"""

import io
import urllib.request
import zipfile

from bank_marketing.config import DATA_URL, RAW_DIR, RAW_FILE


def find_csv(zf: zipfile.ZipFile, name: str) -> bytes | None:
    """Ищет файл name в архиве, заходя во вложенные zip."""
    for info in zf.infolist():
        base = info.filename.rsplit("/", 1)[-1]
        if base == name and "__MACOSX" not in info.filename:
            return zf.read(info)
        if info.filename.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(info))) as inner:
                found = find_csv(inner, name)
                if found is not None:
                    return found
    return None


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачивание {DATA_URL}")
    with urllib.request.urlopen(DATA_URL, timeout=120) as resp:
        payload = resp.read()
    print(f"Получено {len(payload) / 1024:.0f} КБ")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        content = find_csv(zf, RAW_FILE.name)
    if content is None:
        raise RuntimeError(f"{RAW_FILE.name} не найден в архиве")

    RAW_FILE.write_bytes(content)
    print(f"Сохранено: {RAW_FILE} ({len(content) / 1024:.0f} КБ)")


if __name__ == "__main__":
    main()
