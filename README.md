# Bank Marketing MLOps

Прогноз отклика клиентов банка на телемаркетинговую кампанию (оформит ли клиент срочный депозит, `y`) по данным UCI Bank Marketing.

## Воспроизведение

```powershell
uv sync                  # окружение Python 3.12 из uv.lock
uv run dvc pull          # данные из DVC-хранилища (../dvc-storage)
uv run dvc repro         # конвейер: split → eda_initial, validate, kpi, feature_candidates
uv run python scripts/eda.py   # EDA всего набора (ПР1): рисунки и report_assets/eda_out.txt
```

Стадии конвейера (`dvc.yaml`, параметры — `params.yaml`):

| Стадия | Скрипт | Результат |
| --- | --- | --- |
| `split` | `scripts/split_data.py` | `data/initial/initial.csv`, `data/incoming/batch_01…04.csv`, рисунок 1.1 |
| `eda_initial` | `scripts/eda_initial.py` | `report_assets/pr2/eda_initial_out.txt`, рисунки 1.2–1.5 |
| `validate` | `scripts/validate.py` | `report_assets/pr2/validate_out.txt`, `reports/validation/*.json`, `models/reference/data_reference.json` |
| `kpi` | `scripts/kpi.py` | `report_assets/pr2/kpi_out.txt` |
| `feature_candidates` | `scripts/feature_candidates.py` | `report_assets/pr2/features_out.txt` |

Замер времени проверки качества (вне конвейера: зависит от машины и нагрузки):
`uv run python scripts/benchmark.py` → `report_assets/pr2/benchmark_out.txt`
(медиана из 5 прогонов по каждой части и по порции из 10 000 записей).

Скачать данные заново: `uv run python scripts/download_data.py`.

Docker: `docker build -t bank-marketing .`, затем
`docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/report_assets:/app/report_assets bank-marketing`.

## Диаграммы

Исходники PlantUML — `docs/architecture/*.puml` (C4 из встроенной stdlib, раскладка smetana, Graphviz не нужен).
Нужны Java 11+ и `tools/plantuml.jar` (каталог `tools/` не версионируется):

```powershell
New-Item -ItemType Directory -Force tools
curl.exe -L -o tools/plantuml.jar https://github.com/plantuml/plantuml/releases/download/v1.2026.8/plantuml.jar
# Java 11+: winget install --id Microsoft.OpenJDK.21 -e
# или переносная JDK в tools/jdk-*: https://aka.ms/download-jdk/microsoft-jdk-21-windows-x64.zip
uv run python scripts/render_diagrams.py
```

Версия PlantUML: **1.2026.8**. Скрипт ищет Java в `tools/jdk-*`, `JAVA_HOME`, затем в `PATH`,
пишет PNG в `docs/architecture/png/` и копирует их в `report_assets/pr2/img/рис_2_1…рис_2_5.png`.
