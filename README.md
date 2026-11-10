# Bank Marketing MLOps

Прогноз отклика клиентов банка на телемаркетинговую кампанию (оформит ли клиент срочный депозит, `y`) по данным UCI Bank Marketing.

## Воспроизведение

```powershell
uv sync                  # окружение Python 3.12 из uv.lock
uv run dvc pull          # данные из DVC-хранилища (../dvc-storage)
uv run dvc repro         # конвейер: split → eda_initial, validate, kpi, feature_candidates → features, experiments → train
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
| `features` (ПР3) | `scripts/build_features.py` | `data/processed/*.parquet`, `models/reference/feature_params.json`, `report_assets/pr3/features_out.txt` |
| `experiments` (ПР3) | `scripts/train_experiments.py` | `reports/experiments.csv`, `reports/selection.json`, `report_assets/pr3/experiments_out.txt`, рисунки 1.1, 3.1–3.3 |
| `train` (ПР3) | `scripts/train_final.py` | `models/model_v1.joblib`, `models/model_v1.json`, `report_assets/pr3/train_out.txt` |
| `quality`, `evaluate` (ПР4) | `scripts/check_quality.py`, `scripts/evaluate.py` | `reports/quality_demo/`, `reports/evaluation.json`, `report_assets/pr4/` |
| `service` (ПР5) | `scripts/run_service.py` | `data/system/bank.db`, `models/registry/`, `models/current.json`, `report_assets/pr5/service_out.txt` |

Замер времени проверки качества (вне конвейера: зависит от машины и нагрузки):
`uv run python scripts/benchmark.py` → `report_assets/pr2/benchmark_out.txt`
(медиана из 5 прогонов по каждой части и по порции из 10 000 записей).

Скачать данные заново: `uv run python scripts/download_data.py`.

Docker: `docker build -t bank-marketing .`, затем
`docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/report_assets:/app/report_assets bank-marketing`.

## Модели (ПР3)

Пайплайн признаков — `src/bank_marketing/features.py`: `prepare(path)` превращает любой файл
(исходную часть или порцию) в данные для модели, `make_pipeline(model)` объединяет новые признаки,
предобработку и модель в один `Pipeline`. Проверить на порции:
`uv run python -m bank_marketing.features data/incoming/batch_01.csv`.

Эксперименты (стадия `experiments`, около 5 минут) пишутся в `reports/experiments.csv` и в MLflow.
Просмотр: `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`, эксперимент
`pr3-model-selection`. Итоговая модель — `models/model_v1.joblib` (под DVC) с описанием
`models/model_v1.json`, в реестре MLflow — `bank-marketing-response`, версия 1.

## Система: обработка порций, панель, API (ПР5)

Код — `src/bank_marketing/service/` (хранилище `storage.py`, показатели и правила
`monitoring.py`, сценарий `process_batch.py`, переобучение `retrain.py`, версии `registry.py`),
панель и API — `app/`, тесты — `tests/`. Пороги и правило замены — `params.yaml` → `service`
и `evaluate.monitor`.

```powershell
uv sync
# 1. новая порция одной командой (повторный запуск той же порции ничего не дублирует)
uv run python -m bank_marketing.service.process_batch data/incoming/batch_01.csv
# 2. если панель показала «Сигнал к переобучению» — претендент и решение о замене
uv run python -m bank_marketing.service.retrain
# 3. версии моделей, откат и возврат
uv run python -m bank_marketing.service.registry status
uv run python -m bank_marketing.service.registry rollback
uv run python -m bank_marketing.service.registry activate v3
# 4. панель (http://localhost:8501) и API (http://127.0.0.1:8000/docs) — в двух окнах
uv run streamlit run app/dashboard.py
uv run uvicorn app.api:app --port 8000
# 5. тесты API
uv run pytest -q
```

Весь сценарий ПР5 с нуля (испорченные порции, порции 1–4, переобучение, замеры, API, тесты) —
`uv run dvc repro service` (около 2 минут). Хранилище — `data/system/bank.db` (SQLite: `raw_data`,
`predictions`, `metrics_history`, `quality`, `alerts`, `runs`, `retrains`), списки обзвона для CRM —
`data/system/call_lists/`, снимки базы после каждой порции — `data/system/snapshots/`.
Модели: v1 — `models/model_v1.joblib` (ПР3), новые — `models/registry/model_vN.joblib` с карточкой
`vN.json` (данные, строки и их MD5, коммит, параметры, метрики сравнения, профиль для PSI);
действующая версия — `models/current.json`.

Скриншоты панели по снимкам базы: `uv run playwright install chromium`, затем
`uv run python scripts/screenshots.py` → `report_assets/pr5/img/рис_3_1…3_4.png`.
Диаграммы ПР5: `uv run python scripts/render_pr5_diagrams.py` → `рис_1_1.png`, `рис_1_2.png`.

## Диаграммы

Исходники PlantUML — `docs/architecture/*.puml` (C4 из встроенной stdlib). Рис. 2.2 (контейнеры)
и 2.3 (варианты использования) раскладываются Graphviz, остальные — встроенной раскладкой smetana.
Нужны Java 11+, `tools/plantuml.jar` и Graphviz в `tools/graphviz` (каталог `tools/` не версионируется):

```powershell
New-Item -ItemType Directory -Force tools
curl.exe -L -o tools/plantuml.jar https://github.com/plantuml/plantuml/releases/download/v1.2026.8/plantuml.jar
# Java 11+: winget install --id Microsoft.OpenJDK.21 -e
# или переносная JDK в tools/jdk-*: https://aka.ms/download-jdk/microsoft-jdk-21-windows-x64.zip
# Graphviz (переносной zip с graphviz.org, распаковать в tools/graphviz):
curl.exe -L -o tools/graphviz.zip https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/16.1.0/windows_10_cmake_Release_Graphviz-16.1.0-win64.zip
uv run python scripts/render_diagrams.py
```

Версии: PlantUML **1.2026.8**, Graphviz **16.1.0**. Скрипт ищет Java в `tools/jdk-*`, `JAVA_HOME`,
затем в `PATH`; `dot` — в `tools/graphviz/bin`, затем в `PATH`, и передаёт его PlantUML через
`GRAPHVIZ_DOT`. PNG пишутся в `docs/architecture/png/` и копируются в
`report_assets/pr2/img/рис_2_1…рис_2_5.png`; для каждого печатаются размер при 200 dpi, кегль
и проверка ограничений книжной страницы (≤ 17 × 19 см, кегль ≥ 12 = 9 pt).
