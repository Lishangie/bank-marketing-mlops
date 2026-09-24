# Bank Marketing MLOps

Прогноз отклика клиентов банка на телемаркетинговую кампанию (оформит ли клиент срочный депозит, `y`) по данным UCI Bank Marketing.

## Воспроизведение

```powershell
uv sync                           # окружение Python 3.12 из uv.lock
uv run dvc pull                   # данные из DVC-хранилища (../dvc-storage)
uv run python scripts/eda.py      # EDA: рисунки и report_assets/eda_out.txt
```

Скачать данные заново: `uv run python scripts/download_data.py`.

Docker: `docker build -t bank-marketing .`, затем
`docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/report_assets:/app/report_assets bank-marketing`.
