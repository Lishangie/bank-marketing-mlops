"""Панель показателей (Streamlit). Читает только хранилище, поэтому после
обработки новой порции обновляется без изменения кода.

    uv run streamlit run app/dashboard.py          # http://localhost:8501
"""

import altair as alt
import pandas as pd
import streamlit as st

from bank_marketing.service import storage

st.set_page_config(page_title="Отклик клиентов банка", layout="wide")
st.markdown(
    "<style>.block-container {padding-top: 1.5rem; padding-bottom: 1rem;}</style>",
    unsafe_allow_html=True,
)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def num(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def delta_pp(cur, prev, key) -> str | None:
    if prev is None or pd.isna(prev.get(key)) or pd.isna(cur.get(key)):
        return None
    return f"{(cur[key] - prev[key]) * 100:+.1f}".replace(".", ",") + " п. п."


def delta_num(cur, prev, key, digits=2) -> str | None:
    if prev is None or pd.isna(prev.get(key)) or pd.isna(cur.get(key)):
        return None
    return f"{cur[key] - prev[key]:+.{digits}f}".replace(".", ",")


con = storage.connect()
hist = storage.history(con)
st.title("Прогноз отклика на телемаркетинг: панель показателей")
if hist.empty:
    st.info("Порции ещё не обрабатывались: uv run python -m bank_marketing.service.process_batch …")
    st.stop()

runs = storage.table(con, "runs")
alerts = storage.table(con, "alerts")
retrains = storage.table(con, "retrains")
batch = hist.index[-1]
cur = hist.loc[batch]
prev = hist.iloc[-2] if len(hist) > 1 else None
periods = con.execute(
    "SELECT MIN(period), MAX(period) FROM raw_data WHERE batch = ?", (batch,)
).fetchone()
run = runs[(runs["batch"] == batch) & (runs["status"] == "ok")].iloc[-1]
rejected = runs[runs["status"] == "rejected"]
st.caption(
    f"Последняя порция: **{batch}** ({periods[0]}…{periods[1]}), записей "
    f"{int(cur['rows_saved'])}; модель **{run['model_version']}**; обработана "
    f"{run['finished_at'].replace('T', ' ')} за {num(run['duration_s'], 1)} с. "
    f"Принято порций: {len(hist)}, отклонено: {len(rejected)}."
)

c = st.columns(5)
c[0].metric(
    "Ожидаемые вклады (10 % базы)",
    num(cur["expected_deposits"], 1),
    delta_num(cur, prev, "expected_deposits", 1),
    help=f"Сумма вероятностей у {int(cur['call_list_size'])} клиентов (10 %)",
)
c[1].metric(
    "Доля клиентов выше порога",
    pct(cur["above_threshold_share"]),
    delta_pp(cur, prev, "above_threshold_share"),
    delta_color="off",
)
c[2].metric(
    "Доля отклика (факт)",
    pct(cur.get("response_rate", float("nan"))),
    delta_pp(cur, prev, "response_rate"),
    delta_color="off",
)
c[3].metric("Lift@10 %", num(cur.get("lift_k", float("nan"))), delta_num(cur, prev, "lift_k"))
c[4].metric(
    "Записи вне области обучения",
    pct(cur["out_of_domain_share"], 0),
    delta_pp(cur, prev, "out_of_domain_share"),
    delta_color="inverse",
)

left, right = st.columns([3, 2])
with left:
    st.subheader("Предупреждения")
    now_alerts = alerts[alerts["batch"] == batch]
    if now_alerts.empty:
        st.success("Пороги не нарушены")
    hard = now_alerts.loc[now_alerts["retrain"] == 1, "message"].tolist()
    soft = now_alerts.loc[now_alerts["retrain"] == 0, "message"].tolist()
    if hard:
        st.error("**Сигнал к переобучению**\n" + "".join(f"\n- {m}" for m in hard))
    if soft:
        st.warning("**Предупреждения**\n" + "".join(f"\n- {m}" for m in soft))
with right:
    st.subheader("Переобучение")
    r = retrains[retrains["batch"] == batch]
    if r.empty:
        st.info("После этой порции не запускалось")
    else:
        r = r.iloc[-1]
        verdict = (
            f"замена: {r['current_version']} → **{r['new_version']}**"
            if isinstance(r["new_version"], str)
            else f"действующая {r['current_version']} остаётся"
        )
        st.markdown(
            f"Претендент — {r['candidate']}, обучен на {int(r['train_rows'])} записях; "
            f"сравнение на последних {int(r['holdout_rows'])} записях порции.\n\n"
            f"| | Действующая | Претендент |\n|---|---|---|\n"
            f"| PR-AUC | {num(r['cur_pr_auc'], 3)} | {num(r['cand_pr_auc'], 3)} |\n"
            f"| Lift@10 % | {num(r['cur_lift_k'])} | {num(r['cand_lift_k'])} |\n"
            f"| Прогноз − факт | {delta_pp({'x': r['cur_calib']}, {'x': 0}, 'x')} | "
            f"{delta_pp({'x': r['cand_calib']}, {'x': 0}, 'x')} |\n\n"
            f"Решение: {verdict}"
        )

# динамика по порциям
labels = {b: f"{i + 1}. {b}" for i, b in enumerate(hist.index)}
h = hist.rename(index=labels).reset_index(names="Порция")


def line(cols: dict, title: str, fmt: str):
    data = h.melt("Порция", list(cols), var_name="Показатель", value_name="Значение")
    data["Показатель"] = data["Показатель"].map(cols)
    scale = alt.Scale(zero=False)
    axis = alt.Axis(format=fmt)
    return (
        alt.Chart(data, title=title)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Порция:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("Значение:Q", axis=axis, title=None, scale=scale),
            color=alt.Color(
                "Показатель:N",
                legend=alt.Legend(orient="bottom", title=None, columns=3, labelLimit=150),
            ),
        )
        .properties(height=260)
    )


g = st.columns(3)
g[0].altair_chart(
    line(
        {"pred_mean": "прогноз модели", "response_rate": "факт"},
        "Доля отклика: прогноз и факт",
        ".0%",
    ),
    use_container_width=True,
)
g[1].altair_chart(
    line(
        {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "lift_k": "Lift@10 %"},
        "Качество модели по меткам",
        ".1f",
    ),
    use_container_width=True,
)
psi = pd.DataFrame(
    {
        "Признак": [c[4:] for c in hist.columns if c.startswith("psi_")],
        "PSI": [cur[c] for c in hist.columns if c.startswith("psi_")],
    }
)
psi["Уровень"] = pd.cut(
    psi["PSI"], [-1, 0.1, 0.25, 1e9], labels=["норма", "предупреждение", "переобучение"]
).astype(str)
psi["Подпись"] = psi["PSI"].map(lambda v: f"{v:.2f}".replace(".", ","))
bars = alt.Chart(
    psi,
    title=alt.Title(
        f"PSI признаков: {batch}", subtitle="красный — выше 0,25 (переобучение), жёлтый — 0,1–0,25"
    ),
).encode(
    y=alt.Y("Признак:N", sort="-x", title=None),
    x=alt.X(
        "PSI:Q",
        scale=alt.Scale(type="symlog"),
        title=None,
        axis=alt.Axis(values=[0, 0.25, 1, 5, 20], format="~g"),
    ),
)
g[2].altair_chart(
    (
        bars.mark_bar().encode(
            color=alt.Color(
                "Уровень:N",
                scale=alt.Scale(
                    domain=["норма", "предупреждение", "переобучение"],
                    range=["#7fb3d5", "#f5b041", "#e74c3c"],
                ),
                legend=None,
            )
        )
        + bars.mark_text(align="left", dx=3).encode(text="Подпись:N")
    ).properties(height=260),
    use_container_width=True,
)

st.subheader(f"Список обзвона: первые 10 из {int(cur['call_list_size'])} клиентов")
top = pd.read_sql(
    "SELECT p.rank AS 'Ранг', p.row_id AS 'Клиент', ROUND(p.probability, 3) AS 'Вероятность', "
    "r.age AS 'Возраст', r.job AS 'Род занятий', r.contact AS 'Связь', "
    "r.poutcome AS 'Прошлая кампания', p.model_version AS 'Модель' "
    "FROM predictions p JOIN raw_data r USING (batch, row_id) "
    "WHERE p.batch = ? AND p.in_call_list = 1 ORDER BY p.rank LIMIT 10",
    con,
    params=(batch,),
)
top["Вероятность"] = top["Вероятность"].map(lambda v: f"{v:.3f}".replace(".", ","))
st.dataframe(top, hide_index=True, use_container_width=True)
con.close()
