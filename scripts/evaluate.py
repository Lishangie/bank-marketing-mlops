"""Оценка итоговой модели ПР4 на тестовой выборке и на порциях новых данных.

Протокол (метрики, цена ошибок, пороги приемки и мониторинга) задан в
params.yaml → evaluate до запуска этого скрипта. Порядок:

1. Метрики model_v1 и моделей сравнения на тесте, бутстреп-интервалы,
   значимость разностей, сравнение с оценкой по кросс-валидации.
2. Порог по цене ошибок — по прогнозам вне блока на обучающей выборке
   (тест для выбора порога не используется), проверка на тесте.
3. Срезы теста, перестановочная важность, частичная зависимость, локальные
   объяснения, устойчивость к шуму и пропускам.
4. Порции 1–4: метрики, калибровка, PSI признаков модели, правила мониторинга.
5. Проверка порогов приемки, паспорт модели, reports/evaluation.json.

Результаты: report_assets/pr4/evaluate_out.txt, report_assets/pr4/img/рис_*.png
(копия в reports/figures/pr4), reports/evaluation.json.
"""

import importlib.util
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

from bank_marketing.config import (
    ASSETS_DIR,
    FIGURES_DIR,
    INITIAL_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT,
)
from bank_marketing.data import batch_files
from bank_marketing.evaluation import (
    at_threshold,
    bootstrap,
    feature_psi,
    load_eval_params,
    local_contributions,
    loss_curve,
    metrics,
    oof_scores,
    partial_dependence,
    permutation_importance,
    perturb_missing,
    perturb_noise,
    slice_table,
    typical_values,
)
from bank_marketing.experiments import load_params, split_train_test, time_cv
from bank_marketing.features import load_reference, make_pipeline, prepare
from bank_marketing.models import CatBoostModel
from bank_marketing.plotting import (
    C_ACCENT,
    C_MEAN,
    C_NO,
    C_YES,
    CM,
    LABEL_BOX,
    apply_style,
    comma_axes,
    num,
    pct_formatter,
    save,
)
from bank_marketing.report import emit

warnings.filterwarnings("ignore")
apply_style(font_size=8.5, tick_size=8)

PR4_DIR = ASSETS_DIR / "pr4"
IMG_DIRS = [PR4_DIR / "img", FIGURES_DIR / "pr4"]
OUT_JSON = REPORTS_DIR / "evaluation.json"

_spec = importlib.util.spec_from_file_location("pr3", ROOT / "scripts" / "train_experiments.py")
pr3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr3)
rule_score, lr = pr3.rule_score, pr3.lr  # правило и логистическая регрессия — как в ПР3

NAMES = {
    "v1": "CatBoost после подбора (model_v1)",
    "cb": "CatBoost по умолчанию",
    "lr": "Логистическая регрессия после подбора",
    "rule": "Правило из разведочного анализа",
    "prior": "Наиболее частый класс",
}
PART_TITLE = {
    "test": "Тест",
    "batch_01": "Порция 1",
    "batch_02": "Порция 2",
    "batch_03": "Порция 3",
    "batch_04": "Порция 4",
}
PSI_RU = {
    "macro_pc1": "macro_pc1",
    "cons.conf.idx": "Индекс доверия",
    "campaign_capped": "Число контактов",
    "age": "Возраст",
    "contact": "Тип связи",
}


def f3(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else num(v, 3)


def ci(t) -> str:
    return f"{f3(t[0])} [{f3(t[1])}; {f3(t[2])}]"


# ================================================================ рисунки
def fig_curves(y, scores: dict, thr, loss_te, loss_oof, t_star, n_te, n_oof) -> None:
    """Рисунок 2.1: ROC, «точность – полнота», потери на клиента от порога."""
    fig, axs = plt.subplots(1, 3, figsize=(16.5 * CM, 6.0 * CM))
    style = {
        "v1": (C_MEAN, "-", "model_v1"),
        "lr": (C_ACCENT, "--", "Лог. регрессия"),
        "rule": (C_NO, ":", "Правило"),
    }
    for key, (c, ls, lab) in style.items():
        fpr, tpr, _ = roc_curve(y, scores[key])
        axs[0].plot(fpr, tpr, color=c, ls=ls, lw=1.4, label=lab)
        p, r, _ = precision_recall_curve(y, scores[key])
        axs[1].plot(r, p, color=c, ls=ls, lw=1.4, label=lab)
    axs[0].plot([0, 1], [0, 1], color="0.6", lw=0.8)
    axs[0].set_xlabel("Доля ложных срабатываний")
    axs[0].set_ylabel("Полнота")
    axs[0].legend(loc="upper left", frameon=False, fontsize=7)
    base = float(np.mean(y))
    axs[1].axhline(base, color="0.6", lw=0.8)
    axs[1].set_ylim(0, 0.35)
    axs[1].set_xlabel("Полнота")
    axs[1].set_ylabel("Точность")
    axs[2].plot(thr, loss_te / n_te, color=C_MEAN, lw=1.4, label="тест")
    axs[2].plot(thr, loss_oof / n_oof, color=C_ACCENT, lw=1.2, ls="--", label="обучающая")
    axs[2].axvline(t_star, color="0.4", lw=0.8, ls=":")
    axs[2].set_xlabel("Порог")
    axs[2].set_ylabel("Потери на клиента, руб.")
    axs[2].legend(
        loc="center right", bbox_to_anchor=(1.0, 0.56), frameon=False, fontsize=7, handlelength=1.6
    )
    lo, hi = axs[2].get_ylim()
    axs[2].text(
        t_star + 0.003,
        hi - (hi - lo) * 0.06,
        f"t* = {num(t_star, 3)}",
        fontsize=7,
        color="0.25",
        va="top",
    )
    for ax in axs:
        comma_axes(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, "рис_2_1.png", IMG_DIRS)


def fig_confusion(cms: list[tuple[str, dict]]) -> None:
    """Рисунок 2.2: матрицы ошибок при двух порогах."""
    cmap = LinearSegmentedColormap.from_list("b", ["#ffffff", C_ACCENT])
    fig, axs = plt.subplots(1, len(cms), figsize=(14 * CM, 5.6 * CM))
    for ax, (title, r) in zip(axs, cms, strict=True):
        m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(np.log1p(m), cmap=cmap, vmin=0, vmax=np.log1p(m.sum()))
        for i in range(2):
            for j in range(2):
                v = m[i, j]
                ax.text(
                    j,
                    i,
                    num(v, 0),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if np.log1p(v) > 0.6 * np.log1p(m.sum()) else "black",
                )
        ax.set_xticks([0, 1], ["не звонить", "звонить"])
        ax.set_yticks([0, 1], ["отказ", "отклик"])
        ax.set_xlabel("Решение модели")
        ax.set_title(title, fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("Факт")
    fig.tight_layout(w_pad=2)
    save(fig, "рис_2_2.png", IMG_DIRS)


def fig_importance(imp: pd.DataFrame) -> None:
    """Рисунок 3.1: перестановочная важность групп признаков на тесте."""
    d = imp.sort_values("d_roc_auc")
    fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 7.0 * CM), sharey=True)
    for ax, col, title in [
        (axs[0], "d_roc_auc", "Снижение ROC-AUC"),
        (axs[1], "d_pr_auc", "Снижение PR-AUC"),
    ]:
        colors = [C_MEAN if v > 0 else C_NO for v in d[col]]
        ax.barh(
            d["group"],
            d[col],
            xerr=d[col + "_std"],
            color=colors,
            error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "0.3"},
        )
        ax.axvline(0, color="0.3", lw=0.7)
        ax.set_xlabel(title)
        comma_axes(ax, y=False)
    fig.tight_layout(w_pad=1)
    save(fig, "рис_3_1.png", IMG_DIRS)


def fig_pdp(panels: list[dict]) -> None:
    """Рисунок 3.2: частичная зависимость вероятности отклика."""
    fig, axs = plt.subplots(2, 2, figsize=(16 * CM, 10.5 * CM))
    for ax, p in zip(axs.ravel(), panels, strict=True):
        if p["kind"] == "num":
            ax.plot(p["grid"], p["pd"], color=C_MEAN, lw=1.5)
            if "train_range" in p:
                lo, hi = p["train_range"]
                ax.axvspan(lo, hi, color=C_NO, alpha=0.15, lw=0)
                ax.text(
                    (lo + hi) / 2,
                    0.02,
                    "обучение",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    fontsize=7,
                    color="0.3",
                )
            for x, lab, ha in p.get("marks", []):
                ax.axvline(x, color=C_YES, lw=0.8, ls="--")
                ax.text(
                    x + (0.15 if ha == "left" else -0.15),
                    0.97,
                    lab,
                    transform=ax.get_xaxis_transform(),
                    ha=ha,
                    va="top",
                    fontsize=7,
                    color=C_YES,
                    bbox=LABEL_BOX,
                )
        else:
            ax.bar(
                range(len(p["grid"])), p["pd"], color=[C_ACCENT, C_NO][: len(p["grid"])], width=0.55
            )
            ax.set_xticks(range(len(p["grid"])), p["ticks"])
        ax.set_xlabel(p["label"])
        ax.set_ylabel("Вероятность отклика")
        ax.yaxis.set_major_formatter(pct_formatter(1))
        if p["kind"] == "num":
            comma_axes(ax, y=False)
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, "рис_3_2.png", IMG_DIRS)


def fig_batches(rows: list[dict], mon: dict) -> None:
    """Рисунок 4.1: качество, калибровка и PSI по тесту и порциям."""
    labels = [PART_TITLE[r["part"]] for r in rows]
    x = np.arange(len(rows))
    fig = plt.figure(figsize=(16 * CM, 11.5 * CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    a1.plot(x, [r["roc_auc"] for r in rows], "o-", color=C_MEAN, lw=1.3, label="ROC-AUC")
    a1.plot(x, [r["lift_k"] for r in rows], "s--", color=C_ACCENT, lw=1.3, label="Lift@10 %")
    a1.plot(
        x,
        [r["pr_auc"] / r["base"] for r in rows],
        "^:",
        color=C_YES,
        lw=1.3,
        label="PR-AUC / доля отклика",
    )
    a1.axhline(mon["lift_k"], color="0.5", lw=0.8, ls="-")
    a1.axhline(0.5, color="0.75", lw=0.6)
    a1.set_xticks(x, labels, rotation=30, ha="right")
    a1.set_ylabel("Значение метрики")
    a1.legend(frameon=False, fontsize=6.8, loc="upper left")
    a1.set_ylim(0.3, 2.4)
    w = 0.38
    a2.bar(x - w / 2, [r["base"] for r in rows], w, color=C_YES, label="фактическая")
    a2.bar(x + w / 2, [r["mean_pred"] for r in rows], w, color=C_ACCENT, label="прогнозная")
    a2.set_xticks(x, labels, rotation=30, ha="right")
    a2.set_ylabel("Доля отклика")
    a2.yaxis.set_major_formatter(pct_formatter())
    a2.legend(frameon=False, fontsize=7)
    feats = list(rows[0]["psi"])
    wb = 0.8 / len(feats)
    palette = [C_MEAN, C_ACCENT, C_YES, C_NO, "#5f9e6e"]
    for i, f in enumerate(feats):
        vals = [max(r["psi"][f], 1e-3) for r in rows]
        a3.bar(x + (i - (len(feats) - 1) / 2) * wb, vals, wb, color=palette[i], label=PSI_RU[f])
    a3.set_yscale("log")
    a3.axhline(mon["psi_warn"], color="0.5", lw=0.8, ls="--")
    a3.axhline(mon["psi_alert"], color=C_MEAN, lw=0.9, ls="--")
    a3.set_xticks(x, labels)
    a3.set_ylabel("PSI (логарифмическая шкала)")
    a3.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    a3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num(v, 3).rstrip("0").rstrip(",")))
    comma_axes(a1, x=False)
    fig.tight_layout(h_pad=1.0, w_pad=1.5)
    save(fig, "рис_4_1.png", IMG_DIRS)


# ================================================================ основной расчёт
def main() -> None:
    tp_, ep = load_params(), load_eval_params()
    k, seed, c_fp, c_fn = tp_["top_k"], ep["seed"], ep["cost_fp"], ep["cost_fn"]
    ref = load_reference()
    X, y = prepare(INITIAL_FILE, ref)
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, tp_["test_from"])
    X_tr, y_tr = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
    X_te, y_te = X_te.reset_index(drop=True), y_te.reset_index(drop=True)
    info = json.loads((MODELS_DIR / "model_v1.json").read_text(encoding="utf-8"))
    hp = info["hyperparameters"]
    v1 = joblib.load(MODELS_DIR / "model_v1.joblib")
    out = ["Оценка итоговой модели ПР4", ""]
    out.append(
        f"Протокол (params.yaml → evaluate): цена ложного срабатывания {c_fp} руб., "
        f"пропуска {c_fn} руб.; бутстреп {ep['bootstrap']}, seed {seed}"
    )
    out.append(f"Пороги приемки: {ep['accept']}")
    out.append(f"Пороги мониторинга: {ep['monitor']}")
    out.append(
        f"Обучающая: {len(X_tr)} ({y_tr.mean():.2%}); тест: {len(X_te)} "
        f"({y_te.mean():.2%}, откликов {int(y_te.sum())}); периоды теста "
        + ", ".join(f"{p} {n}" for p, n in X_te["period"].value_counts().sort_index().items())
    )

    # ---------- 1. Метрики на тесте
    cb = make_pipeline(CatBoostModel(random_seed=tp_["seed"]), native_cat=True).fit(X_tr, y_tr)
    lrm = make_pipeline(lr(1, l1=False, cw=None)).fit(X_tr, y_tr)
    scores = {
        "v1": v1.predict_proba(X_te)[:, 1],
        "cb": cb.predict_proba(X_te)[:, 1],
        "lr": lrm.predict_proba(X_te)[:, 1],
        "rule": rule_score(X_te).astype(float),
        "prior": np.full(len(X_te), float(y_tr.mean())),
    }
    pairs = [("v1", "cb"), ("v1", "lr"), ("v1", "rule"), ("lr", "rule")]
    bs = bootstrap(y_te, scores, ep["bootstrap"], seed, k, pairs)
    out += ["", "1. Тестовая выборка: оценка [95 %-ный бутстреп-интервал]"]
    for key, name in NAMES.items():
        b = bs[key]
        out.append(
            f"  {name:<40} PR-AUC {ci(b['pr_auc'])}; ROC-AUC {ci(b['roc_auc'])}; "
            f"Lift@10 {ci(b['lift_k'])}; средняя оценка {scores[key].mean():.4f}"
        )
    out.append("  Разности (a − b), PR-AUC / ROC-AUC / Lift@10:")
    for a, b_ in pairs:
        d = bs[f"{a} − {b_}"]
        out.append(f"    {a} − {b_}: {ci(d['pr_auc'])}; {ci(d['roc_auc'])}; {ci(d['lift_k'])}")
    cv = info["cv"]
    out.append(
        f"  КВ model_v1 (ПР3): PR-AUC {cv['pr_auc']:.4f} ± {cv['pr_auc_std']:.4f} "
        f"(базовое {cv['baseline_pr_auc']:.4f}, "
        f"отношение {cv['pr_auc'] / cv['baseline_pr_auc']:.2f}); "
        f"ROC-AUC {cv['roc_auc']:.4f}; Lift@10 {cv['lift_at_10']:.3f}"
    )
    m_v1 = metrics(y_te, scores["v1"], k)
    out.append(
        f"  тест model_v1: PR-AUC / доля отклика = {m_v1['pr_auc'] / m_v1['base']:.2f}; "
        f"КВ-оценка внутри интервала: "
        f"{'да' if bs['v1']['pr_auc'][1] <= cv['pr_auc'] <= bs['v1']['pr_auc'][2] else 'нет'}"
    )
    for per in sorted(X_te["period"].unique()):
        msk = (X_te["period"] == per).to_numpy()
        parts = []
        for key in ["v1", "lr", "rule"]:
            r = metrics(y_te[msk], scores[key][msk], k)
            parts.append(f"{key} PR {r['pr_auc']:.3f} ROC {r['roc_auc']:.3f}")
        out.append(
            f"  {per}: {int(msk.sum())} записей, отклик {y_te[msk].mean():.2%}; " + "; ".join(parts)
        )

    # ---------- 2. Порог по цене ошибок
    def factory():
        return make_pipeline(CatBoostModel(random_seed=tp_["seed"], **hp), native_cat=True)

    oof_idx, oof, folds = oof_scores(factory, X_tr, y_tr, time_cv(tp_))
    y_oof = y_tr.to_numpy()[oof_idx]
    thr = np.round(np.arange(0, 0.2001, 0.001), 3)
    loss_oof = loss_curve(y_oof, oof, thr, c_fp, c_fn)
    t_star = float(thr[int(np.argmin(loss_oof))])
    loss_te = loss_curve(y_te, scores["v1"], thr, c_fp, c_fn)
    t_te = float(thr[int(np.argmin(loss_te))])
    bayes = c_fp / (c_fp + c_fn)
    out += ["", "2. Порог по цене ошибок"]
    out.append(
        f"  прогнозы вне блока: {len(oof)} записей, отклик {y_oof.mean():.2%}, "
        f"средняя оценка {oof.mean():.4f}; теоретический порог c_FP/(c_FP + c_FN) = {bayes:.4f}"
    )
    out.append(
        f"  t* по обучающей (минимум потерь вне блока) = {t_star:.3f}; потери вне блока: "
        f"при t* {loss_oof.min() / len(oof):.1f} руб./клиент, «звонить всем» "
        f"{loss_oof[0] / len(oof):.1f}, «никому» {int(y_oof.sum()) * c_fn / len(oof):.1f}"
    )
    thr_rows = {}
    for title, t in [("0,5", 0.5), (f"t* = {t_star:.3f}", t_star), ("всем (t = 0)", 0.0)]:
        r = at_threshold(y_te, scores["v1"], t)
        r["loss"] = r["fn"] * c_fn + r["fp"] * c_fp
        thr_rows[title] = r
        out.append(
            f"  тест, порог {title}: TP {r['tp']} FP {r['fp']} FN {r['fn']} TN {r['tn']}; "
            f"обзвон {r['called']} ({r['called'] / len(y_te):.1%}); точность "
            f"{r['precision']:.3f}, полнота {r['recall']:.3f}, F1 {r['f1']:.3f}, "
            f"доля верных {r['accuracy']:.3f}; потери {r['loss'] / 1000:.1f} тыс. руб. "
            f"({r['loss'] / len(y_te):.1f} руб./клиент)"
        )
    r_lr = at_threshold(y_te, scores["lr"], t_star)
    out.append(
        f"  справочно: минимум потерь на тесте при t = {t_te:.3f} "
        f"({loss_te.min() / 1000:.1f} тыс. руб.); лог. регрессия при t*: полнота "
        f"{r_lr['recall']:.3f}, точность {r_lr['precision']:.3f}, потери "
        f"{(r_lr['fn'] * c_fn + r_lr['fp'] * c_fp) / 1000:.1f} тыс. руб."
    )
    fig_curves(y_te, scores, thr, loss_te, loss_oof, t_star, len(y_te), len(oof))
    fig_confusion(
        [
            ("Порог 0,5", thr_rows["0,5"]),
            (f"Порог t* = {num(t_star, 3)}", thr_rows[f"t* = {t_star:.3f}"]),
        ]
    )

    # ---------- 3. Срезы
    camp = pd.cut(X_te["campaign"], [0, 1, 3, np.inf], labels=["1", "2–3", "4 и более"]).astype(str)
    slices = {"Тип связи": X_te["contact"], "Месяц": X_te["period"], "Число контактов": camp}
    st = slice_table(X_te, y_te, scores["v1"], t_star, slices, k)
    out += ["", f"3. Срезы тестовой выборки (порог t* = {t_star:.3f})"]
    for _, r in st.iterrows():
        out.append(
            f"  {r['slice']:<16} {str(r['group']):<10} n {r['n']:>5}  откликов {r['pos']:>4}  "
            f"доля {r['base']:.1%}  ср. оценка {r['mean_pred']:.4f}  PR-AUC {f3(r['pr_auc'])}  "
            f"ROC-AUC {f3(r['roc_auc'])}  полнота {f3(r['recall_t'])}  в обзвоне "
            f"{r['called_share']:.1%}"
        )
    tr_ct = pd.DataFrame({"c": X_tr["contact"], "y": y_tr}).groupby("c")["y"].agg(["size", "mean"])
    out.append(
        "  справочно, обучающая по типу связи: "
        + "; ".join(f"{i} {int(r['size'])} ({r['mean']:.2%})" for i, r in tr_ct.iterrows())
    )

    # ---------- 4. Интерпретация
    imp = permutation_importance(v1, X_te, y_te, n_repeats=30, seed=seed)
    imp_tr = permutation_importance(v1, X_tr, y_tr, n_repeats=10, seed=seed)
    out += ["", "4. Перестановочная важность групп (тест, 30 повторов): снижение ROC-AUC / PR-AUC"]
    for _, r in imp.iterrows():
        rt = imp_tr.set_index("group").loc[r["group"]]
        out.append(
            f"  {r['group']:<24} ΔROC {r['d_roc_auc']:+.4f} ± {r['d_roc_auc_std']:.4f}  "
            f"ΔPR {r['d_pr_auc']:+.4f} ± {r['d_pr_auc_std']:.4f}   | на обучающей ΔROC "
            f"{rt['d_roc_auc']:+.4f}"
        )
    fig_importance(imp)

    fb = v1.named_steps["features"]
    tail = v1[1:]
    Z_te, Z_tr = fb.transform(X_te), fb.transform(X_tr)
    lo_m, hi_m = float(Z_tr["macro_pc1"].min()), float(Z_tr["macro_pc1"].max())
    macro_te = Z_te.groupby("period")["macro_pc1"].first()
    g_macro = np.linspace(min(macro_te.min(), lo_m) - 0.3, max(macro_te.max(), hi_m) + 0.3, 60)
    g_camp = np.arange(1, int(fb.cap_) + 1)
    g_age = np.arange(20, 71)
    panels = [
        {
            "kind": "num",
            "label": "macro_pc1",
            "grid": g_macro,
            "pd": partial_dependence(tail, Z_te, "macro_pc1", g_macro),
            "train_range": (lo_m, hi_m),
            "marks": [
                (v, *{"2008-08": ("август", "right"), "2008-10": ("октябрь", "left")}[p])
                for p, v in macro_te.items()
            ],
        },
        {
            "kind": "num",
            "label": "Число контактов в кампании",
            "grid": g_camp,
            "pd": partial_dependence(tail, Z_te, "campaign_capped", g_camp),
        },
        {
            "kind": "num",
            "label": "Возраст, лет",
            "grid": g_age,
            "pd": partial_dependence(tail, Z_te.assign(age_group="25-59"), "age", g_age),
        },
        {
            "kind": "cat",
            "label": "Тип связи",
            "grid": ["cellular", "telephone"],
            "ticks": ["мобильная", "стационарная"],
            "pd": partial_dependence(tail, Z_te, "contact", ["cellular", "telephone"]),
        },
    ]
    out += ["", "  Частичная зависимость (тест, средняя вероятность):"]
    for p in panels:
        vals = p["pd"]
        if p["kind"] == "cat":
            out.append(
                f"    {p['label']}: "
                + "; ".join(f"{g} {v:.4f}" for g, v in zip(p["grid"], vals, strict=True))
            )
        else:
            gi = np.asarray(p["grid"])
            pick = np.linspace(0, len(gi) - 1, 7).astype(int)
            out.append(
                f"    {p['label']}: "
                + "; ".join(f"{gi[i]:.2f} → {vals[i]:.4f}" for i in pick)
                + f"; размах {vals.max() - vals.min():.4f}"
            )
    out.append(
        f"    macro_pc1: обучение [{lo_m:.3f}; {hi_m:.3f}], тест "
        + ", ".join(f"{p} {v:.3f}" for p, v in macro_te.items())
    )
    fig_pdp(panels)

    typical = typical_values(X_tr)
    s_v1 = scores["v1"]
    yv = y_te.to_numpy()
    pred = s_v1 >= t_star
    cases = {
        "Верное срабатывание": int(np.argmax(np.where(pred & (yv == 1), s_v1, -1))),
        "Ложное срабатывание": int(np.argmax(np.where(pred & (yv == 0), s_v1, -1))),
        # пропуск из среза с худшим качеством: стационарная связь, наименьшая оценка
        "Пропуск": int(
            np.argmin(
                np.where(~pred & (yv == 1) & (X_te["contact"] == "telephone").to_numpy(), s_v1, 2)
            )
        ),
    }
    out += ["", "  Локальные объяснения: вклад = p(x) − p(x с типичным значением группы)"]
    out.append(
        "    типичные (обучающая): "
        + ", ".join(
            f"{c} {typical[c]}"
            for c in [
                "age",
                "job",
                "contact",
                "month",
                "campaign",
                "poutcome",
                "euribor3m",
                "cons.conf.idx",
            ]
        )
    )
    local_rows = []
    for title, i in cases.items():
        row = X_te.iloc[[i]]
        contrib = local_contributions(v1, row, typical).sort_values(key=np.abs, ascending=False)
        r0 = row.iloc[0]
        desc = (
            f"{int(r0['age'])} лет, {r0['job']}, {r0['marital']}, {r0['education']}, "
            f"{r0['contact']}, {r0['period']} ({r0['day_of_week']}), "
            f"контактов {int(r0['campaign'])}, "
            f"poutcome {r0['poutcome']}, housing {r0['housing']}, loan {r0['loan']}"
        )
        top = contrib.head(4)
        local_rows.append(
            {
                "case": title,
                "p": float(s_v1[i]),
                "y": int(yv[i]),
                "desc": desc,
                "top": {g: float(v) for g, v in top.items()},
            }
        )
        out.append(
            f"    {title}: строка {int(r0['row_id'])}; p = {s_v1[i]:.4f}; факт {int(yv[i])}; {desc}"
        )
        out.append("      вклады: " + "; ".join(f"{g} {v:+.4f}" for g, v in top.items()))
        alt = "telephone" if r0["contact"] == "cellular" else "cellular"
        p_alt = v1.predict_proba(row.assign(contact=alt))[:, 1][0]
        out.append(f"      если бы тип связи был {alt}: p = {p_alt:.4f}")
        local_rows[-1]["p_alt_contact"] = float(p_alt)

    # ---------- 5. Устойчивость
    base_m = metrics(y_te, s_v1, k)
    y_oof_m = metrics(y_oof, oof, k)

    def scen(rng):
        return {
            "шум: возраст ×(1 + N(0; 0,1)), контакты ± 1": lambda D: perturb_noise(D, rng),
            "20 % пропусков в возрасте и числе контактов (медиана)": lambda D: perturb_missing(
                D, rng, ["age", "campaign"], 0.2, typical
            ),
            "20 % пропусков в типе связи (мода)": lambda D: perturb_missing(
                D, rng, ["contact"], 0.2, typical
            ),
        }

    out += ["", "5. Устойчивость (20 повторов, среднее ± ст. откл.)"]
    out.append(
        f"  без возмущений: тест PR-AUC {base_m['pr_auc']:.3f}, ROC-AUC {base_m['roc_auc']:.3f}, "
        f"Lift@10 {base_m['lift_k']:.3f}; обучающая вне блока PR-AUC {y_oof_m['pr_auc']:.3f}, "
        f"ROC-AUC {y_oof_m['roc_auc']:.3f}, Lift@10 {y_oof_m['lift_k']:.3f} "
        f"(доля отклика {y_oof.mean():.3f})"
    )
    rng = np.random.default_rng(seed)
    robust = []
    for title in scen(rng):
        te_v, oof_v = [], []
        for _ in range(20):
            fn = scen(rng)[title]
            te_v.append(metrics(y_te, v1.predict_proba(fn(X_te))[:, 1], k))
            s_o = np.concatenate([m.predict_proba(fn(X_tr.iloc[va]))[:, 1] for m, va in folds])
            oof_v.append(metrics(y_oof, s_o, k))
        row = {"scenario": title}
        for tag, vals in [("te", te_v), ("oof", oof_v)]:
            for mname in ["pr_auc", "roc_auc", "lift_k"]:
                arr = np.array([v[mname] for v in vals])
                row[f"{tag}_{mname}"], row[f"{tag}_{mname}_std"] = arr.mean(), arr.std()
        robust.append(row)
        out.append(f"  {title}:")
        out.append(
            f"    тест: PR-AUC {row['te_pr_auc']:.3f} ± {row['te_pr_auc_std']:.3f}; ROC-AUC "
            f"{row['te_roc_auc']:.3f} ± {row['te_roc_auc_std']:.3f}; Lift@10 {row['te_lift_k']:.3f}"
        )
        out.append(
            f"    вне блока: PR-AUC {row['oof_pr_auc']:.3f} ± {row['oof_pr_auc_std']:.3f}; ROC-AUC "
            f"{row['oof_roc_auc']:.3f} ± {row['oof_roc_auc_std']:.3f}; "
            f"Lift@10 {row['oof_lift_k']:.3f}"
        )

    # ---------- 6. Порции
    mon = ep["monitor"]
    feats = ep["psi_features"]
    parts = {"test": (X_te, y_te)}
    for path in batch_files():
        parts[path.stem] = prepare(path, ref)
    rows = []
    out += ["", "6. Тест и порции: качество, калибровка, PSI признаков модели (бины по обучающей)"]
    for name, (Xp, yp) in parts.items():
        s = v1.predict_proba(Xp)[:, 1]
        r = metrics(yp, s, k)
        r.update(
            part=name,
            n=len(yp),
            mean_pred=float(s.mean()),
            period=f"{Xp['period'].min()}…{Xp['period'].max()}",
            psi=feature_psi(Z_tr, fb.transform(Xp), feats),
            thr=at_threshold(yp, s, t_star),
        )
        lrs = lrm.predict_proba(Xp)[:, 1]
        r["lr"] = metrics(yp, lrs, k)
        flags = []
        if r["lift_k"] < mon["lift_k"]:
            flags.append("Lift@10 ниже порога")
        if abs(r["mean_pred"] - r["base"]) > mon["calib_pp"]:
            flags.append("калибровка")
        alert = [f for f, v in r["psi"].items() if v > mon["psi_alert"]]
        warn = [f for f, v in r["psi"].items() if mon["psi_warn"] < v <= mon["psi_alert"]]
        if alert:
            flags.append("PSI > 0,25: " + ", ".join(alert))
        if warn:
            flags.append("PSI 0,1–0,25: " + ", ".join(warn))
        r["flags"] = flags
        hard = alert or any(f.startswith(("Lift", "калибр")) for f in flags)
        r["decision"] = "переобучение" if hard else ("предупреждение" if warn else "норма")
        rows.append(r)
        out.append(
            f"  {PART_TITLE[name]:<9} {r['period']:<17} n {r['n']:>5}  отклик {r['base']:.2%}  "
            f"прогноз {r['mean_pred']:.2%}  PR-AUC {r['pr_auc']:.3f} "
            f"({r['pr_auc'] / r['base']:.2f}×)  "
            f"ROC-AUC {r['roc_auc']:.3f}  Lift@10 {r['lift_k']:.2f}  полнота@t* "
            f"{r['thr']['recall']:.3f}  в обзвоне {r['thr']['called'] / r['n']:.1%}"
        )
        out.append(
            "            PSI: "
            + "; ".join(f"{f} {v:.3f}" for f, v in r["psi"].items())
            + f"  | лог. регрессия ROC-AUC {r['lr']['roc_auc']:.3f}, "
            f"Lift@10 {r['lr']['lift_k']:.2f}"
        )
        out.append(f"            сигналы: {', '.join(flags) or 'нет'} → {r['decision']}")
    fig_batches(rows, mon)

    # ---------- 7. Пороги приемки и решение
    acc = ep["accept"]
    b = bs["v1"]
    d_rule = bs["v1 − rule"]["pr_auc"]
    checks = [
        (
            "PR-AUC ≥ 1,2 × доля отклика",
            f"{b['pr_auc'][0]:.4f} ≥ {acc['pr_auc_lift'] * m_v1['base']:.4f}",
            b["pr_auc"][0] >= acc["pr_auc_lift"] * m_v1["base"],
        ),
        (
            "Превосходство над правилом по PR-AUC ≥ 0,005, ДИ без нуля",
            f"{d_rule[0]:+.4f} [{d_rule[1]:+.4f}; {d_rule[2]:+.4f}]",
            d_rule[0] >= acc["pr_auc_gain_rule"] and d_rule[1] > 0,
        ),
        ("Lift@10 % ≥ 1,2", f"{b['lift_k'][0]:.3f}", b["lift_k"][0] >= acc["lift_k"]),
        ("ROC-AUC ≥ 0,55", f"{b['roc_auc'][0]:.4f}", b["roc_auc"][0] >= acc["roc_auc"]),
    ]
    out += ["", "7. Пороги приемки (model_v1 на тесте)"]
    for title, val, ok in checks:
        out.append(f"  {'[ok]  ' if ok else '[нет] '}{title}: {val}")
    for key in ["lr", "cb"]:
        bb = bs[key]
        dr = bs["lr − rule"]["pr_auc"] if key == "lr" else None
        ok = [
            bb["pr_auc"][0] >= acc["pr_auc_lift"] * m_v1["base"],
            bb["lift_k"][0] >= acc["lift_k"],
            bb["roc_auc"][0] >= acc["roc_auc"],
        ]
        out.append(
            f"  справочно {NAMES[key]}: PR-AUC {'ok' if ok[0] else 'нет'}, "
            f"Lift {'ok' if ok[1] else 'нет'}, "
            f"ROC-AUC {'ok' if ok[2] else 'нет'}"
            + (f"; разность с правилом {ci(dr)}" if dr else "")
        )
    passed = all(ok for *_, ok in checks)
    decision = "используется в системе" if passed else "возвращается на доработку"
    out.append(f"  Решение: модель {decision}")

    OUT_JSON.write_text(
        json.dumps(
            {
                "model": "models/model_v1.joblib",
                "threshold": t_star,
                "threshold_rule": "минимум FN·c_FN + FP·c_FP по прогнозам вне блока "
                "на обучающей выборке",
                "costs": {"fp": c_fp, "fn": c_fn},
                "test": {key: {m: list(v) for m, v in bs[key].items()} for key in NAMES},
                "acceptance": [{"check": t, "value": v, "passed": bool(ok)} for t, v, ok in checks],
                "decision": decision,
                "monitoring": mon,
                "psi_features": feats,
                "batches": [
                    {
                        "part": r["part"],
                        "n": r["n"],
                        "base": r["base"],
                        "mean_pred": r["mean_pred"],
                        "pr_auc": r["pr_auc"],
                        "roc_auc": r["roc_auc"],
                        "lift_k": r["lift_k"],
                        "psi": r["psi"],
                        "signals": r["flags"],
                        "decision": r["decision"],
                    }
                    for r in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    out.append(f"  сохранено: {OUT_JSON.relative_to(ROOT).as_posix()}")
    import inspect

    import bank_marketing.evaluation as E

    (PR4_DIR / "code").mkdir(parents=True, exist_ok=True)
    (PR4_DIR / "code" / "листинг_2_1_bootstrap.py").write_text(
        inspect.getsource(E.bootstrap) + "\n\n" + inspect.getsource(E._ci), encoding="utf-8"
    )
    emit(out, PR4_DIR / "evaluate_out.txt")


if __name__ == "__main__":
    main()
