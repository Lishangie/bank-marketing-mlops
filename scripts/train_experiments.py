"""Эксперименты ПР3: базовые решения, линейные модели, классификаторы, ансамбли.

Все модели сравниваются по одной схеме: обучающая выборка исходной части
(май – июль 2008 г.), TimeSeriesSplit на 5 блоков, метрики PR-AUC (основная),
ROC-AUC, Lift@k и Recall@k. Тестовая выборка не используется.

Результаты:
  reports/experiments.csv              — все эксперименты (учёт; дублируется в MLflow);
  reports/selection.json               — итоговая модель по правилу выбора;
  report_assets/pr3/experiments_out.txt — сводка для отчёта;
  report_assets/pr3/img/рис_*.png       — рисунки (копия в reports/figures/pr3).
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from bank_marketing.config import ASSETS_DIR, FIGURES_DIR, REPORTS_DIR
from bank_marketing.experiments import (
    METRICS,
    Tracker,
    evaluate,
    load_params,
    load_train,
    score_rule,
    time_cv,
)
from bank_marketing.features import TIME_PROXIES, make_pipeline
from bank_marketing.models import CatBoostModel
from bank_marketing.plotting import C_ACCENT, C_MEAN, C_NO, C_YES, CM, apply_style, num, save
from bank_marketing.report import emit

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PR3_DIR = ASSETS_DIR / "pr3"
IMG_DIRS = [PR3_DIR / "img", FIGURES_DIR / "pr3"]
C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
DEPTHS = list(range(1, 13))
# Порядок сложности для правила выбора: при близком качестве берётся более простая.
COMPLEXITY = ["linear", "nb", "knn", "tree", "rf", "boosting"]
TIME_DROP = (*TIME_PROXIES, "emp.var.rate", "euribor3m", "nr.employed")


def lr(c=1.0, l1=False, cw="balanced"):
    return LogisticRegression(
        C=c,
        l1_ratio=1 if l1 else 0,
        solver="liblinear",
        class_weight=cw,
        max_iter=5000,
        random_state=23,
        intercept_scaling=100,  # liblinear штрафует свободный член; так штраф пренебрежимо мал
    )


def rule_score(X: pd.DataFrame) -> np.ndarray:
    """Правило по итогам EDA (ПР2): мобильная связь, студенты и пенсионеры, младше 25 и 60+."""
    return (
        (X["contact"] == "cellular").astype(int)
        + X["job"].isin(["student", "retired"]).astype(int)
        + ((X["age"] < 25) | (X["age"] >= 60)).astype(int)
    ).to_numpy()


def nonzero_coefs(pipe, X, y) -> tuple[int, int]:
    pipe.fit(X, y)
    coef = pipe.named_steps["model"].coef_.ravel()
    return int((np.abs(coef) > 1e-8).sum()), len(coef)


def coef_table(pipe, X, y, top=12) -> pd.DataFrame:
    pipe.fit(X, y)
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_.ravel()
    t = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coef": coef})
    return t.reindex(t["coef"].abs().sort_values(ascending=False).index).head(top)


def mean(res, m="pr_auc") -> float:
    return float(np.mean(res.folds[m]))


def fmt_res(res) -> str:
    v = {m: np.asarray(res.folds[m]) for m in METRICS}
    return (
        f"PR-AUC {v['pr_auc'].mean():.4f} ± {v['pr_auc'].std():.4f}; "
        f"ROC-AUC {v['roc_auc'].mean():.4f} ± {v['roc_auc'].std():.4f}; "
        f"Lift@k {v['lift_k'].mean():.2f}; Recall@k {v['recall_k'].mean():.3f}; "
        f"время {res.fit_time:.2f} с"
    )


# ---------------------------------------------------------------- рисунки
def fig_split(X: pd.DataFrame, params: dict, cv) -> None:
    """Рисунок 1.1: временное разбиение исходной части и блоки кросс-валидации."""
    from bank_marketing.config import INITIAL_FILE
    from bank_marketing.features import prepare

    full, _ = prepare(INITIAL_FILE)
    n_all, n_tr = len(full), len(X)
    months = full["period"].to_numpy()
    fig, ax = plt.subplots(figsize=(16 * CM, 7.2 * CM))
    rows = params["cv_splits"]
    # верхняя полоса: обучающая / тестовая выборка
    y0 = rows + 1.2
    ax.barh(y0, n_tr, left=0, height=0.7, color=C_ACCENT)
    ax.barh(y0, n_all - n_tr, left=n_tr, height=0.7, color=C_YES)
    ax.text(n_tr / 2, y0, f"обучающая: {num(n_tr, 0)}", ha="center", va="center", color="white")
    ax.text(
        n_tr + (n_all - n_tr) / 2,
        y0,
        f"тест: {num(n_all - n_tr, 0)}",
        ha="center",
        va="center",
        color="white",
    )
    for i, (tr, va) in enumerate(cv.split(X)):
        yy = rows - i
        ax.barh(yy, len(tr), left=0, height=0.6, color=C_NO)
        ax.barh(yy, len(va), left=va[0], height=0.6, color=C_MEAN)
    # границы месяцев
    for j in np.flatnonzero(months[1:] != months[:-1]) + 1:
        ax.axvline(j, color="black", lw=0.6, ls=":")
    starts = [0, *(np.flatnonzero(months[1:] != months[:-1]) + 1)]
    ends = [*starts[1:], n_all]
    names = {"05": "май", "06": "июнь", "07": "июль", "08": "авг.", "10": "окт."}
    for s, e in zip(starts, ends, strict=True):
        label = names[months[s][-2:]]
        if e - s > 500:
            ax.text((s + e) / 2, rows + 2.05, label, ha="center", va="bottom")
    ax.set_yticks([y0, *range(rows, 0, -1)])
    ax.set_yticklabels(["Исходная часть", *[f"Блок {i + 1}" for i in range(rows)]])
    ax.set_xlabel("Записи исходной части по порядку времени")
    ax.set_xlim(0, n_all)
    ax.set_ylim(0.4, rows + 2.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num(v, 0)))
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=C_NO, label="обучение в блоке"),
            Patch(color=C_MEAN, label="проверка"),
        ],
        loc="lower right",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(1, -0.36),
    )
    ax.spines["left"].set_visible(False)
    save(fig, "рис_1_1.png", IMG_DIRS)


def fig_regularization(curve: pd.DataFrame, base_pr: float) -> None:
    """Рисунок 3.1: сила регуляризации → PR-AUC (L1, L2) и число ненулевых коэффициентов."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16 * CM, 6.5 * CM))
    for pen, color, mk in [("L2", C_ACCENT, "o"), ("L1", C_YES, "s")]:
        d = curve[curve["penalty"] == pen]
        a1.plot(d["C"], d["pr_auc"], color=color, marker=mk, ms=4, lw=1.2, label=pen)
    a1.axhline(base_pr, color=C_NO, ls="--", lw=1, label="базовое")
    a1.set_xscale("log")
    a1.set_xlabel("Параметр C (меньше — сильнее\nрегуляризация)")
    a1.set_ylabel("PR-AUC, среднее по блокам")
    a1.legend(frameon=False, loc="lower right", handlelength=1.5, borderaxespad=0.2)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num(v, 4)))
    d = curve[curve["penalty"] == "L1"]
    a2.plot(d["C"], d["nonzero"], color=C_YES, marker="s", ms=4)
    a2.axhline(d["total"].iloc[0], color=C_NO, ls="--", lw=1, label="всего коэффициентов")
    a2.set_xscale("log")
    a2.set_xlabel("Параметр C")
    a2.set_ylabel("Ненулевых коэффициентов L1")
    a2.legend(frameon=False, loc="center right")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num(v, 0)))
    for a in (a1, a2):
        a.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: num(v, 4).rstrip("0").rstrip(","))
        )
    fig.tight_layout()
    save(fig, "рис_3_1.png", IMG_DIRS)


def fig_depth(depth: pd.DataFrame) -> None:
    """Рисунок 3.2: PR-AUC дерева на обучающих и проверочных блоках по глубине."""
    fig, ax = plt.subplots(figsize=(16 * CM, 6.5 * CM))
    ax.plot(
        depth["depth"], depth["train"], color=C_ACCENT, marker="o", ms=4, label="обучающие блоки"
    )
    ax.plot(depth["depth"], depth["val"], color=C_YES, marker="s", ms=4, label="проверочные блоки")
    ax.set_yscale("log")
    ax.set_xlabel("Максимальная глубина дерева")
    ax.set_ylabel("PR-AUC (логарифмическая шкала)")
    ax.set_xticks(depth["depth"])
    ax.set_yticks([0.04, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num(v, 2)))
    ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: ""))
    ax.legend(frameon=False)
    save(fig, "рис_3_2.png", IMG_DIRS)


def fig_compare(summary: pd.DataFrame, base_pr: float, rule_pr: float) -> None:
    """Рисунок 3.3: средний PR-AUC моделей сводной таблицы и два базовых уровня."""
    d = summary[summary["family"] != "baseline"].sort_values("pr_auc_mean")
    fig, ax = plt.subplots(figsize=(16 * CM, 0.62 * CM * len(d) + 2.2 * CM))
    colors = [C_YES if s else C_ACCENT for s in d["selected"]]
    ax.barh(d["label"], d["pr_auc_mean"], color=colors, height=0.6)
    for x, label, ls in [(base_pr, "наиболее частый класс", "--"), (rule_pr, "правило", ":")]:
        ax.axvline(x, color=C_MEAN if ls == ":" else C_NO, ls=ls, lw=1.2)
        ax.text(
            x,
            len(d) - 0.3,
            label,
            color="#444444",
            va="bottom",
            ha="right" if ls == "--" else "left",
        )
    lo = 0.045
    ax.set_xlim(lo, d["pr_auc_mean"].max() * 1.06)
    ax.set_xlabel("PR-AUC, среднее по блокам")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: num(v, 3)))
    for i, v in enumerate(d["pr_auc_mean"]):
        ax.text(v, i, f" {num(v, 4)}", va="center", fontsize=8)
    save(fig, "рис_3_3.png", IMG_DIRS)


# ---------------------------------------------------------------- эксперименты
def main() -> None:
    apply_style()
    params = load_params()
    k, seed = params["top_k"], params["seed"]
    X, y = load_train(params)
    cv = time_cv(params)
    tr = Tracker()
    out = [
        "Эксперименты ПР3: обучающая выборка исходной части, TimeSeriesSplit "
        f"({params['cv_splits']} блоков), k = {k:.0%}",
        f"Обучающая выборка: {len(X)} записей, {X['period'].min()}…{X['period'].max()}, "
        f"доля отклика {y.mean():.2%}",
        "",
        "Блоки кросс-валидации (обучение / проверка, месяцы проверки, доля отклика):",
    ]
    for i, (a, b) in enumerate(cv.split(X)):
        out.append(
            f"  {i + 1}: {len(a):>6} / {len(b)}; {', '.join(X['period'].iloc[b].unique())}; "
            f"{y.iloc[b].mean():.2%}"
        )
    fig_split(X, params, cv)

    # 1. Базовые решения
    out += ["", "1. Базовые решения"]
    dummy = tr.add(
        evaluate(
            "Наиболее частый класс",
            "baseline",
            "сравнение",
            make_pipeline(DummyClassifier(strategy="prior")),
            X,
            y,
            cv,
            k,
            params={"strategy": "prior"},
        )
    )
    rule = tr.add(
        score_rule(
            "Правило: мобильная связь, студенты и пенсионеры, возраст",
            "baseline",
            X,
            y,
            cv,
            k,
            rule_score,
        )
    )
    out += [f"  {r.name}: {fmt_res(r)}" for r in (dummy, rule)]
    base_pr = mean(dummy)

    # 2. Линейные модели: сила регуляризации
    out += ["", "2. Логистическая регрессия (class_weight = balanced): C → метрики"]
    curve = []
    for pen in ["L2", "L1"]:
        for c in C_GRID:
            pipe = make_pipeline(lr(c, l1=pen == "L1"))
            r = tr.add(
                evaluate(
                    f"LogReg {pen} C={c:g}",
                    "linear",
                    "кривая",
                    pipe,
                    X,
                    y,
                    cv,
                    k,
                    params={"penalty": pen, "C": c, "class_weight": "balanced"},
                )
            )
            nz, tot = nonzero_coefs(make_pipeline(lr(c, l1=pen == "L1")), X, y)
            curve.append(
                {
                    "penalty": pen,
                    "C": c,
                    "pr_auc": mean(r),
                    "pr_std": float(np.std(r.folds["pr_auc"])),
                    "roc_auc": mean(r, "roc_auc"),
                    "lift": mean(r, "lift_k"),
                    "nonzero": nz,
                    "total": tot,
                }
            )
    curve = pd.DataFrame(curve)
    for _, c in curve.iterrows():
        out.append(
            f"  {c['penalty']} C = {c['C']:<7g} PR-AUC {c['pr_auc']:.4f} ± {c['pr_std']:.4f}"
            f"  ROC-AUC {c['roc_auc']:.4f}  Lift@k {c['lift']:.2f}  ненулевых {c['nonzero']} "
            f"из {c['total']}"
        )
    fig_regularization(curve, base_pr)
    best = {
        pen: curve[curve["penalty"] == pen].sort_values("pr_auc").iloc[-1] for pen in ["L1", "L2"]
    }
    lin = {}
    for pen in ["L2", "L1"]:
        c = float(best[pen]["C"])
        lin[pen] = tr.add(
            evaluate(
                f"Логистическая регрессия {pen} (C = {num(c, 4).rstrip('0').rstrip(',')})",
                "linear",
                "сравнение",
                make_pipeline(lr(c, l1=pen == "L1")),
                X,
                y,
                cv,
                k,
                params={"penalty": pen, "C": c, "class_weight": "balanced"},
            )
        )
    out.append(f"  лучшие: L2 C = {best['L2']['C']:g}, L1 C = {best['L1']['C']:g}")

    # 3. Классификаторы разной природы
    out += ["", "3. Модели классификации и ансамбли"]
    depth_rows = []
    for d in DEPTHS:
        r = tr.add(
            evaluate(
                f"Дерево глубины {d}",
                "tree",
                "кривая",
                make_pipeline(DecisionTreeClassifier(max_depth=d, random_state=seed), scale=False),
                X,
                y,
                cv,
                k,
                params={"max_depth": d},
                train=True,
            )
        )
        depth_rows.append(
            {
                "depth": d,
                "train": float(np.mean(r.train_folds["pr_auc"])),
                "val": mean(r),
                "val_std": float(np.std(r.folds["pr_auc"])),
            }
        )
    depth = pd.DataFrame(depth_rows)
    fig_depth(depth)
    out.append("  Дерево: глубина → PR-AUC обучающие / проверочные блоки")
    out += [
        f"    {int(r['depth']):>2}: {r['train']:.4f} / {r['val']:.4f} ± {r['val_std']:.4f}"
        for _, r in depth.iterrows()
    ]
    d_best = int(depth.sort_values("val").iloc[-1]["depth"])
    models = [
        (
            "k ближайших соседей (k = 100)",
            "knn",
            make_pipeline(KNeighborsClassifier(100)),
            {"n_neighbors": 100},
        ),
        ("Наивный байесовский классификатор", "nb", make_pipeline(GaussianNB()), {}),
        (
            f"Дерево решений (глубина {d_best})",
            "tree",
            make_pipeline(DecisionTreeClassifier(max_depth=d_best, random_state=seed), scale=False),
            {"max_depth": d_best},
        ),
        (
            "Дерево решений без ограничения глубины",
            "tree",
            make_pipeline(DecisionTreeClassifier(random_state=seed), scale=False),
            {"max_depth": None},
        ),
        (
            "Случайный лес (300 деревьев)",
            "rf",
            make_pipeline(
                RandomForestClassifier(300, min_samples_leaf=20, n_jobs=-1, random_state=seed),
                scale=False,
            ),
            {"n_estimators": 300, "min_samples_leaf": 20},
        ),
        (
            "CatBoost (по умолчанию)",
            "boosting",
            make_pipeline(CatBoostModel(random_seed=seed), native_cat=True),
            {"iterations": 500, "depth": 6},
        ),
    ]
    clf = {}
    for name, fam, pipe, p in models:
        clf[name] = tr.add(evaluate(name, fam, "сравнение", pipe, X, y, cv, k, params=p))
        out.append(f"  {name}: {fmt_res(clf[name])}")

    # 4. Польза новых признаков и признаки времени
    out += ["", "4. Польза новых признаков (base — исходные столбцы, new — с признаками плана)"]
    c_l2 = float(best["L2"]["C"])
    feat_rows = []
    variants = [
        ("Логистическая регрессия L2", lambda fs, drop: make_pipeline(lr(c_l2), fs, drop)),
        (
            "CatBoost",
            lambda fs, drop: make_pipeline(
                CatBoostModel(random_seed=seed), fs, drop, native_cat=True
            ),
        ),
    ]
    for mname, factory in variants:
        for fs, drop, label in [
            ("base", (), "исходные"),
            ("new", (), "с новыми"),
            ("new", TIME_DROP, "с новыми, без признаков времени"),
        ]:
            r = tr.add(
                evaluate(
                    f"{mname}: {label}",
                    "features",
                    "признаки",
                    factory(fs, drop),
                    X,
                    y,
                    cv,
                    k,
                    feature_set=fs if not drop else "new-time",
                    params={"model": mname},
                )
            )
            feat_rows.append(
                {
                    "model": mname,
                    "set": label,
                    "pr": mean(r),
                    "pr_std": float(np.std(r.folds["pr_auc"])),
                    "roc": mean(r, "roc_auc"),
                    "lift": mean(r, "lift_k"),
                }
            )
            out.append(f"  {mname:<28} {label:<32} {fmt_res(r)}")

    # 5. Контроль: случайная кросс-валидация вместо временной
    out += ["", "5. Контроль схемы: случайная стратифицированная КВ (5 блоков) вместо временной"]
    rnd = StratifiedKFold(5, shuffle=True, random_state=seed)
    for mname, factory in variants:
        r = tr.add(
            evaluate(
                f"{mname}: случайная КВ",
                "control",
                "контроль",
                factory("new", ()),
                X,
                y,
                rnd,
                k,
                params={"cv": "StratifiedKFold"},
            )
        )
        out.append(f"  {mname:<28} {fmt_res(r)}")

    # 6. Подбор гиперпараметров двух лучших семейств
    cands = [r for r in tr.results if r.stage == "сравнение" and r.family != "baseline"]
    by_family = {}
    for r in sorted(cands, key=mean, reverse=True):
        by_family.setdefault(r.family, r)
    top2 = list(by_family)[:2]
    out += ["", f"6. Подбор гиперпараметров: семейства {', '.join(top2)} (GridSearchCV, PR-AUC)"]
    grids = {
        "linear": (
            make_pipeline(lr()),
            {
                "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
                "model__l1_ratio": [0, 1],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "boosting": (
            make_pipeline(CatBoostModel(random_seed=seed), native_cat=True),
            {
                "model__iterations": [200, 500],
                "model__depth": [3, 4, 6],
                "model__learning_rate": [0.03, 0.1],
                "model__l2_leaf_reg": [3, 30],
            },
        ),
        "rf": (
            make_pipeline(RandomForestClassifier(300, n_jobs=-1, random_state=seed), scale=False),
            {
                "model__min_samples_leaf": [5, 20, 50, 100],
                "model__max_features": ["sqrt", 0.3],
                "model__max_depth": [6, None],
            },
        ),
        "tree": (
            make_pipeline(DecisionTreeClassifier(random_state=seed), scale=False),
            {
                "model__max_depth": [2, 3, 4, 5, 6],
                "model__min_samples_leaf": [1, 20, 100, 300],
            },
        ),
        "knn": (
            make_pipeline(KNeighborsClassifier()),
            {
                "model__n_neighbors": [25, 50, 100, 200, 400],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "nb": (make_pipeline(GaussianNB()), {"model__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
    }
    tuned = {}
    for fam in top2:
        pipe, grid = grids[fam]
        gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, refit=False, n_jobs=1)
        gs.fit(X, y)
        bp = {kk.replace("model__", ""): v for kk, v in gs.best_params_.items()}
        n_comb = len(gs.cv_results_["params"])
        out.append(
            f"  {fam}: перебрано {n_comb} сочетаний; лучшее {bp}; PR-AUC {gs.best_score_:.4f}"
        )
        cvr = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
        out += ["    5 лучших сочетаний:"]
        for _, row in cvr.head(5).iterrows():
            pp = {kk.replace("model__", ""): v for kk, v in row["params"].items()}
            out.append(f"      {pp}: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        best_pipe = pipe.set_params(**gs.best_params_)
        label = {
            "linear": "Логистическая регрессия",
            "boosting": "CatBoost",
            "rf": "Случайный лес",
            "tree": "Дерево решений",
            "knn": "k ближайших соседей",
            "nb": "Наивный байесовский классификатор",
        }[fam]
        tuned[fam] = tr.add(
            evaluate(
                f"{label} (после подбора)", fam, "сравнение", best_pipe, X, y, cv, k, params=bp
            )
        )
        out.append(f"    после подбора: {fmt_res(tuned[fam])}")
        if fam == "linear":
            ct = coef_table(best_pipe, X, y, top=14)
            out.append("    коэффициенты на всей обучающей выборке, 14 наибольших по модулю:")
            out += [f"      {r['feature']:<28} {r['coef']:+.3f}" for _, r in ct.iterrows()]

    # 7. Сводная таблица и выбор
    summ = [r for r in tr.results if r.stage == "сравнение"]
    rows = []
    for r in summ:
        rr = r.row()
        rr["label"] = r.name
        rows.append(rr)
    summary = pd.DataFrame(rows)
    models_only = summary[summary["family"] != "baseline"]
    top = models_only.sort_values("pr_auc_mean", ascending=False).iloc[0]
    close = models_only[models_only["pr_auc_mean"] >= top["pr_auc_mean"] - params["select_delta"]]
    close = close.assign(cx=close["family"].map(COMPLEXITY.index))
    fold_top = np.array([float(x) for x in top["pr_auc_folds"].split()])

    def wins(row) -> int:
        """В скольких блоках лучшая модель превосходит данную по PR-AUC."""
        return int((fold_top > np.array([float(x) for x in row["pr_auc_folds"].split()])).sum())

    # Правило: самая простая из близких, если лучшая обгоняет её меньше чем в
    # select_min_folds блоках и у неё Lift@k > 1; иначе — лучшая.
    chosen = top
    for _, cand in close.sort_values(["cx", "pr_auc_mean"], ascending=[True, False]).iterrows():
        if wins(cand) < params["select_min_folds"] and cand["lift_k_mean"] > 1:
            chosen = cand
            break
    rule_lift = float(np.mean(rule.folds["lift_k"]))
    summary["selected"] = summary["model"] == chosen["model"]
    out += ["", "7. Сводная таблица (сравнение по TimeSeriesSplit)"]
    head = f"{'PR-AUC':>17}{'ROC-AUC':>17}{'Lift@k':>8}{'Recall@k':>10}{'время, с':>10}"
    out.append(f"  {'модель':<58}{head}")
    for _, r in summary.sort_values("pr_auc_mean", ascending=False).iterrows():
        mark = " ←" if r["selected"] else ""
        out.append(
            f"  {r['model']:<58}{r['pr_auc_mean']:>9.4f} ± {r['pr_auc_std']:.4f}"
            f"{r['roc_auc_mean']:>9.4f} ± {r['roc_auc_std']:.4f}{r['lift_k_mean']:>8.2f}"
            f"{r['recall_k_mean']:>10.3f}{r['fit_time_s']:>10.2f}{mark}"
        )
    out += [
        "",
        f"Наибольший PR-AUC: {top['model']} ({top['pr_auc_mean']:.4f}); "
        f"в пределах {params['select_delta']} от него: {', '.join(close['model'])}",
        "Сколько блоков из "
        f"{len(fold_top)} лучшая модель выигрывает у близких, Lift@k: "
        + "; ".join(
            f"{r['model']} — {wins(r)}, {r['lift_k_mean']:.2f}"
            for _, r in close.iterrows()
            if r["model"] != top["model"]
        ),
        f"Выбрана по правилу: {chosen['model']} ({chosen['pr_auc_mean']:.4f})",
        f"Выше наиболее частого класса по PR-AUC: {chosen['pr_auc_mean'] - base_pr:+.4f}; "
        f"Lift@k {chosen['lift_k_mean']:.2f} против {rule_lift:.2f} у правила",
    ]
    fig_compare(summary, base_pr, float(np.mean(rule.folds["pr_auc"])))

    selection = {
        "model": chosen["model"],
        "family": chosen["family"],
        "params": json.loads(chosen["params"]),
        "feature_set": chosen["feature_set"],
        "cv_pr_auc": chosen["pr_auc_mean"],
        "cv_pr_auc_std": chosen["pr_auc_std"],
        "cv_roc_auc": chosen["roc_auc_mean"],
        "cv_lift_k": chosen["lift_k_mean"],
        "cv_recall_k": chosen["recall_k_mean"],
        "baseline_pr_auc": base_pr,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tr.frame().to_csv(REPORTS_DIR / "experiments.csv", index=False, encoding="utf-8")
    out.append(f"\nЭкспериментов записано: {len(tr.rows)} (reports/experiments.csv, MLflow)")
    emit(out, PR3_DIR / "experiments_out.txt")


if __name__ == "__main__":
    main()
