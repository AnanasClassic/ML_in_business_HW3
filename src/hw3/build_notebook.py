from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOK_CELLS = [
    (
        "markdown",
        """# HW3: M5 Price Elasticity and Forecasting

Цель:

- выбрать несколько SKU с частыми продажами и как минимум 10 разными ценами;
- оценить ценовую эластичность каждого SKU и решить, подходит ли он для промо;
- сравнить 3 способа прогноза спроса на отложенном горизонте 28 дней.
""",
    ),
    (
        "markdown",
        """## План анализа

1. Скачать и подготовить датасет M5.
2. Автоматически выбрать SKU с высокой частотой продаж и богатой ценовой историей.
3. Построить кривую эластичности по каждому SKU.
4. Сравнить три прогноза:
   - без знания будущей цены;
   - с будущей ценой как признаком модели;
   - базовая модель + кривая спроса от цены.
""",
    ),
    (
        "code",
        """from pathlib import Path
import json

import pandas as pd
from IPython.display import Image, Markdown, display

from hw3.pipeline import default_config, run_pipeline

PROJECT_DIR = Path.cwd()
CONFIG = default_config(PROJECT_DIR)
TABLES_DIR = CONFIG.tables_dir
FIGURES_DIR = CONFIG.figures_dir
""",
    ),
    (
        "code",
        """if not (TABLES_DIR / "analysis_summary.json").exists():
    run_pipeline(CONFIG)

with open(TABLES_DIR / "analysis_summary.json", "r", encoding="utf-8") as file:
    summary_payload = json.load(file)

selected_skus = pd.read_csv(TABLES_DIR / "selected_skus.csv")
elasticity_summary = pd.read_csv(TABLES_DIR / "elasticity_summary.csv")
metrics_by_sku = pd.read_csv(TABLES_DIR / "forecast_metrics_by_sku.csv")
method_summary = pd.read_csv(TABLES_DIR / "forecast_method_summary.csv")
final_recommendations = pd.read_csv(TABLES_DIR / "final_recommendations.csv")
predictions = pd.read_csv(TABLES_DIR / "holdout_predictions.csv", parse_dates=["date"])
""",
    ),
    ("markdown", "## Выбранные SKU"),
    (
        "code",
        """selected_skus[
    [
        "sku_id",
        "dept_id",
        "cat_id",
        "positive_sales_ratio",
        "total_sales",
        "unique_prices",
        "min_price",
        "median_price",
        "max_price",
    ]
].round(3)
""",
    ),
    (
        "markdown",
        """## Методика

- `Модель 1`: градиентный бустинг по лагам, календарю и SNAP, но без цены будущего периода.
- `Модель 2`: тот же тип модели, но с признаками текущей цены.
- `Модель 3`: прогноз `Модели 1`, скорректированный через отдельную монотонную кривую эластичности.

Кривая эластичности оценивается как связь между ценой и мультипликатором спроса относительно базового прогноза без цены.
""",
    ),
    ("markdown", "## Итоги по эластичности"),
    (
        "code",
        """elasticity_summary.sort_values("local_elasticity")[[
    "sku_id",
    "unique_prices",
    "local_elasticity",
    "promo_lift",
    "recommendation",
]]
""",
    ),
    ("markdown", "## Качество прогноза"),
    (
        "code",
        """method_summary.assign(
    mae=lambda df: df["mae"].round(3),
    rmse=lambda df: df["rmse"].round(3),
    wape=lambda df: df["wape"].round(3),
    bias=lambda df: df["bias"].round(3),
)
""",
    ),
    (
        "code",
        """metrics_by_sku.assign(
    mae=lambda df: df["mae"].round(3),
    rmse=lambda df: df["rmse"].round(3),
    wape=lambda df: df["wape"].round(3),
    bias=lambda df: df["bias"].round(3),
).sort_values(["sku_id", "wape"])
""",
    ),
    ("markdown", "## Выводы по SKU"),
    (
        "code",
        """for narrative in summary_payload["narratives"]:
    display(Markdown(f"- {narrative}"))
""",
    ),
    ("markdown", "## Визуализации"),
    (
        "code",
        """for sku_id in selected_skus["sku_id"]:
    display(Markdown(f"### {sku_id}"))
    display(Image(filename=str(FIGURES_DIR / f"elasticity_{sku_id.replace('|', '_')}.png")))
    display(Image(filename=str(FIGURES_DIR / f"forecast_{sku_id.replace('|', '_')}.png")))
""",
    ),
    (
        "markdown",
        """## Общий анализ

- Если `Модель 2` лучше, значит будущая цена действительно несёт полезный сигнал и модель смогла его выучить.
- Если сильнее `Модель 3`, значит явная кривая спроса добавляет структуру, которую бустинг сам не вытащил.
- Если выигрывает `Модель 1`, цена либо слабо влияет на спрос, либо влияние нестабильно и тонет в шуме.

Что можно улучшить дальше:

- добавить иерархические признаки по категории и магазину;
- использовать rolling validation для настройки гиперпараметров;
- заменить локальную изотоническую кривую на байесовскую или сплайновую модель с доверительными интервалами;
- учесть каннибализацию, праздники и глубину промо отдельно.
""",
    ),
]

NOTEBOOK_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}


def make_cell(kind: str, source: str) -> nbf.NotebookNode:
    return getattr(nbf.v4, f"new_{kind}_cell")(source)


def build_notebook() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        cells=[make_cell(kind, source) for kind, source in NOTEBOOK_CELLS],
        metadata=NOTEBOOK_METADATA,
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_path = project_dir / "hw3_m5_elasticity_forecasting.ipynb"
    with output_path.open("w", encoding="utf-8") as file:
        nbf.write(build_notebook(), file)
    print(output_path)


if __name__ == "__main__":
    main()
