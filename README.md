# HW3

Домашняя работа по ценовой эластичности и прогнозированию спроса на датасете M5 Forecasting.

Сделано:

- автоматически выбраны 5 SKU с высокой частотой продаж и 13-17 ценами;
- построены кривые эластичности и рекомендации `promo`/`regular`;
- на holdout 28 дней сравнены 3 подхода: без будущей цены, с будущей ценой, базовая модель + кривая спроса.

Итог:

- лучший средний метод: `with_future_price`, `WAPE = 0.463`;
- `no_future_price`: `WAPE = 0.479`;
- `base_plus_curve`: `WAPE = 0.566`;
- явный promo-кандидат: `TX_2|FOODS_1_129`, остальные SKU лучше держать в regular.

Ключевые файлы:

- `hw3_m5_elasticity_forecasting.ipynb`;
- `src/hw3/pipeline.py`;
- `output/tables/final_recommendations.csv`;
- `output/tables/forecast_method_summary.csv`;
- `output/figures/`.

Как воспроизвести:

```bash
cd /Users/vlad/projects/HSSE/ML_in_business/HW3
uv sync
uv run hw3-run --selected-skus 5
uv run hw3-build-notebook
uv run jupyter nbconvert --to notebook --execute --inplace hw3_m5_elasticity_forecasting.ipynb
```
