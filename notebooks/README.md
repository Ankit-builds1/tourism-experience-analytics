# Notebooks

`tourism-experience-analytics.ipynb` — the end-to-end training pipeline that produces everything in [`../artifacts/`](../artifacts/).

## Pipeline stages

| Cell | Stage | Output |
|---|---|---|
| 1 | Imports, slug-independent Kaggle path discovery | — |
| 2 | Load 9 Excel tables; repair padding rows, duplicate files, coerced text columns | 9 DataFrames |
| 3 | Relational merge with a fan-out assertion | 52,930 × 22 |
| 4 | Cleaning, outlier removal, calendar features | `tourism_clean.csv` |
| 5 | EDA — distributions, cross-tabs, correlations, **variance decomposition** | the R² ceiling of 0.26 |
| 6 | **Out-of-fold target encoding**, per-key smoothing, automated leak check | 84 columns |
| 7 | Regression (6 models), satisfaction model, threshold sweep | `regression_results.csv` |
| 8 | Classification (6 models), baseline-relative reporting | `classification_results.csv` |
| 9 | Recommendation — CF, SVD, content-based, hybrid, popularity, demographic | `recommender_results.csv` |
| 10 | Artifact export with reload verification | `artifacts/*.pkl` |
| 11 | Generate `app.py` | the Streamlit app |

## Running it

The notebook expects the nine source tables (`Transaction.xlsx`, `User.xlsx`, `City.xlsx`,
`Updated_Item.xlsx`, `Type.xlsx`, `Mode.xlsx`, `Continent.xlsx`, `Region.xlsx`, `Country.xlsx`)
in a Kaggle input dataset. Cell 1 discovers them by filename, so the dataset slug does not matter.

Runtime is roughly 20 minutes on Kaggle CPU — Cell 2 spends ~40s in `openpyxl` reading the
52,930-row transaction sheet, and Cells 7–9 train 13 models in total.

> Cell 6 is the pivot point of the dependency graph: it creates `train`, `test`, `GLOBAL_MEAN`,
> `CLASSES` and `FOLD`. Changing anything there invalidates Cells 7 through 11. Use
> **Restart & Run All** before trusting the reported numbers.
