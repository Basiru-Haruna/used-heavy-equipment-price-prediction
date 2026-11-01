# Technical audit and reproducibility notes

## Scope and source

This portfolio was reorganized from the analyst's original `01_Bulldozer_Data_Understanding.ipynb`. It retains the substantive EDA, data-quality rules, five selected predictors, Random Forest/CatBoost workflows, chronological development windows, error diagnostics, and CatBoost explanations. The original experimental/teaching notebook is not part of the publication package.

## Verified locally by the analyst (outputs reviewed in conversation)

- Project source data: **401,125 rows**, **53 columns**, transactions dated **1989-01-17 to 2011-12-30**.
- Preparation: **38,198** invalid/unknown equipment ages, **378** operating-hours readings flagged by the physical-feasibility check, **401,125** rows retained.
- Automated tests: **2 passed** under **pytest 9.2.0** after replacing a pandas-legacy dtype assertion.
- Three chronological validation experiments executed; all **six** models fitted and metrics exported locally.
- Development-validation MAE, A/B/C: Random Forest **7,316.82 / 7,879.70 / 7,464.70**; CatBoost **7,386.08 / 8,676.73 / 7,200.14** (USD).
- Three-period **unweighted** mean MAE: Random Forest **$7,553.74**; CatBoost **$7,754.32**.
- Evaluation notebook loaded generated predictions and reproduced CatBoost 2010–2011 MAE **$7,200.14**, with MedianAE **$4,189.99** and MAPE **23.87%**.
- Interpretability notebook refitted CatBoost in period C, reproduced the same MAE, and generated global feature importance plus a local SHAP explanation whose contributions reconstruct the model prediction.
- The EDA printout was reviewed: key counts, correlations, group medians and missingness are consistent with the source data. The Markdown **inside the original packaged `.ipynb`** includes the correct dollar amounts; some `$` amounts were stripped when copying notebook text into chat. The notebook itself therefore does not require the proposed dollar-value corrections.

These confirmations derive from the analyst's locally executed outputs; the raw dataset was **not supplied** with the publication ZIP and complete training has not been rerun independently in this preparation environment.

## Environment reported for the local run

| Library | Version |
|---|---|
| Python | 3.14.8 |
| pandas | 3.0.6 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 |
| CatBoost | 1.2.10 |
| Matplotlib | 3.11.2 |
| nbformat | 5.11.0 |
| pytest | 9.2.0 |

`requirements.txt` states allowable ranges, **not an exact lockfile**. Future library versions must be tested; these ranges do not guarantee identical floating-point scores.

## Outstanding discrepancy: Random Forest

Historical experimental notebook RF MAEs for periods A/B/C: **$7,539.81 / $8,151.64 / $7,639.24** (mean $7,776.90). Current reproduced RF MAEs: **$7,316.82 / $7,879.70 / $7,464.70** (mean $7,553.74). The source of this difference has **not been established**. The repository uses per-training-window imputation and one-hot encoding; changes in preprocessing implementation, categorical handling, and software versions warrant examination. The CatBoost MAEs match the historical results to two decimals. Do not claim the RF difference is proven improvement or version drift.

## Interpretation and leakage limits

- A/B/C periods were consulted during feature selection and model development, so metrics are **development-validation**, not untouched holdout performance.
- The 2012 validation outcome was also inspected previously; it is not a genuinely unseen independent test.
- Historical nominal auction prices do not represent today's fair-market values.
- Missing `ProductSize` is strongly category-dependent; missing operating hours are extensive.
- CatBoost PredictionValuesChange and SHAP measure model attributions, not causal effects or directly explained variance.

## Before GitHub publication

1. Preserve the analyst's executed local notebooks. Do not overwrite them with blank-output templates.
2. Copy the corrected `README.md`, `requirements.txt`, `tests/test_preprocessing.py`, and `reports/TECHNICAL_AUDIT.md` from the update package.
3. Optionally execute every notebook from a fresh kernel one final time, especially the EDA notebook, and save outputs.
4. Run `python -m pytest -q` from the repository root.
5. Verify Git excludes raw data, caches, checkpoint files, saved models and any private information before publishing.
6. Reconcile the Random Forest difference if claiming exact reproduction of the historical experiment; otherwise clearly retain this documented limitation.
