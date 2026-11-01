# Used Heavy Equipment Price Prediction & Market Intelligence

**Python · Machine Learning · CatBoost · Random Forest · Time-Series Validation · SHAP**

## Business question

Can historical auction records provide useful estimates of used heavy equipment selling prices, and where do these estimates become unreliable? This independent portfolio case study applies a reproducible regression workflow to the **Kaggle Blue Book for Bulldozers** dataset; it is not commissioned work for an auction company.

## Dataset and setup

The original `Train.csv` contains **401,125 transactions and 53 columns**, covering **1989–2011**. The raw dataset is **not redistributed in this repository**. Obtain `Train.csv` from the Blue Book for Bulldozers Kaggle competition (subject to its terms) and place it at `data/raw/Train.csv`.

```bash
python -m venv .venv
# Activate .venv using instructions for your operating system
pip install -r requirements.txt
python -m pytest -q
jupyter lab
```

Start Jupyter from the repository root, or open the notebooks from the `notebooks/` directory. Execute in order: `01_exploratory_data_analysis`, `02_data_preparation`, `03_model_development`, `04_model_evaluation`, `05_model_interpretability`.

**Model development fits six relatively expensive models.** You can run EDA and data preparation first and postpone the expensive fits.

## Data preparation

- Interpret `YearMade == 1000` and manufacturing years after sale as invalid.
- Define `EquipmentAge = SaleYear − YearMade_Clean`, preserving missing values.
- Retain zero operating hours as different from missing hours; invalidate readings that exceed an age-based physical limit when age is known.
- Treat `ModelID` as a **categorical identifier** rather than a continuous numerical feature.
- Use `EquipmentAge`, `MachineHours_Clean`, `ProductGroup`, `ModelID`, `ProductSize` for the two final benchmarks.

## Historical validation design

| Experiment | Train | Validation |
|---|---|---|
| A | 2001–2005 | 2006–2007 |
| B | 2003–2007 | 2008–2009 |
| C | 2005–2009 | 2010–2011 |

Random Forest fits imputation and one-hot encoding on each **training** fold only. CatBoost handles missing numerical predictors and explicitly labeled missing categorical values. The models were selected through repeated development-stage comparisons.

## Reproduced development-validation results (locally verified, November 2026)

The notebook `03_model_development.ipynb` was executed against the 401,125-row source data in the analyst's local environment (Python 3.14.8; pandas 3.0.6; scikit-learn 1.9.1; CatBoost 1.2.10). Its output was checked in `04_model_evaluation.ipynb`. These are **development-validation**, not independent test, results.

| Validation window | Random Forest MAE | CatBoost MAE |
|---|---:|---:|
| A — 2006–2007 | **$7,316.82** | $7,386.08 |
| B — 2008–2009 | **$7,879.70** | $8,676.73 |
| C — 2010–2011 | $7,464.70 | **$7,200.14** |
| **Unweighted mean across the three periods** | **$7,553.74** | **$7,754.32** |

The three-period mean gives equal weight to each period, **not** each transaction. Random Forest has the lower mean MAE in this reproduction, while CatBoost leads in the 2010–2011 window. Neither model can be declared superior on truly unseen future transactions from these experiments alone.

For CatBoost in 2010–2011, the reproduced **MedianAE was $4,189.99**, **MAPE was 23.87%**, and **22.08%** of predictions had absolute error exceeding $10,000. Global CatBoost `PredictionValuesChange` importance ranked `ModelID` (38.48), `EquipmentAge` (32.87), `ProductGroup` (13.39), `ProductSize` (12.92), and `MachineHours_Clean` (2.34); these are model-attribution scores, **not** variance explained or causal effects.

### Historical benchmark discrepancy

The earlier research notebook reported Random Forest MAEs of **$7,539.81 / $8,151.64 / $7,639.24** for periods A/B/C (mean **$7,776.90**). The reorganized project's Random Forest results differ, whereas its CatBoost values match the historical results to two decimals. This difference is **unresolved**. The new Random Forest pipeline fits preprocessing within each training fold; changes in feature handling and library versions are plausible explanations, **not verified causes**. Do not describe this discrepancy as a demonstrated model improvement. See [`reports/TECHNICAL_AUDIT.md`](reports/TECHNICAL_AUDIT.md).

All metrics should be computed from model predictions. `03_model_development.ipynb` writes `reports/tables/walkforward_recomputed.csv`, `model_comparison_recomputed.csv`, and `catboost_2010_2011_predictions.csv` locally; rerun it to regenerate them. These files are **not included in the distribution ZIP**, because the full raw dataset and local prediction outputs were not supplied with it.

## Insights and business implications

- Overall historical median selling price: **$24,000**, but category medians range from **$10,000 (SSL)** to **$40,000 (MG)**.
- Price is right-skewed; a single overall price is inadequate for valuing individual equipment models.
- Equipment age has an overall Pearson correlation of approximately **−0.248** with price, with stronger negative within-category associations.
- `MachineHoursCurrentMeter` is missing in approximately **64.4%** of training records; `ProductSize` in **52.6%**. `ProductSize` is entirely missing in BL, MG, and SSL categories.
- Historical price shifts and extreme individual errors require monitoring bias and uncertainty, not only average error.

## Repository structure

```text
notebooks/   # EDA, data preparation, model development, evaluation, explanation
src/         # Shared cleaning, modelling, reporting and path utilities
tests/       # Synthetic-data unit tests
reports/     # Outputs regenerated by notebooks; technical audit and limitations
data/raw/    # User-provided Kaggle CSV (ignored by Git)
models/      # Optional locally fitted models (ignored by Git)
```

## Model limitations and integrity

Historical periods and the 2012 validation records were already consulted during feature development and model selection. Neither is an untouched independent estimate of deployment performance. The historical prices are nominal (not inflation-adjusted), equipment completeness varies by category, and selected transactions have large errors. Feature importance and SHAP do **not** imply causality. An independent future holdout is required before claiming generalizable accuracy.

## Reproducibility status

**Execution reported and outputs reviewed:** The analyst executed the 401,125-row preparation workflow, all six walk-forward model fits, evaluation diagnostics, and CatBoost interpretability in a local Windows/Jupyter environment. The associated output tables and screenshots were reviewed. Both pytest tests passed locally after making the dtype-compatible assertion. EDA statistical outputs were also reviewed, although an independently inspected fresh-kernel full run has not been performed in this package. Full model fits have **not** been independently rerun by the package author; published scores should be attributed to the analyst’s reproduced local outputs. See the technical audit for the unresolved Random Forest discrepancy.
