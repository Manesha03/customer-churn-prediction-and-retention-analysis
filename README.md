# Customer Churn Risk and Retention Analysis

An end-to-end, reproducible churn-ranking project for the supplied snapshot. It preserves all observations, excludes `customer_id` from predictors, tunes three model families, and simulates a constrained 20% contact list.

## Run

Use Python 3.10 or newer. From the repository root:

```powershell
python -m pip install -r requirements.txt
python src/run_analysis.py
```

The original input is bundled at `data/raw/customer_churn_advanced_dataset.csv`. The script uses seed `42`, stratified 75/25 train/test split, and 4-fold stratified cross-validation. Grid search selects by average precision. The final test split is held apart until final model evaluation. Running the notebook top-to-bottom performs the same workflow; the script is its source of truth.

## Deliverables

- `notebooks/churn_analysis.ipynb`: executable, sectioned analysis with audit, EDA, feature engineering, model search, evaluation, explanations, and simulation.
- `data/cleaned_customer_churn.csv`: all original rows with whitespace-normalized categories and eight engineered predictors. Source missing values remain visible; no rows are silently removed.
- `models/churn_model.joblib`: fitted sklearn pipeline, operating threshold, and metadata. Load with `joblib.load`; use the `pipeline` entry's `predict_proba` on raw predictor columns (exclude `customer_id` and `churn`). The custom transformer is defined in the importable `src.features` module.
- `churn_business_report.pdf`: six-page decision report.
- `reports/figures/`: EDA and model evaluation figures.
- `reports/`: consolidated `analysis_details.json` for audit tables, model comparison, threshold metrics, permutation importance, local explanations, and risk ranking; individual JSON files hold quality checks, final metrics, robustness, package versions, and intervention results. Figures are stored in `reports/figures/`.

## Data and Modeling Decisions

The dataset has 1,200 rows, 20 columns, 1,036 churn labels (86.3%), and missingness in monthly charges (30), total charges (24), satisfaction (48), and monthly usage (36). The workflow checks duplicate rows/IDs, binary labels and plausible numeric ranges. It records out-of-range counts and retains all rows. Extreme but plausible usage or charge values are not automatically treated as errors. Missing numeric and categorical inputs are imputed inside each cross-validation fold; one-hot encoding handles unseen categories. The cleaned CSV preserves the original missing values.

Engineered features use only customer attributes available at scoring time: average charge per tenure month; late-payment plus electronic-check risk indicator; complaint intensity relative to tenure; complaint/payment interaction; low-satisfaction flag; charge-to-usage ratio; tenure/charge interaction; and month-to-month indicator. None uses churn or a post-churn field. Confirm that `total_charges` is available as of the intended scoring date before production use.

Models: class-weighted Logistic Regression and Random Forest, plus HistGradientBoosting. Each has an untuned baseline and a small grid search evaluated by four-fold average precision. The selected operating threshold maximizes assumed net value on training-only out-of-fold scores, subject to the training contact-capacity constraint. The final holdout is not used to select the model or threshold. A capacity-based top-20% ranking is separately evaluated on the test split.

## Business Assumptions and Limits

The intervention calculation assumes an $8 contact cost, $180 value for a retained customer, and 25% intervention save rate. These are hypothetical inputs, not measured treatment effects. The model uses historical labels, so its feature importance and customer-level perturbations describe predictive association, not causation. A randomized retention pilot is needed to estimate incremental impact. Validate on a later time period, review calibration and subgroup performance, and confirm data availability and label timing before deployment. The churn prevalence is unusually high and may reflect the data generation process; do not assume it represents a live customer portfolio.

## Environment

Python package versions used for the generated outputs are recorded in `reports/package_versions.json`. Dependencies are listed in `requirements.txt`.
