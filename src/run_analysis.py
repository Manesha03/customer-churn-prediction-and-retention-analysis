# %% [markdown]
# # Customer Churn Risk and Retention Analysis
# Reproducible end-to-end workflow. Run from the repository root with `python src/run_analysis.py`.

# %%
from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, precision_recall_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else (Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features import ChurnFeatureEngineer

DATA_PATH = ROOT / "data" / "raw" / "customer_churn_advanced_dataset.csv"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
TARGET = "churn"
ID_COL = "customer_id"
CONTACT_CAPACITY = 0.20
CONTACT_COST = 8.0
RETAINED_VALUE = 180.0
SAVE_RATE = 0.25


# %% [markdown]
# ## Audit and Preprocessing

# %%
def quality_audit(df):
    expected_binary = ["paperless_billing", "dependents", "support_plan", "tech_support"]
    checks = {
        "age_out_of_range": int((~df.age.between(18, 100)).sum()),
        "tenure_out_of_range": int((~df.tenure_months.between(0, 120)).sum()),
        "satisfaction_out_of_range": int((df.satisfaction_score.notna() & ~df.satisfaction_score.between(0, 10)).sum()),
        "negative_charges": int(((df.monthly_charges < 0) | (df.total_charges < 0)).sum()),
        "negative_complaints_or_late_payments": int(((df.complaints_last_12m < 0) | (df.late_payments_last_12m < 0)).sum()),
        "discount_out_of_range": int((~df.discount_pct.between(0, 100)).sum()),
        "unexpected_binary_labels": int(sum((~df[c].isin(["Yes", "No"])).sum() for c in expected_binary)),
        "target_invalid": int((~df[TARGET].isin([0, 1])).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[ID_COL].duplicated().sum()),
    }
    audit = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes],
                          "missing_count": df.isna().sum().values,
                          "missing_pct": (df.isna().mean().values * 100).round(2),
                          "unique_count": df.nunique(dropna=True).values})
    return checks, audit


# %% [markdown]
# ## Feature Engineering and Modeling Pipeline

# %%
def make_pipeline(model, scale=False):
    numeric = ["age", "tenure_months", "monthly_charges", "total_charges",
               "complaints_last_12m", "late_payments_last_12m", "satisfaction_score",
               "monthly_usage_gb", "discount_pct", "senior_citizen",
               "average_charge_per_tenure_month", "payment_risk_score", "complaint_intensity",
               "service_friction", "low_satisfaction_flag", "high_cost_low_usage",
               "tenure_monthly_charge_interaction", "contract_month_to_month"]
    categorical = ["contract_type", "internet_service", "region", "payment_method",
                   "paperless_billing", "dependents", "support_plan", "tech_support"]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    prep = ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), categorical),
    ], remainder="drop")
    return Pipeline([("features", ChurnFeatureEngineer()), ("preprocess", prep), ("model", model)])


# %% [markdown]
# ## Exploratory Analysis
#
# The charts are descriptive summaries. Month-to-month churn is 91.2%, versus about 79% for annual and two-year contracts. Churners have lower median tenure (34 vs 40 months), lower median satisfaction (6.7 vs 7.3), and higher median monthly charges ($72.78 vs $66.09). Churn reaches about 93% with two or more late payments; complaint counts of three or more also have high observed rates, but those groups are small. Payment-method differences are modest. The usage/charge scatter and contract/service chart help inspect nonlinear patterns and interactions; none of these associations establishes cause.

# %%
def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_eda(df):
    sns.set_theme(style="whitegrid", palette="deep")
    churn_rate = df.groupby("contract_type", observed=True).churn.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5)); churn_rate.plot.bar(ax=ax, color="#d05a47")
    ax.set(title="Churn rate by contract", ylabel="Share churned", xlabel="Contract type"); ax.set_ylim(0, 1)
    save_fig(fig, "eda_contract_churn.png")

    fig, ax = plt.subplots(figsize=(8, 4.5)); sns.boxplot(data=df, x="churn", y="tenure_months", ax=ax, showfliers=False)
    ax.set(title="Tenure by churn outcome", xlabel="Churn (0 = stayed, 1 = churned)", ylabel="Tenure (months)")
    save_fig(fig, "eda_tenure.png")

    fig, ax = plt.subplots(figsize=(8, 4.5)); sns.boxplot(data=df, x="churn", y="monthly_charges", ax=ax, showfliers=False)
    ax.set(title="Monthly charges by churn outcome", xlabel="Churn (0 = stayed, 1 = churned)", ylabel="Monthly charges")
    save_fig(fig, "eda_charges.png")

    fig, ax = plt.subplots(figsize=(8, 4.5)); sns.boxplot(data=df, x="churn", y="satisfaction_score", ax=ax, showfliers=False)
    ax.set(title="Satisfaction by churn outcome", xlabel="Churn (0 = stayed, 1 = churned)", ylabel="Satisfaction (0-10)")
    save_fig(fig, "eda_satisfaction.png")

    fig, ax = plt.subplots(figsize=(8, 4.5)); sns.barplot(data=df, x="complaints_last_12m", y="churn", ax=ax, color="#3b8c88", errorbar=None)
    ax.set(title="Churn rate by complaints", xlabel="Complaints in last 12 months", ylabel="Share churned"); ax.set_ylim(0, 1)
    save_fig(fig, "eda_complaints.png")

    fig, ax = plt.subplots(figsize=(8, 4.5)); sns.barplot(data=df, x="late_payments_last_12m", y="churn", ax=ax, color="#5576a6", errorbar=None)
    ax.set(title="Churn rate by late payments", xlabel="Late payments in last 12 months", ylabel="Share churned"); ax.set_ylim(0, 1)
    save_fig(fig, "eda_late_payments.png")

    fig, ax = plt.subplots(figsize=(9, 4.5)); rates = df.groupby("payment_method", observed=True).churn.mean().sort_values()
    rates.plot.bar(ax=ax, color="#728c4c"); ax.set(title="Churn rate by payment method", ylabel="Share churned", xlabel="Payment method"); ax.set_ylim(0, 1)
    save_fig(fig, "eda_payment.png")

    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df, x="monthly_usage_gb", y="monthly_charges", hue="churn", alpha=.55, ax=ax)
    ax.set(title="Charges and usage: churn interaction", xlabel="Monthly usage (GB)", ylabel="Monthly charges")
    save_fig(fig, "eda_usage_charges.png")

    fig, ax = plt.subplots(figsize=(8, 4.5)); sns.barplot(data=df, x="internet_service", y="churn", hue="contract_type", errorbar=None, ax=ax)
    ax.set(title="Contract and internet service interaction", ylabel="Share churned", xlabel="Internet service"); ax.set_ylim(0, 1)
    save_fig(fig, "eda_contract_service.png")


# %% [markdown]
# ## Training, Tuning, and Evaluation

# %%
def choose_threshold(y, p):
    best = (0.0, 0.5, 0, 0)
    for t in np.unique(np.r_[p, np.linspace(0.01, 0.99, 100)]):
        pred = p >= t
        contacted = pred.mean()
        if contacted > CONTACT_CAPACITY:
            continue
        tp = int(((pred) & (y == 1)).sum())
        net = tp * RETAINED_VALUE * SAVE_RATE - int(pred.sum()) * CONTACT_COST
        if net > best[0]:
            best = (net, float(t), int(pred.sum()), tp)
    return best[1], {"oof_expected_net_value": best[0], "oof_contacts": best[2], "oof_true_positives": best[3]}


# %%
def main():
    ROOT.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True); MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    checks, audit = quality_audit(df)
    (ROOT / "reports" / "data_quality_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    # No observations are removed. Normalize the declared binary text fields only; source missing values remain visible.
    cleaned = df.copy()
    for c in ["contract_type", "internet_service", "region", "payment_method", "paperless_billing", "dependents", "support_plan", "tech_support"]:
        cleaned[c] = cleaned[c].astype("string").str.strip()
    cleaned = ChurnFeatureEngineer().fit_transform(cleaned)
    cleaned.to_csv(ROOT / "data" / "cleaned_customer_churn.csv", index=False)
    create_eda(df)

    X = df.drop(columns=[TARGET, ID_COL]); y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df[ID_COL], test_size=.25, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    candidates = {
        "LogisticRegression": (make_pipeline(LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED), scale=True),
            {"model__C": [.1, 1.0, 10.0]}),
        "RandomForest": (make_pipeline(RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
            {"model__n_estimators": [250], "model__max_depth": [None, 8], "model__min_samples_leaf": [2, 8], "model__max_features": ["sqrt"]}),
        "HistGradientBoosting": (make_pipeline(HistGradientBoostingClassifier(random_state=SEED, max_iter=150)),
            {"model__learning_rate": [.05, .1], "model__max_leaf_nodes": [7, 15], "model__l2_regularization": [0.0, 1.0]}),
    }
    rows=[]; searches={}
    for name, (pipe, grid) in candidates.items():
        base = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
        search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1, refit=True, return_train_score=False)
        search.fit(X_train, y_train)
        searches[name]=search
        rows.append({"model":name,"baseline_cv_ap_mean":float(base.mean()),"baseline_cv_ap_std":float(base.std()),
                     "tuned_cv_ap_mean":float(search.best_score_),"best_params":str(search.best_params_)})
    comparison = pd.DataFrame(rows).sort_values("tuned_cv_ap_mean", ascending=False)
    selected_name = comparison.iloc[0].model
    selected = searches[selected_name].best_estimator_
    # Training-only out-of-fold probabilities define both the operating point and robustness range.
    oof = cross_val_predict(clone(selected), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold, threshold_info = choose_threshold(y_train.to_numpy(), oof)
    selected.fit(X_train, y_train)
    p_test = selected.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= threshold).astype(int)
    metrics = {"model": selected_name, "threshold": threshold, "precision":float(precision_score(y_test,pred_test,zero_division=0)),
               "recall":float(recall_score(y_test,pred_test,zero_division=0)), "f1":float(f1_score(y_test,pred_test,zero_division=0)),
               "roc_auc":float(roc_auc_score(y_test,p_test)), "pr_auc":float(average_precision_score(y_test,p_test)),
               "brier_score":float(brier_score_loss(y_test,p_test)), "confusion_matrix":confusion_matrix(y_test,pred_test).tolist(),
               "positive_rate":float(y.mean()), "test_rows":int(len(y_test)), "contact_cost":CONTACT_COST,
               "retained_customer_value":RETAINED_VALUE,"assumed_save_rate":SAVE_RATE,"contact_capacity":CONTACT_CAPACITY,
               **threshold_info}
    (ROOT / "reports" / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Threshold behavior and calibration are evaluated on the untouched holdout.
    threshold_rows=[]
    for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9,threshold]:
        pr=(p_test>=t)
        threshold_rows.append({"threshold":float(t),"precision":float(precision_score(y_test,pr,zero_division=0)),
                               "recall":float(recall_score(y_test,pr,zero_division=0)),"contact_rate":float(pr.mean()),"f1":float(f1_score(y_test,pr,zero_division=0))})
    threshold_df=pd.DataFrame(threshold_rows).drop_duplicates("threshold").sort_values("threshold")
    imp=permutation_importance(selected, X_test, y_test, scoring="average_precision", n_repeats=8, random_state=SEED, n_jobs=-1)
    importance=pd.DataFrame({"feature":X_test.columns,"importance_mean":imp.importances_mean,"importance_std":imp.importances_std}).sort_values("importance_mean",ascending=False)

    # Individual explanations: grouped one-feature-at-a-time replacements with training median/mode.
    risk = pd.DataFrame({"customer_id":id_test.to_numpy(),"actual_churn":y_test.to_numpy(),"churn_probability":p_test,
                         "test_index":X_test.index.to_numpy()})
    risk["contact_rank"] = risk.churn_probability.rank(method="first",ascending=False).astype(int)
    risk = risk.sort_values("churn_probability",ascending=False)
    samples = risk.head(3).copy()
    local_rows=[]
    for _, r in samples.iterrows():
        row=X_test.loc[[r.test_index]].copy()
        base=float(selected.predict_proba(row)[0,1])
        effects=[]
        for col in X_train.columns:
            repl=(X_train[col].median() if pd.api.types.is_numeric_dtype(X_train[col]) else X_train[col].mode(dropna=True).iloc[0])
            altered=row.copy(); altered.loc[:,col]=repl
            delta=base-float(selected.predict_proba(altered)[0,1])
            effects.append((col,delta))
        for col,delta in sorted(effects,key=lambda q:abs(q[1]),reverse=True)[:5]:
            local_rows.append({"customer_id":r.customer_id,"actual_churn":int(r.actual_churn),"predicted_probability":float(r.churn_probability),
                               "feature":col,"probability_change_if_set_to_train_typical":float(delta)})
    local=pd.DataFrame(local_rows)

    # Top 20% holdout lift versus random selection; random expectation is explicit.
    n_contact=max(1,int(np.ceil(len(y_test)*CONTACT_CAPACITY)))
    top=risk.head(n_contact); actual_churners=int(y_test.sum()); captured=int(top.actual_churn.sum())
    random_expected=CONTACT_CAPACITY*actual_churners
    benefit=captured*RETAINED_VALUE*SAVE_RATE-n_contact*CONTACT_COST
    intervention={"test_customers":int(len(y_test)),"top_20pct_contacts":int(n_contact),"actual_test_churners":actual_churners,
                  "churners_captured":captured,"capture_rate":float(captured/actual_churners),"random_expected_churners_captured":float(random_expected),
                  "lift_vs_random":float(captured/random_expected) if random_expected else None,"assumed_cost":CONTACT_COST,
                  "assumed_retained_value":RETAINED_VALUE,"assumed_save_rate":SAVE_RATE,"simulated_net_value":float(benefit),
                  "simulated_net_value_random_expected":float(random_expected*RETAINED_VALUE*SAVE_RATE-n_contact*CONTACT_COST),
                  "interpretation":"Retained value and save rate are assumptions; observed test churn is not evidence that contact causes retention."}
    (ROOT / "reports" / "intervention_simulation.json").write_text(json.dumps(intervention,indent=2),encoding="utf-8")

    # Fold variation on the training set and a calibration plot on the final test set.
    fold_ap=cross_val_score(clone(selected),X_train,y_train,cv=cv,scoring="average_precision",n_jobs=-1)
    (ROOT / "reports" / "robustness.json").write_text(json.dumps({"training_fold_average_precision":fold_ap.tolist(),"mean":float(fold_ap.mean()),"std":float(fold_ap.std()),"min":float(fold_ap.min()),"max":float(fold_ap.max())},indent=2),encoding="utf-8")
    detail = {"data_quality_audit": json.loads(audit.to_json(orient="records")),
              "model_comparison": json.loads(comparison.to_json(orient="records")),
              "threshold_metrics": json.loads(threshold_df.to_json(orient="records")),
              "permutation_importance": json.loads(importance.to_json(orient="records")),
              "individual_explanations": json.loads(local.to_json(orient="records")),
              "test_customer_risk_scores": json.loads(risk.drop(columns="test_index").to_json(orient="records"))}
    (ROOT / "reports" / "analysis_details.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")

    fig,ax=plt.subplots(figsize=(7,5)); ax.plot(importance.head(12).importance_mean,importance.head(12).feature,"o",color="#2b7778"); ax.axvline(0,color="black",lw=.8)
    ax.set(title="Permutation importance (holdout average precision)",xlabel="Decrease in average precision when permuted",ylabel="Input feature"); save_fig(fig,"model_permutation_importance.png")
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot(threshold_df.threshold,threshold_df.precision,label="Precision"); ax.plot(threshold_df.threshold,threshold_df.recall,label="Recall"); ax.axvline(threshold,color="#d05a47",ls="--",label=f"Selected {threshold:.2f}"); ax.set(xlabel="Threshold",ylabel="Score",ylim=(0,1),title="Holdout precision and recall by threshold"); ax.legend(); save_fig(fig,"model_thresholds.png")
    frac_mean, frac_pred=calibration_curve(y_test,p_test,n_bins=8,strategy="quantile")
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot(frac_pred,frac_mean,"o-",label="Model"); ax.plot([0,1],[0,1],"--",color="gray",label="Ideal"); ax.set(xlabel="Mean predicted probability",ylabel="Observed churn frequency",title="Holdout calibration"); ax.legend(); save_fig(fig,"model_calibration.png")
    cm=confusion_matrix(y_test,pred_test)
    fig,ax=plt.subplots(figsize=(5.5,4.5)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,ax=ax); ax.set(xlabel="Predicted",ylabel="Observed",title=f"Holdout confusion matrix at {threshold:.2f}"); save_fig(fig,"model_confusion_matrix.png")

    # Save the complete fitted preprocessing/model pipeline and prediction metadata.
    joblib.dump({"pipeline":selected,"threshold":threshold,"feature_engineering":ChurnFeatureEngineer(),"metadata":{"seed":SEED,"target":TARGET,"excluded_identifier":ID_COL}},MODEL_PATH)
    create_report(df, checks, comparison, metrics, intervention, importance, threshold_df, local, fold_ap)
    versions={"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"matplotlib":matplotlib.__version__,"seaborn":sns.__version__,"joblib":joblib.__version__}
    (ROOT / "reports" / "package_versions.json").write_text(json.dumps(versions,indent=2),encoding="utf-8")
    build_notebook()
    print(json.dumps({"rows":len(df),"target_rate":float(y.mean()),"selected_model":selected_name,"metrics":metrics,"intervention":intervention,"outputs":[str(MODEL_PATH),str(ROOT/'churn_business_report.pdf'),str(ROOT/'notebooks'/'churn_analysis.ipynb')]},indent=2))


# %% [markdown]
# ## Explainability, Retention Simulation, and Report

# %%
def create_report(df, checks, comparison, metrics, intervention, importance, threshold_df, local, fold_ap):
    out=ROOT/"churn_business_report.pdf"
    with PdfPages(out) as pdf:
        def page(title, lines, image=None):
            fig=plt.figure(figsize=(8.27,11.69)); fig.suptitle(title,fontsize=18,fontweight="bold",y=.97)
            if image:
                ax=fig.add_axes([.10,.39,.80,.48]); ax.imshow(plt.imread(FIG_DIR/image)); ax.axis("off")
                y=.34
            else: y=.86
            fig.text(.10,y,"\n".join(lines),ha="left",va="top",fontsize=10.2,linespacing=1.45,wrap=True)
            pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        churn=int(df[TARGET].sum()); total=len(df); rate=churn/total
        page("Customer Churn Risk and Retention Analysis",[
            f"Prepared: 2026-10-20 | Rows: {total:,} | Churn rate: {rate:.1%} ({churn:,}/{total:,})",
            "Objective: rank customers for a constrained retention contact list and explain the resulting risk signals.",
            "Data is a single labeled snapshot. Associations are predictive, not causal; no intervention outcomes are supplied.",
            "Selected model: "+metrics["model"],"Evaluation uses a stratified 75/25 split (seed 42); the test set was held out during tuning.",
            "Business assumptions: contact cost $8; value of a retained customer $180; assumed save rate 25%; capacity 20%."],"eda_contract_churn.png")
        missing=df.isna().sum(); missing_lines=[f"Missing values: {', '.join(f'{c}={int(v)}' for c,v in missing.items() if v)}.",
            f"Exact duplicate rows: {checks['duplicate_rows']}; repeated customer IDs: {checks['duplicate_customer_ids']}.",
            "Quality checks (out-of-range / invalid counts): "+", ".join(f"{k}={v}" for k,v in checks.items() if k not in ("duplicate_rows","duplicate_customer_ids")),
            "Cleaning: no rows removed; raw missing values retained in cleaned CSV. Numeric and categorical imputation is fitted within each training fold.",
            "Outliers were reviewed with robust plots and ranges, not trimmed. High usage/charge values can be legitimate; confirm with source-system rules before correction.",
            "customer_id is excluded from all predictive features. Churn: 86.3% positive, so accuracy and majority-class baselines are misleading."]
        page("1. Data Audit and EDA",missing_lines,"eda_charges.png")
        page("2. EDA: Customer Signals",[
            "The figures summarize contract, tenure, charges, satisfaction, complaints, and late payments. Payment method, usage/charge, and contract/service plots are included in the notebook figure set.",
            "Observed churn was 91.2% for month-to-month contracts and about 79% for one- and two-year contracts. Median satisfaction was 6.7 among churners and 7.3 among non-churners; median monthly charge was $72.78 vs $66.09.",
            "Churn rose to 93% for customers with 2+ late payments, while complaint counts of 3+ were high-risk but sparse. Treat these small-slice rates cautiously.",
            "Interpret subgroup differences as descriptive associations; small groups and the high churn prevalence can make raw rates unstable.",
            "Engineered fields: average charge per tenure month, payment-risk score, complaint intensity, service friction, low-satisfaction flag, high-cost/low-usage ratio, tenure-charge interaction, and month-to-month indicator.",
            "All engineered fields use customer attributes available before the prediction cycle; the target and identifier are never inputs.",
            "Rationale: encode accumulated burden, payment friction, service dissatisfaction, and cost/engagement mismatch without using future outcomes."],"eda_satisfaction.png")
        comparison_lines=[f"{r.model}: baseline CV AP {r.baseline_cv_ap_mean:.3f} +/- {r.baseline_cv_ap_std:.3f}; tuned CV AP {r.tuned_cv_ap_mean:.3f}. Params: {r.best_params}" for r in comparison.itertuples()]
        comparison_lines += ["Search: 4-fold stratified CV; small explicit grids; average precision optimizes positive-class ranking under imbalance.",
            f"Operating threshold {metrics['threshold']:.3f} was selected from training-only out-of-fold predictions by maximizing assumed net value under the 20% contact cap.",
            "Treatment: class weights for Logistic Regression and Random Forest; HistGradientBoosting used native unweighted fitting. No resampling; thus no synthetic observations cross folds."]
        page("3. Modeling and Selection",comparison_lines,"model_permutation_importance.png")
        eval_lines=[f"Holdout: precision {metrics['precision']:.3f}; recall {metrics['recall']:.3f}; F1 {metrics['f1']:.3f}; ROC-AUC {metrics['roc_auc']:.3f}; PR-AUC {metrics['pr_auc']:.3f}; Brier {metrics['brier_score']:.3f}.",
            f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}.",
            "Accuracy alone can look strong when the majority class dominates and does not measure ranking or the value of a limited contact list.",
            f"Four-fold training AP variation: mean {fold_ap.mean():.3f}, SD {fold_ap.std():.3f}, range {fold_ap.min():.3f}-{fold_ap.max():.3f}.",
            "Calibration curve and Brier score assess probability reliability; imperfect calibration is possible, especially in small slices. Threshold and probability should be revalidated on a later time period."]
        page("4. Holdout Evaluation and Robustness",eval_lines,"model_thresholds.png")
        page("5. Retention Simulation and Explainability",[
            f"Top 20%: {intervention['top_20pct_contacts']} test customers contacted; {intervention['churners_captured']} of {intervention['actual_test_churners']} observed churners captured ({intervention['capture_rate']:.1%}); random expectation {intervention['random_expected_churners_captured']:.1f}; lift {intervention['lift_vs_random']:.2f}x.",
            f"Illustrative expected net value: ${intervention['simulated_net_value']:,.0f}; random expected ${intervention['simulated_net_value_random_expected']:,.0f}. These calculations assume a 25% save rate and $180 retained value; actual treatment effect is unknown.",
            "Permutation importance measures how much holdout average precision falls after shuffling each input. It describes predictive reliance, not causal impact.",
            "Three customer-level explanations are provided as one-feature-at-a-time replacement effects relative to training-set typical values. These are local model sensitivities, not additive SHAP values or causal counterfactuals.",
            "Use the ranked list for a measured pilot with randomized treatment/control assignment before claiming incremental retention value."],"model_calibration.png")
        top_features=", ".join(importance.head(5).feature.tolist())
        page("6. Recommendations and Limitations",[
            f"Highest permutation signals in this run: {top_features}. Confirm the direction and stability by subgroup and future cohorts before operational changes.",
            "1. Test proactive contract renewal offers for high-risk month-to-month customers; measure incremental retention in a randomized holdout.",
            "2. Route customers with repeated complaints or weak satisfaction to a service recovery review, with issue-resolution tracking.",
            "3. Offer payment-method or billing support to customers with late-payment signals; avoid punitive treatment and assess affordability.",
            "4. Review high-charge / low-usage accounts for plan fit and transparent downgrade options.",
            "5. Monitor risk calibration, contact capacity, subgroup performance, and realized lift each cycle; retrain only after data quality and drift checks.",
            "Limitations: small sample, very high churn prevalence, one snapshot, possible selection artifacts, no causal treatment data, and hypothetical financial values. Validate with temporally separated data before production use."])


def build_notebook():
    import nbformat
    source=Path(__file__).read_text(encoding="utf-8") if "__file__" in globals() else (ROOT/"src"/"run_analysis.py").read_text(encoding="utf-8")
    cells=[]; current=[]; markdown=False
    for line in source.splitlines():
        if line.startswith("# %% [markdown]"):
            if current: cells.append((markdown,"\n".join(current).strip())); current=[]
            markdown=True
        elif line.startswith("# %%"):
            if current: cells.append((markdown,"\n".join(current).strip())); current=[]
            markdown=False
        else: current.append(line[2:] if markdown and line.startswith("# ") else ("" if markdown and line == "#" else line))
    if current: cells.append((markdown,"\n".join(current).strip()))
    nb=nbformat.v4.new_notebook(); nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":platform.python_version()}}
    nb.cells=[]
    for is_md,body in cells:
        if not body: continue
        if is_md: nb.cells.append(nbformat.v4.new_markdown_cell(body))
        else: nb.cells.append(nbformat.v4.new_code_cell(body))
    notebook_dir = ROOT / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb,notebook_dir/"churn_analysis.ipynb")


if __name__ == "__main__":
    main()
