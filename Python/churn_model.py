"""
Customer Churn / Propensity Model
=====================================
Trains and rigorously validates two classifiers (Logistic Regression --
an interpretable "scorecard"-style model directly analogous to credit
scoring practice, and Random Forest -- a stronger but less interpretable
comparison) to predict customer churn from account and engagement
features.

This is the supervised-modelling counterpart to the unsupervised
clustering project elsewhere in this portfolio: here the target (churn)
is known and the model is validated against held-out ground truth with
standard classification metrics, not just descriptive segmentation.

Note on data: real customer/account data is confidential. This runs on a
synthetic dataset built with a genuine but IMPERFECT signal (substantial
irreducible noise is deliberately included in the data-generating process
-- see generate_data.py) so model performance here reflects an honest,
realistic ceiling rather than an artificially easy problem inflating the
reported metrics.

Run: python3 churn_model.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score, confusion_matrix,
                              classification_report, brier_score_loss)
from sklearn.calibration import calibration_curve
import os
import json

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bank_customers.csv")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
COLORS = ["#2E86AB", "#4CAF50", "#D62839", "#E9A23B"]

df = pd.read_csv(DATA_PATH)
feature_cols = ["age", "tenure_months", "num_products", "has_credit_card", "is_active_member",
                 "estimated_monthly_salary_zmw", "balance_zmw", "credit_score",
                 "num_complaints_last_year", "avg_monthly_transactions", "digital_engagement_score"]
X = df[feature_cols]
y = df["churned"]

print(f"Dataset: {len(df)} customers, churn rate {y.mean()*100:.1f}%")

# ---------------------------------------------------------------
# 1. Train/test split, stratified to preserve churn rate in both sets
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print(f"Train: {len(X_train)} ({y_train.mean()*100:.1f}% churn) | Test: {len(X_test)} ({y_test.mean()*100:.1f}% churn)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 2. Logistic Regression -- the interpretable "scorecard" model
# ---------------------------------------------------------------
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
logreg_auc = roc_auc_score(y_test, logreg_proba)

# 5-fold CV on training set to check the AUC isn't a lucky single split
cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5, scoring="roc_auc")
print(f"\nLogistic Regression: test AUC = {logreg_auc:.3f} | 5-fold CV AUC = {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# ---------------------------------------------------------------
# 3. Random Forest -- comparison model
# ---------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
rf.fit(X_train, y_train)  # tree models don't need scaling
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
print(f"Random Forest:       test AUC = {rf_auc:.3f} | 5-fold CV AUC = {rf_cv_scores.mean():.3f} +/- {rf_cv_scores.std():.3f}")

# ---------------------------------------------------------------
# 4. ROC curves
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, proba, auc, color in [("Logistic Regression", logreg_proba, logreg_auc, COLORS[0]),
                                  ("Random Forest", rf_proba, rf_auc, COLORS[1])]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve: Churn Prediction Models")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "roc_curve.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Precision-Recall curve (more informative than ROC under class imbalance)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, proba, color in [("Logistic Regression", logreg_proba, COLORS[0]), ("Random Forest", rf_proba, COLORS[1])]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)
ax.axhline(y_test.mean(), linestyle="--", color="gray", label=f"Baseline (churn rate={y_test.mean():.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve: Churn Prediction Models")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "precision_recall_curve.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Calibration check: are predicted probabilities trustworthy, not
#    just well-RANKED? (A model can have good AUC but badly miscalibrated
#    probabilities -- worth checking separately, since a propensity score
#    used for business decisions needs to mean what it says.)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, proba, color in [("Logistic Regression", logreg_proba, COLORS[0]), ("Random Forest", rf_proba, COLORS[1])]:
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", color=color, label=name)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed churn rate")
ax.set_title("Calibration Curve")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "calibration_curve.png"), dpi=150)
plt.close()

brier_logreg = brier_score_loss(y_test, logreg_proba)
brier_rf = brier_score_loss(y_test, rf_proba)
print(f"\nBrier score (lower=better calibrated): Logistic={brier_logreg:.4f} | RF={brier_rf:.4f}")

# ---------------------------------------------------------------
# 7. Decile / lift analysis -- the standard propensity-model business
#    view: if we contact the top N% highest-risk customers, how much of
#    total churn do we actually capture?
# ---------------------------------------------------------------
results_df = X_test.copy()
results_df["actual_churn"] = y_test.values
results_df["predicted_proba"] = logreg_proba  # use the interpretable model for the business-facing view
results_df["decile"] = pd.qcut(results_df["predicted_proba"], 10, labels=False, duplicates="drop")
results_df["decile"] = 9 - results_df["decile"]  # 0 = highest risk

decile_summary = results_df.groupby("decile").agg(
    n_customers=("actual_churn", "count"),
    n_churned=("actual_churn", "sum"),
    avg_predicted_proba=("predicted_proba", "mean"),
).reset_index()
decile_summary["churn_rate"] = decile_summary["n_churned"] / decile_summary["n_customers"]
decile_summary["cumulative_churn_captured"] = decile_summary["n_churned"].cumsum() / decile_summary["n_churned"].sum()
decile_summary["pct_customers_contacted"] = (decile_summary["decile"] + 1) / decile_summary["decile"].nunique()

top3_capture = decile_summary[decile_summary.decile <= 2]["n_churned"].sum() / decile_summary["n_churned"].sum()
print(f"\nContacting the top 3 deciles (30% of customers) captures {top3_capture*100:.1f}% of actual churners")

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(decile_summary["decile"], decile_summary["churn_rate"]*100, color=COLORS[0], alpha=0.75)
ax1.set_xlabel("Risk decile (0 = highest predicted risk)")
ax1.set_ylabel("Observed churn rate in decile (%)", color=COLORS[0])
ax1.axhline(y_test.mean()*100, color="gray", linestyle="--", label=f"Overall churn rate ({y_test.mean()*100:.1f}%)")
ax1.legend(loc="upper right")
ax1.set_title("Decile Analysis: Model Successfully Ranks Risk")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "decile_analysis.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(decile_summary["pct_customers_contacted"]*100, decile_summary["cumulative_churn_captured"]*100,
        marker="o", color=COLORS[2], linewidth=2, label="Model")
ax.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random contact")
ax.set_xlabel("% of customers contacted (highest risk first)")
ax.set_ylabel("% of actual churners captured")
ax.set_title("Lift Chart: Value of Risk-Ranked Targeting")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "lift_chart.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 8. Feature importance / interpretability
# ---------------------------------------------------------------
logreg_coefs = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": logreg.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)
print("\nLogistic Regression coefficients (standardized features, so directly comparable):")
print(logreg_coefs.to_string(index=False))

rf_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
c = [COLORS[1] if v > 0 else COLORS[2] for v in logreg_coefs["coefficient"]]
axes[0].barh(logreg_coefs["feature"], logreg_coefs["coefficient"], color=c)
axes[0].set_title("Logistic Regression Coefficients\n(positive = increases churn risk)")
axes[0].axvline(0, color="black", linewidth=0.5)
axes[0].invert_yaxis()

axes[1].barh(rf_importance["feature"], rf_importance["importance"], color=COLORS[0])
axes[1].set_title("Random Forest Feature Importance")
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 9. Confusion matrix at a business-relevant threshold
# ---------------------------------------------------------------
THRESHOLD = 0.35  # chosen to prioritize recall given asymmetric cost (see findings.md)
y_pred_business = (logreg_proba >= THRESHOLD).astype(int)
cm = confusion_matrix(y_test, y_pred_business)
print(f"\nConfusion matrix at threshold={THRESHOLD} (Logistic Regression):")
print(cm)
report = classification_report(y_test, y_pred_business, output_dict=True)

# ---------------------------------------------------------------
# 10. Save outputs
# ---------------------------------------------------------------
results_df.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "test_predictions.csv"), index=False)
decile_summary.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "decile_summary.csv"), index=False)
logreg_coefs.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "logreg_coefficients.csv"), index=False)

summary = {
    "dataset_size": len(df),
    "churn_rate_pct": round(y.mean()*100, 2),
    "logreg_test_auc": round(logreg_auc, 4),
    "logreg_cv_auc_mean": round(cv_scores.mean(), 4),
    "logreg_cv_auc_std": round(cv_scores.std(), 4),
    "rf_test_auc": round(rf_auc, 4),
    "rf_cv_auc_mean": round(rf_cv_scores.mean(), 4),
    "rf_cv_auc_std": round(rf_cv_scores.std(), 4),
    "brier_score_logreg": round(brier_logreg, 4),
    "brier_score_rf": round(brier_rf, 4),
    "top3decile_churn_capture_pct": round(top3_capture*100, 1),
    "business_threshold": THRESHOLD,
    "confusion_matrix_at_threshold": cm.tolist(),
    "precision_recall_at_threshold": {
        "precision": round(report["1"]["precision"], 3),
        "recall": round(report["1"]["recall"], 3),
        "f1": round(report["1"]["f1-score"], 3),
    },
}
with open(os.path.join(os.path.dirname(__file__), "..", "results", "stats_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nCharts and results saved to results/")
