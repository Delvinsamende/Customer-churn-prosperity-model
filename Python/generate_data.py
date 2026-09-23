"""
Generates a synthetic retail-bank customer dataset with a genuine,
learnable (but not perfectly separable) churn signal -- realistic
irreducible noise is included deliberately, so model performance in
this project reflects honest predictive limits rather than an
artificially easy synthetic problem.

Run: python3 generate_data.py
"""
import numpy as np
import pandas as pd
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bank_customers.csv")


def main():
    rng = np.random.default_rng(2026)
    n = 3000

    age = np.clip(rng.normal(40, 12, n), 18, 80).round(0)
    tenure_months = np.clip(rng.normal(48, 30, n), 1, 180).round(0)
    num_products = rng.choice([1, 2, 3, 4], size=n, p=[0.45, 0.35, 0.15, 0.05])
    has_credit_card = rng.choice([0, 1], size=n, p=[0.3, 0.7])
    is_active_member = rng.choice([0, 1], size=n, p=[0.4, 0.6])
    estimated_salary = np.clip(rng.normal(9500, 4500, n), 1500, 40000).round(2)
    balance = np.clip(rng.normal(15000, 12000, n), 0, None).round(2)
    credit_score = np.clip(rng.normal(650, 80, n), 300, 850).round(0)
    num_complaints_last_year = rng.poisson(0.4, n)
    avg_monthly_transactions = np.clip(rng.normal(12, 6, n), 0, None).round(0)
    digital_engagement_score = np.clip(rng.normal(55, 20, n), 0, 100).round(1)

    tenure_z = (tenure_months - tenure_months.mean()) / tenure_months.std()
    digital_z = (digital_engagement_score - digital_engagement_score.mean()) / digital_engagement_score.std()
    age_z = (age - age.mean()) / age.std()
    balance_z = (balance - balance.mean()) / balance.std()

    z = (
        -1.35
        - 0.55 * tenure_z
        + 0.55 * (num_products == 1).astype(int)
        - 0.45 * is_active_member
        + 0.35 * num_complaints_last_year
        - 0.40 * digital_z
        + 0.12 * age_z
        - 0.05 * balance_z
        + rng.normal(0, 0.8, n)  # irreducible noise -- keeps the problem realistically hard
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churned = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i+1:05d}" for i in range(n)],
        "age": age.astype(int),
        "tenure_months": tenure_months.astype(int),
        "num_products": num_products,
        "has_credit_card": has_credit_card,
        "is_active_member": is_active_member,
        "estimated_monthly_salary_zmw": estimated_salary,
        "balance_zmw": balance,
        "credit_score": credit_score.astype(int),
        "num_complaints_last_year": num_complaints_last_year,
        "avg_monthly_transactions": avg_monthly_transactions.astype(int),
        "digital_engagement_score": digital_engagement_score,
        "churned": churned,
    })
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Generated {len(df)} customer records, churn rate {df.churned.mean()*100:.1f}% -> {OUT_PATH}")


if __name__ == "__main__":
    main()
