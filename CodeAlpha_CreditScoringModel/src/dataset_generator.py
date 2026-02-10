import numpy as np
import pandas as pd
import os

def generate_dataset(n=20000, seed=42):
    np.random.seed(seed)

    age = np.random.randint(21, 65, n)
    monthly_income = np.random.lognormal(mean=10, sigma=0.4, size=n)
    employment_tenure = np.random.randint(0, 360, n)
    income_variance = np.random.uniform(0.05, 0.6, n)

    total_debt = monthly_income * np.random.uniform(2, 10, n)
    credit_utilization = np.clip(np.random.beta(2, 5, n), 0, 1)
    num_active_loans = np.random.randint(0, 8, n)

    on_time_payment_rate = np.random.beta(5, 2, n)
    missed_payment_count = np.random.poisson(1.2, n)

    savings_balance = monthly_income * np.random.uniform(0, 6, n)
    monthly_expenses = monthly_income * np.random.uniform(0.4, 0.9, n)

    credit_dependency_ratio = credit_utilization
    payment_discipline_momentum = np.random.normal(0, 0.15, n)
    financial_shock_resilience_score = savings_balance / (3 * monthly_expenses + 1)

    risk_score = (
        0.35 * credit_utilization +
        0.25 * credit_dependency_ratio +
        0.20 * (missed_payment_count / (missed_payment_count.max() + 1)) -
        0.30 * financial_shock_resilience_score -
        0.20 * payment_discipline_momentum
    )

    risk_score += np.random.normal(0, 0.1, n)
    creditworthy = (risk_score < np.percentile(risk_score, 70)).astype(int)

    return pd.DataFrame({
        "age": age,
        "monthly_income": monthly_income,
        "employment_tenure_months": employment_tenure,
        "income_variance_6m": income_variance,
        "total_debt": total_debt,
        "credit_utilization": credit_utilization,
        "num_active_loans": num_active_loans,
        "credit_dependency_ratio": credit_dependency_ratio,
        "on_time_payment_rate_6m": on_time_payment_rate,
        "missed_payment_count": missed_payment_count,
        "payment_discipline_momentum": payment_discipline_momentum,
        "savings_balance": savings_balance,
        "monthly_expenses": monthly_expenses,
        "financial_shock_resilience_score": financial_shock_resilience_score,
        "creditworthy": creditworthy
    })

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/hybrid_credit_scoring_dataset.csv", index=False)
    print("Dataset generated.")
