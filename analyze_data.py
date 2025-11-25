import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

# Load results
df = pd.read_csv("batch_test_results.csv")

# 1. Overall Rates
total = len(df)
baseline_refusals = df['baseline_refused'].sum()
ablated_refusals = df['ablated_refused'].sum()

print(f"Total Prompts: {total}")
print(f"Baseline Refusal Rate: {baseline_refusals/total:.1%} ({baseline_refusals}/{total})")
print(f"Ablated Refusal Rate: {ablated_refusals/total:.1%} ({ablated_refusals}/{total})")

# 2. Category Breakdown
print("\n--- Breakdown by Category ---")
categories = df['category'].unique()
cat_stats = []
for cat in categories:
    sub = df[df['category'] == cat]
    n = len(sub)
    b_ref = sub['baseline_refused'].sum()
    a_ref = sub['ablated_refused'].sum()
    cat_stats.append({
        "Category": cat,
        "N": n,
        "Baseline Refusal": f"{b_ref/n:.1%}",
        "Ablated Refusal": f"{a_ref/n:.1%}",
        "Success Rate": f"{1 - a_ref/n:.1%}"
    })

df_cat = pd.DataFrame(cat_stats)
print(df_cat.to_string(index=False))

# 3. Statistical Significance
# Contingency Table
#           Refused | Complied
# Baseline    A     |    B
# Ablated     C     |    D

A = baseline_refusals
B = total - baseline_refusals
C = ablated_refusals
D = total - ablated_refusals

table = [[A, B], [C, D]]
chi2, p, dof, expected = chi2_contingency(table)

print(f"\n--- Statistical Validation ---")
print(f"Contingency Table: {table}")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
if p < 0.05:
    print("Result is Statistically Significant (p < 0.05)")
else:
    print("Result is NOT Statistically Significant")

# 4. Find Examples
print("\n--- Examples ---")
# Held Refusals (Ablated Refused = True)
held = df[df['ablated_refused'] == True]
print(f"Held Refusals (N={len(held)}):")
if not held.empty:
    print(held[['id', 'prompt', 'ablated_response']].head(3).to_string())
else:
    print("None found.")

# Broke Refusals (Baseline=True, Ablated=False)
broke = df[(df['baseline_refused'] == True) & (df['ablated_refused'] == False)]
print(f"\nBroke Refusals (N={len(broke)}):")
if not broke.empty:
    print(broke[['id', 'prompt', 'ablated_response']].sample(3).to_string())

